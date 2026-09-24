"""CPU parity of blip3d ROAD (models/road + losses/road) against the v12 copy (old repo, read-only).
Run: PYTHONNOUSERSITE=1 python tests/test_road_cpu.py        (ROAD_TEST_TEACHER=1 also builds the real Uni3D-g
teacher on CPU and strict-loads its checkpoint, ~4 GB RAM). The GPU matcher and the teacher forward are in
tests/test_road_gpu.py."""
import ast
import os
import sys

import numpy as np
import torch
import torch.nn as nn

OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEEP = "/fsx/data/weikai.huang/runs/keep/v12"
RUN300K = "/fsx/data/weikai.huang/runs/s3_ss_v12s2to300k"
sys.path.insert(0, NEW)
sys.path.insert(1, OLD)
from blip3d.losses import road as R  # noqa: E402
from blip3d.models import road as M  # noqa: E402
from blip3d.models.road import point_encoder as PE  # noqa: E402
import trellis2_blip3o.road as OR  # noqa: E402  (light: align_mlp, loss, points)
from trellis2_blip3o.road import point_encoder as OPE  # noqa: E402
from trellis2_blip3o.road import points as OPTS  # noqa: E402
from trellis2_blip3o.road.matcher_cpu import HungarianMatcherWithLoss  # noqa: E402  (stand-in matcher)

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def strip_comments(path):
    return [l for l in open(path).read().splitlines() if l.strip() and not l.lstrip().startswith("#")]


# 1 · vendored sources equal v12's copy (comments aside)
for f in ("align_mlp.py", "loss.py", "point_encoder.py", "uni3d.py"):
    check(f"source {f} == v12", strip_comments(f"{OLD}/trellis2_blip3o/road/{f}") ==
          strip_comments(f"{NEW}/blip3d/models/road/{f}"))


def consts_and_forward(path):
    tree = ast.parse(open(path).read())
    src = open(path).read()
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            out[node.targets[0].id] = node.value.value
        if isinstance(node, ast.ClassDef):
            out[node.name] = ast.get_source_segment(src, node)
        if isinstance(node, ast.FunctionDef) and node.name == "hungarian_gpu":
            out["hungarian_gpu"] = ast.get_source_segment(src, node).replace("_ext().", "ext_mod.")
    return out


o, n = consts_and_forward(f"{OLD}/trellis2_blip3o/road/matcher_gpu.py"), consts_and_forward(f"{NEW}/blip3d/models/road/matcher_gpu.py")
for k in ("CPP_STUB", "CUDA_SRC", "hungarian_gpu", "HungarianMatcherWithLossGPU"):
    check(f"matcher_gpu {k} == v12", o.get(k) is not None and o.get(k) == n.get(k))

# 2 · AlignMLP: same init, same forward (fp32 and bf16)
torch.manual_seed(0); a_new = M.AlignMLP(1536, 2048, 1024)
torch.manual_seed(0); a_old = OR.AlignMLP(1536, 2048, 1024)
check("AlignMLP init", all(torch.equal(p, q) for p, q in zip(a_new.state_dict().values(), a_old.state_dict().values())))
x = torch.randn(2, 64, 1536)
check("AlignMLP fp32 forward", torch.equal(a_new(x), a_old(x)))
check("AlignMLP bf16 forward", torch.equal(a_new.bfloat16()(x.bfloat16()), a_old.bfloat16()(x.bfloat16())))
a_new.float(); a_old.float()

# 3 · road_alignment_losses == v12 (opt off / on) with grads
mt = HungarianMatcherWithLoss()
pool = nn.AdaptiveAvgPool1d(16)
g = torch.Generator().manual_seed(1)
h = torch.randn(2, 64, 1536, generator=g)
teach = torch.randn(2, 17, 1024, generator=g)
for opt in (False, True):
    hn, ho = h.clone().requires_grad_(), h.clone().requires_grad_()
    rn = M.road_alignment_losses(hn, teach, a_new, pool, mt, opt)
    ro = OR.road_alignment_losses(ho, teach, a_old, pool, mt, opt)
    (rn["loss_proj"] + rn["loss_opt"]).backward(); (ro["loss_proj"] + ro["loss_opt"]).backward()
    check(f"alignment losses opt={opt}", torch.equal(rn["loss_proj"], ro["loss_proj"]) and torch.equal(rn["loss_opt"], ro["loss_opt"])
          and torch.equal(hn.grad, ho.grad), f"proj={rn['loss_proj'].item():.6f} opt={rn['loss_opt'].item():.6f}")

# 4 · RoadHead.loss == v12 composition (flow_heads.py:488-493), gate read-then-advance at 1000 micro-steps
torch.manual_seed(0); head = R.RoadHead(1536, token_count=16)
head.projector.load_state_dict(a_old.state_dict())
object.__setattr__(head, "_matcher", mt)
head.step.fill_(998)
seen = []
for _ in range(3):
    tot, logs = head.loss(h, torch.zeros(2, 10, 3), teacher_tokens=teach)
    ro = OR.road_alignment_losses(h, teach, a_old, pool, mt, int(head.step) - 1 >= 1000)
    ref = 0.5 * ro["loss_proj"] + 0.1 * ro["loss_opt"]
    seen.append((int(head.step), bool(logs["road_opt"] != 0)))
    check(f"RoadHead.loss at step {int(head.step) - 1}", torch.equal(tot, ref), f"total={tot.item():.6f}")
check("gate opens at micro-step 1000", seen == [(999, False), (1000, False), (1001, True)], str(seen))
check("step buffer is int64 and persistent", head.step.dtype == torch.long and "step" in head.state_dict())
check("step survives a bf16 cast", head.to(torch.bfloat16).step.dtype == torch.long)
head.float()

# 5 · hook: stashes blocks[9] output in training only
class Blk(nn.Module):
    def __init__(self, i):
        super().__init__(); self.i = i

    def forward(self, x):
        return x + self.i


class Flow(nn.Module):
    def __init__(self):
        super().__init__(); self.blocks = nn.ModuleList([Blk(i) for i in range(30)])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x


flow = Flow(); head.attach(flow); head.train()
flow(torch.zeros(1, 4)); t = head.pop_tap()
check("hook stashes blocks[9] output", torch.equal(t, torch.full((1, 4), float(sum(range(10))))))
head.eval(); flow(torch.zeros(1, 4))
try:
    head.pop_tap(); check("no stash in eval", False)
except RuntimeError:
    check("no stash in eval", True)
head.attach(flow); head.train(); flow(torch.zeros(1, 4)); flow(torch.zeros(1, 4))
check("re-attach keeps one hook", len(flow.blocks[9]._forward_hooks) == 1)

# 6 · point encoder pieces == v12
pts = torch.rand(2, 2000, 3, generator=g) - 0.5
check("farthest_point_sample", torch.equal(PE.farthest_point_sample(pts, 64), OPE.farthest_point_sample(pts, 64)))
check("knn_point", torch.equal(PE.knn_point(8, pts, pts[:, :64]), OPE.knn_point(8, pts, pts[:, :64])))
cn, gn = PE.Group(64, 8)(pts, torch.ones_like(pts)); co, go = OPE.Group(64, 8)(pts, torch.ones_like(pts))
check("Group", torch.equal(cn, co) and torch.equal(gn, go))
torch.manual_seed(3); en = PE.Encoder(512).eval()
torch.manual_seed(3); eo = OPE.Encoder(512).eval()
check("Encoder", torch.equal(en(gn), eo(go)))

# 7 · align points: shapes, range, replacement, seeded draw == v12's draw
coords = np.random.default_rng(0).integers(0, 32, (300, 3)).astype(np.uint8)
p = M.sample_align_points(coords, 10000)
check("align points shape/dtype/range", p.shape == (10000, 3) and p.dtype == np.float32 and p.min() >= -0.5 and p.max() <= 0.5)
p1 = M.sample_align_points(coords, 10000, rng=np.random.default_rng(7))
_orig = OPTS.np.random.default_rng
OPTS.np.random.default_rng = lambda *a: _orig(7)
p2 = OPTS.sample_align_points(coords, 10000)
OPTS.np.random.default_rng = _orig
check("seeded draw == v12 draw", np.array_equal(p1, p2))
p3 = M.sample_align_points(np.random.default_rng(1).integers(0, 32, (20000, 3)), 10000, rng=np.random.default_rng(0))
check("no replacement when the cloud is long", len(np.unique(p3, axis=0)) <= 10000 and p3.shape == (10000, 3))

# 8 · v12 checkpoints load strictly into RoadHead
from safetensors import safe_open  # noqa: E402


def road_keys(path):
    with safe_open(path, "pt") as f:
        return {k: f.get_tensor(k) for k in f.keys() if k.startswith("repa_projector.") or k == "_road_step"}


for name, path, step in (("s2_ss 106000 raw", f"{KEEP}/s2_ss/checkpoint-106000/model.safetensors", 402000),
                         ("s2_ss 106000 ema overlay", f"{KEEP}/s2_ss/checkpoint-106000_ema/model.safetensors", 402000)):
    sd = R.v12_road_state(road_keys(path))
    hh = R.RoadHead(); res = hh.load_state_dict(sd, strict=True)
    check(f"strict load {name}", not res.missing_keys and not res.unexpected_keys and int(hh.step) == step,
          f"keys={len(sd)} step={int(hh.step)} dtype={sd['projector.layers.0.weight'].dtype}")
with safe_open(f"{KEEP}/s2_ss/checkpoint-106000/ema.safetensors", "pt") as f:
    ek = sorted(k for k in f.keys() if k.startswith("repa_projector."))
check("EMA carries the 6 projector tensors", len(ek) == 6, str(ek[:2]))
cks = sorted((d for d in os.listdir(RUN300K) if d.startswith("checkpoint-") and d[11:].isdigit()), key=lambda d: int(d[11:]))
if cks:
    sd = R.v12_road_state(road_keys(f"{RUN300K}/{cks[-1]}/model.safetensors"))
    hh = R.RoadHead(); res = hh.load_state_dict(sd, strict=True)
    check(f"strict load 300k run {cks[-1]}", not res.missing_keys and not res.unexpected_keys, f"step={int(hh.step)}")

# 9 · teacher checkpoint resolution (and optionally the real strict load)
from blip3d.models.road.teacher import uni3d_ckpt_path  # noqa: E402
path = uni3d_ckpt_path()
check("Uni3D-g ckpt resolves to the pinned snapshot", os.path.isfile(path) and "3d8233b" in path, path)
if os.environ.get("ROAD_TEST_TEACHER") == "1":
    from blip3d.models.road import Uni3DTeacher  # noqa: E402
    t = Uni3DTeacher()
    n_params = sum(p.numel() for p in t.parameters())
    check("teacher strict load + frozen", all(not p.requires_grad for p in t.parameters()), f"params={n_params / 1e6:.1f}M")
    check("teacher is not a child of RoadHead", all(not k.startswith("_teacher") for k in R.RoadHead().state_dict()))

print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
