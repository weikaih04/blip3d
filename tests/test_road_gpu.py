"""GPU parity of blip3d ROAD against the v12 copy. Needs one free GPU (~12 GB) and nvcc for the matcher JIT.
Run: PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=<free gpu> python tests/test_road_gpu.py
Checks: (1) the Uni3D-g teacher forward equals v12's Uni3DTeacher bit-for-bit under bf16 autocast; (2) the GPU
Hungarian matcher finds the optimal assignment (scipy reference on the same cosine cost); (3) RoadHead.loss equals the
v12 flow_heads composition on real teacher tokens, gate closed and open; (4) the hook on the real TRELLIS.2 SS flow
stashes blocks[9]'s (B, 4096, 1536) output and the loss backpropagates into the flow and the projector."""
import os
import sys

import numpy as np
import torch

OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NEW)
from blip3d.utils import backend  # noqa: E402
backend.setup("train")
from blip3d.utils.paths import get_paths  # noqa: E402
sys.path.insert(1, OLD)
from blip3d.losses import road as R  # noqa: E402
from blip3d.models.road.teacher import uni3d_ckpt_path  # noqa: E402
import trellis2_blip3o.road as OR  # noqa: E402
from trellis2_blip3o.road.teacher import Uni3DTeacher as OldTeacher  # noqa: E402

FAIL = []
dev = torch.device("cuda")


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


g = torch.Generator().manual_seed(0)
B = 2
coords = [torch.randint(0, 32, (n, 3), generator=g).numpy() for n in (1500, 12000)]
pts = torch.from_numpy(np.stack([OR.sample_align_points(c, R.N_POINTS) for c in coords])).to(dev)

# 1 · teacher tokens == v12
head = R.RoadHead().to(dev)
tn = head.teacher_tokens(pts)
old_t = OldTeacher(uni3d_ckpt_path()).to(dev)
with torch.autocast("cuda", dtype=torch.bfloat16):
    to = old_t(pts)
check("teacher tokens == v12", torch.equal(tn, to), f"shape={tuple(tn.shape)} dtype={tn.dtype}")
del old_t
torch.cuda.empty_cache()

# 2 · GPU matcher reaches the optimum
from scipy.optimize import linear_sum_assignment  # noqa: E402
import torch.nn.functional as F  # noqa: E402
m = head._ensure_matcher()
f1 = torch.randn(B, 512, 1024, device=dev)
f2 = torch.randn(B, 512, 1024, device=dev)
loss_gpu = m(f1, f2)
sim = torch.bmm(F.normalize(f1, dim=-1), F.normalize(f2, dim=-1).transpose(1, 2)).cpu().numpy()
ref = []
for b in range(B):
    r, c = linear_sum_assignment(1.0 - sim[b])
    ref.append(sim[b][r, c])
loss_ref = 1.0 - float(np.concatenate(ref).mean())
check("GPU matcher == scipy optimum", abs(loss_gpu.item() - loss_ref) < 1e-5, f"gpu={loss_gpu.item():.7f} ref={loss_ref:.7f}")

# 3 · RoadHead.loss == v12 composition on real teacher tokens (bf16 projector, as trained)
head = head.to(torch.bfloat16)
old_proj = OR.AlignMLP(1536, 2048, 1024).to(dev, torch.bfloat16)
old_proj.load_state_dict(head.projector.state_dict())
pool = torch.nn.AdaptiveAvgPool1d(512)
h = torch.randn(B, 4096, 1536, device=dev, dtype=torch.bfloat16)
for step in (999, 1000):
    head.step.fill_(step)
    tot, logs = head.loss(h, pts)
    ro = OR.road_alignment_losses(h, tn, old_proj, pool, m, step >= 1000)
    ref = 0.5 * ro["loss_proj"] + 0.1 * ro["loss_opt"]
    check(f"RoadHead.loss == v12 at micro-step {step}", torch.equal(tot, ref),
          f"proj={logs['road_proj'].item():.5f} opt={logs['road_opt'].item():.5f}")

# 4 · the tap on the real SS flow
from blip3d.models.towers import released_flow, to_bf16_keep_complex  # noqa: E402
flow = to_bf16_keep_complex(released_flow("ss", get_paths().trellis2_ckpt)).to(dev).train()
head.attach(flow); head.train()
x = torch.randn(B, flow.in_channels, *[flow.resolution] * 3, device=dev, dtype=torch.bfloat16)
cond = torch.randn(B, 77, 1024, device=dev, dtype=torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    v = flow(x, torch.full((B,), 500.0, device=dev), cond)
tap = head.pop_tap()
check("tap shape", tuple(tap.shape) == (B, 4096, 1536), str(tuple(tap.shape)))
tot, logs = head.loss(tap, pts)
tot.backward()   # only the tap path: blocks after 9 must stay out of the graph
check("grad reaches the projector and the tapped block",
      head.projector.layers[0].weight.grad is not None and any(p.grad is not None for p in flow.blocks[9].parameters())
      and all(p.grad is None for p in flow.blocks[10].parameters()))

print(f"\n{'ALL PASS' if not FAIL else 'FAILED: ' + ', '.join(FAIL)}")
sys.exit(1 if FAIL else 0)
