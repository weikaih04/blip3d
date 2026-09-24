"""GPU parity of blip3d.losses.unified on the real S3 model (needs one free GPU with ~40 GB; NOT run on the training
nodes). Run:
    CUDA_VISIBLE_DEVICES=<free gpu> PYTHONNOUSERSITE=1 python tests/test_unified_loss_gpu.py

The S3 17k checkpoint is loaded once into Blip3DUnified (training layout, fused attention, trainable) plus its three
connectors (bf16, trainable). One synthetic batch per modality (random latents on random 32^3 coords, random encoder
records) goes through
  * v12 ``flow_heads.compute_unified_geotex_loss`` (the old loss function, called on the SAME model object) and
  * BLIP3D ``unified_loss`` with ``compat_v12=True``,
from the same CUDA seed. Total loss, every log value and the gradients of a fixed sample of parameters (model and
connectors) must be bit-identical. The model itself is checked separately (G2 goldens); this isolates the loss.
Then the BLIP3D default (geo replays the tex drops) is compared with v12 run with ``joint_cond_drop=True``.
"""
import os
import sys

os.environ["GEOTEX_XATTN_FLASH"] = "1"
import torch  # noqa: E402

OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S3 = "/fsx/data/weikai.huang/runs/keep/v12/s3_unify_4n/checkpoint-17000"
sys.path.insert(0, NEW)
from blip3d.utils import backend  # noqa: E402
backend.setup("train")
sys.path.insert(1, OLD)
from trellis2.modules import sparse as sp  # noqa: E402
import trellis2_blip3o.flow_heads as FH  # noqa: E402
from trellis2_blip3o.loss import TRELLIS2FlowMatchingLoss  # noqa: E402
from blip3d.cond import assemble as A  # noqa: E402
from blip3d.losses import unified as U  # noqa: E402
from blip3d.models.unified.loading import load_unified  # noqa: E402
from blip3d.utils.paths import get_paths  # noqa: E402

assert torch.cuda.is_available(), "needs a GPU"
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAIL.append(name)


P = get_paths()
bundle = load_unified(S3, trellis2_ckpt=P.trellis2_ckpt, use_ema=False, device="cuda", connector_dtype=torch.bfloat16)
model, conns, views = bundle.model.train().requires_grad_(True), bundle.connectors, bundle.views
for c in conns.values():
    c.train().requires_grad_(True)
named = [(f"model.{n}", p) for n, p in model.named_parameters()] + \
        [(f"{lane}.{n}", p) for lane, c in conns.items() for n, p in c.named_parameters()]
g = torch.Generator().manual_seed(0)
probe_params = [named[i] for i in sorted(torch.randperm(len(named), generator=g)[:48].tolist())]
if os.environ.get("UNIFIED_TEST_LIGHT") == "1":
    # only the probed tensors carry gradients (their values do not depend on which other leaves require grad);
    # cuts ~8 GB of gradients and the activations of frozen branches, for a GPU shared with a training job
    for _, p in named:
        p.requires_grad_(False)
    for _, p in probe_params:
        p.requires_grad_(True)


def rec(mod, T, K, seed):
    g = torch.Generator().manual_seed(seed)
    r = {"qwen": torch.randn(T, 2048, generator=g).half(), "qwen_keep": torch.rand(T, generator=g) > 0.1, "modality": mod}
    if mod != "t":
        rc = torch.full((T, 2), -1.0); rc[4:4 + 256 * K] = torch.rand(256 * K, 2, generator=g); r["qwen_rc"] = rc
        r["dino"] = torch.randn(K * 1029, 1024, generator=g).half(); r["dino_keep"] = torch.ones(K * 1029, dtype=torch.bool)
        r["dino_views"] = torch.arange(K).repeat_interleave(1029)
        if mod == "im":
            qv = torch.full((T,), -1, dtype=torch.long)
            for k in range(K):
                qv[4 + 256 * k:4 + 256 * (k + 1)] = k
            r["qwen_views"] = qv
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in r.items()}


def targets(B, seed, tex_valid):
    g = torch.Generator().manual_seed(1000 + seed)
    cs = []
    for b in range(B):
        n = int(torch.randint(800, 3000, (1,), generator=g))
        xyz = torch.randperm(32 ** 3, generator=g)[:n].sort().values
        cs.append(torch.stack([torch.full((n,), b), xyz // 1024, (xyz // 32) % 32, xyz % 32], 1).int())
    coords = torch.cat(cs).cuda()
    N = coords.shape[0]
    shape = sp.SparseTensor(feats=torch.randn(N, 32, generator=g).cuda(), coords=coords)
    tv = torch.tensor(tex_valid)
    rows = torch.cat([torch.full((sl.stop - sl.start,), bool(tv[b])) for b, sl in enumerate(shape.layout)]).cuda()
    tex = sp.SparseTensor(feats=torch.randn(N, 32, generator=g).cuda() * rows[:, None].float(), coords=coords)
    ss = torch.randn(B, 8, 16, 16, 16, generator=g).to(torch.bfloat16).cuda()
    return {"shape": shape, "tex": tex, "ss": ss, "tex_valid": tv.cuda()}


def run_old(cb, tg, joint):
    FH.compute_unified_geotex_loss._reported = True
    FH.compute_unified_geotex_loss._probe_n = 0
    fuse = "dino" in cb
    loss, logs = FH.compute_unified_geotex_loss(
        unified_model=model, connector_geo=conns["geo"], connector_tex=conns["tex"],
        loss_fn_slat=TRELLIS2FlowMatchingLoss(t_schedule="uniform"),
        cond_hidden=cb["qwen"].to(torch.bfloat16), cond_key_mask=cb["qwen_keep"],
        target_shape_slat_512=tg["shape"], target_tex_slat_512=tg["tex"], tex_valid=tg["tex_valid"],
        dino_hidden=cb["dino"].to(torch.bfloat16) if fuse else None, dino_key_mask=cb["dino_keep"].bool() if fuse else None,
        dino_view_ids=cb.get("dino_views") if fuse else None, qwen_view_ids=cb.get("qwen_views") if fuse else None,
        qwen_img_rc=cb.get("qwen_rc") if fuse else None, dino_view_embed=views.table if fuse else None,
        mask_drop_prob=0.1, dino_drop_prob=0.3, qwen_drop_prob=0.1, cond_max_length=10240,
        p_corner=0.1, p_corner2=0.2, geo_loss_w=1.0, joint_cond_drop=joint,
        ss_flow_present=True, target_ss_latent=tg["ss"], connector_ss=conns["ss"],
        loss_fn_ss=TRELLIS2FlowMatchingLoss(t_schedule="logitNormal"), ss_loss_w=1.0, ss_dual=True,
        p_solo=0.0, p_lag=0.3, k0_lo=3, k0_hi=11, probe_every=200)
    return loss, logs


def run_new(cb, tg, compat):
    return U.unified_loss(model, conns, views, tg, cb, step=0, generator=None,
                          cfg=U.UnifiedLossCfg(compat_v12=compat), state=U.UnifiedLossState(reported=True))


def step(fn):
    for _, p in named:
        p.grad = None
    torch.manual_seed(1234); torch.cuda.manual_seed(1234)
    loss, logs = fn()
    loss.backward()
    torch.cuda.synchronize()
    return loss.detach().float().cpu(), {k: float(v) for k, v in logs.items()}, \
        {n: (p.grad.detach().float().cpu().clone() if p.grad is not None else None) for n, p in probe_params}


for mod, K, tv in (("i1", 1, [True, False]), ("im", 3, [True, True]), ("t", 0, [False, True])):
    B = 2
    cb = A.collate([rec(mod, T=(1050 if mod == "i1" else 1056 if mod == "im" else 60), K=K, seed=s) for s in range(B)],
                   device="cuda")
    tg = targets(B, 0, tv)
    for compat in (True, False):
        lo, go, gro = step(lambda: run_old(cb, tg, joint=not compat))
        ln, gn, grn = step(lambda: run_new(cb, tg, compat))
        ok_l = torch.equal(lo, ln)
        bad_logs = [k for k in go if gn.get(k) != go[k]]
        ok_g = all((a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                   for a, b in zip(gro.values(), grn.values()))
        check(f"{mod} {'compat_v12' if compat else 'fixed vs v12 joint_cond_drop'}", ok_l and not bad_logs and ok_g,
              f"loss {float(lo):.6f}/{float(ln):.6f} logs={'ok' if not bad_logs else bad_logs} "
              f"grads({len(probe_params)})={'ok' if ok_g else 'DIFF'}")

print(f"\n{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
