"""CPU parity of blip3d.losses.unified against v12 flow_heads (old repo, read-only).

Run (CPU only, never touches a GPU):
    CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 python tests/test_unified_loss_cpu.py

1. sample_timestep_pairs / sample_timestep_triples: bit-exact vs v12 for many B, seeds and knob sets; a passed
   generator reproduces the default-stream draw with the same seed.
2. row / voxel balanced MSE vs v12 (single process), including empty selections.
3. End-to-end loss with a deterministic stub model (same interface as Blip3DUnified) and the REAL S3 connectors of
   runs/keep/v12/s3_unify_4n/checkpoint-17000: the old compute_unified_geotex_loss and the new unified_loss are run
   for 3 micro-steps from the same seed, and every model input (per-lane conditions, SS mask, noisy states,
   timesteps, ss_read_on), the total loss, every log value and every gradient must be bit-identical.
   compat_v12 is compared with v12 (joint_cond_drop False); the BLIP3D default with v12 run with joint_cond_drop True.
"""
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["GEOTEX_XATTN_FLASH"] = "1"          # the v12 S3 launcher value (flow_heads._mask_or_none)
import warnings  # noqa: E402

warnings.filterwarnings("ignore", message=".*CUDA is not available.*")
warnings.filterwarnings("ignore", message=".*autocast.*")

import torch  # noqa: E402

OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S3 = "/fsx/data/weikai.huang/runs/keep/v12/s3_unify_4n/checkpoint-17000"
sys.path.insert(0, NEW)
from blip3d.utils import backend  # noqa: E402
backend.setup("eval")
sys.path.insert(1, OLD)
from trellis2.modules import sparse as sp  # noqa: E402
import trellis2_blip3o.flow_heads as FH  # noqa: E402
from trellis2_blip3o.loss import TRELLIS2FlowMatchingLoss  # noqa: E402
from trellis2_blip3o.connector import TRELLIS2TransformerAdapter  # noqa: E402
from blip3d.cond import assemble as A  # noqa: E402
from blip3d.cond.connector import Connector  # noqa: E402
from blip3d.cond.stamp import ViewCodes  # noqa: E402
from blip3d.losses import unified as U  # noqa: E402
from blip3d.train.ckpt import connector_state  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def same(a, b):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.shape == b.shape
                and a.dtype == b.dtype and torch.equal(a, b))
    return a == b


# ── 1. timestep samplers ──
ok_p = ok_t = ok_g = True
knobs = [dict(p_corner=0.1, p_corner2=0.2, p_solo=0.0, p_lag=0.3), dict(p_corner=0.2, p_corner2=0.2, p_solo=0.2, p_lag=0.2),
         dict(p_corner=0.0, p_corner2=0.0, p_solo=0.0, p_lag=0.0), dict(p_corner=0.1, p_corner2=0.2, p_solo=0.3, p_lag=0.4, k0_lo=2, k0_hi=11)]
for B in (1, 3, 4, 8, 64, 4096):
    for seed in range(3):
        for pc, pc2 in ((0.2, 0.2), (0.1, 0.2), (0.0, 0.0), (0.5, 0.5)):
            torch.manual_seed(seed); o = FH.sample_timestep_pairs(B, "cpu", p_corner=pc, p_corner2=pc2)
            torch.manual_seed(seed); n = U.sample_timestep_pairs(B, "cpu", p_corner=pc, p_corner2=pc2)
            ok_p &= all(same(a, b) for a, b in zip(o, n))
        for kn in knobs:
            torch.manual_seed(seed); o = FH.sample_timestep_triples(B, "cpu", **kn)
            torch.manual_seed(seed); n = U.sample_timestep_triples(B, "cpu", **kn)
            ok_t &= all(same(a, b) for a, b in zip(o, n))
            g = torch.Generator().manual_seed(seed)
            n2 = U.sample_timestep_triples(B, "cpu", generator=g, **kn)
            ok_g &= all(same(a, b) for a, b in zip(o, n2))
check("sample_timestep_pairs == v12", ok_p, "(6 B x 3 seeds x 4 corner sets)")
check("sample_timestep_triples == v12", ok_t, "(6 B x 3 seeds x 4 knob sets)")
check("triples with a seeded generator == default stream", ok_g)

# ── 2. balanced MSE ──
ok = True
for seed in range(5):
    g = torch.Generator().manual_seed(seed)
    p, t = torch.randn(6, 8, 4, 4, 4, generator=g), torch.randn(6, 8, 4, 4, 4, generator=g)
    for keep in (torch.rand(6, generator=g) > 0.5, torch.zeros(6, dtype=torch.bool), torch.ones(6, dtype=torch.bool)):
        ok &= same(FH._row_balanced_mse(p, t, keep), U.row_balanced_mse(p, t, keep))
    for n in (0, 1, 777):
        a, b = torch.randn(n, 32, generator=g), torch.randn(n, 32, generator=g)
        ok &= same(FH._voxel_balanced_mse(a, b), U.voxel_balanced_mse(a, b))
check("row/voxel balanced MSE == v12", ok)


# ── 3. end-to-end with a stub model ──
class Stub(torch.nn.Module):
    """Deterministic stand-in with Blip3DUnified's interface. Every output depends on every input it is given
    (states, timesteps, per-row conditions, SS mask, ss_read_on, the SS gate) so a wrong input changes the loss."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.w_s = torch.nn.Parameter(torch.randn(32, 32, generator=g) * 0.1)
        self.w_x = torch.nn.Parameter(torch.randn(64, 32, generator=g) * 0.1)
        self.a_ss = torch.nn.Parameter(torch.randn(8, generator=g) * 0.1)
        self.ss_reads_gate = torch.nn.Parameter(torch.randn(30, 2, 12, generator=g) * 0.01)
        self.calls = []

    @staticmethod
    def _crow(cond, mask=None):
        """(B,) per-row summary of a condition (list of (T_b, C) or dense (B, T, C) + optional (B,1,1,T) mask)."""
        if isinstance(cond, list):
            return torch.stack([c.float()[:, :8].mean() for c in cond])
        c = cond.float()[..., :8].mean(-1)
        if mask is None:
            return c.mean(1)
        m = mask[:, 0, 0, :].float()
        return (c * m).sum(1) / m.sum(1).clamp_min(1)

    @staticmethod
    def _bidx(x):
        return torch.cat([torch.full((sl.stop - sl.start,), b, dtype=torch.long) for b, sl in enumerate(x.layout)])

    def _rec(self, **kw):
        self.calls.append({k: ([c.detach().clone() for c in v] if isinstance(v, list)
                               else v.feats.detach().clone() if hasattr(v, "feats")
                               else v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()})

    def forward(self, x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=None, x_ss=None, t_ss=None,
                cond_ss=None, ss_cond_mask=None, ss_read_on=None):
        self._rec(x_s=x_s, x_x=x_x, t_s=t_s, t_x=t_x, cond_s=cond_s, cond_x=cond_x, cc=tex_concat_cond,
                  x_ss=x_ss, t_ss=t_ss, cond_ss=cond_ss, ss_cond_mask=ss_cond_mask, ss_read_on=ss_read_on)
        bi = self._bidx(x_s)
        cs, cx = self._crow(cond_s), self._crow(cond_x)
        v_s = x_s.replace(x_s.feats @ self.w_s + (t_s.float() / 1000)[bi, None] + 0.1 * cs[bi, None])
        v_x = x_x.replace(torch.cat([x_x.feats, tex_concat_cond.feats], -1) @ self.w_x
                          + (t_x.float() / 1000)[bi, None] + 0.1 * cx[bi, None] + 0.01 * cs[bi, None])
        if x_ss is None:
            return v_s, v_x
        css = self._crow(cond_ss, ss_cond_mask)
        r = ss_read_on.float() if ss_read_on is not None else torch.zeros(x_ss.shape[0])
        v_ss = (x_ss * self.a_ss.view(1, -1, 1, 1, 1) + (t_ss.float() / 1000).view(-1, 1, 1, 1, 1)
                + 0.1 * css.view(-1, 1, 1, 1, 1) + (0.01 * r + self.ss_reads_gate.sum() * 1e-3).view(-1, 1, 1, 1, 1))
        return v_s, v_x, v_ss

    def ss_forward(self, x, t, cond, ss_cond_mask=None):
        self._rec(dual_x=x, dual_t=t, dual_cond=cond, dual_mask=ss_cond_mask)
        css = self._crow(cond, ss_cond_mask)
        return x * self.a_ss.view(1, -1, 1, 1, 1) + (t.float() / 1000).view(-1, 1, 1, 1, 1) + 0.1 * css.view(-1, 1, 1, 1, 1)


# real S3 connectors (bf16, the training dtype) in both implementations
from safetensors import safe_open  # noqa: E402
f = safe_open(os.path.join(S3, "model.safetensors"), "pt")
PRE = {"geo": "geo_connector", "tex": "diffusion_connector", "ss": "ss_connector"}
sd = {k: f.get_tensor(k) for k in f.keys() if k.split(".")[0] in PRE.values() or k == "dino_view_embed"}
views = ViewCodes(16, 1024, 0.2, table=sd["dino_view_embed"])


def make_conns():
    old, new = {}, {}
    for lane, pre in PRE.items():
        oc = TRELLIS2TransformerAdapter(2048, 1024, n_blocks=2, seg_embed=True, patch_pos="sincos2d", patch_lattice=32)
        oc.load_state_dict({k[len(pre) + 1:]: v for k, v in sd.items() if k.startswith(pre + ".")}, strict=True)
        nc = Connector()
        nc.load_state_dict(connector_state(sd, pre), strict=True)
        old[lane], new[lane] = oc.to(torch.bfloat16).train(), nc.to(torch.bfloat16).train()
    return old, new


def rec(mod, T, K, seed):
    g = torch.Generator().manual_seed(seed)
    r = {"qwen": torch.randn(T, 2048, generator=g).half(), "qwen_keep": torch.rand(T, generator=g) > 0.2, "modality": mod}
    if mod != "t":
        rc = torch.full((T, 2), -1.0); rc[4:4 + 16 * K] = torch.rand(16 * K, 2, generator=g); r["qwen_rc"] = rc
        r["dino"] = torch.randn(K * 30, 1024, generator=g).half(); r["dino_keep"] = torch.ones(K * 30, dtype=torch.bool)
        r["dino_views"] = torch.arange(K).repeat_interleave(30)
        if mod == "im":
            qv = torch.full((T,), -1, dtype=torch.long)
            for k in range(K):
                qv[4 + 16 * k:4 + 16 * (k + 1)] = k
            r["qwen_views"] = qv
    return r


def targets(B, seed, tex_valid):
    g = torch.Generator().manual_seed(1000 + seed)
    cs = []
    for b in range(B):
        n = int(torch.randint(20, 60, (1,), generator=g))
        xyz = torch.randperm(32 ** 3, generator=g)[:n].sort().values
        cs.append(torch.stack([torch.full((n,), b), xyz // 1024, (xyz // 32) % 32, xyz % 32], 1).int())
    coords = torch.cat(cs)
    N = coords.shape[0]
    shape = sp.SparseTensor(feats=torch.randn(N, 32, generator=g), coords=coords)
    tex = sp.SparseTensor(feats=torch.randn(N, 32, generator=g), coords=coords)
    if tex_valid is not None:   # placeholder rows carry zeros
        tv = torch.tensor(tex_valid)
        rows = torch.cat([torch.full((sl.stop - sl.start,), bool(tv[b])) for b, sl in enumerate(tex.layout)])
        tex = tex.replace(tex.feats * rows[:, None].float())
    ss = torch.randn(B, 8, 16, 16, 16, generator=g).to(torch.bfloat16)   # HF's dense bf16 cast (T-09)
    out = {"shape": shape, "tex": tex, "ss": ss}
    if tex_valid is not None:
        out["tex_valid"] = torch.tensor(tex_valid)
    return out


def run_old(model, conns, cb, tg, kn, joint, steps):
    FH.compute_unified_geotex_loss._reported = True
    FH.compute_unified_geotex_loss._probe_n = 0
    if hasattr(FH.compute_unified_geotex_loss, "_probe_last"):
        del FH.compute_unified_geotex_loss._probe_last
    lf_slat, lf_ss = TRELLIS2FlowMatchingLoss(t_schedule="uniform"), TRELLIS2FlowMatchingLoss(t_schedule="logitNormal")
    fuse = "dino" in cb
    outs = []
    for _ in range(steps):
        loss, logs = FH.compute_unified_geotex_loss(
            unified_model=model, connector_geo=conns["geo"], connector_tex=conns["tex"], loss_fn_slat=lf_slat,
            cond_hidden=cb["qwen"].to(torch.bfloat16), cond_key_mask=cb["qwen_keep"],
            target_shape_slat_512=tg["shape"], target_tex_slat_512=tg["tex"], tex_valid=tg.get("tex_valid"),
            dino_hidden=cb["dino"].to(torch.bfloat16) if fuse else None,
            dino_key_mask=cb["dino_keep"].bool() if fuse else None,
            dino_view_ids=cb.get("dino_views") if fuse else None, qwen_view_ids=cb.get("qwen_views") if fuse else None,
            qwen_img_rc=cb.get("qwen_rc") if fuse else None, dino_view_embed=views.table if fuse else None,
            mask_drop_prob=kn["p_cfg"], dino_drop_prob=kn["p_dino"], qwen_drop_prob=kn["p_qwen"], cond_max_length=10240,
            p_corner=kn["p_corner"], p_corner2=kn["p_corner2"], geo_loss_w=1.0, joint_cond_drop=joint,
            ss_flow_present=True, target_ss_latent=tg["ss"], connector_ss=conns["ss"], loss_fn_ss=lf_ss,
            ss_loss_w=1.0, ss_dual=True, p_solo=kn["p_solo"], p_lag=kn["p_lag"], k0_lo=3, k0_hi=11,
            probe_every=kn["probe_every"])
        loss.backward()
        outs.append((loss.detach(), logs))
    return outs


def run_new(model, conns, cb, tg, kn, cfg, steps):
    st = U.UnifiedLossState(reported=True)
    outs = []
    for i in range(steps):
        loss, logs = U.unified_loss(model, conns, views, tg, cb, step=i, generator=None, cfg=cfg, state=st)
        loss.backward()
        outs.append((loss.detach(), logs))
    return outs


def grads(mods):
    return {f"{n}:{k}": p.grad.clone() if p.grad is not None else None
            for n, m in mods.items() for k, p in m.named_parameters()}


KN = [dict(p_cfg=0.1, p_dino=0.3, p_qwen=0.1, p_corner=0.1, p_corner2=0.2, p_solo=0.0, p_lag=0.3, probe_every=200),
      dict(p_cfg=0.5, p_dino=0.5, p_qwen=0.5, p_corner=0.2, p_corner2=0.2, p_solo=0.2, p_lag=0.3, probe_every=1)]
CASES = [("i1", 1, [True, False, True, True]), ("im", 3, [True, True, True, True]), ("t", 0, None),
         ("i1", 1, [False, False, False, False]), ("t", 0, [True, False, False, True])]
for ki, kn in enumerate(KN):
    for mod, K, tv in CASES:
        for compat in (True, False):
            B = 4
            rs = [rec(mod, T=40 + 7 * s + 16 * K, K=K, seed=s) for s in range(B)]
            cb = A.collate(rs, device="cpu")
            tg = targets(B, ki, tv)
            conns_o, conns_n = make_conns()
            mo, mn = Stub(), Stub()
            torch.manual_seed(1234)
            oo = run_old(mo, conns_o, cb, tg, kn, joint=not compat, steps=3)
            cfg = U.UnifiedLossCfg(p_cfg=kn["p_cfg"], p_dino=kn["p_dino"], p_qwen=kn["p_qwen"], p_corner=kn["p_corner"],
                                   p_corner2=kn["p_corner2"], p_solo=kn["p_solo"], p_lag=kn["p_lag"],
                                   probe_every=kn["probe_every"], compat_v12=compat)
            torch.manual_seed(1234)
            nn_ = run_new(mn, conns_n, cb, tg, kn, cfg, steps=3)
            tag = f"knobs{ki} {mod} tex_valid={tv} {'compat_v12' if compat else 'fixed(=v12 joint_cond_drop)'}"
            # model inputs, call by call
            ok_in = len(mo.calls) == len(mn.calls)
            bad = ""
            for ci, (a, b) in enumerate(zip(mo.calls, mn.calls)):
                for k in a:
                    va, vb = a[k], b.get(k)
                    eqv = (all(same(x, y) for x, y in zip(va, vb)) and len(va) == len(vb)) if isinstance(va, list) else same(va, vb)
                    if not eqv:
                        ok_in, bad = False, bad or f"call {ci} key {k}"
            # losses, logs
            ok_l = all(same(a[0], b[0]) for a, b in zip(oo, nn_))
            ok_logs, badlog = True, ""
            for (_, lo), (_, ln) in zip(oo, nn_):
                for k, v in lo.items():
                    w = ln.get(k)
                    w = float(w) if isinstance(w, torch.Tensor) else w
                    if w != v:
                        ok_logs, badlog = False, badlog or f"{k}: {v} vs {w}"
                extra = set(ln) - set(lo)
                if extra:
                    ok_logs, badlog = False, f"extra keys {extra}"
            # grads (connectors: same parameter names in both implementations)
            go, gn = grads({**conns_o, "stub": mo}), grads({**conns_n, "stub": mn})
            ok_g = go.keys() == gn.keys() and all(same(go[k], gn[k]) for k in go)
            check(tag, ok_in and ok_l and ok_logs and ok_g,
                  f"calls={len(mn.calls)} inputs={'ok' if ok_in else bad} loss={'ok' if ok_l else 'DIFF'} "
                  f"logs={'ok' if ok_logs else badlog} grads={'ok' if ok_g else 'DIFF'}")

# fixed default: the geo lane really sees the tex drops (and compat does not)
rs = [rec("i1", T=60, K=1, seed=s) for s in range(4)]
cb = A.collate(rs, device="cpu")
_, cn = make_conns()
for compat in (True, False):
    torch.manual_seed(5)
    lanes = U.build_lane_conds(cn, views, cb, U.UnifiedLossCfg(p_cfg=1.0, compat_v12=compat))
    n_geo = [c.shape[0] for c in lanes.geo]
    torch.manual_seed(5)
    ref = U.build_lane_conds(cn, views, cb, U.UnifiedLossCfg(p_cfg=0.0, compat_v12=True))
    geo_changed = any(not same(a, b) for a, b in zip(lanes.geo, ref.geo))
    check(f"geo lane dropped with the tex lane: {'no (compat)' if compat else 'yes (fixed)'}",
          geo_changed == (not compat), f"geo tokens {n_geo}")

print(f"\n{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
