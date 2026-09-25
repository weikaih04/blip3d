"""S3 loss of the unified three-tower model (geo = shape SLAT, tex = pbr SLAT, ss = dense 16^3 structure).

What one step does:

1. Conditioning, one ``assemble`` per lane. The tex lane DRAWS the CFG / DINO / Qwen drops; the SS lane
   REPLAYS them; the geo lane replays them too (fixed default) or, with ``compat_v12``, is never
   dropped (v12 still consumed one ``rand(B)`` there on image batches — reproduced). geo/tex get per-row lists of the
   kept tokens (varlen cross-attention); SS gets the padded tensor and a (B,1,1,T) sdpa mask, or None when no row is
   padded (``GEOTEX_XATTN_FLASH=1`` in the v12 S3 launcher).
2. Timesteps: row classes solo / lag / clean (``sample_timestep_triples``) with the invariant t_ss <= t_s <= t_x.
   The lag pairing uses the constant ``TRAIN_LAG_GRID`` (12 nodes, SS shift 5, slat shift 3).
3. Noise, in the v12 order: geo, tex, SS (all on the default CUDA stream). t is cast to the TARGET dtype before it
   is used and before x1000 (fp32 in S3, because SparseTensor targets bypass HF's bf16 cast).
4. One unified forward under bf16 autocast; SS reads the slat lanes only on lag rows.
5. Losses: tex voxel-balanced MSE on rows (tex_valid & not solo); SS row-balanced MSE on not-clean rows;
   SS dual term (a second ``ss_forward`` on the clean rows at a logitNormal t); geo voxel-balanced MSE on rows
   (t_s != 0) & not solo. Every helper that all-reduces is called unconditionally (a guarded collective is a hang).
6. Every ``probe_every`` micro-steps, the condition-sensitivity probe (two no-grad forwards at t = 1, conditions
   rolled by one row) — it draws noise, so it is part of the RNG stream.

Timestep RNG: v12 drew timesteps on the default CUDA generator, which the trainer reseeds per rank, so S3 was never
affected by the shared-timestep issue. A ``generator`` may be passed to own the timestep draws (and the SS-dual t); ``compat_v12`` ignores
it. ``compat_v12=True`` reproduces v12 bit for bit.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ..cond.assemble import Drops, assemble, draw_drops
from ..infer.presets import TRAIN_LAG_GRID
from .flow import diffuse, velocity

CLS_CLEAN, CLS_SOLO, CLS_LAG = 0, 1, 2


@dataclass(frozen=True)
class UnifiedLossCfg:
    """v12 S3 values. ``compat_v12`` restores the v12 behaviour of the
    one number-changing fix here (geo lane never dropped) and ignores a passed timestep generator."""
    p_cfg: float = 0.1                 # mask_drop_prob (tex draws, SS and geo replay)
    p_dino: float = 0.3
    p_qwen: float = 0.1
    p_corner: float = 0.1              # t_s = 0 edge, share of slat-supervised rows
    p_corner2: float = 0.2             # t_x = 1 edge
    p_solo: float = 0.0
    p_lag: float = 0.3
    k0_lo: int = 3
    k0_hi: int = 11
    ss_logit_mean: float = 1.0
    ss_logit_std: float = 1.0
    ss_loss_w: float = 1.0
    ss_dual: bool = True
    geo_loss_w: float = 1.0
    joint_cond_drop: bool = True       # geo replays the tex drops; forced False by compat_v12
    xattn_flash_unpadded: bool = True  # SS sdpa mask -> None when no row is padded (v12 GEOTEX_XATTN_FLASH=1)
    probe_every: int = 200             # 0 = off
    cond_max_length: int = 10240       # Qwen tokens; v12 truncated with a warning (never fired), BLIP3D raises
    compat_v12: bool = False

    @property
    def geo_replays_drops(self) -> bool:
        return self.joint_cond_drop and not self.compat_v12


@dataclass
class UnifiedLossState:
    """Per-process state v12 kept as function attributes: the one-time schedule report and the last probe values
    (carried every step so a logging window averages to the value)."""
    reported: bool = False
    probe_last: Optional[Tuple[float, float]] = None


_DEFAULT_STATE = UnifiedLossState()


# ── timesteps ────────────────────────────────────────────────────────────────

def _rand(B: int, device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.rand(B, device=device)
    return torch.rand(B, device=generator.device, generator=generator).to(device)


def _randn(B: int, device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randn(B, device=device)
    return torch.randn(B, device=generator.device, generator=generator).to(device)


def _randint(lo: int, hi: int, B: int, device, generator: Optional[torch.Generator]) -> torch.Tensor:
    if generator is None:
        return torch.randint(lo, hi, (B,), device=device)
    return torch.randint(lo, hi, (B,), device=generator.device, generator=generator).to(device)


def sample_timestep_pairs(B: int, device, p_corner: float = 0.2, p_corner2: float = 0.2,
                          generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """(t_s, t_x) with t_s <= t_x: t_s ~ U[0,1] first (uniform marginal), then t_x ~ U[t_s, 1]. Two edges on one
    uniform u: u < p_corner pins t_s = 0 and REDRAWS t_x ~ U[0,1] (the tex | given-mesh regime);
    p_corner <= u < p_corner + p_corner2 pins t_x = 1 (mesh-only / the first joint step)."""
    assert p_corner + p_corner2 <= 1.0 + 1e-6, f"p_corner {p_corner} + p_corner2 {p_corner2} > 1"
    t_s = _rand(B, device, generator)
    t_x = t_s + (1.0 - t_s) * _rand(B, device, generator)
    u = _rand(B, device, generator)
    edge_s = u < p_corner
    edge_x = (u >= p_corner) & (u < p_corner + p_corner2)
    t_x = torch.where(edge_x, torch.ones_like(t_x), t_x)
    t_x = torch.where(edge_s, _rand(B, device, generator), t_x)
    t_s = torch.where(edge_s, torch.zeros_like(t_s), t_s)
    return t_s, t_x


def shift_grid(rescale_t: float, u: torch.Tensor) -> torch.Tensor:
    """The sampler's node transform t' = r u / (1 + (r - 1) u), as a continuous map."""
    return rescale_t * u / (1.0 + (rescale_t - 1.0) * u)


def sample_timestep_triples(B: int, device, p_corner: float = 0.1, p_corner2: float = 0.2,
                            p_solo: float = 0.0, p_lag: float = 0.3, k0_lo: int = 3, k0_hi: int = 11,
                            ss_logit_mean: float = 1.0, ss_logit_std: float = 1.0,
                            generator: Optional[torch.Generator] = None):
    """(t_ss, t_s, t_x, cls), invariant t_ss <= t_s <= t_x by construction.

    solo  t_ss ~ logitNormal, slat pinned to noise (t_s = t_x = 1), slat losses masked.
    lag   u ~ U(k0/S, 1], k0 ~ U{k0_lo..k0_hi}: t_s = shift_3(u), t_ss = shift_5(u - k0/S), t_x = t_s + (1-t_s) U.
    clean t_ss = 0 (raw latent), (t_s, t_x) from ``sample_timestep_pairs`` with the corners rescaled by
          (1 - p_solo) / (1 - p_solo - p_lag) so p_corner keeps meaning "share of slat-supervised rows".
    Every primitive is drawn for every row in a fixed order and composed with torch.where (RNG discipline).
    S and the two shifts are the constant training grid (``TRAIN_LAG_GRID``), never the inference preset."""
    assert 0.0 <= p_solo and 0.0 <= p_lag and p_solo + p_lag <= 1.0 + 1e-6, f"p_solo {p_solo} + p_lag {p_lag} > 1"
    assert k0_lo >= 2, f"k0_lo={k0_lo}: t_ss <= t_s fails for k0 < 2"
    assert k0_hi >= k0_lo
    slat_sup, clean = 1.0 - p_solo, 1.0 - p_solo - p_lag
    scale = (slat_sup / clean) if clean > 1e-9 else 0.0
    p_c_eff, p_c2_eff = p_corner * scale, p_corner2 * scale
    assert p_c_eff + p_c2_eff <= 1.0 + 1e-6, "corner rescale overflows the clean class"

    u_cls = _rand(B, device, generator)
    g_solo = _randn(B, device, generator)
    u_lag = _rand(B, device, generator)
    k0 = _randint(k0_lo, k0_hi + 1, B, device, generator).float()
    w_lag = _rand(B, device, generator)
    t_s9, t_x9 = sample_timestep_pairs(B, device, p_corner=p_c_eff, p_corner2=p_c2_eff, generator=generator)

    t_solo = torch.sigmoid(ss_logit_mean + ss_logit_std * g_solo)
    steps = float(TRAIN_LAG_GRID["steps"])
    assert k0_hi <= steps - 1, f"k0_hi={k0_hi} > steps-1={steps - 1:.0f}"
    lo = k0 / steps
    u_eff = lo + (1.0 - lo) * u_lag
    t_s_lag = shift_grid(TRAIN_LAG_GRID["slat_rescale_t"], u_eff)
    t_ss_lag = shift_grid(TRAIN_LAG_GRID["ss_rescale_t"], u_eff - lo)
    t_x_lag = t_s_lag + (1.0 - t_s_lag) * w_lag

    solo = u_cls < p_solo
    lag = (u_cls >= p_solo) & (u_cls < p_solo + p_lag)
    cls = torch.where(solo, torch.full_like(u_cls, CLS_SOLO, dtype=torch.long),
                      torch.where(lag, torch.full_like(u_cls, CLS_LAG, dtype=torch.long),
                                  torch.full_like(u_cls, CLS_CLEAN, dtype=torch.long)))
    zero, one = torch.zeros_like(t_s9), torch.ones_like(t_s9)
    t_ss = torch.where(solo, t_solo, torch.where(lag, t_ss_lag, zero))
    t_s = torch.where(solo, one, torch.where(lag, t_s_lag, t_s9))
    t_x = torch.where(solo, one, torch.where(lag, t_x_lag, t_x9))
    assert bool((t_ss <= t_s + 1e-6).all()) and bool((t_s <= t_x + 1e-6).all()), "triple order invariant violated"
    return t_ss, t_s, t_x, cls


# ── balanced MSE (collectives: call unconditionally on every rank) ───────────

def _dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def row_balanced_mse(pred: torch.Tensor, target: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Dense (B, ...) MSE over the kept rows, normalised by the GLOBAL kept-row count (x world, which DDP/ZeRO's
    gradient averaging cancels). Row selection, not multiply-by-zero: an empty local selection adds 0 to both
    numerator and denominator and still carries a grad_fn."""
    p_sel, t_sel = pred[keep], target[keep]
    n_local = torch.tensor(float(p_sel.shape[0]), device=pred.device)
    sq = ((p_sel.float() - t_sel.float()) ** 2).flatten(1).mean(1).sum()
    if _dist():
        world = torch.distributed.get_world_size()
        n_glob = n_local.clone()
        torch.distributed.all_reduce(n_glob)
        return sq * world / n_glob.clamp_min(1.0)
    return sq / n_local.clamp_min(1.0)


def voxel_balanced_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sparse (N, C) MSE with per-voxel weight 1 / N_global regardless of how many voxels this rank drew.
   """
    per_voxel = (pred - target).pow(2).mean(-1)
    n_local = per_voxel.numel()
    if not _dist():
        return per_voxel.mean() if n_local else per_voxel.sum()
    n = torch.tensor([float(n_local)], device=per_voxel.device)
    torch.distributed.all_reduce(n, op=torch.distributed.ReduceOp.SUM)
    world = torch.distributed.get_world_size()
    return per_voxel.sum() * world / n.clamp_min(1.0).squeeze()


def _voxel_rows(layout, row_mask: List[bool], device) -> torch.Tensor:
    """Per-row bool -> per-voxel bool along a SparseTensor layout."""
    return torch.cat([torch.full((sl.stop - sl.start,), bool(row_mask[b]), dtype=torch.bool, device=device)
                      for b, sl in enumerate(layout)])


# ── conditioning ─────────────────────────────────────────────────────────────

@dataclass
class LaneConds:
    geo: List[torch.Tensor]            # per-row kept tokens (varlen)
    tex: List[torch.Tensor]
    ss: torch.Tensor                   # (B, T, C) padded
    ss_mask: Optional[torch.Tensor]    # (B, 1, 1, T) bool, or None when no row is padded (flash path)
    drops: Drops                       # the tex lane's realised drops


def _param_dtype(m: torch.nn.Module) -> torch.dtype:
    return next(m.parameters()).dtype


def _masked_list(cond: torch.Tensor, key_mask: torch.Tensor) -> List[torch.Tensor]:
    m = key_mask.bool()
    return [cond[b, m[b]] for b in range(cond.shape[0])]


def build_lane_conds(connectors: Dict[str, torch.nn.Module], views, cond_batch: Dict[str, torch.Tensor],
                     cfg: UnifiedLossCfg) -> LaneConds:
    """The three lane conditions in v12's order: tex (draws), geo (replays or, compat, none), SS (replays)."""
    q = cond_batch["qwen"]
    if q.shape[1] > cfg.cond_max_length:
        raise ValueError(f"cond length {q.shape[1]} > cond_max_length {cfg.cond_max_length}")
    B, dev = q.shape[0], q.device
    has_dino = "dino" in cond_batch

    dt_x = _param_dtype(connectors["tex"])
    d = draw_drops(B, has_dino=has_dino, p_cfg=cfg.p_cfg, p_dino=cfg.p_dino, p_qwen=cfg.p_qwen,
                   device=dev, text_draw="bernoulli", dtype=dt_x)
    c_x, k_x = assemble(connectors["tex"], views, cond_batch, d, dt_x)

    conn_g = connectors["geo"]
    if cfg.geo_replays_drops:
        d_g = Drops(d.drop, d.ddrop, d.qdrop)
    else:
        # v12: geo was assembled with every drop probability 0, and the image branch still drew rand(B) for the
        # (always-false) CFG drop. That draw is kept so the rest of the RNG stream lines up.
        d_g = draw_drops(B, has_dino=has_dino, p_cfg=0.0, p_dino=0.0, p_qwen=0.0, device=dev,
                         text_draw="bernoulli", dtype=_param_dtype(conn_g))
    ctx = contextlib.nullcontext() if any(p.requires_grad for p in conn_g.parameters()) else torch.no_grad()
    with ctx:
        c_g, k_g = assemble(conn_g, views, cond_batch, d_g, _param_dtype(conn_g))
        geo = _masked_list(c_g, k_g)

    c_ss, k_ss = assemble(connectors["ss"], views, cond_batch, Drops(d.drop, d.ddrop, d.qdrop),
                          _param_dtype(connectors["ss"]))
    ss_mask = None if (cfg.xattn_flash_unpadded and bool(k_ss.all())) else k_ss[:, None, None, :]
    return LaneConds(geo=geo, tex=_masked_list(c_x, k_x), ss=c_ss, ss_mask=ss_mask, drops=d)


# ── the loss ─────────────────────────────────────────────────────────────────

def unified_loss(model, connectors: Dict[str, torch.nn.Module], views, batch: Dict, cond_batch: Dict[str, torch.Tensor],
                 *, step: int, generator: Optional[torch.Generator] = None, cfg: UnifiedLossCfg = UnifiedLossCfg(),
                 state: Optional[UnifiedLossState] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """One S3 micro-step. ``batch``: targets (``shape`` shape-norm SparseTensor, ``tex``
    pbr-norm SparseTensor on the same coords, ``ss`` raw dense latent, optional ``tex_valid``), on the GPU.
    ``cond_batch``: ``cond.assemble.collate`` of the encoder records. ``step``: this process's micro-step counter
    from 0 (the probe fires when ``step % probe_every == 0``, as v12's call counter did).
    Returns (total, logs); logs are detached 0-dim tensors (probe values are floats), one key set per config."""
    st = state if state is not None else _DEFAULT_STATE
    gen = None if cfg.compat_v12 else generator
    lanes = build_lane_conds(connectors, views, cond_batch, cfg)
    B = cond_batch["qwen"].shape[0]
    dev = cond_batch["qwen"].device

    t_ss, t_s, t_x, cls = sample_timestep_triples(
        B, dev, p_corner=cfg.p_corner, p_corner2=cfg.p_corner2, p_solo=cfg.p_solo, p_lag=cfg.p_lag,
        k0_lo=cfg.k0_lo, k0_hi=cfg.k0_hi, ss_logit_mean=cfg.ss_logit_mean, ss_logit_std=cfg.ss_logit_std,
        generator=gen)
    m_solo, m_lag, m_clean = cls == CLS_SOLO, cls == CLS_LAG, cls == CLS_CLEAN
    m_ss_sup, m_slat_sup = ~m_clean, ~m_solo
    t_x_sampled = t_x.clone()
    tex_valid = batch.get("tex_valid")
    if tex_valid is not None:
        tex_valid = tex_valid.to(dev).bool()
        t_x = torch.where(tex_valid, t_x, torch.ones_like(t_x))
    if not st.reported:
        st.reported = True
        sup = t_s != 0
        tri = sup & (t_s <= t_x_sampled)
        print(f"[timestep] corner {cfg.p_corner} corner2 {cfg.p_corner2} | geo-supervised {sup.float().mean():.0%}, "
              f"inference-aligned {(tri.sum() / sup.sum().clamp_min(1)).item():.0%} | solo {m_solo.float().mean():.2f} "
              f"lag {m_lag.float().mean():.2f} clean {m_clean.float().mean():.2f} (B={B}, this batch)", flush=True)

    # targets, noise, x_t — t cast to the target dtype first
    x0_x = batch["tex"]
    common_dt = x0_x.feats.dtype
    x0_s = batch["shape"].replace(batch["shape"].feats.to(common_dt))
    t_s, t_x = t_s.to(common_dt), t_x.to(common_dt)
    noise_s = x0_s.replace(torch.randn_like(x0_s.feats))
    noise_x = x0_x.replace(torch.randn_like(x0_x.feats))
    x_ts = diffuse(x0_s, t_s, noise_s)
    x_tx = diffuse(x0_x, t_x, noise_x)
    v_target = velocity(x0_x, noise_x)
    x0_ss = batch["ss"].to(dev).to(common_dt)
    noise_ss = torch.randn_like(x0_ss)
    t_ss = t_ss.to(common_dt)
    x_tss = diffuse(x0_ss, t_ss, noise_ss)
    v_ss_target = velocity(x0_ss, noise_ss)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=x_tx.feats.is_cuda):
        v_s_pred, v_x_pred, v_ss_pred = model(
            x_ts, x_tx, (t_s * 1000.0).to(x_ts.feats.dtype), (t_x * 1000.0).to(x_tx.feats.dtype),
            lanes.geo, lanes.tex, tex_concat_cond=x_ts,
            x_ss=x_tss, t_ss=(t_ss * 1000.0).to(x_tss.dtype), cond_ss=lanes.ss, ss_cond_mask=lanes.ss_mask,
            ss_read_on=m_lag)

    # tex: rows with tex supervision and not solo
    tv = tex_valid.tolist() if tex_valid is not None else [True] * B
    slat_sup = m_slat_sup.tolist()
    rows_x = _voxel_rows(x0_x.layout, [a and b for a, b in zip(tv, slat_sup)], dev)
    loss = voxel_balanced_mse(v_x_pred.feats[rows_x].float(), v_target.feats[rows_x].float())
    tex_only = loss.detach()

    # SS: interleaved term on not-clean rows, dual term on the clean rows
    ss_loss = row_balanced_mse(v_ss_pred.float(), v_ss_target.float(), m_ss_sup)
    loss = loss + cfg.ss_loss_w * ss_loss
    ss_loss2 = None
    if cfg.ss_dual:
        sel = m_clean
        n_sel = int(sel.sum())
        if n_sel > 0:
            g2 = _randn(n_sel, dev, gen)
            t2 = torch.sigmoid(cfg.ss_logit_mean + cfg.ss_logit_std * g2)
            x0_sel = x0_ss[sel]
            n2 = torch.randn_like(x0_sel)
            x_t2 = diffuse(x0_sel, t2, n2)
            tgt2 = velocity(x0_sel, n2)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v2 = model.ss_forward(x_t2, (t2 * 1000.0).to(x_t2.dtype), lanes.ss[sel],
                                      ss_cond_mask=None if lanes.ss_mask is None else lanes.ss_mask[sel])
            keep2 = torch.ones(n_sel, dtype=torch.bool, device=dev)
        else:
            v2, tgt2 = v_ss_pred[:0].float(), v_ss_target[:0].float()
            keep2 = torch.zeros(0, dtype=torch.bool, device=dev)
        ss_loss2 = row_balanced_mse(v2.float(), tgt2.float(), keep2)
        loss = loss + cfg.ss_loss_w * ss_loss2

    # global row counts and schedule logs (one collective, unconditional)
    b_slat = float(len(x0_s.layout))
    ssum = m_ss_sup.float().sum()
    rows = torch.stack([torch.as_tensor(b_slat, device=dev), torch.as_tensor(float(B), device=dev),
                        torch.as_tensor(b_slat, device=dev) if cfg.ss_dual else ssum,
                        m_slat_sup.float().sum(), ssum]).float()
    if _dist():
        torch.distributed.all_reduce(rows)
    rden = rows[0].clamp_min(1.0)
    gate = model.ss_reads_gate.detach().abs().float()
    zero = torch.zeros((), device=dev)
    logs = {
        "tex_flow_loss": tex_only,
        "global_rows": rows[0],
        "ss_sup_frac": rows[2] / rden,
        "slat_sup_frac": rows[3] / rden,
        "ss_sup_interleaved": rows[4] / rden,
        "rows_mismatch": (rows[0] - rows[1]).abs(),
        "t_s_mean": t_s.float().mean(), "t_x_mean": t_x_sampled.float().mean(),
        "corner_frac": (t_s == 0).float().mean(),
        "corner2_frac": (t_x_sampled == 1).float().mean(),
        "corner2_frac_effective": (t_x == 1).float().mean(),
        "tex_valid_frac": tex_valid.float().mean() if tex_valid is not None else torch.ones((), device=dev),
        "ss_flow_loss": ss_loss.detach().float(),
        "t_ss_mean": t_ss.float().mean(),
        "ss_solo_frac": m_solo.float().mean(), "lag_frac": m_lag.float().mean(), "clean_frac": m_clean.float().mean(),
        "ss_dual_loss": zero if ss_loss2 is None else ss_loss2.detach().float(),
        "ss_reads_gate": gate.mean(), "ss_reads_gate_max": gate.max(),
    }

    # geo: rows with t_s != 0 and not solo (the t_s = 0 corner is a trivial negation nothing reads)
    if cfg.geo_loss_w > 0:
        v_s_target = velocity(x0_s, noise_s)
        keep = ((t_s != 0) & m_slat_sup).tolist()
        rows_s = _voxel_rows(x0_s.layout, keep, dev)
        geo_loss = voxel_balanced_mse(v_s_pred.feats[rows_s].float(), v_s_target.feats[rows_s].float())
        loss = loss + cfg.geo_loss_w * geo_loss
        logs["geo_flow_loss"] = geo_loss.detach()

    if cfg.probe_every > 0 and B > 1:
        if step % cfg.probe_every == 0:
            st.probe_last = _probe(model, lanes, x0_s, x0_x, rows_x, common_dt, st.probe_last)
        if st.probe_last is not None:
            logs["cond_sens_geo"], logs["cond_sens_tex"] = st.probe_last
    return loss, logs


@torch.no_grad()
def _probe(model, lanes: LaneConds, x0_s, x0_x, rows_x, common_dt, prev):
    """MSE(x0 | condition of the next row) / MSE(x0 | own condition) at t = 1 for geo and tex (the SS lane is not
    run). 1.0 = the lane ignores its condition. The tex arm uses only the tex-supervised voxels; an all-invalid
    batch carries the previous tex value."""
    B = len(lanes.tex)
    one = torch.ones(B, device=x0_s.feats.device, dtype=common_dt)
    ns = x0_s.replace(torch.randn_like(x0_s.feats))
    nx = x0_x.replace(torch.randn_like(x0_x.feats))
    roll = lambda c: c[1:] + c[:1]
    out = {}
    for tag, (cs, cx) in (("ok", (lanes.geo, lanes.tex)), ("bad", (roll(lanes.geo), roll(lanes.tex)))):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=nx.feats.is_cuda):
            ps, px = model(ns, nx, one * 1000.0, one * 1000.0, cs, cx, tex_concat_cond=ns)
        e_x = ((nx.feats.float() - px.feats.float()) - x0_x.feats.float()).pow(2)[rows_x]
        out[tag] = (float(((ns.feats.float() - ps.feats.float()) - x0_s.feats.float()).pow(2).mean()),
                    float(e_x.mean()) if e_x.numel() else None)
    prev = prev or (1.0, 1.0)
    geo = out["bad"][0] / max(out["ok"][0], 1e-8)
    tex = (out["bad"][1] / max(out["ok"][1], 1e-8)
           if out["ok"][1] is not None and out["bad"][1] is not None else prev[1])
    return geo, tex
