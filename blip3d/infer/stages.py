"""Pieces shared by the cascade and the unified sampler: SS sampling, occupancy -> slat coords, the slat Euler loops."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .presets import StageParams
from .schedule import guided, needs_uncond, t_seq


@torch.no_grad()
def sample_ss_released(flow, cond, uncond, p: StageParams, seed: int) -> torch.Tensor:
    """SS latent (1, 8, 16, 16, 16) with the released TRELLIS.2 sampler under bf16 autocast — exactly the v12 cascade.
    (The unified model samples SS with its own loop; the two differ in reduced-precision details, see infer/unified.py.)"""
    from trellis2.pipelines.samplers import FlowEulerGuidanceIntervalSampler
    g = torch.Generator(device="cuda").manual_seed(seed)
    r, c = flow.resolution, flow.in_channels
    noise = torch.randn(1, c, r, r, r, generator=g, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return FlowEulerGuidanceIntervalSampler(sigma_min=1e-5).sample(
            flow, noise, cond=cond, neg_cond=uncond, verbose=False, steps=p.steps, guidance_strength=p.cfg,
            guidance_rescale=p.rescale, guidance_interval=list(p.interval), rescale_t=p.rescale_t).samples


@torch.no_grad()
def occ_to_coords(ss_decoder, z: torch.Tensor, *, max_voxels: Optional[int] = None,
                  generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, Dict]:
    """SS latent (raw) -> 64³ occupancy (decoder logits > 0) -> 32³ (max-pool 2, > 0.5) -> (N, 4) int32 coords with a
    batch column. ``max_voxels``: random subsample (sorted) when exceeded, reported in ``info`` (unified path only)."""
    z = z[None] if z.dim() == 4 else z
    occ64 = (ss_decoder(z) > 0)[0, 0]
    occ32 = F.max_pool3d(occ64.float()[None, None], 2, 2)[0, 0] > 0.5
    c3 = torch.argwhere(occ32).int()
    info = {"voxels_64": int(occ64.sum()), "voxels_generated": int(c3.shape[0]), "truncated": False,
            "empty": int(c3.shape[0]) == 0}
    if max_voxels is not None and c3.shape[0] > max_voxels:
        keep = torch.randperm(c3.shape[0], generator=generator, device=c3.device)[:max_voxels]
        c3 = c3[keep.sort().values]
        info["truncated"] = True
    info["voxels_kept"] = int(c3.shape[0])
    coords = torch.cat([torch.zeros(c3.shape[0], 1, dtype=torch.int32, device=c3.device), c3], 1)
    return coords, info


@torch.no_grad()
def sample_shape(flow, cond, uncond, coords, p: StageParams, seed: int):
    """Separate shape tower on given coords: fp32 noise, flow in its own layout (no autocast), fp32 guidance."""
    from trellis2.modules import sparse as sp
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = sp.SparseTensor(torch.randn(coords.shape[0], flow.in_channels, generator=g, device="cuda",
                                    dtype=torch.float32), coords.cuda())
    ts = t_seq(p.steps, p.rescale_t)
    for i in range(p.steps):
        t, tp = float(ts[i]), float(ts[i + 1])
        tt = torch.tensor([t * 1000.0], device="cuda")
        vp = flow(x, tt, cond).feats.float()
        vn = flow(x, tt, uncond).feats.float() if needs_uncond(p, t) else None
        x = x.replace(x.feats - (t - tp) * guided(vp, vn, x.feats.float(), t, p))
    return x


@torch.no_grad()
def sample_tex(flow, cond, uncond, coords, shape_norm: torch.Tensor, p: StageParams, seed: int) -> torch.Tensor:
    """Separate tex tower: input = cat[x_tex ; shape latent in the tex tower's concat-cond normalisation]."""
    from trellis2.modules import sparse as sp
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(coords.shape[0], 32, generator=g, device="cuda", dtype=torch.float32)
    sz = shape_norm.float()
    ts = t_seq(p.steps, p.rescale_t)
    for i in range(p.steps):
        t, tp = float(ts[i]), float(ts[i + 1])
        tt = torch.tensor([t * 1000.0], device="cuda")
        xin = sp.SparseTensor(torch.cat([x, sz], 1), coords.cuda())
        vp = flow(xin, tt, cond).feats.float()
        vn = flow(xin, tt, uncond).feats.float() if needs_uncond(p, t) else None
        x = x - (t - tp) * guided(vp, vn, x, t, p)
    return x
