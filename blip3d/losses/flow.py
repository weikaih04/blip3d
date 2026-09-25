"""Rectified-flow training math of the three TRELLIS.2 towers and the per-tower loss (S1, S2, SS continuation).

Upstream TRELLIS.2 (trainers/flow_matching/flow_matching.py) with sigma_min 1e-5:
    x_t = (1 - t) x0 + (sigma_min + (1 - sigma_min) t) eps,   v* = (1 - sigma_min) eps - x0,   model input t*1000.
Timesteps: SS logitNormal(mean 1, std 1), shape/tex uniform. v12 contracts kept on purpose:
* the noise is drawn (CUDA) BEFORE t (CPU) and t is cast to the target dtype (bf16) before x_t and t*1000 are formed;
* loss = plain MSE over every element (dense) or every voxel of the batch (sparse), in fp32, under bf16 autocast.
Timestep RNG: v12 drew t from the process CPU generator, identical on every rank. BLIP3D draws from a
per-rank ``generator`` the trainer seeds from (seed, rank, step); ``generator=None`` reproduces the old shared draw.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

SIGMA_MIN = 1e-5
T_SCHEDULE = {"ss": ("logitNormal", 1.0, 1.0), "shape": ("uniform", 0.0, 0.0), "tex": ("uniform", 0.0, 0.0)}


def sample_t(kind: str, B: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    name, mean, std = T_SCHEDULE[kind]
    if name == "uniform":
        return torch.rand(B, generator=generator)
    return torch.sigmoid(torch.randn(B, generator=generator) * std + mean)


def diffuse(x_0, t: torch.Tensor, noise):
    """Upstream expression verbatim (works for dense tensors and SparseTensors: t broadcasts per batch row)."""
    t = t.view(-1, *[1 for _ in range(len(x_0.shape) - 1)])
    return (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise


def velocity(x_0, noise):
    return (1 - SIGMA_MIN) * noise - x_0


def flow_loss(kind: str, flow, x_0, cond: torch.Tensor, key_mask: torch.Tensor, *,
              generator: Optional[torch.Generator] = None, concat_cond=None) -> Tuple[torch.Tensor, Dict]:
    """One flow-matching step of one tower. ``x_0``: dense (B,C,16,16,16) for SS, SparseTensor for shape/tex (already in
    the cond dtype). ``key_mask`` (B,T) bool. The dense SS cross-attention always receives the mask (sdpa), the sparse
    towers a per-row list of the kept tokens (varlen), exactly as v12."""
    sparse = hasattr(x_0, "feats")
    noise = x_0.replace(torch.randn_like(x_0.feats)) if sparse else torch.randn_like(x_0)
    dtype = x_0.feats.dtype if sparse else x_0.dtype
    t = sample_t(kind, x_0.shape[0], generator).to(cond.device).to(dtype)
    x_t = diffuse(x_0, t, noise)
    t_in = (t * 1000.0).to(dtype)
    kw = {} if concat_cond is None else {"concat_cond": concat_cond}
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if sparse:
            m = key_mask.bool()
            v = flow(x_t, t_in, [cond[b, m[b]] for b in range(cond.shape[0])], **kw)
            loss = F.mse_loss(v.feats.float(), velocity(x_0, noise).feats.float())
        else:
            v = flow(x_t, t_in, cond, cond_mask=key_mask[:, None, None, :], **kw)
            loss = F.mse_loss(v.float(), velocity(x_0, noise).float())
    return loss, {"t_mean": t.float().mean().detach()}
