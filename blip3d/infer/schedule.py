"""Time grids, guidance and the Euler update shared by every sampler."""
from __future__ import annotations

from typing import List

import numpy as np
import torch

from .presets import StageParams

SIGMA_MIN = 1e-5


def t_seq(steps: int, rescale_t: float) -> List[float]:
    """TRELLIS flow_euler grid: uniform 1→0, then t' = r t / (1 + (r-1) t)."""
    t = np.linspace(1, 0, steps + 1)
    return (rescale_t * t / (1 + (rescale_t - 1) * t)).tolist()


def warp(t: float, alpha: float) -> float:
    """Follower time for the joint loop: t_x = alpha t / (1 + (alpha-1) t) (noisier than t for alpha > 1)."""
    return alpha * t / (1.0 + (alpha - 1.0) * t)


def pred_to_x0(x_t, t, v):
    return (1 - SIGMA_MIN) * x_t - (SIGMA_MIN + (1 - SIGMA_MIN) * t) * v


def x0_to_pred(x_t, t, x0):
    return ((1 - SIGMA_MIN) * x_t - x0) / (SIGMA_MIN + (1 - SIGMA_MIN) * t)


def needs_uncond(p: StageParams, t: float) -> bool:
    return p.cfg != 1 and p.interval[0] <= t <= p.interval[1]


def guided(v_pos: torch.Tensor, v_neg, x_t: torch.Tensor, t: float, p: StageParams) -> torch.Tensor:
    """CFG with interval and rescale. Outside the interval (or at cfg 1) the conditional prediction alone.
    The rescale std is ONE population std over every voxel and channel (the SparseTensor semantics), not per row."""
    if not needs_uncond(p, t):
        return v_pos
    pred = p.cfg * v_pos + (1 - p.cfg) * v_neg
    if p.rescale > 0:
        x0_pos, x0_cfg = pred_to_x0(x_t, t, v_pos), pred_to_x0(x_t, t, pred)
        gstd = lambda z: (z.pow(2).mean() - z.mean() ** 2).clamp_min(0).sqrt()
        x0 = p.rescale * (x0_cfg * (gstd(x0_pos) / gstd(x0_cfg))) + (1 - p.rescale) * x0_cfg
        pred = x0_to_pred(x_t, t, x0)
    return pred
