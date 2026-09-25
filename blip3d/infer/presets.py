"""Frozen sampler presets. No module-level mutable parameter tables: a run picks a preset, may derive a
new one with ``dataclasses.replace``, and records it next to its outputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal, Optional, Tuple

import torch


@dataclass(frozen=True)
class StageParams:
    steps: int
    cfg: float
    rescale: float
    interval: Tuple[float, float]
    rescale_t: float


RELEASED_SS = StageParams(12, 7.5, 0.7, (0.6, 1.0), 5.0)
RELEASED_SHAPE = StageParams(12, 7.5, 0.5, (0.6, 1.0), 3.0)
RELEASED_TEX = StageParams(12, 1.0, 0.0, (0.6, 0.9), 3.0)
# text path (calibrated 2026-09-15): SS 24 steps / cfg 10; tex reads its condition only at t >= 0.95, so its CFG window
# moves to [0.9, 1]
TEXT_SS = replace(RELEASED_SS, steps=24, cfg=10.0)
TEXT_TEX = StageParams(24, 7.5, 0.5, (0.9, 1.0), 5.0)


@dataclass(frozen=True)
class Preset:
    ss: StageParams
    shape: StageParams
    tex: StageParams
    alpha: float = 32.0                   # unified joint loop: tex time t_x = alpha t / (1 + (alpha-1) t)
    max_voxels: Optional[int] = None      # v12: cascade None, unified 8192 (random subsample, reported)
    cond_dtype: torch.dtype = torch.float32   # v12 evaluated the connectors in fp32

    def as_dict(self):
        d = asdict(self)
        d["cond_dtype"] = str(self.cond_dtype)
        return d


IMAGE = Preset(RELEASED_SS, RELEASED_SHAPE, RELEASED_TEX)
TEXT = Preset(TEXT_SS, RELEASED_SHAPE, TEXT_TEX)


def preset_for(modality: Literal["i1", "im", "t"], *, unified: bool = False) -> Preset:
    p = TEXT if modality == "t" else IMAGE
    return replace(p, max_voxels=8192) if unified else p


# The lag schedule the unified model was trained on (12 nodes, SS rescale_t 5, slat rescale_t 3). Interleave inference
# pairs SS and slat nodes on THIS grid regardless of the preset's SS steps.
TRAIN_LAG_GRID = dict(steps=12, ss_rescale_t=5.0, slat_rescale_t=3.0)
