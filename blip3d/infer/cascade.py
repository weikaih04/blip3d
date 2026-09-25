"""Separate-tower cascade (the v12 product path): SS -> occupancy -> shape -> tex, each tower with its own connector."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ..cond.assemble import cond_uncond
from ..models.decoders import Decoders, LatentStats
from ..models.flows import Tower
from .export import Latents
from .presets import Preset
from .stages import occ_to_coords, sample_shape, sample_ss_released, sample_tex


def _cu(tower: Tower, record, preset: Preset):
    return cond_uncond(tower.connector, tower.views, record, dtype=preset.cond_dtype)


@torch.no_grad()
def run_cascade(towers: Dict[str, Tower], dec: Decoders, stats: LatentStats, record, preset: Preset,
                seed: int = 0) -> Latents:
    """All three stages use ``seed`` (v12). Empty structure -> ``info['empty']`` and no slat stages."""
    c, u = _cu(towers["ss"], record, preset)
    z = sample_ss_released(towers["ss"].flow, c, u, preset.ss, seed)
    coords, info = occ_to_coords(dec.ss, z, max_voxels=preset.max_voxels)
    info["preset"] = preset.as_dict()
    if info["empty"]:
        return Latents(coords, torch.empty(0, 32), torch.empty(0, 32), info)
    c, u = _cu(towers["shape"], record, preset)
    shape_raw = sample_shape(towers["shape"].flow, c, u, coords, preset.shape, seed).feats.float() * stats.shape_std + stats.shape_mean
    tex_raw = tex_on_shape(towers["tex"], stats, record, preset, coords, shape_raw, seed)
    return Latents(coords, shape_raw, tex_raw, info)


@torch.no_grad()
def tex_on_shape(tex: Tower, stats: LatentStats, record, preset: Preset, coords, shape_raw, seed: int = 0) -> torch.Tensor:
    """Tex tower on a given geometry (the cascade's last stage, and the controlled-texture track on GT shape)."""
    c, u = _cu(tex, record, preset)
    tex_n = sample_tex(tex.flow, c, u, coords, (shape_raw - stats.tex_shape_mean) / stats.tex_shape_std, preset.tex, seed)
    return tex_n * stats.tex_std + stats.tex_mean
