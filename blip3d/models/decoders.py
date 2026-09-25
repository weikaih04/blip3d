"""The latent spaces BLIP3D generates in, both from TRELLIS.2 and never trained here: the frozen decoders
(sparse structure -> 64³ occupancy, shape latent -> mesh, texture latent -> PBR attributes) and the per-channel
normalisation of the shape / texture latents (the SS latent has none)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from .flows import RELEASED

DECODERS = {"shape": "shape_dec_next_dc_f16c32_fp16", "tex": "tex_dec_next_dc_f16c32_fp16"}


@dataclass
class LatentStats:
    """Per-channel (1, C) statistics from the TRELLIS.2 stage configs."""
    shape_mean: torch.Tensor
    shape_std: torch.Tensor
    tex_mean: torch.Tensor
    tex_std: torch.Tensor
    tex_shape_mean: torch.Tensor     # the texture flow's concat-cond normalisation (== shape stats, bit-identical)
    tex_shape_std: torch.Tensor

    def to(self, device):
        return LatentStats(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


def latent_stats(trellis2_repo: str) -> LatentStats:
    def read(name, key):
        with open(os.path.join(trellis2_repo, "configs", "gen", name + ".json")) as f:
            n = json.load(f)["dataset"]["args"][key]      # KeyError if absent: never silently un-normalised
        return torch.tensor(n["mean"]).reshape(1, -1), torch.tensor(n["std"]).reshape(1, -1)
    sm, ss = read(RELEASED["shape"], "normalization")
    tm, ts = read(RELEASED["tex"], "pbr_slat_normalization")
    xm, xs = read(RELEASED["tex"], "shape_slat_normalization")
    return LatentStats(sm, ss, tm, ts, xm, xs)


@dataclass
class Decoders:
    ss: nn.Module
    shape: nn.Module
    tex: nn.Module


def load_decoders(trellis2_ckpt: str, ss_decoder: str, device: str = "cuda") -> Decoders:
    from trellis2 import models

    def frozen(path):
        return models.from_pretrained(path).to(device).eval().requires_grad_(False)
    return Decoders(frozen(ss_decoder),
                    frozen(os.path.join(trellis2_ckpt, "ckpts", DECODERS["shape"])),
                    frozen(os.path.join(trellis2_ckpt, "ckpts", DECODERS["tex"])))
