"""The three TRELLIS.2-4B flows (SS 16³ dense, shape SLAT 32³, tex SLAT 32³), their frozen decoders and latent statistics.

Two precision layouts exist in v12 and both are part of the numbers (ISSUES I-13):
* ``released`` — the model as TRELLIS.2 builds it (blocks in bf16, boundary layers / norms in fp32); every v12
  separate-tower evaluation ran this layout, with the trained bf16 weights copied in.
* ``bf16``     — every parameter cast to bf16 except complex buffers (SS ``rope_phases``); what training used, and what
  the unified model is evaluated in.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn

from ..cond.connector import Connector
from ..cond.stamp import ViewCodes
from ..train.ckpt import tower_state

FLOWS = {
    "ss": ("ss_flow_img_dit_1_3B_64_bf16", "ss_flow."),
    "shape": ("slat_flow_img2shape_dit_1_3B_512_bf16", "shape_slat_512."),
    "tex": ("slat_flow_imgshape2tex_dit_1_3B_512_bf16", "tex_slat_512."),
}
DECODERS = {"shape": "shape_dec_next_dc_f16c32_fp16", "tex": "tex_dec_next_dc_f16c32_fp16"}
Layout = Literal["released", "bf16"]


def to_bf16_keep_complex(model: nn.Module) -> nn.Module:
    """bf16 everywhere except complex buffers (a blunt .to(bf16) drops the SS flow's RoPE rotation)."""
    keep = {n: b.clone() for n, b in model.named_buffers() if b is not None and b.is_complex()}
    model = model.to(torch.bfloat16)
    for name, buf in keep.items():
        mod = model
        *path, leaf = name.split(".")
        for p in path:
            mod = getattr(mod, p)
        setattr(mod, leaf, buf)
    return model


def released_flow(kind: str, trellis2_ckpt: str) -> nn.Module:
    from trellis2 import models
    return models.from_pretrained(os.path.join(trellis2_ckpt, "ckpts", FLOWS[kind][0]))


def flow_state(sd: Dict[str, torch.Tensor], kind: str) -> Dict[str, torch.Tensor]:
    pre = FLOWS[kind][1]
    out = {}
    for k, v in sd.items():
        if k.startswith(pre):
            k = k[len(pre):]
            out[k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k] = v
    if not out:
        raise KeyError(f"no '{pre}*' keys in the checkpoint")
    return out


@dataclass
class Tower:
    kind: str
    flow: nn.Module
    connector: Connector
    views: Optional[ViewCodes]


def load_tower(kind: str, ckpt_dir: str, *, trellis2_ckpt: str, use_ema: bool = True, layout: Layout = "released",
               device: str = "cuda", connector_dtype: torch.dtype = torch.float32) -> Tower:
    """A trained v12 tower: flow + its connector + the model-level view codes. Strict: every flow and connector key must
    be present and consumed (the old loaders only printed the missing count)."""
    sd = tower_state(ckpt_dir, kind, use_ema=use_ema)          # v12 or BLIP3D names -> BLIP3D names
    flow = released_flow(kind, trellis2_ckpt)
    missing, unexpected = flow.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("flow.")}, strict=False)
    if missing or unexpected:
        raise KeyError(f"{ckpt_dir} [{kind}]: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
    if layout == "bf16":
        flow = to_bf16_keep_complex(flow)
    conn = Connector()
    conn.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("connector.")}, strict=True)
    views = ViewCodes(table=sd["views.table"]) if "views.table" in sd else None
    flow = flow.to(device).eval().requires_grad_(False)
    conn = conn.to(device).to(connector_dtype).eval().requires_grad_(False)
    return Tower(kind, flow, conn, views.to(device) if views is not None else None)


@dataclass
class LatentStats:
    """Per-channel (1, C) statistics from the TRELLIS.2 stage configs. The SS latent has none (consumed raw)."""
    shape_mean: torch.Tensor
    shape_std: torch.Tensor
    tex_mean: torch.Tensor
    tex_std: torch.Tensor
    tex_shape_mean: torch.Tensor     # tex tower's concat-cond normalisation (== shape stats, bit-identical)
    tex_shape_std: torch.Tensor

    def to(self, device):
        return LatentStats(*(getattr(self, f).to(device) for f in self.__dataclass_fields__))


def latent_stats(trellis2_repo: str) -> LatentStats:
    def read(name, key):
        with open(os.path.join(trellis2_repo, "configs", "gen", name + ".json")) as f:
            n = json.load(f)["dataset"]["args"][key]      # KeyError if absent: never silently un-normalised
        return torch.tensor(n["mean"]).reshape(1, -1), torch.tensor(n["std"]).reshape(1, -1)
    sm, ss = read(FLOWS["shape"][0], "normalization")
    tm, ts = read(FLOWS["tex"][0], "pbr_slat_normalization")
    xm, xs = read(FLOWS["tex"][0], "shape_slat_normalization")
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
