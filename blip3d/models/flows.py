"""The three flow networks BLIP3D is built on: the TRELLIS.2 sparse-structure flow (16³ dense) and the shape / texture
structured-latent flows (32³ sparse). Architectures come from ``configs/model/{ss,shape,tex}_flow.json`` (the
TRELLIS.2-4B configurations); weights come from a BLIP3D checkpoint, or, to start stage 1, from the released
TRELLIS.2-4B files.

Two precision layouts:
* ``released`` — as TRELLIS.2 builds the model (blocks in bf16, boundary layers and norms in fp32); used for sampling
  the separate towers.
* ``bf16`` — every parameter bf16 except complex buffers (the SS flow's RoPE phases); the training layout, also used by
  the unified model.
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
from ..utils.paths import REPO_ROOT

KINDS = ("ss", "shape", "tex")
RELEASED = {"ss": "ss_flow_img_dit_1_3B_64_bf16", "shape": "slat_flow_img2shape_dit_1_3B_512_bf16",
            "tex": "slat_flow_imgshape2tex_dit_1_3B_512_bf16"}
Layout = Literal["released", "bf16"]


def arch_config(kind: str) -> Dict:
    with open(REPO_ROOT / "configs" / "model" / f"{kind}_flow.json") as f:
        return json.load(f)


def build_flow(kind: str) -> nn.Module:
    """The architecture only (fresh initialisation); load weights afterwards."""
    from trellis2 import models
    cfg = arch_config(kind)
    return getattr(models, cfg["name"])(**cfg["args"])


def _strict_load(flow: nn.Module, sd: Dict[str, torch.Tensor], what: str) -> None:
    missing, unexpected = flow.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise KeyError(f"{what}: missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")


def released_flow(kind: str, trellis2_ckpt: str) -> nn.Module:
    """The released TRELLIS.2-4B weights (stage-1 starting point)."""
    from safetensors.torch import load_file
    flow = build_flow(kind)
    _strict_load(flow, load_file(os.path.join(trellis2_ckpt, "ckpts", RELEASED[kind] + ".safetensors")),
                 f"TRELLIS.2-4B {RELEASED[kind]}")
    return flow


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


@dataclass
class Tower:
    kind: str
    flow: nn.Module
    connector: Connector
    views: Optional[ViewCodes]


def load_tower(kind: str, ckpt_dir: str, *, use_ema: bool = True, layout: Layout = "released", device: str = "cuda",
               connector_dtype: torch.dtype = torch.float32) -> Tower:
    """A trained tower for sampling: flow + its connector + the view codes, all keys strict."""
    sd = tower_state(ckpt_dir, kind, use_ema=use_ema)
    flow = build_flow(kind)
    _strict_load(flow, {k[5:]: v for k, v in sd.items() if k.startswith("flow.")}, f"{ckpt_dir} [{kind}]")
    if layout == "bf16":
        flow = to_bf16_keep_complex(flow)
    conn = Connector()
    conn.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("connector.")}, strict=True)
    views = ViewCodes(table=sd["views.table"]) if "views.table" in sd else None
    flow = flow.to(device).eval().requires_grad_(False)
    conn = conn.to(device).to(connector_dtype).eval().requires_grad_(False)
    return Tower(kind, flow, conn, views.to(device) if views is not None else None)
