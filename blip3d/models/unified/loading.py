"""Load a v12 S3 checkpoint into Blip3DUnified + its three connectors (strict; ISSUES U-20..U-22)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from ...cond.connector import Connector
from ...cond.stamp import ViewCodes
from ...train.ckpt import connector_state, load_state
from ..towers import released_flow, to_bf16_keep_complex
from .model import Blip3DUnified

PREFIX = "unified_geotex."
CONNECTORS = {"geo": "geo_connector", "tex": "diffusion_connector", "ss": "ss_connector"}


@dataclass
class UnifiedBundle:
    model: Blip3DUnified
    connectors: Dict[str, Connector]     # keys geo / tex / ss
    views: Optional[ViewCodes]


def skeleton(trellis2_ckpt: str) -> Blip3DUnified:
    """Three flows in the training (uniform bf16) layout, wrapped. The released weights are placeholders here —
    every tensor is overwritten by the strict load that follows."""
    geo = to_bf16_keep_complex(released_flow("shape", trellis2_ckpt))
    tex = to_bf16_keep_complex(released_flow("tex", trellis2_ckpt))
    ss = to_bf16_keep_complex(released_flow("ss", trellis2_ckpt))
    return Blip3DUnified(geo, tex, ss)


def load_unified(ckpt_dir: str, *, trellis2_ckpt: str, use_ema: bool = True, device: str = "cuda",
                 connector_dtype: torch.dtype = torch.float32) -> UnifiedBundle:
    sd = load_state(ckpt_dir, use_ema=use_ema)
    model = skeleton(trellis2_ckpt)
    uni = {k[len(PREFIX):]: v for k, v in sd.items() if k.startswith(PREFIX)}
    if not uni:
        raise KeyError(f"{ckpt_dir}: no '{PREFIX}*' keys (not a unified checkpoint)")
    model.load_state_dict(uni, strict=True)
    conns = {}
    for lane, pre in CONNECTORS.items():
        c = Connector()
        c.load_state_dict(connector_state(sd, pre), strict=True)
        conns[lane] = c.to(device).to(connector_dtype).eval().requires_grad_(False)
    views = ViewCodes(table=sd["dino_view_embed"]).to(device) if "dino_view_embed" in sd else None
    return UnifiedBundle(model.to(device).eval().requires_grad_(False), conns, views)
