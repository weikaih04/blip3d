"""Build the trainable module of a training config: a Blip3DTower (S1, S2, continuation) or the unified model (S3)."""
from __future__ import annotations

import torch

from ..cond.connector import Connector
from ..cond.stamp import ViewCodes
from ..models.blip3d import Blip3DTower, enable_elastic_gc
from ..models.towers import released_flow, to_bf16_keep_complex
from .ckpt import tower_state
from .config import TrainConfig


def build_tower(r: TrainConfig, trellis2_ckpt: str) -> Blip3DTower:
    """Stock TRELLIS.2 flow in the uniform-bf16 training layout + a fresh connector + view codes (+ ROAD on SS), then
    ``init`` weights if given. Buffers (codes, ROAD step) are never trainable."""
    flow = to_bf16_keep_complex(released_flow(r.tower, trellis2_ckpt))
    road = None
    if r.road:
        from ..losses.road import RoadHead
        road = RoadHead()
    m = Blip3DTower(r.tower, flow, Connector(), ViewCodes(), road)
    if r.init:
        sd = tower_state(r.resolve(r.init["from"]), r.tower, use_ema=bool(r.init.get("use_ema", False)),
                         require_ema=False)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        allowed = {"road.step"}   # a checkpoint trained without ROAD starts the counter at 0
        bad_missing = [k for k in missing if k not in allowed]
        if bad_missing or unexpected:
            raise KeyError(f"init {r.init['from']}: missing {bad_missing[:5]} ({len(bad_missing)}), "
                           f"unexpected {unexpected[:5]} ({len(unexpected)})")
    for n, p in m.named_parameters():
        p.requires_grad_(r.trainable == "all" or not n.startswith("flow."))
    if r.elastic:                     # v12: shape/tex in S1 (frozen flow, activations kept for the connector) and S2
        enable_elastic_gc(m.flow)
    return m
