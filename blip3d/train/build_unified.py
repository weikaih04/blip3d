"""S3: the unified three-tower model, warm-started from the three S2 finals, wrapped with its three connectors.

As v12 (``assemble_unified_tri`` + ``load_tri_connectors``): geo = the shape tower, tex = the tex tower, ss = the SS
tower (strict loads); the new coupling parameters (t mixers, cross gates, SS read gate) are created fresh by
Blip3DUnified; connectors geo / tex / ss come from each tower's checkpoint; the view-code table from the tex tower.
Every block of all three towers is gradient-checkpointed (v12 GC=1.0). Checkpoint names: ``model.*``,
``connectors.{geo,tex,ss}.*``, ``views.table``.
"""
from __future__ import annotations

import contextlib
from typing import Dict

import torch
import torch.nn as nn

from ..cond.connector import Connector
from ..cond.stamp import ViewCodes
from ..losses.unified import UnifiedLossCfg, UnifiedLossState, unified_loss
from ..models.flows import build_flow, to_bf16_keep_complex
from ..models.unified.model import Blip3DUnified
from .ckpt import tower_state
from .config import TrainConfig


class Blip3DUnifiedTrainable(nn.Module):
    def __init__(self, model: Blip3DUnified, connectors: Dict[str, Connector], views: ViewCodes, cfg: UnifiedLossCfg):
        super().__init__()
        self.model = model
        self.connectors = nn.ModuleDict(connectors)
        self.views = views
        self.cfg = cfg
        self.t_generator = None          # S3 draws its timesteps on the per-rank CUDA stream
        self._micro = 0
        self._state = UnifiedLossState()

    def memory_context(self):
        return contextlib.nullcontext()

    def forward(self, batch: Dict):
        loss, logs = unified_loss(self.model, dict(self.connectors), self.views, batch, batch["cond"],
                                  step=self._micro, generator=None, cfg=self.cfg, state=self._state)
        self._micro += 1
        return loss, logs


def _tower(kind: str, r: TrainConfig, trellis2_ckpt: str):
    sd = tower_state(r.resolve(r.init[kind]), kind, use_ema=bool(r.init.get("use_ema", False)), require_ema=False)
    flow = to_bf16_keep_complex(build_flow(kind))
    flow.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("flow.")}, strict=True)
    conn = Connector()
    conn.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("connector.")}, strict=True)
    return flow, conn, sd.get("views.table")


def build_unified_trainable(r: TrainConfig, trellis2_ckpt: str) -> Blip3DUnifiedTrainable:
    geo, c_geo, _ = _tower("shape", r, trellis2_ckpt)
    tex, c_tex, table = _tower("tex", r, trellis2_ckpt)
    ss, c_ss, _ = _tower("ss", r, trellis2_ckpt)
    model = Blip3DUnified(geo, tex, ss)
    model.fused_attn = True
    for f in (model.geo_flow, model.tex_flow, model.ss_flow):
        for b in f.blocks:
            b.use_checkpoint = True
    cfg = UnifiedLossCfg(**{**r.unified, "compat_v12": bool(r.compat.unified_v12_drops)})
    m = Blip3DUnifiedTrainable(model, {"geo": c_geo, "tex": c_tex, "ss": c_ss},
                               ViewCodes(table=table) if table is not None else ViewCodes(), cfg)
    for p in m.parameters():
        p.requires_grad_(True)
    return m
