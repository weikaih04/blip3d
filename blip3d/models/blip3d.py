"""Blip3D: one TRELLIS.2 tower with its conditioning connector, as trained in S1 / S2 / the SS continuation.

Registration order is flow -> connector -> ROAD head, the order of the v12 module (``ss_flow`` / ``shape_slat_512`` /
``tex_slat_512`` were built before ``diffusion_connector`` and ``repa_projector``), so the trainable parameters of a
ZeRO-1 optimizer flatten in the same order. Checkpoint names: ``flow.*``, ``connector.*``, ``views.table``,
``road.*``; v12 names are mapped by :func:`blip3d.train.ckpt.from_v12`.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..cond.assemble import assemble, draw_drops
from ..cond.connector import Connector
from ..cond.stamp import ViewCodes
from ..losses.flow import flow_loss

P_CFG = 0.1      # whole-row CFG drop (TRELLIS p_uncond)
P_DINO = 0.3     # DINO segment absent
P_QWEN = 0.0     # Qwen segment absent: S3 only


class Blip3DTower(nn.Module):
    def __init__(self, kind: str, flow: nn.Module, connector: Connector, views: ViewCodes,
                 road: Optional[nn.Module] = None):
        super().__init__()
        assert kind in ("ss", "shape", "tex"), kind
        self.kind = kind
        self.flow = flow
        self.connector = connector
        self.views = views
        if road is not None:
            self.road = road
            road.attach(flow)                 # stashes ss_flow.blocks[9] in training forwards
        self.t_generator: Optional[torch.Generator] = None   # set by the trainer (per-rank); None = v12 shared CPU draw

    @property
    def has_road(self) -> bool:
        return hasattr(self, "road")

    def memory_context(self):
        ctrl = getattr(self.flow, "_memory_controller", None)
        return ctrl.record() if ctrl is not None else contextlib.nullcontext()

    def conditioning(self, cond_batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training conditioning with the v12 dropout stack (drawn here, in v12's RNG order)."""
        B = cond_batch["qwen"].shape[0]
        has_dino = "dino" in cond_batch
        drops = draw_drops(B, has_dino=has_dino, p_cfg=P_CFG, p_dino=P_DINO if (has_dino and self.training) else 0.0,
                           p_qwen=P_QWEN, device=cond_batch["qwen"].device,
                           text_draw="rand" if self.has_road else "bernoulli", dtype=torch.bfloat16)
        return assemble(self.connector, self.views, cond_batch, drops, dtype=torch.bfloat16)

    def forward(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cond, mask = self.conditioning(batch["cond"])
        logs: Dict[str, torch.Tensor] = {}
        if self.kind == "ss":
            loss, lg = flow_loss("ss", self.flow, batch["ss"].to(cond.dtype), cond, mask, generator=self.t_generator)
            logs["loss_ss"] = loss.detach()
            total = loss
            if self.has_road and self.training:   # ROAD aligns CFG- and DINO-dropped rows too (v12, deliberate)
                r_total, r_logs = self.road.loss(self.road.pop_tap(), batch["align_points"])
                total = total + r_total
                logs.update({f"loss_{k}": v for k, v in r_logs.items()})
        elif self.kind == "shape":
            x0 = batch["shape"]
            loss, lg = flow_loss("shape", self.flow, x0.replace(x0.feats.to(cond.dtype)), cond, mask,
                                 generator=self.t_generator)
            logs["loss_shape"] = loss.detach()
            total = loss
        else:
            x0, cc = batch["tex"], batch["tex_concat"]
            loss, lg = flow_loss("tex", self.flow, x0.replace(x0.feats.to(cond.dtype)), cond, mask,
                                 generator=self.t_generator, concat_cond=cc.replace(cc.feats.to(cond.dtype)))
            logs["loss_tex"] = loss.detach()
            total = loss
        logs["t_mean"] = lg["t_mean"]
        return total, logs


def enable_elastic_gc(flow: nn.Module, target_ratio: float = 0.75) -> nn.Module:
    """TRELLIS.2's elastic activation checkpointing on a SLAT flow (one controller per flow, as upstream)."""
    from trellis2.models.structured_latent_flow import ElasticSLatFlowModel
    from trellis2.utils.elastic_utils import LinearMemoryController
    flow.__class__ = ElasticSLatFlowModel
    flow._memory_controller = None
    flow.register_memory_controller(LinearMemoryController(buffer_size=1000, update_every=500,
                                                           target_ratio=target_ratio, max_mem_ratio_start=0.5))
    return flow
