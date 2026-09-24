"""ROAD alignment on the SS tower (v12 S2 SS and the SS continuation).

What v12 did, per training micro-step (trellis_native_vlm.py _road_kit + flow_heads.py:320-384, 488-493):
  1. a forward hook on ``ss_flow.blocks[9]`` (1-indexed depth num_blocks // 3 = 10 of 30) stashes that block's output
     (B, 4096, 1536) during the SS flow forward;
  2. the frozen Uni3D-g teacher encodes the row's GT point cloud (``align_points``, 10k voxel centres) under bf16
     autocast -> (B, 513, 1024), CLS first;
  3. ``road_alignment_losses``: projector (AlignMLP 1536 -> 2048 -> 2048 -> 1024) on the tap; loss_proj = 1 - cos of
     the mean-normalized global tokens; loss_opt = Hungarian cosine matching of 512 pooled student tokens against the
     teacher's patch tokens, only once the micro-step counter reached ``opt_start`` (read BEFORE it is advanced);
  4. total += 0.5 * loss_proj + 0.1 * loss_opt, outside the flow-stage weighting.
Every row is aligned, CFG-dropped and DINO-dropped rows included (deliberate in v12: ROAD has no cond dropout).
The counter is a persistent int64 buffer, so the gate survives resume and carries across stages (402000 at the end of
S2 SS); its unit is micro-steps, not optimizer steps.

Checkpoint key map from v12 (model.safetensors and ema.safetensors of the SS tower runs):
  repa_projector.layers.{0,2,4}.{weight,bias}  ->  road.projector.layers.{0,2,4}.{weight,bias}
  _road_step                                   ->  road.step
``v12_road_state(sd)`` returns the RoadHead-local state (``projector.*``, ``step``) for ``RoadHead.load_state_dict``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..models.road import AlignMLP, road_alignment_losses

TAP_DEPTH = 10                  # 1-indexed: num_blocks // 3 for the 30-block SS flow -> ss_flow.blocks[9]
TEACHER_DIM = 1024              # Uni3D-g embed_dim
PROJECTOR_HIDDEN = 2048         # release projector_dim
LAMBDA_PROJ, LAMBDA_OPT = 0.5, 0.1
OPT_START_MICRO_STEPS = 1000    # v12 SS runs (release: epoch 3 of 600)
TOKEN_COUNT = 512
N_POINTS = 10000

V12_KEY_MAP = {"repa_projector.": "road.projector.", "_road_step": "road.step"}


def v12_road_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """v12 SS-tower state dict -> RoadHead state dict (keys relative to the head)."""
    out = {}
    for k, v in sd.items():
        if k.startswith("repa_projector."):
            out["projector." + k[len("repa_projector."):]] = v
        elif k == "_road_step":
            out["step"] = v
    return out


class RoadHead(nn.Module):
    """Trainable projector + gate counter; the teacher and the matcher are built on the first loss call and held
    outside the module tree (never in state_dict / EMA / optimizer)."""

    def __init__(self, in_dim: int = 1536, *, token_count: int = TOKEN_COUNT, lambda_proj: float = LAMBDA_PROJ,
                 lambda_opt: float = LAMBDA_OPT, opt_start: int = OPT_START_MICRO_STEPS,
                 uni3d_ckpt: Optional[str] = None, tap_depth: int = TAP_DEPTH):
        super().__init__()
        self.projector = AlignMLP(input_dim=in_dim, hidden_dim=PROJECTOR_HIDDEN, output_dim=TEACHER_DIM)
        self.token_pool = nn.AdaptiveAvgPool1d(token_count)          # no parameters
        self.register_buffer("step", torch.zeros((), dtype=torch.long), persistent=True)
        self.lambda_proj, self.lambda_opt, self.opt_start = float(lambda_proj), float(lambda_opt), int(opt_start)
        self.tap_depth = int(tap_depth)
        object.__setattr__(self, "_uni3d_ckpt", uni3d_ckpt)
        object.__setattr__(self, "_stash", {})
        object.__setattr__(self, "_hook", None)
        object.__setattr__(self, "_teacher", None)
        object.__setattr__(self, "_matcher", None)

    # ── the tap ──
    def attach(self, ss_flow: nn.Module) -> None:
        """Hook ``ss_flow.blocks[tap_depth - 1]``: its output is stashed during training forwards only."""
        nb = len(ss_flow.blocks)
        if not 1 <= self.tap_depth <= nb:
            raise ValueError(f"[road] tap depth {self.tap_depth} out of range 1..{nb}")
        if self._hook is not None:
            self._hook.remove()
        stash, head = self._stash, self

        def _hook(_mod, _inp, out):
            if head.training:                       # sampling never stashes
                stash["h"] = out

        object.__setattr__(self, "_hook", ss_flow.blocks[self.tap_depth - 1].register_forward_hook(_hook))

    def pop_tap(self) -> torch.Tensor:
        h = self._stash.pop("h", None)
        if h is None:
            raise RuntimeError("[road] the SS block hook stashed nothing: hook not attached, head not in training "
                               "mode, or the SS flow did not run this micro-step")
        return h

    # ── frozen parts ──
    def _ensure_matcher(self):
        if self._matcher is None:
            from ..models.road.matcher_gpu import HungarianMatcherWithLossGPU
            object.__setattr__(self, "_matcher", HungarianMatcherWithLossGPU())
        return self._matcher

    def _build(self, device) -> None:
        """Teacher (2 GB fp32, moved lazily to the row's device) and matcher, on the first training call, as v12."""
        if self._teacher is None:
            from ..models.road.teacher import UNI3D_G_ARGS, Uni3DTeacher
            assert int(UNI3D_G_ARGS.embed_dim) == self.projector.layers[-1].out_features, \
                "[road] projector out_dim no longer matches the teacher embed_dim"
            object.__setattr__(self, "_teacher", Uni3DTeacher(self._uni3d_ckpt))
            self._ensure_matcher()
        if next(self._teacher.parameters()).device != torch.device(device):
            self._teacher.to(device)

    def teacher_tokens(self, align_points: torch.Tensor) -> torch.Tensor:
        self._build(align_points.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self._teacher(align_points)                         # (B, 1 + M, 1024)

    def next_gate(self) -> bool:
        """Read the micro-step counter, then advance it (v12 _road_kit order)."""
        step = int(self.step.item())
        self.step += 1
        return step >= self.opt_start

    # ── the loss ──
    def loss(self, block_tokens: torch.Tensor, align_points: torch.Tensor, opt_enabled: Optional[bool] = None,
             teacher_tokens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """-> (lambda_proj * loss_proj + lambda_opt * loss_opt, {"road_proj", "road_opt"} detached).
        ``opt_enabled`` None = use and advance the persistent counter (one call per training micro-step)."""
        if opt_enabled is None:
            opt_enabled = self.next_gate()
        if teacher_tokens is None:
            teacher_tokens = self.teacher_tokens(align_points.to(block_tokens.device))
        elif opt_enabled:
            self._ensure_matcher()
        pdt = next(self.projector.parameters()).dtype
        r = road_alignment_losses(intermediate=block_tokens.to(pdt), teacher_tokens=teacher_tokens,
                                  align_mlp=self.projector, token_pool=self.token_pool, matcher=self._matcher,
                                  opt_enabled=bool(opt_enabled))
        total = self.lambda_proj * r["loss_proj"] + self.lambda_opt * r["loss_opt"]
        return total, {"road_proj": r["loss_proj"].detach().float(), "road_opt": r["loss_opt"].detach().float()}
