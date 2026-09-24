# VENDORED from ROAD @ c847391 — step1x3d_geometry/systems/shape_rectified_flow.py:175-217,
# refactored from an inline training_step block into a callable. Every tensor op is kept
# verbatim (order, .float() placements, [:, 1:].detach() CLS handling, the AdaptiveAvgPool1d
# transpose dance). The refactor only names the inputs the block read from `self`/locals:
#   intermediate     <- the tap'd DiT hidden states  (ROAD: 6th double-stream block output)
#   align_mlp        <- self.align_mlp (AlignMLP)
#   token_pool       <- self.token_pool = nn.AdaptiveAvgPool1d(alignment_token_count)
#   matcher          <- self.matcher (Hungarian matcher, cpu or gpu variant)
#   teacher_tokens   <- self.align_model(batch)[0]  (computed by the caller under no_grad)
#   opt_enabled      <- (self.current_epoch >= self.cfg.alignment_start_epoch); our trainer
#                       is step-based, so the caller derives this from a step threshold —
#                       the release used epoch 3 of 600 (~0.5% of the run).
# Their training_step then combines: total = L_diff*λ_diff + L_proj*λ_proj + L_opt*λ_opt
# (release config: λ_proj=0.5, λ_opt=0.1) — the caller does that part.
from typing import Dict

import torch
import torch.nn.functional as F


def road_alignment_losses(
    intermediate: torch.Tensor,        # (B, N, C) tap'd hidden states (grad flows)
    teacher_tokens: torch.Tensor,      # (B, 1+M, z) Uni3D output, CLS first (no grad)
    align_mlp,                         # AlignMLP: C -> z
    token_pool,                        # nn.AdaptiveAvgPool1d(M)
    matcher,                           # HungarianMatcherWithLoss{,GPU}
    opt_enabled: bool,
) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}

    # ── verbatim block (shape_rectified_flow.py:176-195) ──────────────────────
    student_tokens = align_mlp(intermediate)

    teacher_global = F.normalize(teacher_tokens.float(), dim=-1).mean(dim=1)
    student_global = F.normalize(student_tokens.float(), dim=-1).mean(dim=1)
    result["loss_proj"] = 1.0 - F.cosine_similarity(
        teacher_global, student_global, dim=-1
    ).mean()

    if opt_enabled:
        pooled_student = token_pool(student_tokens.transpose(1, 2)).transpose(1, 2)
        # Uni3D token 0 is the CLS token; local matching uses patch tokens only.
        teacher_local = teacher_tokens[:, 1:].detach()
        result["loss_opt"] = matcher(
            pooled_student.float(), teacher_local.float()
        )
    else:
        result["loss_opt"] = intermediate.new_zeros(())
    # ──────────────────────────────────────────────────────────────────────────
    return result
