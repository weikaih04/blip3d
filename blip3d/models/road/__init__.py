"""ROAD representation alignment for the SS tower, vendored from the official release.

Upstream: github.com/H-EmbodVis/ROAD @ c84739164df2698ce1da63caecffcd9b ("Reciprocal-Objective Alignment of
Discriminative Semantics for 3D Shape Generation", arXiv:2607.28581); teacher from BAAI Uni3D (arXiv:2310.06773).

Policy: the algorithm is verbatim (loss formulas, matcher, projector, teacher forward with colors = ones and points in
the dataset frame, token pooling, CLS handling). The one substitution is a torch-native farthest-point sample in place
of pointnet2_ops (a compiled extension the environment does not carry). Infrastructure only: the teacher is held
outside the module tree, the token term is gated on micro-steps instead of epochs, the GPU matcher compiles on first
use instead of at import.

Source map (ROAD @ c847391):
  point_encoder.py <- training/uni3d/models/point_encoder.py
  uni3d.py         <- training/uni3d/models/uni3d.py
  teacher.py       <- step1x3d_geometry/models/transformers/uni3d_align_transformer.py
  align_mlp.py     <- step1x3d_geometry/systems/shape_rectified_flow.py:29-41 (AlignMLP)
  matcher_gpu.py   <- step1x3d_geometry/utils/hungarianmatcher_gpus.py
  loss.py          <- step1x3d_geometry/systems/shape_rectified_flow.py:175-217
Dropped from v12's copy: matcher_cpu.py (scipy, L2 cost; never used), pool.py (shape tap), evaluation/.
Release config: lambda_proj 0.5, lambda_opt 0.1, 512 tokens, matcher "gpu", 10000 points.
"""
from .align_mlp import AlignMLP          # torch.nn only
from .loss import road_alignment_losses  # torch only
from .points import sample_align_points  # numpy only: safe in dataloader workers

__all__ = ["AlignMLP", "road_alignment_losses", "sample_align_points", "Uni3DTeacher", "UNI3D_G_ARGS",
           "point_encoder_state_dict", "HungarianMatcherWithLossGPU"]


def __getattr__(name):
    # teacher.py imports timm (~7 s, +300 MB per process) and matcher_gpu.py pulls torch.utils.cpp_extension:
    # neither may load in dataloader workers that only need sample_align_points.
    if name in ("Uni3DTeacher", "UNI3D_G_ARGS", "point_encoder_state_dict"):
        from . import teacher as _t
        return getattr(_t, name)
    if name == "HungarianMatcherWithLossGPU":
        from .matcher_gpu import HungarianMatcherWithLossGPU
        return HungarianMatcherWithLossGPU
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
