# VENDORED VERBATIM from ROAD @ c847391 —
# step1x3d_geometry/systems/shape_rectified_flow.py:29-41 (class AlignMLP).
# Vendored verbatim from upstream ROAD; do not edit.
#
# Note: structurally this is the same 3-layer SiLU MLP as the official REPA
# build_mlp (Linear→SiLU→Linear→SiLU→Linear, hidden 2048). We vendor ROAD's own class anyway so the ROAD path is traceable
# line-for-line to their release, not to an equivalence argument.
import torch.nn as nn


class AlignMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features):
        return self.layers(features)
