# VENDORED from ROAD @ c847391 — step1x3d_geometry/models/transformers/uni3d_align_transformer.py (class Uni3DAlign),
# stripped of the step1x3d registry scaffolding. configure(), the train() override and forward() are verbatim.
#
# Teacher-input conventions inherited AS-IS from the release (not "fixed"):
#   * points in the dataset's native frame, no recentering or unit-sphere rescale; ours is the 32^3 voxel-centre
#     frame (idx + 0.5)/32 - 0.5 in [-0.5, 0.5]^3;
#   * colors = ones_like(points);
#   * uniform sampling with replacement when the cloud is short (points.py).
#
# Checkpoint: an explicit path, never a glob. ``uni3d_ckpt_path()`` resolves, in order, the argument, paths.yaml
# ``uni3d_ckpt``, then the pinned HF snapshot BAAI/Uni3D@3d8233b (the file v12 trained against).
import os
from types import SimpleNamespace
from typing import Optional

import torch

from .uni3d import create_uni3d

# configs/uni3d_g.json + uni3d_align_transformer.py load_uni3d_config defaults.
UNI3D_G_ARGS = SimpleNamespace(
    pc_model="eva_giant_patch14_560",
    pretrained_pc="",
    drop_path_rate=0.0,
    patch_dropout=0.0,
    pc_feat_dim=1408,
    embed_dim=1024,
    num_group=512,
    group_size=64,
    pc_encoder_dim=512,
)
UNI3D_G_SNAPSHOT = "BAAI/Uni3D@3d8233b76aa350d72f6213ecd2123c2026b42355"
UNI3D_G_FILE = os.path.join("modelzoo", "uni3d-g", "model.pt")


def uni3d_ckpt_path(explicit: Optional[str] = None) -> str:
    if explicit:
        path = explicit
    else:
        from ...utils.paths import get_paths
        p = get_paths()
        path = p.uni3d_ckpt or os.path.join(p.hf_snapshot(UNI3D_G_SNAPSHOT), UNI3D_G_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[road] Uni3D-g checkpoint not found: {path} "
                                "(download BAAI/Uni3D modelzoo/uni3d-g/model.pt, 2.03 GB)")
    return path


def point_encoder_state_dict(checkpoint):
    """Verbatim from uni3d_align_transformer.py:32-43."""
    state = checkpoint.get("module", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for name, value in state.items():
        if name == "logit_scale":
            continue
        if name.startswith("module."):
            name = name[len("module."):]
        if name.startswith("point_encoder."):
            name = name[len("point_encoder."):]
        cleaned[name] = value
    return cleaned


class Uni3DTeacher(torch.nn.Module):
    """Frozen Uni3D-g point encoder. forward(points (B, N, 3)) -> (B, 1 + num_group, embed_dim), CLS first.

    The owner holds it outside its module tree (``object.__setattr__``) so it never enters state_dict, EMA or the
    optimizer; it stays fp32 and runs under bf16 autocast (lightning "bf16-mixed", the release precision).
    """

    def __init__(self, ckpt_path: Optional[str] = None, args: SimpleNamespace = UNI3D_G_ARGS):
        super().__init__()
        self.model = create_uni3d(args)
        ckpt_path = uni3d_ckpt_path(ckpt_path)
        print(f"[road] Uni3D teacher ckpt: {ckpt_path}", flush=True)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.model.load_state_dict(point_encoder_state_dict(checkpoint), strict=False)
        if missing or unexpected:                       # verbatim: hard-fail, never half-load
            raise RuntimeError(f"Uni3D checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.model.requires_grad_(False).eval()

    def train(self, mode: bool = True):                 # verbatim: the teacher never leaves eval
        super().train(False)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, align_points: torch.Tensor) -> torch.Tensor:
        # verbatim semantics (uni3d_align_transformer.py:76-80): xyz only, colors = ones.
        points = align_points[..., :3].contiguous().float()
        colors = torch.ones_like(points)
        output = self.model(points, colors)
        return output.detach()
