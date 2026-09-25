# VENDORED VERBATIM from ROAD @ c847391 — training/uni3d/models/uni3d.py
# Vendored verbatim from upstream ROAD; do not edit.
# Modified from Uni3D: reduced to point-encoder construction for ROAD training.
import timm

from .point_encoder import PointcloudEncoder


def create_uni3d(args):
    point_transformer = timm.create_model(
        args.pc_model,
        checkpoint_path=args.pretrained_pc,
        drop_path_rate=args.drop_path_rate,
    )
    return PointcloudEncoder(point_transformer, args)
