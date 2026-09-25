# Dataset-side ROAD teacher input: numpy only, so dataloader workers never import timm.
# Mirrors ROAD @ c847391 step1x3d_geometry/data/base.py:63-65,76-78 (_sample_rows): uniform random choice, with
# replacement only when the cloud is short. The candidate pool is the occupied 32^3 voxel centres in the TRELLIS frame
# (idx + 0.5)/R - 0.5.
from typing import Optional

import numpy as np


def sample_align_points(coords: np.ndarray, n_points: int, resolution: int = 32,
                        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """coords (N, 3) uint8/int voxel indices -> (n_points, 3) float32 in [-0.5, 0.5]^3.
    ``rng`` None = a fresh unseeded ``default_rng()``, exactly v12; pass one to make the draw reproducible."""
    centers = (coords.astype(np.float32) + 0.5) / float(resolution) - 0.5
    indices = (rng if rng is not None else np.random.default_rng()).choice(
        len(centers), n_points, replace=len(centers) < n_points)
    return centers[indices].astype(np.float32)
