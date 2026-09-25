"""Latent targets: loading, normalization, voxel cap, tex placeholder, ROAD align points. Runs in dataloader workers.

Contracts:
* SS latent ``ss_latent_64`` npz key ``z`` (fallback ``latent``), (8, 16, 16, 16) float32, NOT normalized
  (the TRELLIS.2 SS config has no normalization).
* shape / pbr npz: ``coords`` (N, 3) uint8 -> int32, ``feats`` (N, 32) float32, normalized per channel with the
  TRELLIS.2 stage-config statistics (:func:`blip3d.models.towers.latent_stats`, which raises when a config or key is
  missing — v12's loader silently skipped normalization in that case).
* tex target = pbr in pbr-normalization plus ``concat_cond`` = the shape latent in the tex stage's
  shape-normalization (bit-identical to the shape tower's), both with a zero batch column; coords must match.
* placeholder tex (unified model only): zeros on the shape's coords, ``concat_cond`` = the shape feats already
  loaded (shape-normalized; same statistics), ``tex_valid = False``.
* ROAD align points: 10 000 voxel centres of the 32^3 GT occupancy in [-0.5, 0.5]^3, sampled with replacement only
  when the asset has fewer voxels.
"""
from __future__ import annotations

import os
import zipfile
import zlib
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import torch


# ── skip reasons: the ONLY failures a task resamples on (everything else is a bug and propagates) ──
class SkipSample(Exception):
    reason = "skip"


class VoxelCapExceeded(SkipSample):
    reason = "voxel_cap"


class CoordsMismatch(SkipSample):
    reason = "coords_mismatch"


class MissingField(SkipSample):
    reason = "missing_field"


class MissingFile(SkipSample):
    reason = "missing_file"


# I/O-level failures of one asset: missing / truncated / corrupt files, undecodable images
IO_ERRORS = (OSError, EOFError, zipfile.BadZipFile, zlib.error)


def skip_reason(e: BaseException) -> Optional[str]:
    """Whitelist of per-sample failures that justify a resample; ``None`` = propagate."""
    if isinstance(e, SkipSample):
        return e.reason
    if isinstance(e, IO_ERRORS):
        return "io:" + type(e).__name__
    return None


def sparse():
    """``trellis2.modules.sparse`` (requires :func:`blip3d.utils.backend.setup` to have put TRELLIS.2 on the path)."""
    try:
        from trellis2.modules import sparse as sp
    except ImportError as e:  # pragma: no cover
        raise ImportError("trellis2 is not importable: call blip3d.utils.backend.setup(...) before building "
                          "the data pipeline") from e
    return sp


@lru_cache(maxsize=1)
def norm_stats():
    """Per-channel (1, C) statistics: shape, pbr, and the tex stage's shape statistics."""
    from ..models.towers import latent_stats
    from ..utils.backend import TRELLIS2_ROOT
    return latent_stats(str(TRELLIS2_ROOT))


def load_ss(path: Optional[str]) -> torch.Tensor:
    if not path:
        raise MissingField("ss_latent_64")
    a = np.load(path)
    key = "z" if "z" in a.files else "latent"
    return torch.tensor(a[key]).float()


def read_coords(path: str) -> np.ndarray:
    """Only the ``coords`` member (the cheap voxel-count pre-check)."""
    with np.load(path) as a:
        return np.asarray(a["coords"])


def load_slat_raw(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    a = np.load(path)
    return torch.tensor(a["coords"]).int(), torch.tensor(a["feats"]).float()


def normalize(feats: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (feats - mean) / std


def load_shape(path: Optional[str], max_voxels: int = 0) -> Dict[str, torch.Tensor]:
    """-> {"coords": (N, 3) int32, "feats": (N, 32) float32 shape-normalized}; voxel cap raises
    :class:`VoxelCapExceeded` (0 disables the cap)."""
    if not path:
        raise MissingField("shape_latent_512")
    coords, feats = load_slat_raw(path)
    if max_voxels > 0 and coords.shape[0] > max_voxels:
        raise VoxelCapExceeded(f"{coords.shape[0]} voxels > {max_voxels}")
    st = norm_stats()
    return {"coords": coords, "feats": normalize(feats, st.shape_mean, st.shape_std)}


def _with_batch_col(coords: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=1)


def load_tex(pbr_path: str, shape_path: str, shape_raw: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """-> {"x_0": SparseTensor pbr-normalized, "concat_cond": SparseTensor tex-shape-normalized}.
    ``shape_raw``: the (coords, feats) of ``shape_path`` already read by the caller (v12 re-read the file; the
    values are identical)."""
    sp = sparse()
    st = norm_stats()
    pc, pf = load_slat_raw(pbr_path)
    pbr = sp.SparseTensor(normalize(pf, st.tex_mean, st.tex_std), _with_batch_col(pc))
    sc, sf = shape_raw if shape_raw is not None else load_slat_raw(shape_path)
    shp = sp.SparseTensor(normalize(sf, st.tex_shape_mean, st.tex_shape_std), _with_batch_col(sc))
    if not torch.equal(shp.coords, pbr.coords):
        raise CoordsMismatch(f"shape {tuple(shp.coords.shape)} vs pbr {tuple(pbr.coords.shape)}")
    return {"x_0": pbr, "concat_cond": shp}


def tex_placeholder(shape_item: Dict[str, torch.Tensor]):
    """Zeros on the shape's coords; ``concat_cond`` = the shape feats (shape-normalized, v12 behaviour)."""
    sp = sparse()
    coords = _with_batch_col(shape_item["coords"])
    feats = shape_item["feats"]
    return {"x_0": sp.SparseTensor(torch.zeros(coords.shape[0], feats.shape[1], dtype=feats.dtype), coords),
            "concat_cond": sp.SparseTensor(feats, coords)}


def sample_align_points(coords: np.ndarray, n_points: int, rng: Optional[np.random.Generator] = None,
                        resolution: int = 32) -> np.ndarray:
    """ROAD teacher cloud: (n_points, 3) float32 voxel centres in [-0.5, 0.5]^3 (ROAD's ``_sample_rows``).
    ``rng=None`` draws from a fresh unseeded generator, as v12 did."""
    centers = (np.asarray(coords).astype(np.float32) + 0.5) / float(resolution) - 0.5
    g = rng if rng is not None else np.random.default_rng()
    idx = g.choice(len(centers), n_points, replace=len(centers) < n_points)
    return centers[idx].astype(np.float32)


def render_path(renders_dir: str, view: int) -> Optional[str]:
    for ext in ("webp", "png", "jpg"):
        c = os.path.join(renders_dir, f"{view:03d}.{ext}")
        if os.path.isfile(c):
            return c
    return None
