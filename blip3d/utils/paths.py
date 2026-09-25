"""Filesystem locations, read from ``paths.yaml``. Package code never hard-codes an absolute path.

Lookup order for the file: ``$BLIP3D_PATHS`` → ``<repo>/configs/paths.yaml``. See ``configs/paths.example.yaml``.
Model identities are written as ``<hf id>@<revision>`` and resolved to the local snapshot in the HF cache,
so a checkpoint can record exactly which encoder weights it was trained against.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    data_root: str
    manifests: str
    runs: str
    hf_cache: str
    trellis2_ckpt: str
    ss_decoder: str
    encoders: dict = field(default_factory=dict)
    uni3d_ckpt: Optional[str] = None
    voxel_table: Optional[str] = None

    def hf_snapshot(self, ident: str) -> str:
        """``'org/name@rev'`` → local snapshot dir in ``hf_cache`` (rev may be a prefix). Raises if absent."""
        return resolve_hf_snapshot(ident, self.hf_cache)


def resolve_hf_snapshot(ident: str, hf_cache: str) -> str:
    if "@" not in ident:
        raise ValueError(f"model identity must be pinned as '<hf id>@<revision>', got {ident!r}")
    repo, rev = ident.split("@", 1)
    snaps = Path(hf_cache) / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        snaps = Path(hf_cache) / ("models--" + repo.replace("/", "--")) / "snapshots"
    hits = sorted(p for p in snaps.glob(rev + "*")) if snaps.is_dir() else []
    if len(hits) != 1:
        raise FileNotFoundError(f"{ident}: expected exactly one snapshot matching {rev!r} under {snaps}, found {len(hits)}")
    return str(hits[0])


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    f = os.environ.get("BLIP3D_PATHS") or str(REPO_ROOT / "configs" / "paths.yaml")
    if not os.path.isfile(f):
        raise FileNotFoundError(f"paths file not found: {f} (copy configs/paths.example.yaml to configs/paths.yaml)")
    with open(f) as fh:
        d = yaml.safe_load(fh)
    return Paths(**d)
