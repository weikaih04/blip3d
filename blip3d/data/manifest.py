"""Training manifests: JSONL -> a compact columnar store.

v12 kept every manifest row as a Python dict (~3.9 KB each, ~14 GB per rank across the four tasks, dirtied page by
page by refcounting in every dataloader worker). Here only the fields training reads are kept, and every string
column is one contiguous byte buffer plus offsets, so a forked worker shares the whole store copy-on-write without
touching it.

Fields: ``sha256`` · ``captions`` (non-empty ones, fixed tier order [long, medium, short,
long+texture]) · ``ss_latent_64`` · ``shape_latent_<res>`` · ``pbr_latent_<res>`` · ``renders_dir`` · ``n_views``
(default 16) · ``aesthetic_score`` (None kept) · optional ``n_vox``.

Filters, in v12 order: ``min_aesthetic`` (rows with no score are KEPT), then ``require_caption`` (text tasks).
``validate=True`` (the default outside ``compat_v12``) additionally drops rows that could only ever fail at load
time — missing SS latent, missing renders for image tasks, non-hex id for text tasks (the template hash needs hex),
over-cap voxel counts when ``n_vox`` or a voxel table is known — and prints how many. ``validate=False`` keeps the
exact v12 record list, which the v12 index streams depend on.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import orjson as _json

    def _loads(line: bytes):
        return _json.loads(line)
except ImportError:  # pragma: no cover
    import json as _json

    def _loads(line: bytes):
        return _json.loads(line)

_HEX = re.compile(r"^[0-9a-fA-F]{8,}$")


class StrColumn:
    """Immutable column of strings: one uint8 buffer + int64 offsets. ``""`` stands for absent."""

    __slots__ = ("_buf", "_off")

    def __init__(self, values: Sequence[Optional[str]]):
        enc = [(v or "").encode("utf-8") for v in values]
        off = np.zeros(len(enc) + 1, dtype=np.int64)
        np.cumsum([len(b) for b in enc], out=off[1:])
        self._off = off
        self._buf = np.frombuffer(b"".join(enc), dtype=np.uint8).copy() if off[-1] else np.zeros(0, np.uint8)

    @classmethod
    def _from_parts(cls, buf: np.ndarray, off: np.ndarray) -> "StrColumn":
        c = cls.__new__(cls)
        c._buf, c._off = buf, off
        return c

    def __len__(self) -> int:
        return len(self._off) - 1

    def __getitem__(self, i: int) -> str:
        a, b = int(self._off[i]), int(self._off[i + 1])
        return self._buf[a:b].tobytes().decode("utf-8")

    def take(self, idx: np.ndarray) -> "StrColumn":
        return StrColumn([self[int(i)] for i in idx])


class RaggedStrColumn:
    """Per-row list of strings (the non-empty captions)."""

    __slots__ = ("_flat", "_rows")

    def __init__(self, rows: Sequence[Sequence[str]]):
        flat: List[str] = []
        starts = np.zeros(len(rows) + 1, dtype=np.int64)
        for i, r in enumerate(rows):
            flat.extend(r)
            starts[i + 1] = len(flat)
        self._flat = StrColumn(flat)
        self._rows = starts

    def __len__(self) -> int:
        return len(self._rows) - 1

    def __getitem__(self, i: int) -> List[str]:
        return [self._flat[j] for j in range(int(self._rows[i]), int(self._rows[i + 1]))]

    def count(self, i: int) -> int:
        return int(self._rows[i + 1] - self._rows[i])

    def take(self, idx: np.ndarray) -> "RaggedStrColumn":
        return RaggedStrColumn([self[int(i)] for i in idx])


@dataclass(frozen=True)
class Record:
    """One asset as a task sees it. Absent paths are ``None``."""
    sha: str
    captions: List[str]
    ss: Optional[str]
    shape: Optional[str]
    pbr: Optional[str]
    renders_dir: Optional[str]
    n_views: int
    aesthetic: Optional[float]
    n_vox: Optional[int]


class Manifest:
    """Columnar manifest. ``len(m)`` rows; ``m.record(i)`` builds one :class:`Record` on demand."""

    def __init__(self, path: str, cols: Dict[str, object], n_raw: int, notes: List[str]):
        self.path = path
        self.sha: StrColumn = cols["sha"]
        self.captions: RaggedStrColumn = cols["captions"]
        self.ss: StrColumn = cols["ss"]
        self.shape: StrColumn = cols["shape"]
        self.pbr: StrColumn = cols["pbr"]
        self.renders_dir: StrColumn = cols["renders_dir"]
        self.n_views: np.ndarray = cols["n_views"]
        self.aesthetic: np.ndarray = cols["aesthetic"]      # float64 (exact v12 comparison), NaN = no score
        self.n_vox: np.ndarray = cols["n_vox"]              # int32, -1 = unknown
        self.n_raw = n_raw
        self.notes = notes

    def __len__(self) -> int:
        return len(self.sha)

    def record(self, i: int) -> Record:
        a = float(self.aesthetic[i])
        v = int(self.n_vox[i])
        return Record(sha=self.sha[i], captions=self.captions[i], ss=self.ss[i] or None,
                      shape=self.shape[i] or None, pbr=self.pbr[i] or None,
                      renders_dir=self.renders_dir[i] or None, n_views=int(self.n_views[i]),
                      aesthetic=None if np.isnan(a) else a, n_vox=None if v < 0 else v)

    def shas(self) -> List[str]:
        return [self.sha[i] for i in range(len(self))]

    def summary(self) -> str:
        extra = ("; " + "; ".join(self.notes)) if self.notes else ""
        return f"{self.path}: {self.n_raw} -> {len(self)} rows{extra}"


def _parse(path: str, res: int) -> Tuple[Dict[str, list], int]:
    shas, caps, has_caps, ss, shape, pbr, rdir, nv, aes, nvox = [], [], [], [], [], [], [], [], [], []
    n = 0
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            r = _loads(line)
            shas.append(str(r.get("sha256") or ""))
            raw_caps = r.get("captions") or []
            has_caps.append(bool(raw_caps))
            caps.append([c for c in raw_caps if c])
            ss.append(r.get("ss_latent_64"))
            shape.append(r.get(f"shape_latent_{res}"))
            pbr.append(r.get(f"pbr_latent_{res}"))
            rdir.append(r.get("renders_dir"))
            nv.append(int(r.get("n_views", 16) or 16))
            a = r.get("aesthetic_score")
            aes.append(np.nan if a is None else float(a))
            v = r.get("n_vox")
            nvox.append(-1 if v is None else int(v))
    return dict(sha=shas, captions=caps, has_caps=has_caps, ss=ss, shape=shape, pbr=pbr, renders_dir=rdir, n_views=nv,
                aesthetic=aes, n_vox=nvox), n


_CACHE: Dict[tuple, Manifest] = {}


def load_manifest(path: str, *, slat_resolution: int = 512, min_aesthetic: Optional[float] = None,
                  require_caption: bool = False, validate: bool = False, need_renders: bool = False,
                  need_hex_sha: bool = False, voxel_cap: int = 0,
                  vox_lookup: Optional[Callable[[str], Optional[int]]] = None) -> Manifest:
    """Read and filter one manifest. Results are cached per process, so two tasks on the same pool (single and
    multi image) share one store. ``require_caption`` keeps rows whose raw ``captions`` list is non-empty, exactly
    as v12 (a list of empty strings passes; such a row fails later)."""
    key = (os.path.abspath(path), slat_resolution, min_aesthetic, require_caption, validate, need_renders,
           need_hex_sha, voxel_cap, id(vox_lookup) if vox_lookup else None)
    if key in _CACHE:
        return _CACHE[key]
    raw, n_raw = _parse(path, slat_resolution)
    keep = np.ones(n_raw, dtype=bool)
    notes: List[str] = []
    aes = np.asarray(raw["aesthetic"], dtype=np.float64)
    if min_aesthetic is not None:
        k = np.isnan(aes) | (aes >= float(min_aesthetic))
        notes.append(f"min_aesthetic {min_aesthetic}: -{int((~k).sum())} (no-score rows kept)")
        keep &= k
    if require_caption:
        # v12 tested the RAW list (`if r.get("captions")`): a row of empty strings passes here and fails at load
        k = np.asarray(raw["has_caps"], dtype=bool)
        notes.append(f"require_caption: -{int((keep & ~k).sum())}")
        keep &= k
    if validate:
        def drop(mask: np.ndarray, why: str):
            nonlocal keep
            m = keep & mask
            if m.any():
                notes.append(f"{why}: -{int(m.sum())}")
            keep &= ~mask
        drop(np.array([not s for s in raw["ss"]]), "no ss_latent_64")
        if require_caption:
            drop(np.array([len(c) == 0 for c in raw["captions"]]), "only empty captions")
        if need_renders:
            drop(np.array([not s for s in raw["renders_dir"]]), "no renders_dir")
        if need_hex_sha:
            drop(np.array([not _HEX.match(s or "") for s in raw["sha"]]), "non-hex id")
        if voxel_cap > 0:
            vox = np.asarray(raw["n_vox"], dtype=np.int64)
            if vox_lookup is not None:
                unknown = vox < 0
                if unknown.any():
                    idx = np.flatnonzero(unknown & keep)
                    for i in idx:
                        v = vox_lookup(raw["sha"][i])
                        vox[i] = -1 if v is None else int(v)
                raw["n_vox"] = vox.tolist()
            drop(vox > voxel_cap, f"n_vox > {voxel_cap}")
    idx = np.flatnonzero(keep)
    cols = {
        "sha": StrColumn([raw["sha"][i] for i in idx]),
        "captions": RaggedStrColumn([raw["captions"][i] for i in idx]),
        "ss": StrColumn([raw["ss"][i] for i in idx]),
        "shape": StrColumn([raw["shape"][i] for i in idx]),
        "pbr": StrColumn([raw["pbr"][i] for i in idx]),
        "renders_dir": StrColumn([raw["renders_dir"][i] for i in idx]),
        "n_views": np.asarray(raw["n_views"], dtype=np.int16)[idx],
        "aesthetic": aes[idx],
        "n_vox": np.asarray(raw["n_vox"], dtype=np.int32)[idx],
    }
    m = Manifest(path, cols, n_raw, notes)
    _CACHE[key] = m
    return m


def clear_cache() -> None:
    _CACHE.clear()
