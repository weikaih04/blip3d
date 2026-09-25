"""Items -> one training batch. Runs in dataloader workers and needs nothing beyond torch.

| key | when |
|---|---|
| ``task``, ``modality``, ``ids``, ``prep`` | always |
| ``ss`` (B, 8, 16, 16, 16) float32, raw | always |
| ``shape`` packed sparse ``{"coords": (N, 4) int32 [b, x, y, z], "feats": (N, 32)}`` | shape is a target |
| ``tex``, ``tex_concat`` packed sparse on the shape's coords, ``tex_valid`` (B,) bool | every row carries tex |
| ``align_points`` (B, 10000, 3) | ROAD on |

Sparse targets stay plain tensors here; :func:`to_sparse` turns them into TRELLIS.2 SparseTensors in the training
process (built exactly as TRELLIS.2's own SLat collate builds them: int32 batch column, rows in batch order, layout
registered). Workers therefore never import TRELLIS.2, whose kernels initialise CUDA on import.
Rules the downstream code relies on: one task and one modality per batch; tex all-or-none (a batch where only some
rows carry tex drops it, and says so once); align points all-or-none (raises).
``tex_valid`` is emitted whenever tex is (all True unless placeholder rows are present).
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

_PARTIAL_TEX_WARNED = False
SPARSE_KEYS = ("shape", "tex", "tex_concat")


def pack(coords: Sequence[torch.Tensor], feats: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Per-row (N_i, 3) coords and (N_i, C) feats -> one packed batch with a leading int32 batch column."""
    return {"coords": torch.cat([torch.cat([torch.full((c.shape[0], 1), i, dtype=torch.int32), c.to(torch.int32)], 1)
                                 for i, c in enumerate(coords)]),
            "feats": torch.cat(list(feats))}


def to_sparse(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Packed sparse entries -> trellis2 SparseTensors (training process only)."""
    from trellis2.modules.sparse import SparseTensor
    B = len(batch["ids"])
    for k in SPARSE_KEYS:
        d = batch.get(k)
        if not isinstance(d, dict):
            continue
        coords, feats = d["coords"], d["feats"]
        counts = torch.bincount(coords[:, 0].long().cpu(), minlength=B).tolist()
        layout, start = [], 0
        for n in counts:
            layout.append(slice(start, start + n))
            start += n
        st = SparseTensor(coords=coords, feats=feats)
        st._shape = torch.Size([B, *feats.shape[1:]])
        st.register_spatial_cache("layout", layout)
        batch[k] = st
    return batch


def collate(items: Sequence[Dict[str, Any]], *, strict_shape: bool = True) -> Dict[str, Any]:
    """``strict_shape``: every row must carry the shape target when the first does (v12 gated on row 0 only and
    then failed inside the pack)."""
    global _PARTIAL_TEX_WARNED
    if not items:
        raise ValueError("empty batch")
    task, modality = items[0]["task"], items[0]["modality"]
    for i, it in enumerate(items[1:], 1):
        if it["task"] != task or it["modality"] != modality:
            raise ValueError(f"heterogeneous batch: row 0 is {task}/{modality}, row {i} is "
                             f"{it['task']}/{it['modality']} (sampling must be batch-granular)")
    out: Dict[str, Any] = {"task": task, "modality": modality, "ids": [it["id"] for it in items],
                           "prep": [it["prep"] for it in items],
                           "ss": torch.stack([it["ss"] for it in items], 0)}
    if "shape" in items[0]:
        if strict_shape and not all("shape" in it for it in items):
            raise ValueError("shape target present on some rows only")
        out["shape"] = pack([it["shape"]["coords"] for it in items], [it["shape"]["feats"] for it in items])
    n_tex = sum("tex" in it for it in items)
    if n_tex == len(items):
        coords = [it["shape"]["coords"] for it in items]
        out["tex"] = pack(coords, [it["tex"]["feats"] for it in items])
        out["tex_concat"] = pack(coords, [it["tex"]["concat"] for it in items])
        out["tex_valid"] = torch.tensor([bool(it.get("tex_valid", True)) for it in items])
    elif n_tex and not _PARTIAL_TEX_WARNED:
        _PARTIAL_TEX_WARNED = True
        print(f"[collate] WARNING: {n_tex}/{len(items)} rows carry a tex target; the batch is collated without "
              f"texture supervision (warned once)", flush=True)
    n_pts = sum("align_points" in it for it in items)
    if n_pts:
        if n_pts != len(items):
            raise ValueError("align_points present on some rows only")
        out["align_points"] = torch.stack([it["align_points"] for it in items])
    return out


class Collator:
    """Picklable collate for the DataLoader."""

    def __init__(self, strict_shape: bool = True):
        self.strict_shape = strict_shape

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(items, strict_shape=self.strict_shape)
