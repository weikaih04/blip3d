"""Items -> one training batch. Runs in dataloader workers.

| key | when |
|---|---|
| ``task``, ``modality``, ``ids``, ``prep`` | always |
| ``ss`` (B, 8, 16, 16, 16) float32, raw | always |
| ``shape`` SparseTensor, coords (N, 4) ``[b, x, y, z]`` | shape is a target |
| ``tex``, ``tex_concat`` SparseTensor on the shape's coords, ``tex_valid`` (B,) bool | every row carries tex |
| ``align_points`` (B, 10000, 3) | ROAD on |

Sparse targets are packed with TRELLIS.2's own collate functions (``SLat.collate_fn`` for shape, ``SLatPbr.collate_fn``
for tex), exactly as v12 did, so the batch index column, layout cache and dtypes are identical.
Rules the downstream code relies on: one task and one modality per batch; tex all-or-none (a batch
where only some rows carry tex drops it, and says so once); align points all-or-none (raises).
``tex_valid`` is emitted whenever tex is (all True unless placeholder rows are present).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Sequence

import torch

_PARTIAL_TEX_WARNED = False


@lru_cache(maxsize=1)
def _slat_collate():
    # Imported in the PARENT (Collator.__init__): these modules pull in flex_gemm / Triton, which initialise CUDA on
    # import, and a first import inside a forked worker of a CUDA-initialised trainer raises "Cannot re-initialize
    # CUDA in forked subprocess".
    from trellis2.datasets.structured_latent import SLat
    from trellis2.datasets.structured_latent_svpbr import SLatPbr
    return SLat.collate_fn, SLatPbr.collate_fn


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
    slat_collate, pbr_collate = _slat_collate()
    if "shape" in items[0]:
        if strict_shape and not all("shape" in it for it in items):
            raise ValueError("shape target present on some rows only")
        out["shape"] = slat_collate([it["shape"] for it in items])["x_0"]
    n_tex = sum("tex" in it for it in items)
    if n_tex == len(items):
        pack = pbr_collate([it["tex"] for it in items])
        out["tex"] = pack["x_0"]
        out["tex_concat"] = pack["concat_cond"]
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
        _slat_collate()                     # import in the parent, before the workers fork (see _slat_collate)
        from .latents import sparse
        sparse()

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        return collate(items, strict_shape=self.strict_shape)
