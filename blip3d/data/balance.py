"""Voxel-balanced batching (v12: the S3 unified run only).

Every rank regroups its own stream so that each batch's cost (a fixed SS term + the asset's voxel count) lands near
one shared target; nothing is dropped, only regrouped, and no communication is needed because every rank aims at the
same constant. This removes the rank skew that the loss path's collectives turn into idle time.

Cost of an item = ``median + vox(sha)`` (the median voxel count stands in for the fixed SS cost). Per-task target =
``per_batch * (median + mean vox over the task's records)``. A missing sha costs the median.

Fix over v12 (scan/04 §5.8): items whose cost can never fit a batch near the target (vox above ~11.8k) sat in the
buffer forever, shrinking it. ``max_age`` forces any item that has waited that many pops into the next batch
(``None`` reproduces v12).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class VoxelCost:
    """sha -> voxel count, from a table npz (``sha``, ``vox``) and/or the manifest's ``n_vox`` column."""

    def __init__(self, table_path: Optional[str] = None):
        self._m = {}
        self.median = 0
        if table_path:
            d = np.load(table_path, allow_pickle=False)
            self._m = dict(zip(d["sha"].tolist(), d["vox"].tolist()))
            self.median = int(np.median(d["vox"]))

    def __len__(self) -> int:
        return len(self._m)

    def get(self, sha: Optional[str]) -> int:
        return self._m.get(sha, self.median)

    def lookup(self, sha: str) -> Optional[int]:
        """Known count or ``None`` (used to pre-filter over-cap assets)."""
        return self._m.get(sha)

    def vox(self, records, j: int) -> int:
        v = int(records.n_vox[j])
        return v if v >= 0 else self.get(records.sha[j])

    def item_cost(self, records, j: int) -> float:
        return self.median + self.vox(records, j)


class TargetBatcher:
    """Regroups a stream into batches whose summed cost is close to ``target``; every pushed item is popped once."""

    def __init__(self, target: float, per_batch: int, buf_size: int = 24, max_combos: int = 400, seed: int = 0,
                 max_age: Optional[int] = None):
        assert buf_size >= per_batch
        self.target, self.k, self.buf_size = float(target), int(per_batch), int(buf_size)
        self.max_combos = int(max_combos)
        self.max_age = max_age
        self._rng = np.random.default_rng(seed)
        self._buf: List[list] = []     # [cost, payload, age]

    def push(self, cost: float, payload) -> None:
        self._buf.append([float(cost), payload, 0])

    def ready(self) -> bool:
        return len(self._buf) >= self.buf_size

    def _pick(self, forced: List[int]) -> List[int]:
        n = len(self._buf)
        costs = np.array([c for c, _, _ in self._buf])
        rem, chosen = self.target - costs[forced].sum() if forced else self.target, list(forced)
        avail = [i for i in range(n) if i not in chosen]
        for slot in range(len(chosen), self.k):
            need = rem / (self.k - slot)
            j = min(avail, key=lambda i: abs(costs[i] - need))
            chosen.append(j)
            avail.remove(j)
            rem -= costs[j]
        best = list(chosen)
        best_err = abs(costs[best].sum() - self.target)
        if not forced:
            for _ in range(self.max_combos):
                cand = self._rng.choice(n, self.k, replace=False)
                err = abs(costs[cand].sum() - self.target)
                if err < best_err:
                    best, best_err = cand.tolist(), err
        return sorted(best, reverse=True)

    def pop_batch(self) -> list:
        if len(self._buf) < self.k:
            raise RuntimeError("batcher buffer holds fewer items than one batch")
        if len(self._buf) >= self.buf_size:
            forced = []
            if self.max_age is not None:
                forced = sorted((i for i, e in enumerate(self._buf) if e[2] >= self.max_age),
                                key=lambda i: -self._buf[i][2])[:self.k]
            idxs = self._pick(forced)
        else:
            idxs = list(range(self.k - 1, -1, -1))
        out = [self._buf.pop(i)[1] for i in idxs]
        for e in self._buf:
            e[2] += 1
        return out[::-1]


def make_batcher(records, per_batch: int, cost: VoxelCost, rank: int, *, buf_mult: int = 24,
                 max_age: Optional[int] = None) -> TargetBatcher:
    v = np.array([cost.vox(records, j) for j in range(len(records)) if records.sha[j]], dtype=np.float64)
    target = per_batch * (cost.median + (v.mean() if len(v) else cost.median))
    return TargetBatcher(target, per_batch, buf_size=max(24, per_batch * buf_mult), seed=rank * 7919 + 13,
                         max_age=max_age)
