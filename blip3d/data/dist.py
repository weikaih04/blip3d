"""Rank helpers and the accelerate re-sharding fix.

accelerate wraps every IterableDataset in ``IterableDatasetShard`` when ``num_processes > 1`` and
``dispatch_batches=False``: it pulls ``batch_size * world`` items from the rank's own stream and keeps one slice.
For a dataset that already partitions by rank (the mixture) that throws away (world-1)/world of every load and,
worse, hands each rank a different task draw — ranks then train different parameter sets in the same step
(DDP hangs; ZeRO averages a real gradient with a structural zero). Datasets marked ``_rank_sharded = True`` pass
straight through; everything else keeps accelerate's behaviour (scan/04 §5.4).

The patch replaces ``IterableDatasetShard.__iter__`` of accelerate 1.7.0 and is refused on other versions until
re-checked. The trainer must also set ``accelerator_config.dispatch_batches = False``.
"""
from __future__ import annotations

import os

TESTED_ACCELERATE = ("1.7.0",)
_INSTALLED = False


def env_rank() -> int:
    return int(os.environ.get("RANK", 0))


def env_world() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def install_shard_passthrough() -> bool:
    """Idempotent; returns True when the patch is active (False when accelerate is absent)."""
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        import accelerate
        from accelerate.data_loader import IterableDatasetShard
    except ImportError:
        return False
    if accelerate.__version__ not in TESTED_ACCELERATE and os.environ.get("BLIP3D_ACCELERATE_UNCHECKED") != "1":
        raise RuntimeError(f"accelerate {accelerate.__version__}: the IterableDatasetShard pass-through was verified on "
                           f"{TESTED_ACCELERATE} only; re-check blip3d/data/dist.py (or set "
                           f"BLIP3D_ACCELERATE_UNCHECKED=1)")
    if getattr(IterableDatasetShard, "_rank_aware_passthrough", False):
        _INSTALLED = True
        return True
    orig = IterableDatasetShard.__iter__

    def __iter__(self):
        if getattr(self.dataset, "_rank_sharded", False):
            yield from self.dataset
        else:
            yield from orig(self)

    IterableDatasetShard.__iter__ = __iter__
    IterableDatasetShard._rank_aware_passthrough = True
    _INSTALLED = True
    return True
