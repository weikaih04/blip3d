"""Task mixture: one task per batch, drawn by weight, rank-synced; per-rank index streams; optional voxel balance.

Seeding, identical to v12 for a given base seed:
* task draw   ``default_rng((base * 1_000_003) ^ (worker + 1))`` — no rank term, so every rank trains the same task
  on the same step (a text batch and an image batch touch different parameters; mismatched ranks hang or average
  a gradient against a structural zero). The per-batch multi-image view count comes from the same generator.
* index stream per task ``default_rng(((base * 1_000_003) ^ (rank * 9176) ^ (worker + 1)) ^ (i * 7919 + 1))``,
  an endless sequence of permutations; worker w of W serves batches w, w+W, ... and burns the indices of the rest
  (indices only, nothing is loaded).
* base seed: ``seed + resume_step`` (so a resumed run does not replay the batches it already trained on — v12 always
  used 0 and replayed; the SS-300k continuation used the resume step, which this rule reproduces). ``compat_replay``
  keeps ``seed`` alone and also switches the tasks to v12's loader semantics (see :mod:`blip3d.data.task`).
* per-item draws (view, caption, resample target, ROAD points): a generator per (base, rank, worker, task) unless
  ``compat_replay`` (v12: a fresh unseeded generator per item).

The DataLoader must be built with ``batch_size=per_gpu_bs`` and this dataset; the pass-through patch in
:mod:`blip3d.data.dist` stops accelerate from re-sharding it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from torch.utils.data import IterableDataset, get_worker_info

from .balance import TargetBatcher, VoxelCost, make_batcher
from .collate import Collator
from .dist import env_rank, install_shard_passthrough
from .task import TaskConfig, ThreeDTask

install_shard_passthrough()

ROAD_POINTS = 10000   # ROAD_ALIGN_POINTS default in v12


def temperature_probs(weights: Sequence[float], temperature: float) -> np.ndarray:
    """p ∝ w^(1/τ); τ <= 0 is argmax."""
    w = np.asarray(weights, dtype=np.float64)
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(f"task weights must be non-negative with a positive sum: {w}")
    if temperature <= 0:
        out = np.zeros_like(w)
        out[int(np.argmax(w))] = 1.0
        return out
    p = w ** (1.0 / temperature)
    return p / p.sum()


class MixtureIterableDataset(IterableDataset):
    _rank_sharded = True   # tells the accelerate pass-through patch not to re-shard (dist.py)

    def __init__(self, tasks: List[ThreeDTask], weights: Sequence[float], batch_size: int, *,
                 temperature: float = 1.0, base_seed: int = 0, seeded_items: bool = True,
                 balance: Optional[VoxelCost] = None, balance_buf_mult: int = 24,
                 balance_max_age: Optional[int] = None):
        super().__init__()
        if len(tasks) != len(weights):
            raise ValueError("one weight per task")
        self.tasks = list(tasks)
        self.probs = temperature_probs(weights, temperature)
        self.batch_size = int(batch_size)
        self.base_seed = int(base_seed)
        self.seeded_items = bool(seeded_items)
        self.balance = balance
        self.balance_buf_mult = int(balance_buf_mult)
        self.balance_max_age = balance_max_age

    def task_names(self) -> Tuple[str, ...]:
        return tuple(t.task_name for t in self.tasks)

    # the three generators, exposed for tests
    def task_rng(self, worker_id: int) -> np.random.Generator:
        return np.random.default_rng((self.base_seed * 1_000_003) ^ (worker_id + 1))

    def index_rngs(self, rank: int, worker_id: int) -> List[np.random.Generator]:
        within = (self.base_seed * 1_000_003) ^ (rank * 9176) ^ (worker_id + 1)
        return [np.random.default_rng(within ^ (i * 7919 + 1)) for i in range(len(self.tasks))]

    def item_rngs(self, rank: int, worker_id: int) -> List[Optional[np.random.Generator]]:
        if not self.seeded_items:
            return [None] * len(self.tasks)
        return [np.random.default_rng([self.base_seed, rank, worker_id, i, 0xB3D]) for i in range(len(self.tasks))]

    def __iter__(self):
        wi = get_worker_info()
        worker_id = wi.id if wi is not None else 0
        num_workers = wi.num_workers if wi is not None else 1
        rank = env_rank()
        task_rng = self.task_rng(worker_id)
        idx_rngs = self.index_rngs(rank, worker_id)
        item_rngs = self.item_rngs(rank, worker_id)

        def cycler(ti: int):
            n = len(self.tasks[ti])
            g = idx_rngs[ti]
            while True:
                order = g.permutation(n) if n > 1 else np.array([0])
                for j in order:
                    yield int(j)

        cyclers = [cycler(i) for i in range(len(self.tasks))]
        batchers: Dict[int, TargetBatcher] = {}
        skip = worker_id
        while True:
            t = int(task_rng.choice(len(self.tasks), p=self.probs))
            if skip > 0:
                for _ in range(self.batch_size):
                    next(cyclers[t])
                skip = (skip - 1) % max(1, num_workers)
                continue
            ds = self.tasks[t]
            ds.set_batch_params(task_rng)
            try:
                if self.balance is not None:
                    b = batchers.get(t)
                    if b is None:
                        b = batchers[t] = make_batcher(ds.records, self.batch_size, self.balance, rank,
                                                       buf_mult=self.balance_buf_mult,
                                                       max_age=self.balance_max_age)
                    guard = 0
                    while not b.ready() and guard < b.buf_size * 4:
                        j = next(cyclers[t])
                        guard += 1
                        b.push(self.balance.item_cost(ds.records, j), j)
                    for j in b.pop_batch():
                        yield ds.get(j, item_rngs[t])
                else:
                    for _ in range(self.batch_size):
                        yield ds.get(next(cyclers[t]), item_rngs[t])
            finally:
                ds.clear_batch_params()
            skip = (num_workers - 1) if num_workers > 1 else 0


@dataclass
class Mixture:
    dataset: MixtureIterableDataset
    collate_fn: Collator
    task_names: Tuple[str, ...]
    num_workers: int
    config: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"mixture (base seed {self.dataset.base_seed}, workers {self.num_workers}):"]
        for t, p in zip(self.dataset.tasks, self.dataset.probs):
            lines.append(f"  {t.task_name:<10s} {t.modality:<3s} p={p:.3f}  n={len(t):,}  targets={list(t.cfg.targets)}")
        return "\n".join(lines)


_VAR = re.compile(r"\$\{(\w+)\}")


def _expand(s: str, env: Dict[str, str]) -> str:
    def sub(m):
        k = m.group(1)
        if k not in env:
            raise KeyError(f"unknown variable ${{{k}}} in data config (known: {sorted(env)})")
        return env[k]
    return _VAR.sub(sub, s)


def _paths_env() -> Dict[str, str]:
    try:
        from ..utils.paths import get_paths
        p = get_paths()
        return {k: v for k, v in {"manifests": p.manifests, "data_root": p.data_root, "runs": p.runs,
                                  "voxel_table": p.voxel_table}.items() if v}
    except FileNotFoundError:
        return {}


def load_config(cfg_path: str) -> Dict[str, Any]:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not cfg.get("tasks"):
        raise ValueError(f"{cfg_path}: a data config needs a non-empty 'tasks' list")
    if int(cfg.get("version", 1)) != 1:
        raise ValueError(f"{cfg_path}: unsupported data config version {cfg.get('version')}")
    return cfg


def task_configs(cfg: Dict[str, Any], *, align_points: bool) -> List[TaskConfig]:
    env = _paths_env()
    d = dict(cfg.get("defaults") or {})
    cond = dict(d.pop("cond", None) or {})
    road_points = int(d.pop("road_points", ROAD_POINTS))
    out = []
    names = set()
    for t in cfg["tasks"]:
        t = dict(t)
        name = t.pop("name")
        if name in names:
            raise ValueError(f"duplicate task name {name!r}")
        names.add(name)
        caps = dict(t.pop("captions", None) or {})
        views = dict(t.pop("views", None) or {})
        merged = {**d, **t}
        unknown = set(merged) - {"modality", "weight", "manifest", "targets", "tex_missing", "slat_resolution",
                                 "min_aesthetic", "max_slat_tokens"}
        if unknown:
            raise ValueError(f"task {name}: unknown fields {sorted(unknown)}")
        if views.get("sampler", "uniform" if merged["modality"] == "i1" else "weighted") not in ("uniform", "weighted"):
            raise ValueError(f"task {name}: views.sampler must be uniform (single image) or weighted (multi image)")
        out.append(TaskConfig(
            name=name, modality=merged["modality"], manifest=_expand(str(merged["manifest"]), env),
            weight=float(merged.get("weight", 1.0)), targets=tuple(merged.get("targets", ("ss",))),
            tex_missing=merged.get("tex_missing", "batch_drop"),
            slat_resolution=int(merged.get("slat_resolution", 512)), min_aesthetic=merged.get("min_aesthetic"),
            max_slat_tokens=int(merged.get("max_slat_tokens", 8192)),
            align_points=road_points if align_points else 0,
            caption_weights=caps.get("weights"), caption_fallback=caps.get("fallback", "renormalize"),
            im_view_probs=views.get("n_probs"), max_views=int(views.get("max_views", 4)),
            dino_size=int(cond.get("dino_size", 512)), im_tok_per_view=int(cond.get("im_tok_per_view", 256))))
    return out


def build_mixture(cfg_path: str, *, per_gpu_bs: int, seed: int, resume_step: int = 0, compat_replay: bool = False,
                  align_points: bool = False, qwen_path: str) -> Mixture:
    """Build the training mixture from a data yaml (configs/data/*.yaml).

    ``compat_replay``: v12 data semantics end to end — base seed ``seed`` regardless of ``resume_step`` (every
    resume replays the stream from its start), unseeded per-item draws, any load error resampled, pbr read even when
    tex is not a target. Also forced by ``compat_v12: true`` in the yaml.
    ``align_points``: emit ROAD teacher clouds (SS tower only)."""
    cfg = load_config(cfg_path)
    compat = bool(compat_replay or cfg.get("compat_v12", False))
    sampling = dict(cfg.get("sampling") or {})
    if sampling.get("granularity", "batch") != "batch":
        raise ValueError("only batch granularity is supported (one task per batch)")
    base = int(seed) if compat else int(seed) + int(resume_step)
    loader = dict(cfg.get("loader") or {})
    bal = dict(cfg.get("balance") or {})
    cost = None
    if bal.get("enabled", False):
        table = bal.get("table")
        cost = VoxelCost(_expand(str(table), _paths_env()) if table else None)
    tcfgs = task_configs(cfg, align_points=align_points)
    vox_lookup = cost.lookup if (cost is not None and not compat) else None
    tasks = [ThreeDTask(tc, qwen_path=qwen_path, compat_v12=compat,
                        resample_tries=int(loader.get("resample_tries", 8)), vox_lookup=vox_lookup)
             for tc in tcfgs]
    ds = MixtureIterableDataset(
        tasks, [tc.weight for tc in tcfgs], per_gpu_bs, temperature=float(sampling.get("temperature", 1.0)),
        base_seed=base, seeded_items=not compat, balance=cost, balance_buf_mult=int(bal.get("buf_mult", 24)),
        balance_max_age=None if compat else int(bal.get("max_age", 64)))
    m = Mixture(dataset=ds, collate_fn=Collator(strict_shape=not compat), task_names=ds.task_names(),
                num_workers=int(loader.get("num_workers", 8)), config=cfg)
    if env_rank() == 0:
        print(m.summary(), flush=True)
    return m
