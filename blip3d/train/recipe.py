"""A training recipe = one yaml file (recipes/*.yaml). Everything that changes numbers is here and is written into
every checkpoint (``blip3d_recipe.json``); constants of the v12 recipe are hard-coded in the modules that use them.

Paths: ``init.from`` / ``resume.from`` / ``ema.from`` are absolute or relative to ``paths.runs``; ``data`` names a file
in recipes/data/ (or is a path)."""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..utils.paths import REPO_ROOT, get_paths


@dataclass
class Compute:
    nodes: int = 4
    gpus_per_node: int = 8
    per_gpu_bs: int = 4
    eff_bs: int = 256
    save_steps: int = 5000
    keep: int = 6
    workers: int = 10
    universal: bool = False          # load a DeepSpeed universal checkpoint (world-size change)

    @property
    def grad_accum(self) -> int:
        world = self.nodes * self.gpus_per_node
        ga, rem = divmod(self.eff_bs, self.per_gpu_bs * world)
        if rem or ga < 1:
            raise ValueError(f"eff_bs {self.eff_bs} is not a multiple of per_gpu_bs {self.per_gpu_bs} x {world} GPUs")
        return ga


@dataclass
class Compat:
    """Reproduce a v12 behaviour instead of its fix (for A/B and bit-parity tests)."""
    shared_t: bool = False           # T-01: timesteps from the CPU generator shared by all ranks
    data_replay: bool = False        # T-04: data stream restarts from step 0 on every resume
    unified_v12_drops: bool = False  # U-05: geo lane never sees a CFG drop in S3
    full_model_hidden: bool = False  # C-03: Qwen hidden via the generation model (computes unused LM-head logits)


@dataclass
class Recipe:
    name: str
    stage: str                                  # s1 | s2 | s3 | cont
    data: str
    lr: float
    max_steps: int
    tower: Optional[str] = None                 # ss | shape | tex (towers); None for s3
    trainable: str = "all"                      # all | connector (S1: connector (+ ROAD projector on SS))
    init: Optional[Dict[str, Any]] = None       # {from, use_ema}; S3: {shape, tex, ss, use_ema}
    resume: Optional[str] = None                # checkpoint to resume (weights + optimizer + step); None = auto
    ema: Dict[str, Any] = field(default_factory=lambda: {"decay": 0.9999, "warmup": True})  # + {from, n}
    warmup_steps: int = 100
    decay_steps: int = 6000
    seed: int = 42
    compute: Compute = field(default_factory=Compute)
    compat: Compat = field(default_factory=Compat)
    unified: Dict[str, Any] = field(default_factory=dict)   # S3 loss knobs (losses.unified.UnifiedLossCfg)
    output_dir: Optional[str] = None
    report_to: str = "wandb"

    @property
    def road(self) -> bool:
        return self.tower == "ss"

    @property
    def elastic(self) -> bool:
        return self.tower in ("shape", "tex")

    def resolve(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.join(get_paths().runs, p)

    def data_path(self) -> str:
        if os.path.isfile(self.data):
            return self.data
        return str(REPO_ROOT / "recipes" / "data" / f"{self.data}.yaml")

    def out_dir(self) -> str:
        return self.resolve(self.output_dir or self.name)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, default=str)


def _set(d: Dict, dotted: str, value: str):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def load_recipe(path: str, overrides: Optional[List[str]] = None) -> Recipe:
    """``overrides``: ``key.sub=value`` strings (yaml-parsed values)."""
    if not os.path.isfile(path):
        path = str(REPO_ROOT / "recipes" / (path if path.endswith(".yaml") else path + ".yaml"))
    raw = yaml.safe_load(open(path))
    for o in overrides or []:
        k, v = o.split("=", 1)
        _set(raw, k, v)
    raw["compute"] = Compute(**raw.get("compute", {}))
    raw["compat"] = Compat(**raw.get("compat", {}))
    r = Recipe(**raw)
    if r.stage not in ("s1", "s2", "s3", "cont"):
        raise ValueError(f"stage {r.stage!r}")
    if (r.stage == "s3") != (r.tower is None):
        raise ValueError("towers need `tower`; s3 must not set it")
    if r.trainable not in ("all", "connector"):
        raise ValueError(f"trainable {r.trainable!r}")
    return r
