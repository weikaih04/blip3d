"""Trainer callbacks: per-rank random streams and TRELLIS.2's adaptive gradient clip (state now checkpointed)."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from transformers import TrainerCallback

from .ema import unwrap


def _rank() -> int:
    return dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0


class RankRNGCallback(TrainerCallback):
    """HF seeds every rank identically. CUDA (noise, dropout draws): rank r > 0 reseeded to seed + 10000 (r + 1) at train
    begin, as v12. Timesteps: a dedicated per-rank generator seeded from (seed, rank, resume step), given to the model
    as ``t_generator``. ``shared_t=True`` reproduces v12 (t from the shared CPU generator)."""

    def __init__(self, shared_t: bool = False):
        self.shared_t = shared_t

    def on_train_begin(self, args, state, control, model=None, **kw):
        rk = _rank()
        if rk > 0:
            torch.cuda.manual_seed_all(int(args.seed) + 10_000 * (rk + 1))

    def on_step_begin(self, args, state, control, model=None, **kw):
        base = unwrap(model)
        if self.shared_t or getattr(base, "t_generator", None) is not None:
            return
        rk = _rank()
        seed = int(args.seed) + 7_919 * (rk + 1) + 131 * int(state.global_step)
        base.t_generator = torch.Generator().manual_seed(seed)
        if rk == 0:
            print(f"[rng] per-rank timestep generator from step {state.global_step} (rank-0 seed {seed})", flush=True)


class AdaptiveGradClipCallback(TrainerCallback):
    """TRELLIS.2 AdaptiveGradClipper: after 1000 finite pre-clip norms the DeepSpeed clip threshold becomes
    min(p95(last 1000), 1.0). The engine is bound by the trainer on its first step. The buffer is saved into every
    checkpoint (``gradclip_state.json``) and restored on resume (v12 restarted it empty)."""

    FILE = "gradclip_state.json"

    def __init__(self, max_norm: float = 1.0, percentile: float = 95.0, buffer_size: int = 1000,
                 resume_from: Optional[str] = None):
        self.max_norm, self.percentile, self.buffer_size = float(max_norm), float(percentile), int(buffer_size)
        self.buf = np.zeros(buffer_size, dtype=np.float32)
        self.ptr = self.len = 0
        self.cur = self.max_norm
        self.engine = None
        if resume_from and os.path.isfile(os.path.join(resume_from, self.FILE)):
            s = json.load(open(os.path.join(resume_from, self.FILE)))
            self.buf = np.asarray(s["buf"], dtype=np.float32)
            self.ptr, self.len, self.cur = int(s["ptr"]), int(s["len"]), float(s["cur"])

    def bind(self, engine):
        self.engine = engine
        opt = getattr(engine, "optimizer", None)
        if opt is not None and hasattr(opt, "clip_grad") and self.len >= self.buffer_size:
            opt.clip_grad = self.cur

    def on_step_end(self, args, state, control, **kw):
        e = self.engine
        if e is None or not hasattr(e, "get_global_grad_norm"):
            return
        gn = e.get_global_grad_norm()
        if gn is None or not np.isfinite(float(gn)):
            return
        self.buf[self.ptr] = float(gn)
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.len = min(self.len + 1, self.buffer_size)
        if self.len < self.buffer_size:
            return
        self.cur = min(float(np.percentile(self.buf, self.percentile)), self.max_norm)
        opt = getattr(e, "optimizer", None)
        if opt is not None and hasattr(opt, "clip_grad"):
            opt.clip_grad = self.cur

    def on_save(self, args, state, control, **kw):
        if _rank() != 0:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt):
            json.dump({"buf": self.buf.tolist(), "ptr": self.ptr, "len": self.len, "cur": self.cur},
                      open(os.path.join(ckpt, self.FILE), "w"))
