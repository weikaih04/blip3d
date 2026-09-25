"""EMA of the trainable weights: decay 0.9999 with the (1+n)/(10+n) warm-up, fp32 shadow sharded over ranks.

v12 semantics kept: one update per optimizer step, from the bf16 module weights, shadow sharded by parameter index
(i % world), written only at save time as ``ema.safetensors`` (fp32, trainable names). Fixed: the
update count ``n`` is saved next to it (``ema_state.json``) and a resume restores shadow + n from the checkpoint it
resumes, instead of restarting the average from the raw weights at n = 0. A v12 checkpoint carries no count: pass
``n`` explicitly (the old run's optimizer step, as the SS continuation did with --ema_init_updates).
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
from transformers import TrainerCallback

STATE_FILE = "ema_state.json"


def unwrap(m):
    for _ in range(4):
        inner = getattr(m, "module", None)
        if inner is None:
            break
        m = inner
    return m


def _world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


class EMACallback(TrainerCallback):
    def __init__(self, decay: float = 0.9999, warmup: bool = True, resume_from: Optional[str] = None,
                 init_from: Optional[str] = None, init_n: Optional[int] = None,
                 key_map: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None):
        """``resume_from``: the checkpoint the trainer resumes (shadow + n restored from it).
        ``init_from``/``init_n``: seed the shadow from another checkpoint's EMA (continuations of v12 runs).
        ``key_map``: renames loaded EMA keys (v12 -> BLIP3D)."""
        self.decay, self.warmup = float(decay), bool(warmup)
        self.resume_from, self.init_from, self.init_n = resume_from, init_from, init_n
        self.key_map = key_map
        self.shadow: Dict[str, torch.Tensor] = {}
        self.n = 0
        self._pg = None

    def decay_at(self, n: int) -> float:
        return min(self.decay, (1.0 + n) / (10.0 + n)) if self.warmup else self.decay

    def _trainable(self):
        return [(k, p) for k, p in self._base.named_parameters() if p.requires_grad]

    def _load(self, ckpt: str, n: Optional[int]):
        from safetensors.torch import load_file
        ema = load_file(os.path.join(ckpt, "ema.safetensors"), device="cpu")
        if self.key_map is not None:
            ema = self.key_map(ema)
        missing = [k for k in self.shadow if k not in ema]
        if missing:
            raise KeyError(f"[ema] {ckpt}: {len(missing)} trainable tensors missing from ema.safetensors, e.g. {missing[:3]}")
        for k in self.shadow:
            self.shadow[k].copy_(ema[k].to(self.shadow[k].dtype))
        if n is None:
            f = os.path.join(ckpt, STATE_FILE)
            if not os.path.isfile(f):
                raise FileNotFoundError(f"[ema] {ckpt} has no {STATE_FILE}; pass the EMA update count explicitly")
            n = int(json.load(open(f))["n"])
        self.n = int(n)

    def on_train_begin(self, args, state, control, model=None, **kw):
        self._base = unwrap(model)
        ws, rk = _world()
        names = [k for k, _ in self._trainable()]
        mine = {k for i, k in enumerate(names) if i % ws == rk}
        self.shadow = {k: p.detach().clone().float() for k, p in self._trainable() if k in mine}
        self._pg = dist.new_group(backend="gloo") if ws > 1 else None
        src = None
        if self.resume_from and os.path.isfile(os.path.join(self.resume_from, "ema.safetensors")):
            has_n = os.path.isfile(os.path.join(self.resume_from, STATE_FILE))
            self._load(self.resume_from, None if has_n else self.init_n)
            src = self.resume_from
        elif self.init_from:
            self._load(self.init_from, self.init_n)
            src = self.init_from
        if rk == 0:
            gb = sum(v.numel() for v in self.shadow.values()) * 4 / 2 ** 30
            print(f"[ema] {len(names)} trainable tensors over {ws} rank(s), {gb:.2f} GB here; decay {self.decay}"
                  + (f"; shadow from {src} at n={self.n} (decay {self.decay_at(self.n):.6f})" if src else "; fresh"),
                  flush=True)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kw):
        if not self.shadow:
            return
        d = self.decay_at(self.n)
        self.n += 1
        for k, p in self._trainable():
            sh = self.shadow.get(k)
            if sh is not None:
                sh.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def on_save(self, args, state, control, **kw):
        ws, rk = _world()
        local = {k: v.detach().cpu() for k, v in self.shadow.items()}
        parts = [None] * ws if rk == 0 else None
        if ws > 1:
            dist.gather_object(local, parts, dst=0, group=self._pg)
        else:
            parts = [local]
        if rk != 0:
            return
        ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt):
            return
        from safetensors.torch import save_file
        full = {}
        for p in parts:
            full.update(p)
        save_file(full, os.path.join(ckpt, "ema.safetensors"))
        json.dump({"n": self.n, "decay": self.decay, "warmup": self.warmup, "global_step": state.global_step},
                  open(os.path.join(ckpt, STATE_FILE), "w"))
        print(f"[ema] wrote {len(full)} tensors, n={self.n} -> {ckpt}", flush=True)
