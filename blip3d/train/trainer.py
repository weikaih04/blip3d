"""HF Trainer + DeepSpeed ZeRO-1 (bf16 weights, fp32 master + fp32 grad accumulation) for every BLIP3D stage.

The model's ``forward(batch) -> (loss, logs)``; logs are detached scalars averaged over the logging window and across
ranks. Per-task losses are logged under ``train/task/<name>/loss``. Batches carry no labels, so HF divides the loss by
the accumulation steps (as in v12).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
import torch.distributed as dist
from transformers import Trainer

from .callbacks import AdaptiveGradClipCallback
from .ema import unwrap
from ..data.collate import to_sparse
from .prefetch import LiveCondPrefetch


def _to_device(v, device):
    if torch.is_tensor(v):
        return v.to(device, non_blocking=True)
    if hasattr(v, "feats") and hasattr(v, "to"):          # trellis2 SparseTensor
        return v.to(device)
    return v


class Blip3DTrainer(Trainer):
    def __init__(self, *args, encoder_factory: Callable[[], object], clip: Optional[AdaptiveGradClipCallback] = None,
                 task_names=(), **kw):
        super().__init__(*args, **kw)
        self.model_accepts_loss_kwargs = False
        self._encoder_factory, self._encoder = encoder_factory, None
        self._clip = clip
        self._task_names = tuple(task_names)
        self._sum: Dict[str, torch.Tensor] = {}
        self._n = 0
        self._task_sum = torch.zeros(len(self._task_names))
        self._task_n = torch.zeros(len(self._task_names))

    def encoder(self):
        if self._encoder is None:
            self._encoder = self._encoder_factory()
        return self._encoder

    def get_train_dataloader(self):
        return LiveCondPrefetch(super().get_train_dataloader(), self.encoder(), self.args.device)

    def _prepare_inputs(self, inputs):
        return {k: _to_device(v, self.args.device) for k, v in to_sparse(inputs).items()}

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self._clip is not None and self._clip.engine is None and self.deepspeed is not None:
            self._clip.bind(self.deepspeed)
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        with unwrap(model).memory_context():
            loss, logs = model(inputs)
        for k, v in logs.items():
            v = v.detach().float() if torch.is_tensor(v) else torch.tensor(float(v))
            self._sum[k] = self._sum[k] + v if k in self._sum else v
        self._n += 1
        t = inputs.get("task")
        if t in self._task_names:
            i = self._task_names.index(t)
            self._task_sum[i] += float(loss.detach())
            self._task_n[i] += 1
        return (loss, None) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        if "loss" in logs and self._n:
            keys = sorted(self._sum)
            dev = self.args.device
            v = torch.stack([self._sum[k].to(dev) for k in keys] + [torch.tensor(float(self._n), device=dev)])
            ts = torch.cat([self._task_sum, self._task_n]).to(dev)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(v)
                dist.all_reduce(ts)
            for k, x in zip(keys, v[:-1]):
                logs[f"per_stage/{k}"] = round(float(x / v[-1]), 6)
            nT = len(self._task_names)
            tot = float(ts[nT:].sum())
            for i, name in enumerate(self._task_names):
                if float(ts[nT + i]) > 0:
                    logs[f"train/task/{name}/loss"] = round(float(ts[i] / ts[nT + i]), 6)
                logs[f"train/task/{name}/frac"] = round(float(ts[nT + i]) / max(tot, 1.0), 4)
            self._sum, self._n = {}, 0
            self._task_sum.zero_()
            self._task_n.zero_()
        super().log(logs, *args, **kwargs)
