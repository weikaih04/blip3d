"""Live conditioning one batch ahead: batch N+1 is encoded (frozen Qwen3-VL + DINOv3) on a side stream while batch N
trains. The encode has to be launched a full iteration ahead of the step it hides behind, hence a dataloader wrapper."""
from __future__ import annotations

import torch

from ..cond.assemble import collate


class LiveCondPrefetch:
    def __init__(self, dl, encoder, device):
        self.dl, self.enc, self.device = dl, encoder, device
        self.stream = torch.cuda.Stream(device=device)

    def __len__(self):
        return len(self.dl)

    def __getattr__(self, k):
        return getattr(self.__dict__["dl"], k)

    def _launch(self, batch):
        if batch is None:
            return None
        preps = batch.pop("prep")
        cur = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            cond = collate(self.enc.encode(preps), device=self.device)
        for v in cond.values():
            if torch.is_tensor(v):
                v.record_stream(cur)
        batch["cond"] = cond
        ev = torch.cuda.Event()
        ev.record(self.stream)
        return batch, ev

    def __iter__(self):
        it = iter(self.dl)
        cur = self._launch(next(it, None))
        while cur is not None:
            batch, ev = cur
            torch.cuda.current_stream(self.device).wait_event(ev)
            nxt = self._launch(next(it, None))
            yield batch
            cur = nxt
