"""Conditioning assembly: records -> cross-attention tokens + key mask, for training and for CFG inference.

One function serves both. Token order is [DINO ; Qwen] for image inputs, Qwen only for text.

Addition order (bf16 in training, fp32 at eval — it matters for bit parity):
  Qwen: connector(qwen * keep) -> + view code (multi-image only) -> + patch code (image tokens) -> + seg[1]
  DINO: dino -> + view code (single image: row 0) -> + seg[0] -> * keep
Dropout (independent per row): ``drop`` zeroes the Qwen hidden before the connector and the DINO values after
the codes, keys stay visible (this is the CFG unconditional); ``ddrop`` masks the DINO keys; ``qdrop`` masks the
Qwen keys and never coincides with ``ddrop``. Text rows only have ``drop``.
The CFG unconditional is therefore NOT all zeros: its Qwen half is connector(0) + codes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch

from .connector import Connector
from .stamp import ViewCodes, add_patch_code


@dataclass
class Drops:
    drop: torch.Tensor           # (B,) bool — CFG drop
    ddrop: torch.Tensor          # (B,) bool — DINO segment absent
    qdrop: torch.Tensor          # (B,) bool — Qwen segment absent

    @classmethod
    def none(cls, B: int, device) -> "Drops":
        z = torch.zeros(B, dtype=torch.bool, device=device)
        return cls(z, z.clone(), z.clone())


def draw_drops(B: int, *, has_dino: bool, p_cfg: float, p_dino: float, p_qwen: float, device,
               text_draw: Literal["bernoulli", "rand"] = "bernoulli", dtype=torch.bfloat16) -> Drops:
    """Draw per-row drops in the exact RNG order v12 used.

    Image rows: drop = rand < p_cfg; ddrop = rand < p_dino (only if p_dino > 0); qdrop = (rand < p_qwen) & ~ddrop
    (only if p_qwen > 0). Text rows: ``bernoulli(p_cfg)`` on a ``dtype`` tensor (mask_drop), except the SS tower
    with ROAD in S1/S2, which drew ``rand < p_cfg`` (``text_draw="rand"``)."""
    d = Drops.none(B, device)
    if has_dino:
        d.drop = torch.rand(B, device=device) < p_cfg
        if p_dino > 0:
            d.ddrop = torch.rand(B, device=device) < p_dino
        if p_qwen > 0:
            d.qdrop = (torch.rand(B, device=device) < p_qwen) & ~d.ddrop
    elif p_cfg > 0:
        if text_draw == "rand":
            d.drop = torch.rand(B, device=device) < p_cfg
        else:
            d.drop = torch.bernoulli(torch.zeros(B, device=device, dtype=dtype) + p_cfg).bool()
    return d


def collate(records: Sequence[Dict[str, torch.Tensor]], device=None) -> Dict[str, torch.Tensor]:
    """Pad per-sample records into a batch: features with 0, keep masks with False, views / rc with -1
    (0 is a real view ordinal and a real coordinate). DINO views are padded with 0 as in v12 (masked anyway)."""
    B = len(records)
    dev = device or records[0]["qwen"].device

    def pad(key, fill, dtype=None, width=None):
        xs = [r[key] for r in records]
        T = max(x.shape[0] for x in xs)
        shape = (B, T) + tuple(xs[0].shape[1:])
        out = torch.full(shape, fill, dtype=dtype or xs[0].dtype, device=dev)
        for i, x in enumerate(xs):
            out[i, :x.shape[0]] = x.to(dev)
        return out

    b = {"qwen": pad("qwen", 0), "qwen_keep": pad("qwen_keep", False, torch.bool),
         "modality": records[0]["modality"]}
    if "qwen_rc" in records[0]:
        b["qwen_rc"] = pad("qwen_rc", -1.0, torch.float32)
    if "qwen_views" in records[0]:
        b["qwen_views"] = pad("qwen_views", -1, torch.long)
    if "dino" in records[0]:
        b["dino"] = pad("dino", 0)
        b["dino_keep"] = pad("dino_keep", False, torch.bool)
        b["dino_views"] = pad("dino_views", 0, torch.long)
    return b


def assemble(connector: Connector, views: Optional[ViewCodes], batch: Dict[str, torch.Tensor], drops: Drops,
             dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (cond (B, T, C), key_mask (B, T) bool True = attend). ``dtype``: bf16 in training, fp32 at eval."""
    q = batch["qwen"].to(dtype)
    qmask = batch["qwen_keep"].bool()
    B = q.shape[0]
    keep = (~drops.drop.to(q.device)).to(dtype).view(B, 1, 1)
    if "dino" not in batch:
        cond = connector(q * keep, key_mask=qmask)
        cond = add_patch_code(cond, connector.patch_code, batch.get("qwen_rc"))
        cond = cond + connector.seg_code[1].to(cond.dtype)
        return cond, qmask
    table = None
    if views is not None:
        n = int(max(batch["dino_views"].max().item(),
                    batch["qwen_views"].max().item() if "qwen_views" in batch else 0)) + 1
        table = views.rows(n)
    cq = connector(q * keep, key_mask=qmask)
    if table is not None and "qwen_views" in batch:
        qv = batch["qwen_views"]
        cq = cq + table[qv.clamp_min(0)].to(cq.dtype) * (qv >= 0).unsqueeze(-1).to(cq.dtype)
    ds = batch["dino"].to(cq.dtype)
    if table is not None:
        ds = ds + table[batch["dino_views"]].to(cq.dtype)
    cq = add_patch_code(cq, connector.patch_code, batch.get("qwen_rc"))
    ds = ds + connector.seg_code[0].to(cq.dtype)
    cq = cq + connector.seg_code[1].to(cq.dtype)
    ds = ds * keep
    dmask = batch["dino_keep"].bool() & ~drops.ddrop.to(q.device)[:, None]
    qmask = qmask & ~drops.qdrop.to(q.device)[:, None]
    return torch.cat([ds, cq], 1), torch.cat([dmask, qmask], 1)


@torch.no_grad()
def cond_uncond(connector: Connector, views: Optional[ViewCodes], record: Dict[str, torch.Tensor], *,
                dtype: torch.dtype = torch.float32, drop_dino: bool = False, drop_qwen: bool = False,
                device="cuda") -> Tuple[torch.Tensor, torch.Tensor]:
    """CFG pair for one record, each (1, T_kept, C), compacted to the attended tokens.
    ``drop_dino`` / ``drop_qwen`` select the modality-dropout regimes (analysis only; never both)."""
    b = collate([{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in record.items()}], device=device)
    dd = torch.tensor([bool(drop_dino)], device=device)
    qd = torch.tensor([bool(drop_qwen) and not bool(drop_dino)], device=device)
    out = []
    for cfg_drop in (False, True):
        c, m = assemble(connector, views, b, Drops(torch.tensor([cfg_drop], device=device), dd, qd), dtype)
        out.append(c[0][m[0]][None])
    return out[0], out[1]
