"""CondStamp: fixed codes added to the conditioning tokens.

* segment code  (2, C): row 0 = DINO tokens, row 1 = every Qwen token; two orthogonal random vectors, L2 4.8.
* patch code    (P*P, C): fixed 2D sin-cos table, bilinearly sampled at each Qwen image token's own normalised
                (row, col) in its view (``qwen_rc``); only image tokens receive it.
* view code     (V, C): "hashed" — row v is a seeded random unit vector scaled to ``scale * sqrt(C/2)``; shared by
                the DINO and Qwen segments; ids are view ordinals within a sample.

The checkpoints store all three tables in bf16. Generation here is fp32; ``round_bf16`` gives the stored values,
and tables are always taken from the checkpoint when one is loaded.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

SEG_SEED = 1234
SEG_NORM = 0.15 * 32.0
VIEW_HASH_BASE = 1_000_003


def round_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16)


def segment_codes(C: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEG_SEED)
    a = torch.randn(C, generator=g)
    b = torch.randn(C, generator=g)
    b = b - (b @ a) / (a @ a) * a
    return torch.stack([a / a.norm() * SEG_NORM, b / b.norm() * SEG_NORM])


def patch_codes(P: int, C: int, scale: float) -> torch.Tensor:
    """(P*P, C) 2D sin-cos table over cell centres; half the channels encode the row, half the column."""
    Q = C // 4
    omega = 1.0 / (10000 ** (torch.arange(Q, dtype=torch.float64) / Q))
    pos = (torch.arange(P, dtype=torch.float64) + 0.5)
    r = pos[:, None] * omega[None, :]
    enc = torch.cat([r.sin(), r.cos()], dim=1)
    tab = torch.cat([enc[:, None, :].expand(P, P, C // 2), enc[None, :, :].expand(P, P, C // 2)], dim=-1)
    return (tab.reshape(P * P, C) * scale).float()


def view_codes(n: int, C: int, scale: float, start: int = 0) -> torch.Tensor:
    rows = []
    for v in range(start, start + n):
        g = torch.Generator().manual_seed(VIEW_HASH_BASE + v)
        r = torch.randn(C, generator=g, dtype=torch.float64)
        rows.append(r / r.norm())
    return (torch.stack(rows) * (scale * math.sqrt(C / 2.0))).float()


class ViewCodes(torch.nn.Module):
    """The model-level view-code table. Mode and scale are stored explicitly, so the table can be extended past the
    trained number of views with the exact rule instead of recovering the scale from a bf16 row."""

    def __init__(self, n: int = 16, dim: int = 1024, scale: float = 0.2, table: Optional[torch.Tensor] = None):
        super().__init__()
        self.scale, self.dim = float(scale), int(dim)
        t = table if table is not None else round_bf16(view_codes(n, dim, scale))
        self.register_buffer("table", t, persistent=True)

    def rows(self, n_needed: int) -> torch.Tensor:
        if n_needed <= self.table.shape[0]:
            return self.table
        extra = round_bf16(view_codes(n_needed - self.table.shape[0], self.dim, self.scale,
                                      start=self.table.shape[0])).to(self.table.device)
        return torch.cat([self.table, extra.to(self.table.dtype)], 0)


def image_token_rc(ids: torch.Tensor, grids: torch.Tensor, image_pad_id: int, merge: int) -> torch.Tensor:
    """(T, 2) normalised cell-centre (row, col) of each image token within its own view; -1 elsewhere.
    Raster order within each image, restarting at every image."""
    rc = torch.full((ids.shape[0], 2), -1.0, device=ids.device)
    pos = (ids == image_pad_id).nonzero(as_tuple=False).flatten()
    o = 0
    for g in grids:
        gh, gw = int(g[1]) // merge, int(g[2]) // merge
        c = int(g[0]) * gh * gw
        if c <= 0:
            continue
        k = torch.arange(c, device=ids.device)
        rc[pos[o:o + c], 0] = (((k % (gh * gw)) // gw).float() + 0.5) / max(gh, 1)
        rc[pos[o:o + c], 1] = (((k % (gh * gw)) % gw).float() + 0.5) / max(gw, 1)
        o += c
    return rc


def image_token_views(ids: torch.Tensor, grids: torch.Tensor, image_pad_id: int, merge: int) -> torch.Tensor:
    """(T,) view ordinal of each image token, -1 for text / structural tokens."""
    qv = torch.full((ids.shape[0],), -1, dtype=torch.long, device=ids.device)
    pos = (ids == image_pad_id).nonzero(as_tuple=False).flatten()
    o = 0
    for v, g in enumerate(grids):
        c = int(g[0] * g[1] * g[2]) // (merge * merge)
        qv[pos[o:o + c]] = v
        o += c
    return qv


def add_patch_code(x: torch.Tensor, table: Optional[torch.Tensor], rc: Optional[torch.Tensor]) -> torch.Tensor:
    """Bilinear (align_corners=False) sample of the P×P table at each token's rc; tokens with rc < 0 untouched.
    Table and weights are cast to ``x.dtype`` (bf16 in training), as in v12."""
    if table is None or rc is None:
        return x
    rc = rc.to(x.device).float()
    m = rc[..., 0] >= 0
    if not bool(m.any()):
        return x
    n = table.shape[0]
    P = int(round(n ** 0.5))
    assert P * P == n, f"patch table {n} is not square"
    tab = table.to(x.dtype).view(P, P, -1)
    y = (rc[..., 0] * P - 0.5).clamp(0, P - 1)
    xx = (rc[..., 1] * P - 0.5).clamp(0, P - 1)
    y0, x0 = y.floor().long(), xx.floor().long()
    y1, x1 = (y0 + 1).clamp(max=P - 1), (x0 + 1).clamp(max=P - 1)
    wy, wx = (y - y0.float()).unsqueeze(-1).to(x.dtype), (xx - x0.float()).unsqueeze(-1).to(x.dtype)
    add = (tab[y0, x0] * (1 - wy) * (1 - wx) + tab[y1, x0] * wy * (1 - wx)
           + tab[y0, x1] * (1 - wy) * wx + tab[y1, x1] * wy * wx)
    return x + add * m.unsqueeze(-1).to(x.dtype)
