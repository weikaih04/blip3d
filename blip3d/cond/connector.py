"""xf2 connector: Qwen hidden (B, T, 2048) -> cond tokens (B, T, 1024), plus the fixed segment / patch codes.

in_proj (xavier, zero bias) -> 2 pre-norm blocks (8 heads, QK LayerNorm, MLP x4 tanh-GELU, key-padding mask)
-> output LayerNorm (identity init: matches the LayerNorm'd DINOv3 tokens the TRELLIS.2 flows were trained on).
Parameter names match the v12 checkpoints (`<prefix>.in_proj.*`, `<prefix>.blocks.{i}.*`, `<prefix>.out_norm.*`);
the codes are stored as `seg_code` / `patch_code` (old: `cond_seg_embed` / `cond_patch_pos`, see train/ckpt.py).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from . import stamp


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, dim // n_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.proj(o.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.norm2(x))


class Connector(nn.Module):
    def __init__(self, in_dim: int = 2048, dim: int = 1024, n_blocks: int = 2, n_heads: int = 8,
                 mlp_ratio: float = 4.0, patch_lattice: int = 32, code_scale: float = 0.2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, dim)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        self.blocks = nn.ModuleList([_Block(dim, n_heads, mlp_ratio) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(dim)
        # fixed codes (buffers, never trained); a loaded checkpoint overwrites them with its bf16 values
        self.register_buffer("seg_code", stamp.round_bf16(stamp.segment_codes(dim)), persistent=True)
        self.register_buffer("patch_code", stamp.round_bf16(stamp.patch_codes(patch_lattice, dim, code_scale)),
                             persistent=True)

    def forward(self, hidden: torch.Tensor, key_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.in_proj(hidden)
        m = None if key_mask is None else key_mask[:, None, None, :].to(torch.bool)
        for blk in self.blocks:
            x = blk(x, m)
        return self.out_norm(x)
