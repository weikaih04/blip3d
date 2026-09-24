"""Unified-model building blocks, verbatim from the v12 implementation (op order is part of the numbers).
Sparse/dense block internals, stream tags on the identity-pad rotation pair, and the fused-attention gather plan."""
import torch

from trellis2.modules import sparse as sp
from trellis2.modules.attention import RotaryPositionEmbedder


def _block_mod_params(block, mod: torch.Tensor):
    """share_mod unpack — sparse/transformer/modulated.py:143-147."""
    return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)


def _attn_qkv(attn, h_sp):
    """SparseMultiHeadAttention self-path up to roped (q, k, v) —
    attention/modules.py:110-125."""
    qkv = attn._linear(attn.to_qkv, h_sp)
    qkv = attn._fused_pre(qkv, num_fused=3)          # feats (T, 3, H, C)
    q, k, v = qkv.unbind(dim=-3)
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    if attn.use_rope:
        q, k = attn.rope(q, k)
    return q, k, v


def _attn_out(attn, h_sp):
    """Attention tail: heads -> channels -> to_out."""
    h_sp = attn._reshape_chs(h_sp, (-1,))
    return attn._linear(attn.to_out, h_sp)


# STREAM TAGS, one quarter turn each. The union softmax sees keys from several
# towers at once and cannot otherwise tell which tower is answering, so each
# non-geo stream rotates the IDENTITY-PAD rotation pair (last 2 head dims —
# TRELLIS ropes 63 of 64 pairs, the 64th passes unrotated) by a distinct
# multiple of π/2. Quarter turns are float-EXACT (swap and negate, zero
# rounding), so within-stream attention stays bit-exact and only cross-stream
# logits gain the antisymmetric "which stream" phase.
#
#   geo  0        (never rotated — it is the reference)
#   tex  π/2      TAG_TEX
#   ss   3π/2     TAG_SS
#
# 3π/2 rather than π for SS so that BOTH readers can separate it: a geo query
# (phase 0) sees SS at 3π/2 and tex at π/2; a tex query (phase π/2) sees SS at π
# and geo at −π/2. Every pair is distinct, and distinct from the within-stream 0.
# π stays reserved for the cond stream, which cond_mode="stream" would add.
TAG_TEX, TAG_SS = 1, 3


def _rot_pad(f: torch.Tensor, turns: int) -> torch.Tensor:
    """Rotate the identity-pad pair of a raw (..., D) feature tensor."""
    turns %= 4
    if turns == 0:
        return f
    x, y = f[..., -2:-1], f[..., -1:]
    if turns == 1:
        a, b = -y, x
    elif turns == 2:
        a, b = -x, -y
    else:
        a, b = y, -x
    return torch.cat([f[..., :-2], a, b], dim=-1)


def _rotate_pad_pair(k_sp, turns: int = TAG_TEX):
    """Sparse/VarLen wrapper for _rot_pad. Default is the tex tag, which is what
    every existing call site meant when it took no argument."""
    return k_sp.replace(_rot_pad(k_sp.feats, turns))


def _dense_mod_params(block, mod: torch.Tensor):
    """modulated.py:148-149, share_mod branch. Six (B, C) tensors."""
    assert block.share_mod, "SS blocks are share_mod=True; the adaLN branch is untested here"
    return (block.modulation + mod).type(mod.dtype).chunk(6, dim=1)


def _dense_attn_qkv(attn, h: torch.Tensor, phases: torch.Tensor):
    """modules.py:73-87 for type='self'. Returns (q, k, v, k_pre_rope).

    k_pre_rope is the EXPORT POINT for the cross-tower read: post-qk_rms_norm,
    PRE-rope. It has to be pre-rope because a borrowed key is re-roped into the
    32^3 frame at the consumer; handing over the already-roped k would rotate it
    twice and quietly destroy every relative position it encodes. The rms-then-
    rope order here is identical to the sparse path, so the two towers export
    the same object.
    """
    B, L, _ = h.shape
    qkv = attn.to_qkv(h).reshape(B, L, 3, attn.num_heads, -1)
    q, k, v = qkv.unbind(dim=2)
    if attn.qk_rms_norm:
        q = attn.q_rms_norm(q)
        k = attn.k_rms_norm(k)
    k_pre = k
    if attn.use_rope:
        q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
        k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
    return q, k, v, k_pre


def _dense_attn_out(attn, h: torch.Tensor):
    """modules.py:109-110."""
    B, L = h.shape[:2]
    return attn.to_out(h.reshape(B, L, -1))


def _fusion_plan(layout, corner_on, device, n_ss: int = 0):
    """Gather indices for the FUSED joint-attention path. Source stacking is
    rows [0,T) = geo, [T,2T) = tex (a plain cat of the two lanes' k/v).

      fused_idx / fused_layout : per sample [geo_b ; tex_b]  (kv for the tex lane,
                                 and for JOINT samples in the geo lane)
      geo_kv_idx / geo_layout  : per sample [geo_b] if t_s=0 (corner ⇒ one-way)
                                 else [geo_b ; tex_b]

    Varlen attention takes per-sample kv lengths, so the corner mask is encoded
    in the LAYOUT — no attention mask, no batch grouping, no python loop over
    samples at attention time. Computed ONCE per forward, reused by all 30
    blocks (the plan depends only on the layout and t_s, both step-constant)."""
    T = layout[-1].stop
    geo_rows = [torch.arange(s.start, s.stop, device=device) for s in layout]
    tex_rows = [r + T for r in geo_rows]
    # SS rows: the dense (B, n_ss, H, D) lane flattened to (B*n_ss, H, D) and
    # stacked after the two slat lanes, so sample b owns [2T + b*n_ss, +n_ss).
    ss_rows = ([torch.arange(2 * T + b * n_ss, 2 * T + (b + 1) * n_ss, device=device)
                for b in range(len(layout))] if n_ss else None)

    def _seg(b, with_tex, with_ss):
        parts = [geo_rows[b]]
        if with_tex:
            parts.append(tex_rows[b])
        if with_ss and ss_rows is not None:
            parts.append(ss_rows[b])
        return torch.cat(parts) if len(parts) > 1 else parts[0]

    def _len(b, with_tex, with_ss):
        n = layout[b].stop - layout[b].start
        return n * (2 if with_tex else 1) + (n_ss if (with_ss and n_ss) else 0)

    def _build(sel):
        """sel(b) -> (with_tex, with_ss) for the lane being built."""
        parts, lens = [], []
        for b in range(len(layout)):
            wt, ws = sel(b)
            parts.append(_seg(b, wt, ws)); lens.append(_len(b, wt, ws))
        return torch.cat(parts), sp.VarLenTensor.layout_from_seqlen(lens)

    # tex lane: always sees geo, and SS when there is an SS tower
    fused_idx, fused_layout = _build(lambda b: (True, True))
    # geo lane: at the corner (t_s == 0) it must be EXACTLY the specialist, so it
    # takes neither tex nor SS — the same one-way rule SS now inherits, because
    # a clean geo state makes SS's structure redundant there anyway.
    if all(corner_on):        # corner_on 现在是主机侧 list[bool],不再同步
        return fused_idx, fused_layout, fused_idx, fused_layout
    geo_idx, geo_layout = _build(lambda b: (corner_on[b],) * 2)
    return fused_idx, fused_layout, geo_idx, geo_layout
