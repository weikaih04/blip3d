"""Blip3DUnified: the three TRELLIS.2 towers in one model (S3).

Shape and tex lanes share one union softmax per block (fused varlen attention, corner-masked: at t_s = 0 the geo lane
is exactly the specialist); the SS tower runs as a third, dense lane interleaved block by block and lends its K/V
(re-roped into the 32^3 frame) as a third key segment; streams are told apart by quarter-turn tags on the identity-pad
rotation pair. Only the fused attention path is kept.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules import sparse as sp
from trellis2.modules.attention import RotaryPositionEmbedder
from trellis2.modules.attention.full_attn import scaled_dot_product_attention as _dense_sdpa
from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
from trellis2.modules.sparse.transformer.modulated import fused_norm_modulate, fused_gate_residual
from trellis2.modules.utils import manual_cast

from .ops import (TAG_SS, TAG_TEX, _attn_out, _attn_qkv, _block_mod_params, _dense_attn_out, _dense_attn_qkv,
                  _dense_mod_params, _fusion_plan, _rot_pad, _rotate_pad_pair)


class Blip3DUnified(nn.Module):

    def __init__(self, geo_flow: nn.Module, tex_flow: nn.Module, ss_flow: nn.Module):
        """The v12 S3 configuration only: union coupling, bidirectional (corner-masked), SS as a third stream,
        read-once cross-attention for the condition, fused attention. Parameter names match the v12 checkpoints
        (`unified_geotex.*` without the prefix)."""
        super().__init__()
        for f, name in ((geo_flow, "geo"), (tex_flow, "tex")):
            assert getattr(f, "pe_mode", None) == "rope", f"{name}: pe_mode must be rope"
            assert f.share_mod, f"{name}: share_mod expected"
        assert len(geo_flow.blocks) == len(tex_flow.blocks) == len(ss_flow.blocks), "block count mismatch"
        ga, ta = geo_flow.blocks[0].self_attn, tex_flow.blocks[0].self_attn
        assert ga.channels == ta.channels and ga.use_rope and ta.use_rope
        assert ga.rope.rope_freq == ta.rope.rope_freq and ga.rope.head_dim == ta.rope.head_dim
        assert getattr(ss_flow, "pe_mode", None) == "rope" and ss_flow.rope_phases.is_complex(), \
            "ss.rope_phases lost its imaginary part (cast with to_bf16_keep_complex, not .to(bf16))"
        assert tuple(ga.rope.rope_freq) == (1.0, 10000.0)
        assert ss_flow.model_channels // ss_flow.num_heads == ga.rope.head_dim
        assert ss_flow.num_heads == ga.num_heads == ta.num_heads
        assert ss_flow.blocks[0].self_attn.qk_rms_norm == ga.qk_rms_norm

        self.geo_flow, self.tex_flow, self.ss_flow = geo_flow, tex_flow, ss_flow
        self.coupling, self.bidirectional, self.cond_mode = "union", True, "cross_attn"
        self.fused_attn = True          # the path S3 trained with; False = the loop oracle (compat_v12 attn="loop")
        self.xattn_scale = 1.0
        n_heads = ta.num_heads
        # cross-time mixers (zero-init gates): geo knows t_x, tex knows t_s
        self.t_mixer_s = type(geo_flow.t_embedder)(geo_flow.model_channels)
        self.cross_alpha_s = nn.Parameter(torch.zeros(1))
        self.t_mixer_s.to(next(geo_flow.t_embedder.parameters()).dtype)
        self.t_mixer = type(tex_flow.t_embedder)(tex_flow.model_channels)
        self.cross_alpha = nn.Parameter(torch.zeros(1))
        self.t_mixer.to(next(tex_flow.t_embedder.parameters()).dtype)
        # SS <- slat read (lag rows in training; interleave round 2 at inference), per-head zero-init
        self.ss_reads_gate = nn.Parameter(torch.zeros(len(tex_flow.blocks), 2, n_heads))
        self.ss_reads_enabled = True
        # SS grid mapped into the 32^3 slat frame (c -> 2c+0.5) for re-roping borrowed SS keys; non-persistent
        res = ss_flow.resolution
        c = torch.stack(torch.meshgrid(*[torch.arange(res, dtype=torch.float32)] * 3, indexing="ij"),
                        dim=-1).reshape(-1, 3)
        rp = RotaryPositionEmbedder(ga.rope.head_dim, 3, rope_freq=(1.0, 10000.0))
        self.register_buffer("ss_phases_slat", rp(2.0 * c + 0.5), persistent=False)


    @staticmethod
    def _gc_on(*blocks) -> bool:
        """The ONE rule every lane uses to decide whether to checkpoint.

        Checkpoint iff grad is being recorded AND at least one of the blocks this
        wrapper actually covers asks for it. Both halves matter, and each was
        wrong somewhere before this existed:

          * the three slat lanes read only `tblk.use_checkpoint`, so the geo
            tower's own flags were never read. It happened to work because the
            builder gives both towers 30 blocks and the same fraction, so the
            patterns coincide — a per-tower fraction would have silently made geo
            follow tex.
          * _run_block_triple read NO flag and checkpointed unconditionally, so
            --geotex_gc 0 did not turn it off. The mirror image of the SS bug:
            the knob said one thing and the lane did another.
          * _run_ss_block had no wrapper at all while the builder set its flags
            and the log printed "ss 30/30" (blocks 30, wrapped 0).

        A lane that drives blocks by reaching into blk.norm1 / blk.attn / blk.mlp
        can never inherit checkpointing from blk.forward(), so the wrapper is the
        only thing that can honour the flag. Routing every lane through here
        means the next such lane cannot forget.
        """
        return (torch.is_grad_enabled()
                and any(getattr(b, "use_checkpoint", False) for b in blocks))

    def _union_attn(self, q_x, k_x, v_x, k_s, v_s, k_c=None, v_c=None,
                    seg_on=None, ss_kv=None):
        """ONE shared softmax over K=[k_x;k_s(;k_c)] (MF dit.py:137-148
        semantics, per-stream projections; cond segment only in stream mode).
        seg_on: optional (B,) bool — per-sample HARD inclusion of the foreign
        segments (bidir corner mask: False at t_s=0 ⇒ this sample's softmax is
        pure self, exactly the specialist). Per-sample torch sdpa (sparse
        kernel has no mask arg; the G0 instrument needs one)."""
        outs = []
        for b in range(q_x.shape[0]):
            sl = q_x.layout[b]
            qb = q_x.feats[sl.start:sl.stop]
            segs = [(k_x.feats[sl.start:sl.stop], v_x.feats[sl.start:sl.stop], False)]
            fon = seg_on is None or bool(seg_on[b])
            if k_s is not None and fon:
                ss = k_s.layout[b]
                segs.append((k_s.feats[ss.start:ss.stop],
                             v_s.feats[ss.start:ss.stop], True))
            if k_c is not None and fon:
                cs = k_c.layout[b]
                segs.append((k_c.feats[cs.start:cs.stop],
                             v_c.feats[cs.start:cs.stop], True))
            if ss_kv is not None and fon:
                # dense (B, n_ss, H, D): sample b is just row b, no layout
                segs.append((ss_kv[0][b], ss_kv[1][b], True))
            K = torch.cat([s[0] for s in segs], dim=0)
            V = torch.cat([s[1] for s in segs], dim=0)
            out = F.scaled_dot_product_attention(
                qb.transpose(0, 1)[None], K.transpose(0, 1)[None],
                V.transpose(0, 1)[None],
            )[0].transpose(0, 1)
            outs.append(out)
        return q_x.replace(torch.cat(outs, dim=0))

    def _fused_joint_attn(self, q_s, k_s, v_s, q_x, k_x, v_x, plan, ss_kv=None):
        """Two native varlen flash calls, no mask, no per-sample python loop.
        MF dit.py:138-148 does exactly this (cat q/k/v of all streams → ONE
        attention → split by segment); the sparse kernel's per-sample kv
        lengths let us fold the corner exclusion into the layout instead.

        ss_kv joins as a THIRD key segment rather than a separate gated softmax.
        A separate softmax makes the SS read a pure addition — every slat token
        gets a fixed per-head amount of it and cannot prefer SS over tex — while
        one softmax makes the three towers compete for the same attention mass,
        which is what the geo/tex pair has always done. It also removes the
        zero-init ramp entirely: the coupling is on from step 0 by construction,
        so there is no gate that might never open."""
        fused_idx, fused_layout, geo_idx, geo_layout = plan
        segs_k = [k_s.feats, k_x.feats]
        segs_v = [v_s.feats, v_x.feats]
        if ss_kv is not None:
            k_ss, v_ss = ss_kv                      # (B, n_ss, H, D), already
            segs_k.append(k_ss.flatten(0, 1))       # re-roped into the 32^3 frame
            segs_v.append(v_ss.flatten(0, 1))
        k_all = torch.cat(segs_k, dim=0)
        v_all = torch.cat(segs_v, dim=0)
        k_f = sp.VarLenTensor(k_all[fused_idx], fused_layout)
        v_f = sp.VarLenTensor(v_all[fused_idx], fused_layout)
        a_x = sparse_scaled_dot_product_attention(q_x, k_f, v_f)
        if geo_idx is fused_idx:
            a_s = sparse_scaled_dot_product_attention(q_s, k_f, v_f)
        else:
            a_s = sparse_scaled_dot_product_attention(
                q_s, sp.VarLenTensor(k_all[geo_idx], geo_layout),
                sp.VarLenTensor(v_all[geo_idx], geo_layout))
        return a_s, a_x

    def _run_block_pair_fused(self, idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x,
                              plan, ss_kv=None):
        """Standard-MMDiT block on the fused path. Stream tag is ABSOLUTE here
        (tex's q AND k rotated by π/2, geo's not) — a single shared softmax
        cannot know which stream is querying, so the tag must live on the
        tokens, exactly like MF's stream-id RoPE axis (model.py:70 gives the
        depth stream time_id=1.0 while img keeps 0.0, and apply_rope hits both
        q and k). Within-stream logits are unchanged (both sides rotate);
        cross-stream logits get ±π/2, antisymmetric by direction."""
        gblk = self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]

        def inner(hs_in, hx_in):
            gsh, gsc, gg, gsh2, gsc2, gg2 = _block_mod_params(gblk, mod_s)
            tsh, tsc, tg, tsh2, tsc2, tg2 = _block_mod_params(tblk, mod_x)
            hn_s = fused_norm_modulate(hs_in, gblk.norm1, gsc, gsh)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn_s)
            hn_x = fused_norm_modulate(hx_in, tblk.norm1, tsc, tsh)
            q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn_x)
            q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)   # stream tag
            kv_ss = None if ss_kv is None else (
                _rot_pad(ss_kv[0], TAG_SS), ss_kv[1])   # SS gets its own tag
            a_s, a_x = self._fused_joint_attn(q_s, k_s, v_s, q_x, k_x, v_x,
                                              plan, ss_kv=kv_ss)

            a_s = _attn_out(gblk.self_attn, a_s)
            hs = fused_gate_residual(hs_in, a_s, gg)
            hcs = hs.replace(gblk.norm2(hs.feats))
            hs = hs + gblk.cross_attn(hcs, cond_s) * self.xattn_scale
            hms = fused_norm_modulate(hs, gblk.norm3, gsc2, gsh2)
            hs = fused_gate_residual(hs, gblk.mlp(hms), gg2)

            a_x = _attn_out(tblk.self_attn, a_x)
            hx = fused_gate_residual(hx_in, a_x, tg)
            hcx = hx.replace(tblk.norm2(hx.feats))
            hx = hx + tblk.cross_attn(hcx, cond_x) * self.xattn_scale
            hmx = fused_norm_modulate(hx, tblk.norm3, tsc2, tsh2)
            hx = fused_gate_residual(hx, tblk.mlp(hmx), tg2)
            return hs, hx

        if self._gc_on(gblk, tblk):
            import torch.utils.checkpoint as _ckpt
            return _ckpt.checkpoint(inner, h_s, h_x, use_reentrant=False)
        return inner(h_s, h_x)

    def _run_block_pair_bidir(self, idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x,
                              corner_on, tok_on, ss_kv=None):
        """BIDIRECTIONAL stitched block (user topology 2026-08-11): standard
        MMDiT phase order — both streams' QKV from PRE-update hiddens, geo
        attends [self; tex] (corner-masked), tex attends [self; geo], then both
        advance (residual + read-once cross-attn + MLP, verbatim fused ops).
        Geo WEIGHTS are frozen in S1 but the lane runs IN-GRAPH: b_gates and
        the geo-mediated tex feedback train through the tex loss. The whole
        pair is gradient-checkpointed (two 1.3B lanes live otherwise).
        corner_on: 主机侧 list[bool](B 个),t_s>0;tok_on 是它的逐 token 张量版。
        刻意用 list 而非张量:它在 30 个 block 里恒定,张量会每层同步一次。"""
        gblk = self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]

        def inner(hs_in, hx_in):
            gsh, gsc, gg, gsh2, gsc2, gg2 = _block_mod_params(gblk, mod_s)
            tsh, tsc, tg, tsh2, tsc2, tg2 = _block_mod_params(tblk, mod_x)
            hn_s = fused_norm_modulate(hs_in, gblk.norm1, gsc, gsh)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn_s)
            hn_x = fused_norm_modulate(hx_in, tblk.norm1, tsc, tsh)
            q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn_x)
            # geo lane: self + corner-masked tex read. Corner (reads-OFF) rows
            # take the FLASH pure-self result — bit-exact vs the specialist AND
            # kernel-identical to the tex|mesh cached path (the sdpa union is
            # 1-ULP-class off flash at real scale; G0-bidir caught it).
            # ABSOLUTE stream tag: tex's q/k rotated, geo's untouched (same
            # convention as the fused path and the cached inference path).
            q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)
            k_x_tag = k_x
            # SS keys join the union as a third segment with their own tag; they
            # arrive already re-roped into the 32^3 frame by ss_kv_for_slat.
            kv_ss = None if ss_kv is None else (
                _rot_pad(ss_kv[0], TAG_SS), ss_kv[1])
            if True:   # union coupling (the only one v12 trained)
                qkv_s = q_s.replace(torch.stack(
                    [q_s.feats, k_s.feats, v_s.feats], dim=1))
                a_flash = sparse_scaled_dot_product_attention(qkv_s)
                if any(corner_on):   # list,零同步
                    a_u = self._union_attn(q_s, k_s, v_s, k_x_tag, v_x, ss_kv=kv_ss,
                                           seg_on=corner_on)
                    sel = tok_on.to(a_flash.feats.dtype)
                    a_s = a_flash.replace(a_u.feats * sel
                                          + a_flash.feats * (1 - sel))
                else:
                    a_s = a_flash
            a_s = _attn_out(gblk.self_attn, a_s)
            hs = fused_gate_residual(hs_in, a_s, gg)
            hcs = hs.replace(gblk.norm2(hs.feats))
            hs = hs + gblk.cross_attn(hcs, cond_s) * self.xattn_scale
            hms = fused_norm_modulate(hs, gblk.norm3, gsc2, gsh2)
            hs = fused_gate_residual(hs, gblk.mlp(hms), gg2)
            # tex lane: self + geo read (always on; foreign keys stream-tagged)
            k_s_tag = _rotate_pad_pair(k_s)
            a_x = self._union_attn(q_x, k_x, v_x, k_s_tag, v_s, ss_kv=kv_ss)
            a_x = _attn_out(tblk.self_attn, a_x)
            hx = fused_gate_residual(hx_in, a_x, tg)
            hcx = hx.replace(tblk.norm2(hx.feats))
            hx = hx + tblk.cross_attn(hcx, cond_x) * self.xattn_scale
            hmx = fused_norm_modulate(hx, tblk.norm3, tsc2, tsh2)
            hx = fused_gate_residual(hx, tblk.mlp(hmx), tg2)
            return hs, hx

        if self._gc_on(gblk, tblk):
            import torch.utils.checkpoint as _ckpt
            return _ckpt.checkpoint(inner, h_s, h_x, use_reentrant=False)
        return inner(h_s, h_x)

    def _tex_block_inner(self, idx, h_x, mod_x, cond_x, k_s, v_s, k_c=None, v_c=None,
                         ss_kv=None):
        tblk = self.tex_flow.blocks[idx]
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = _block_mod_params(tblk, mod_x)
        hn = fused_norm_modulate(h_x, tblk.norm1, sc_msa, sh_msa)
        q_x, k_x, v_x = _attn_qkv(tblk.self_attn, hn)
        # ABSOLUTE stream tag (the convention training runs under, via the fused
        # path): rotate THIS stream's q and k; foreign keys stay unrotated.
        # Within-stream logits unchanged, cross-stream get ∓π/2. Keeping the
        # cached tex|mesh inference path on the same convention is mandatory —
        # a relative tag here would silently mismatch the trained model.
        q_x, k_x = _rotate_pad_pair(q_x), _rotate_pad_pair(k_x)
        kv_ss = None if ss_kv is None else (
            _rot_pad(ss_kv[0], TAG_SS), ss_kv[1])
        a = self._union_attn(q_x, k_x, v_x, k_s, v_s, k_c, v_c, ss_kv=kv_ss)
        a = _attn_out(tblk.self_attn, a)
        h_x = fused_gate_residual(h_x, a, g_msa)
        hc = h_x.replace(tblk.norm2(h_x.feats))
        h_x = h_x + tblk.cross_attn(hc, cond_x) * self.xattn_scale
        hm = fused_norm_modulate(h_x, tblk.norm3, sc_mlp, sh_mlp)
        h_x = fused_gate_residual(h_x, tblk.mlp(hm), g_mlp)
        return h_x

    def ss_prologue(self, x_ss, t_ss, cond_ss):
        """sparse_structure_flow.py:234-241 — everything before the block loop.

        Returns (h, mod_ss, cond_ss) in the tower's own dtype. Split out from the
        loop so the caller can interleave SS blocks with the slat pair rather
        than running the tower to completion first: pre-running it would mean
        holding all 30 blocks' (k, v) live across the whole slat pass, which is
        755 MB per sample at 4096 tokens.
        """
        ss = self.ss_flow
        assert list(x_ss.shape) == [x_ss.shape[0], ss.in_channels] + [ss.resolution] * 3, \
            f"SS input {tuple(x_ss.shape)} != (B, {ss.in_channels}, {ss.resolution}^3)"
        h = x_ss.view(*x_ss.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = ss.input_layer(h)
        t_emb = ss.t_embedder(t_ss)
        if ss.share_mod:
            t_emb = ss.adaLN_modulation(t_emb)
        return (manual_cast(h, ss.dtype), manual_cast(t_emb, ss.dtype),
                manual_cast(cond_ss, ss.dtype))

    def _run_ss_block(self, idx, h, mod_ss, cond_ss, ss_cond_mask=None,
                      want_kv: bool = False, ss_read=None):
        """modulated.py:148-165 replicated op-for-op for one dense SS block.

        want_kv returns (k_pre_rope, v) for the slat lanes to borrow.
        ss_read is the SS<-slat term (v10 keeps it dormant); it is ADDED to the
        attention output before to_out, the same place the slat lanes add theirs.

        CHECKPOINTED HERE, not via blk.use_checkpoint. This lane reaches into
        blk.norm1 / blk.self_attn / blk.cross_attn / blk.mlp and never calls
        blk.forward(), so the per-block use_checkpoint flag the builder sets is
        DEAD on this tower — exactly the trap the from_scratch branch of
        trellis_native_vlm.py documents for MMDiT3D, reproduced one tower over.
        The flag was set, the log printed "ss 30/30", and none of the 30 blocks
        was actually checkpointed: the SS tower is dense (4096 tokens x 1536 ch
        x 30 blocks, plus a 4x MLP hidden), so keeping every intermediate live
        costs ~9.5 GB per sample. That, not the 22% parameter growth over v8, is
        why PER_GPU_BS=4 could not fit while v8 ran at 8.
        """
        ss = self.ss_flow
        blk = ss.blocks[idx]
        sh1, sc1, g1, sh2, sc2, g2 = _dense_mod_params(blk, mod_ss)

        def inner(h, ss_read):
            hn = blk.norm1(h)
            hn = hn * (1 + sc1.unsqueeze(1)) + sh1.unsqueeze(1)
            q, k, v, k_pre = _dense_attn_qkv(blk.self_attn, hn, ss.rope_phases)
            a = _dense_sdpa(q, k, v)
            if ss_read is not None:
                a = a + ss_read
            a = _dense_attn_out(blk.self_attn, a)
            h = h + a * g1.unsqueeze(1)

            # * xattn_scale for the same reason the two slat lanes do: it is the
            # knob that anneals the read-once cross-attn away when a cond STREAM
            # takes over. Inert at 1.0 (the cross_attn cond_mode v10 runs), but
            # leaving it off would mean an anneal silently starved two towers of
            # conditioning while the third kept its own — an asymmetry nothing in
            # the logs would show.
            h = h + blk.cross_attn(blk.norm2(h), cond_ss,
                                   attn_mask=ss_cond_mask) * self.xattn_scale

            hm = blk.norm3(h)
            hm = hm * (1 + sc2.unsqueeze(1)) + sh2.unsqueeze(1)
            h = h + blk.mlp(hm) * g2.unsqueeze(1)
            return h, k_pre, v

        if self._gc_on(blk):
            import torch.utils.checkpoint as _ckpt
            # ss_read is passed as a checkpoint ARGUMENT, not captured in the
            # closure: a closure-captured tensor stays live for the whole
            # backward instead of being recomputed, which would leave the
            # cross-tower read holding memory this wrap exists to free.
            h, k_pre, v = _ckpt.checkpoint(inner, h, ss_read,
                                           use_reentrant=False)
        else:
            h, k_pre, v = inner(h, ss_read)
        return (h, k_pre, v) if want_kv else (h, None, None)

    def ss_epilogue(self, h, out_dtype):
        """sparse_structure_flow.py:243-247 — layer_norm, out_layer, reshape back
        to (B, C, res, res, res)."""
        ss = self.ss_flow
        h = manual_cast(h, out_dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = ss.out_layer(h)
        return h.permute(0, 2, 1).view(
            h.shape[0], h.shape[2], *[ss.resolution] * 3).contiguous()

    def ss_forward(self, x_ss, t_ss, cond_ss, ss_cond_mask=None):
        """The SS tower run standalone THROUGH the replicated lane. Exists so the
        replication can be certified against ss_flow(...) directly; the joint
        forward uses the same three pieces interleaved with the slat blocks."""
        h, mod_ss, c = self.ss_prologue(x_ss, t_ss, cond_ss)
        for i in range(len(self.ss_flow.blocks)):
            h, _, _ = self._run_ss_block(i, h, mod_ss, c, ss_cond_mask)
        return self.ss_epilogue(h, x_ss.dtype)

    def _ss_reads_slat(self, idx, h_ss, h_s, h_x, ss_read_on):
        """SS <- slat, the dormant direction. Returns None unless explicitly on.

        THE ROW MASK IS CORRECTNESS, NOT AN OPTIMISATION. At training time the
        slat lanes live on GT-derived coords, which ARE the occupancy the SS
        tower is being asked to predict. A row where SS may read them is a row
        where SS can copy the answer, and the symptom is a BETTER loss curve, so
        nothing downstream will complain. ss_read_on is the per-sample gate: it
        is on only for lag rows (t_ss > 0 with a slat context that is itself
        noised), off for the t_ss=0 clean rows where SS already holds the answer
        and off for solo rows where the slat lanes are pure noise.

        ss_reads_enabled is the separate, global inference switch — v10 samples
        with it False so the SS tower's trajectory is exactly the specialist's.
        """
        if (self.ss_reads_gate is None or not self.ss_reads_enabled
                or ss_read_on is None or not bool(ss_read_on.any())):
            return None
        ss, gblk = self.ss_flow, self.geo_flow.blocks[idx]
        tblk = self.tex_flow.blocks[idx]
        # q from the SS hidden through the SS block's own q projection; k/v from
        # the slat lanes' PRE-update hiddens, so the direction is symmetric with
        # the slat<-SS read that happens in the same block.
        blk = ss.blocks[idx]
        B, L, _ = h_ss.shape

        # ROW SELECTION, not compute-then-zero. ss_read_on is a 0/1 per-sample
        # mask, and the rule this codebase already follows everywhere the mask is
        # 0/1 is: do not compute what you are going to discard.
        #   * the union corner mask drops the foreign segment from the concat, so
        #     the shorter key set never enters the softmax
        #   * the losses select rows (pred[keep]) because multiply-by-zero would
        #     also corrupt the all-reduced denominator
        # This lane used to compute both cross-tower attentions for EVERY row and
        # then multiply 80% of them by zero (p_lag=0.2, and the early-return only
        # fires when a micro-batch draws no lag row at all — 0.8^8 = 17% at B=8).
        # Each discarded row costs a 4096-query attention against the slat keys,
        # twice, in all 30 blocks.
        # The slat<-SS direction has no gate at all now: SS is a segment of the
        # shared union softmax. This direction keeps its 0/1 row mask because
        # that mask is ANTI-LEAK, not an init ramp — the slat lanes sit on
        # GT-derived coords, so a row where SS may read them is a row where SS
        # can copy the answer.
        sel = ss_read_on.bool()
        h_ss_sel = h_ss[sel]
        q = blk.self_attn.to_qkv(h_ss_sel).reshape(
            h_ss_sel.shape[0], L, 3, blk.self_attn.num_heads, -1)[:, :, 0]
        if blk.self_attn.qk_rms_norm:
            q = blk.self_attn.q_rms_norm(q)
        out = 0.0
        for j, (hh, hb) in enumerate(((h_s, gblk), (h_x, tblk))):
            k, v = _attn_qkv(hb.self_attn, hh[sel])[1:]
            r = sparse_scaled_dot_product_attention(q, k, v)
            g = self.ss_reads_gate[idx, j].to(r.dtype).reshape(1, 1, -1, 1)
            out = out + g * r
        # scatter back so the caller still sees (B, ...) with exact zeros where
        # the read is off — index_put keeps the gradient on the selected rows and
        # gives the others none, which is what a mask means.
        full = out.new_zeros(B, *out.shape[1:])
        return full.index_put((sel.nonzero(as_tuple=True)[0],), out)

    def ss_kv_for_slat(self, k_pre, v):
        """Re-rope a borrowed SS key into the 32^3 slat frame.

        The key is rotated by the SS grid mapped through c -> 2c+0.5, so a slat
        query at voxel p and an SS key at cell c see the relative position
        (p - (2c+0.5)) that they would if both lived in the 32^3 grid. v is never
        roped anywhere in TRELLIS, so it passes through untouched.
        """
        ph = self.ss_phases_slat.to(k_pre.device)
        return RotaryPositionEmbedder.apply_rotary_embedding(k_pre, ph), v

    def forward(self, x_s, x_x, t_s, t_x, cond_s, cond_x, tex_concat_cond=None,
                x_ss=None, t_ss=None, cond_ss=None, ss_cond_mask=None,
                ss_read_on=None, ss_kv_cache=None):
        """x_s / x_x: noisy SparseTensors (32 ch, shared coords); t_s / t_x (B,) in flow units (x1000);
        cond_s / cond_x per-lane condition; tex_concat_cond: the geo state fed to tex (x_{t_s});
        x_ss / t_ss / cond_ss: the SS lane (dense); ss_kv_cache: per-block SS K/V from precompute_ss_kv.
        Returns (v_s, v_x) or (v_s, v_x, v_ss) when the SS lane runs."""
        geo, tex = self.geo_flow, self.tex_flow
        if tex_concat_cond is not None:
            assert torch.equal(x_s.coords, x_x.coords), "geo/tex coords mismatch"
            assert torch.equal(x_x.coords, tex_concat_cond.coords), "concat_cond coords mismatch"
            x_x_in = sp.sparse_cat([x_x, tex_concat_cond], dim=-1)
        else:
            assert torch.equal(x_s.coords, x_x.coords), "geo/tex coords mismatch"
            x_x_in = x_x
        if isinstance(cond_s, list):
            cond_s = sp.VarLenTensor.from_tensor_list(cond_s)
        if isinstance(cond_x, list):
            cond_x = sp.VarLenTensor.from_tensor_list(cond_x)
        import contextlib
        geo_frozen = not next(geo.parameters()).requires_grad
        gctx = (torch.no_grad if geo_frozen else contextlib.nullcontext)
        with gctx():
            h_s = manual_cast(geo.input_layer(x_s), geo.dtype)
            cond_s = manual_cast(cond_s, geo.dtype)
        t_emb_s = geo.t_embedder(t_s) + self.cross_alpha_s * self.t_mixer_s(t_x)
        mod_s = manual_cast(geo.adaLN_modulation(t_emb_s), geo.dtype)
        h_x = manual_cast(tex.input_layer(x_x_in), tex.dtype)
        t_emb_x = tex.t_embedder(t_x) + self.cross_alpha * self.t_mixer(t_s)
        mod_x = manual_cast(tex.adaLN_modulation(t_emb_x), tex.dtype)
        cond_x = manual_cast(cond_x, tex.dtype)

        corner_on = (t_s != 0)
        corner_host = corner_on.tolist()
        cached = ss_kv_cache is not None
        ss_on = x_ss is not None and not cached
        if ss_on:
            h_ss, mod_ss, c_ss = self.ss_prologue(x_ss, t_ss, cond_ss)
        if self.fused_attn:
            plan = _fusion_plan(x_s.layout, corner_host, x_s.feats.device,
                                n_ss=(h_ss.shape[1] if ss_on else (ss_kv_cache[0][0].shape[1] if cached else 0)))
            for idx in range(len(geo.blocks)):
                ss_kv = ss_kv_cache[idx] if cached else None
                if ss_on:
                    h_ss, k_pre, v_ss = self._run_ss_block(
                        idx, h_ss, mod_ss, c_ss, ss_cond_mask, want_kv=True,
                        ss_read=self._ss_reads_slat(idx, h_ss, h_s, h_x, ss_read_on))
                    ss_kv = self.ss_kv_for_slat(k_pre, v_ss)
                h_s, h_x = self._run_block_pair_fused(idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x, plan, ss_kv=ss_kv)
        else:
            tok_on = torch.cat([corner_on[b].to(x_s.feats.dtype).expand(sl.stop - sl.start)
                                for b, sl in enumerate(x_s.layout)]).view(-1, 1, 1)
            for idx in range(len(geo.blocks)):
                ss_kv = ss_kv_cache[idx] if cached else None
                if ss_on:
                    h_ss, k_pre, v_ss = self._run_ss_block(
                        idx, h_ss, mod_ss, c_ss, ss_cond_mask, want_kv=True,
                        ss_read=self._ss_reads_slat(idx, h_ss, h_s, h_x, ss_read_on))
                    ss_kv = self.ss_kv_for_slat(k_pre, v_ss)
                h_s, h_x = self._run_block_pair_bidir(idx, h_s, h_x, mod_s, mod_x, cond_s, cond_x,
                                                      corner_host, tok_on, ss_kv=ss_kv)
        with gctx():
            h_s = manual_cast(h_s, x_s.dtype)
            h_s = h_s.replace(F.layer_norm(h_s.feats, h_s.feats.shape[-1:]))
            v_s = geo.out_layer(h_s)
        h_x = manual_cast(h_x, x_x.dtype)
        h_x = h_x.replace(F.layer_norm(h_x.feats, h_x.feats.shape[-1:]))
        v_x = tex.out_layer(h_x)
        if ss_on:
            return v_s, v_x, self.ss_epilogue(h_ss, x_ss.dtype)
        return v_s, v_x


    @torch.no_grad()
    def precompute_ss_kv(self, x_ss, t_ss, cond_ss, ss_cond_mask=None):
        """Run the SS tower once with no SS<-slat read and return the per-block (k, v) the slat lanes
        borrow (already re-roped by ss_kv_for_slat). Pass the list as forward(..., ss_kv_cache=...)."""
        h_ss, mod_ss, c_ss = self.ss_prologue(x_ss, t_ss, cond_ss)
        out = []
        for idx in range(len(self.geo_flow.blocks)):
            h_ss, k_pre, v_ss = self._run_ss_block(idx, h_ss, mod_ss, c_ss, ss_cond_mask,
                                                   want_kv=True, ss_read=None)
            out.append(self.ss_kv_for_slat(k_pre, v_ss))
        return out

    @torch.no_grad()
    def precompute_geo_kv(self, x_s, t_s, cond_s, want_v: bool = False, t_x=None):
        """One full geo pass; returns (kv, v_s|None) with kv[i] = the block-i
        (k_s, v_s) post rms+RoPE — exactly what _run_block_pair borrows.
        tex|mesh mode calls this ONCE (t_s≡0 + clean shape state are constant
        across all tex steps → geo cost amortized to a single pass); joint mode
        calls it per step with want_v=True (v_s integrates the geo stream, kv
        feeds the same step's tex read — no double geo compute).
        Ops are verbatim _run_block_pair's geo lane (G0-certified bit-exact)."""
        geo = self.geo_flow
        if isinstance(cond_s, list):
            cond_s = sp.VarLenTensor.from_tensor_list(cond_s)
        h_s = manual_cast(geo.input_layer(x_s), geo.dtype)
        t_emb_s = geo.t_embedder(t_s)
        if t_x is not None and getattr(self, "t_mixer_s", None) is not None:
            # training adds cross_alpha_s * t_mixer_s(t_x) to the geo time embedding; v12's cache omitted
            # it (t_x=None keeps that). With it the K/V depend on t_x, so the caller recomputes them per tex step.
            t_emb_s = t_emb_s + self.cross_alpha_s * self.t_mixer_s(t_x)
        mod_s = manual_cast(geo.adaLN_modulation(t_emb_s), geo.dtype)
        cond_s = manual_cast(cond_s, geo.dtype)
        kv = []
        for gblk in geo.blocks:
            sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = _block_mod_params(gblk, mod_s)
            hn = fused_norm_modulate(h_s, gblk.norm1, sc_msa, sh_msa)
            q_s, k_s, v_s = _attn_qkv(gblk.self_attn, hn)
            kv.append((k_s, v_s))
            qkv_s = q_s.replace(torch.stack([q_s.feats, k_s.feats, v_s.feats], dim=1))
            a = sparse_scaled_dot_product_attention(qkv_s)
            a = _attn_out(gblk.self_attn, a)
            h_s = fused_gate_residual(h_s, a, g_msa)
            hc = h_s.replace(gblk.norm2(h_s.feats))
            h_s = h_s + gblk.cross_attn(hc, cond_s) * self.xattn_scale
            hm = fused_norm_modulate(h_s, gblk.norm3, sc_mlp, sh_mlp)
            h_s = fused_gate_residual(h_s, gblk.mlp(hm), g_mlp)
        v = None
        if want_v:
            h_s = manual_cast(h_s, x_s.dtype)
            h_s = h_s.replace(F.layer_norm(h_s.feats, h_s.feats.shape[-1:]))
            v = geo.out_layer(h_s)
        return kv, v

    @torch.no_grad()
    def tex_forward_cached(self, x_x, t_x, t_s, cond_x, tex_concat_cond, kv, ss_kv_cache=None):
        """Tex lane only, consuming precomputed geo (k,v). Mirrors forward()'s
        tex lane exactly — including the cross-t mixing, which still needs t_s
        even when the geo pass is cached."""
        tex = self.tex_flow
        assert torch.equal(x_x.coords, tex_concat_cond.coords), "concat_cond coords mismatch"
        x_x_in = sp.sparse_cat([x_x, tex_concat_cond], dim=-1)
        if isinstance(cond_x, list):
            cond_x = sp.VarLenTensor.from_tensor_list(cond_x)
        h_x = manual_cast(tex.input_layer(x_x_in), tex.dtype)
        t_emb_x = tex.t_embedder(t_x) + self.cross_alpha * self.t_mixer(t_s)
        mod_x = manual_cast(tex.adaLN_modulation(t_emb_x), tex.dtype)
        cond_x = manual_cast(cond_x, tex.dtype)
        for idx in range(len(tex.blocks)):
            k_s, v_s = kv[idx]
            h_x = self._tex_block_inner(idx, h_x, mod_x, cond_x, k_s, v_s,
                                        ss_kv=None if ss_kv_cache is None else ss_kv_cache[idx])
        h_x = manual_cast(h_x, x_x.dtype)
        h_x = h_x.replace(F.layer_norm(h_x.feats, h_x.feats.shape[-1:]))
        return tex.out_layer(h_x)
