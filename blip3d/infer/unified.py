"""Sampling the unified model: modes ``refine`` (v12 benchmark default), ``joint`` and ``interleave``.

refine     SS alone -> coords -> shape+tex joint loop (tex time warped by alpha) -> tex re-sampled on the final shape
joint      the same without the tex re-sample
interleave refine's first round, then SS re-runs READING the slat on the trained lag schedule, new coords, joint(+refine)

Numerics follow v12 exactly by default (``UnifiedOptions()``); the fixes from ISSUES U-02/03/04/06 are opt-in until
they are measured (``UnifiedOptions.fixed()``). ``cache_ss_kv`` is exact (verified bit-identical) and on by default.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..cond.assemble import cond_uncond
from ..models.unified.loading import UnifiedBundle
from .export import Latents
from .presets import TRAIN_LAG_GRID, Preset
from .schedule import guided, t_seq, warp
from .stages import occ_to_coords


@dataclass(frozen=True)
class UnifiedOptions:
    cache_ss_kv: bool = True           # exact
    ss_uncond_in_uncond: bool = False  # U-04: v12 fed the SS lane the positive cond in the uncond forward
    tex_reads_ss: bool = False         # U-02: v12 refine / tex-on-mesh ran without the SS K/V training always had
    lag_on_train_grid: bool = False    # U-06: v12 interleave paired nodes on the preset's SS steps (24 on the text path)
    geo_kv_tmix: bool = False          # U-03: v12's cached geo pass dropped the t_mixer_s(t_x) term (K/V recomputed per step)

    @classmethod
    def fixed(cls) -> "UnifiedOptions":
        return cls(True, True, True, True, True)


class UnifiedSampler:
    def __init__(self, bundle: UnifiedBundle, ss_decoder, preset: Preset, opts: UnifiedOptions = UnifiedOptions()):
        self.b, self.m, self.ssdec, self.p, self.o = bundle, bundle.model, ss_decoder, preset, opts

    # ── conditioning ──
    def conds(self, record) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """(cond, uncond) per lane, bf16 (the unified model runs in its training layout under autocast)."""
        out = {}
        for lane in ("ss", "geo", "tex"):
            c, u = cond_uncond(self.b.connectors[lane], self.b.views, record, dtype=self.p.cond_dtype)
            out[lane] = (c.to(torch.bfloat16), u.to(torch.bfloat16))
        return out

    # ── SS alone ──
    @torch.no_grad()
    def ss_alone(self, cs, seed: int) -> Tuple[torch.Tensor, torch.Generator]:
        ss, p = self.m.ss_flow, self.p.ss
        g = torch.Generator(device="cuda").manual_seed(seed)
        was = self.m.ss_reads_enabled
        self.m.ss_reads_enabled = False
        try:
            ts = t_seq(p.steps, p.rescale_t)
            z = torch.randn(1, ss.in_channels, *[ss.resolution] * 3, generator=g, device="cuda", dtype=torch.float32)
            for i in range(p.steps):
                t, tp = float(ts[i]), float(ts[i + 1])
                tt = torch.tensor([t * 1000.0], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vp = self.m.ss_forward(z.to(torch.bfloat16), tt, cs[0])
                    vn = self.m.ss_forward(z.to(torch.bfloat16), tt, cs[1])
                z = z - (t - tp) * guided(vp.float(), vn.float(), z, t, p)
        finally:
            self.m.ss_reads_enabled = was
        return z, g

    def _ss_kw(self, z, c_ss, u_ss):
        """SS lane context for the slat passes: the generated structure at t_ss = 0 (cond, uncond variants)."""
        t0 = torch.tensor([0.0], device="cuda")
        pos = dict(x_ss=z.to(torch.bfloat16), t_ss=t0, cond_ss=c_ss)
        neg = dict(pos, cond_ss=u_ss) if self.o.ss_uncond_in_uncond else pos
        if self.o.cache_ss_kv:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                kv_pos = self.m.precompute_ss_kv(pos["x_ss"], t0, pos["cond_ss"])
                kv_neg = kv_pos if neg is pos else self.m.precompute_ss_kv(neg["x_ss"], t0, neg["cond_ss"])
            return dict(ss_kv_cache=kv_pos), dict(ss_kv_cache=kv_neg)
        return pos, neg

    def _ss_kv_for_tex(self, z, cs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.precompute_ss_kv(z.to(torch.bfloat16), torch.tensor([0.0], device="cuda"), cs["ss"][0])

    # ── the joint loop (+ refine) ──
    @torch.no_grad()
    def joint_on_coords(self, coords, cs, z, seed: int, refine: bool):
        from trellis2.modules import sparse as sp
        geo, ps, px = self.m.geo_flow, self.p.shape, self.p.tex
        g = torch.Generator(device="cuda").manual_seed(seed)
        N = coords.shape[0]
        x_s = sp.SparseTensor(torch.randn(N, geo.in_channels, generator=g, device="cuda", dtype=torch.float32), coords.cuda())
        x_x = torch.randn(N, 32, generator=g, device="cuda", dtype=torch.float32)
        ts_s = np.array(t_seq(ps.steps, ps.rescale_t))
        ts_x = np.array([warp(float(t), self.p.alpha) for t in ts_s])
        kw_pos, kw_neg = self._ss_kw(z, *cs["ss"])
        for i in range(ps.steps):
            tx, txp, tss, tsp = float(ts_x[i]), float(ts_x[i + 1]), float(ts_s[i]), float(ts_s[i + 1])
            tt_x = torch.tensor([tx * 1000.0], device="cuda")
            tt_s = torch.tensor([tss * 1000.0], device="cuda")
            xin = sp.SparseTensor(x_x, coords.cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                op = self.m(x_s, xin, tt_s, tt_x, cs["geo"][0], cs["tex"][0], tex_concat_cond=x_s, **kw_pos)
                on = self.m(x_s, xin, tt_s, tt_x, cs["geo"][1], cs["tex"][1], tex_concat_cond=x_s, **kw_neg)
            x_x = x_x - (tx - txp) * guided(op[1].feats.float(), on[1].feats.float(), x_x, tx, px)
            x_s = x_s.replace(x_s.feats - (tss - tsp) * guided(op[0].feats.float(), on[0].feats.float(),
                                                                x_s.feats.float(), tss, ps))
        if refine:
            # v12 ran the refine pass without the SS K/V that training always had (U-02 fixes it, opt-in)
            ss_kv = self._ss_kv_for_tex(z, cs) if self.o.tex_reads_ss else None
            x_x = self.tex_given_mesh(coords, x_s.feats, cs, seed=seed + 1, ss_kv=ss_kv)
        return x_s, x_x

    @torch.no_grad()
    def tex_given_mesh(self, coords, shape_norm_feats, cs, seed: int, ss_kv=None) -> torch.Tensor:
        """Tex lane on a fixed clean shape (t_s = 0): one geo pass, cached K/V, then the tex Euler loop."""
        from trellis2.modules import sparse as sp
        N = coords.shape[0]
        cc = sp.SparseTensor(shape_norm_feats.float().cuda(), coords.cuda())
        t0 = torch.tensor([0.0], device="cuda")
        kv, _ = self.m.precompute_geo_kv(cc, t0, cs["geo"][0])
        g = torch.Generator(device="cuda").manual_seed(seed)
        exact = self.o.geo_kv_tmix
        x = torch.randn(N, 32, generator=g, device="cuda", dtype=torch.float32)
        p = self.p.tex
        ts = t_seq(p.steps, p.rescale_t)
        for i in range(p.steps):
            t, tp = float(ts[i]), float(ts[i + 1])
            tt = torch.tensor([t * 1000.0], device="cuda")
            xin = sp.SparseTensor(x, coords.cuda())
            if exact:
                kv, _ = self.m.precompute_geo_kv(cc, t0, cs["geo"][0], t_x=tt)
            vp = self.m.tex_forward_cached(xin, tt, t0, cs["tex"][0], cc, kv, ss_kv_cache=ss_kv).feats.float()
            vn = (vp if p.cfg == 1 else  # released tex cfg 1.0: the negative branch is never needed
                  self.m.tex_forward_cached(xin, tt, t0, cs["tex"][1], cc, kv, ss_kv_cache=ss_kv).feats.float())
            x = x - (t - tp) * guided(vp, vn, x, t, p)
        return x

    # ── modes ──
    @torch.no_grad()
    def run(self, record, mode: str = "refine", seed: int = 0, k0: int = 7) -> Tuple[Optional[Latents], Dict]:
        """One asset. Returns ((shape SparseTensor, tex feats, coords) in norm space, info), or (None, info) when
        the SS lane produced an empty structure (a declared failure, not an exception).
        The whole sampler runs under bf16 autocast, including the occupancy decode, exactly as the v12 benchmark
        called it; the connectors run in fp32 before that and are cast to bf16."""
        cs = self.conds(record)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, g = self.ss_alone(cs["ss"], seed)
            coords, info = occ_to_coords(self.ssdec, z, max_voxels=self.p.max_voxels, generator=g)
            info.update(mode=mode, preset=self.p.as_dict(), options=dict(self.o.__dict__))
            if info["empty"]:
                return None, info
            if mode in ("refine", "joint"):
                x_s, x_x = self.joint_on_coords(coords, cs, z, seed, refine=(mode == "refine"))
                return (x_s, x_x, coords), info
            if mode != "interleave":
                raise ValueError(f"unknown mode {mode!r}")
            return self._interleave(cs, z, coords, info, seed, k0)

    @torch.no_grad()
    def _interleave(self, cs, z, coords1, info, seed, k0):
        from trellis2.modules import sparse as sp
        x_s1, x_x1 = self.joint_on_coords(coords1, cs, z, seed, refine=False)
        info["round1_voxels"] = info["voxels_kept"]
        # v12 used a FRESH Generator(seed) here, so the SS re-noise below equals round 1's initial noise. Kept.
        g = torch.Generator(device="cuda").manual_seed(seed)
        shift = lambda r, u: r * u / (1.0 + (r - 1.0) * u)
        p = self.p.ss
        S = TRAIN_LAG_GRID["steps"] if self.o.lag_on_train_grid else p.steps
        assert 2 <= k0 < S, f"k0={k0} outside the trained range [2, {S - 1}]"
        was = self.m.ss_reads_enabled
        self.m.ss_reads_enabled = True
        try:
            t0 = shift(5.0, 1.0 - k0 / S)
            z2 = (1.0 - t0) * z + t0 * torch.randn(z.shape, generator=g, device="cuda", dtype=torch.float32)
            ns = torch.randn(x_s1.feats.shape, generator=g, device="cuda")
            nx = torch.randn(x_x1.shape, generator=g, device="cuda")
            one = torch.ones(1, device="cuda")
            s1f, x1f = x_s1.feats.float(), x_x1.float()
            info.update(lag_k0=k0, lag_t_ss0=round(t0, 4), lag_t_s_min=round(shift(3.0, k0 / S), 4))
            for i in range(k0, S):
                t_ss, t_ssn, t_s = shift(5.0, 1.0 - i / S), shift(5.0, 1.0 - (i + 1) / S), shift(3.0, 1.0 - (i - k0) / S)
                xs_t = sp.SparseTensor(((1 - t_s) * s1f + t_s * ns).to(torch.bfloat16), coords1)
                xx_t = sp.SparseTensor(((1 - t_s) * x1f + t_s * nx).to(torch.bfloat16), coords1)
                tt_s = torch.tensor([t_s * 1000.0], device="cuda")
                tt_ss = torch.tensor([t_ss * 1000.0], device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, _, vp = self.m(xs_t, xx_t, tt_s, tt_s, cs["geo"][0], cs["tex"][0], tex_concat_cond=xs_t,
                                      x_ss=z2.to(torch.bfloat16), t_ss=tt_ss, cond_ss=cs["ss"][0], ss_read_on=one)
                    _, _, vn = self.m(xs_t, xx_t, tt_s, tt_s, cs["geo"][1], cs["tex"][1], tex_concat_cond=xs_t,
                                      x_ss=z2.to(torch.bfloat16), t_ss=tt_ss, cond_ss=cs["ss"][1], ss_read_on=one)
                z2 = z2 - (t_ss - t_ssn) * guided(vp.float(), vn.float(), z2, t_ss, p)
        finally:
            self.m.ss_reads_enabled = was
        coords2, info2 = occ_to_coords(self.ssdec, z2, max_voxels=self.p.max_voxels, generator=g)
        info["round2_voxels"] = info2["voxels_generated"]
        if info2["empty"]:
            info["empty_round2"] = True
            return (x_s1, x_x1, coords1), info
        info.update(truncated=info2["truncated"], voxels_kept=info2["voxels_kept"])
        x_s2, x_x2 = self.joint_on_coords(coords2, cs, z2, seed, refine=True)
        return (x_s2, x_x2, coords2), info
