"""One task class for the three input modalities: text (``t``), single image (``i1``), multi image (``im``).

v12 had five registered dataset classes for this (text_to_3d, text_to_3d_weighted, image_to_3d,
multi_image_to_3d plus the base) and routed on their names. Here the modality is a field and the task name is only
a label for logging (``t_real``, ``t_synth``, ``i1``, ``im``).

Per-item randomness follows v12 draw for draw, so with the same generator the same view,
caption and template come out:
* text, ``fallback: long`` (v12 text_to_3d_weighted): a tier is drawn from the four fixed-order weights; a tier the
  asset lacks falls back to the long caption.
* text, ``fallback: renormalize`` (v12 text_to_3d): weights renormalized over the captions the asset has.
* single image: a view uniform over all ``n_views`` renders (the three below-ground views included); a missing file
  falls back to the first render in the directory.
* multi image: n in {2, 3, 4} drawn once per batch (``set_batch_params``, rank-synced), then n distinct views
  weighted by :data:`IM_VIEW_WEIGHTS`, sorted.
The template of a caption is ``(int(sha[:8], 16) + caption index) % 3``.

``compat_v12`` reproduces v12's loader exactly: an unseeded generator per item, any exception resampled (eight
tries, then the ninth surfaces), pbr loaded whenever the manifest has it. Otherwise (the default) the generator is
the per-worker one the mixture passes in, only whitelisted failures are resampled (see
:func:`blip3d.data.latents.skip_reason`) and counted per reason, anything else fails immediately, and pbr is read
only when tex is a target.
"""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..cond import prompts
from ..cond.prep import prep_render, prep_renders, prep_text
from . import latents as L
from .manifest import Manifest, Record, load_manifest

MODALITIES = ("t", "i1", "im")
TARGETS = ("ss", "shape", "tex")
TEX_MISSING = ("placeholder", "resample", "batch_drop")

DEFAULT_TIER_WEIGHTS = (0.35, 0.20, 0.10, 0.35)   # [long, medium, short, long+texture]

# multi-image view weights over the 16 renders (live_cond_batch.py:64-67)
IM_VIEW_WEIGHTS = tuple([0.15] * 3 +   # 000-002 below ground
                        [0.6] * 2 +    # 003-004
                        [1.0] * 7 +    # 005-011 eye level .. three-quarter
                        [0.5] * 4)     # 012-015


@dataclass
class TaskConfig:
    name: str
    modality: str
    manifest: str
    weight: float = 1.0
    targets: Tuple[str, ...] = ("ss",)
    tex_missing: str = "batch_drop"
    slat_resolution: int = 512
    min_aesthetic: Optional[float] = None
    max_slat_tokens: int = 8192          # applied only when shape or tex is a target (v12 semantics)
    align_points: int = 0                # ROAD teacher cloud size (SS tower: 10000)
    caption_weights: Optional[List[float]] = None
    caption_fallback: str = "renormalize"   # "long" | "renormalize"
    im_view_probs: Optional[List[float]] = None   # p(n) for n = 2..max_views
    max_views: int = 4
    dino_size: int = 512
    im_tok_per_view: int = 256

    def __post_init__(self):
        if self.modality not in MODALITIES:
            raise ValueError(f"task {self.name}: modality must be one of {MODALITIES}, got {self.modality!r}")
        self.targets = tuple(self.targets)
        bad = [t for t in self.targets if t not in TARGETS]
        if bad or "ss" not in self.targets:
            raise ValueError(f"task {self.name}: targets must include 'ss' and be drawn from {TARGETS}, got {self.targets}")
        if "tex" in self.targets and "shape" not in self.targets:
            raise ValueError(f"task {self.name}: a tex target needs the shape latent (concat condition)")
        if self.tex_missing not in TEX_MISSING:
            raise ValueError(f"task {self.name}: tex_missing must be one of {TEX_MISSING}")
        if self.caption_fallback not in ("long", "renormalize"):
            raise ValueError(f"task {self.name}: captions.fallback must be 'long' or 'renormalize'")
        if self.align_points and self.slat_resolution != 512:
            raise ValueError("ROAD align points are defined on the 512 grid (32^3 voxels) only")
        if self.caption_weights is not None:
            w = [float(x) for x in self.caption_weights]
            if not w or min(w) < 0 or sum(w) <= 0:
                raise ValueError(f"task {self.name}: caption weights must be non-negative with a positive sum")
            self.caption_weights = w
        if self.im_view_probs is not None:
            p = [float(x) for x in self.im_view_probs]
            if len(p) != self.max_views - 1 or min(p) < 0 or sum(p) <= 0:
                raise ValueError(f"task {self.name}: views.n_probs must list p(n) for n = 2..{self.max_views}")
            self.im_view_probs = p


class ThreeDTask(Dataset):
    """Map-style dataset over one manifest. ``get(i, rng)`` returns one item (see :mod:`blip3d.data.collate`)."""

    FAIL_WINDOW = 1000          # attempts per health window
    WARN_RATE = 0.05            # non-cap failure rate that is logged loudly
    ABORT_RATE = 0.5            # non-cap failure rate that is treated as a systematic bug

    def __init__(self, cfg: TaskConfig, *, qwen_path: str, compat_v12: bool = False, resample_tries: int = 8,
                 vox_lookup=None, verbose: bool = True):
        super().__init__()
        self.cfg = cfg
        self.task_name = cfg.name
        self.modality = cfg.modality
        self.qwen_path = qwen_path
        self.compat = bool(compat_v12)
        self.tries = int(resample_tries)
        self.cap = int(cfg.max_slat_tokens or 0) if ("shape" in cfg.targets or "tex" in cfg.targets) else 0
        self.records: Manifest = load_manifest(
            cfg.manifest, slat_resolution=cfg.slat_resolution, min_aesthetic=cfg.min_aesthetic,
            require_caption=(cfg.modality == "t"), validate=not self.compat,
            need_renders=(cfg.modality != "t"), need_hex_sha=(cfg.modality == "t"),
            voxel_cap=self.cap, vox_lookup=vox_lookup)
        if len(self.records) == 0:
            raise ValueError(f"task {cfg.name}: no records left in {cfg.manifest}")
        if verbose:
            print(f"[data] {cfg.name} ({cfg.modality}) {self.records.summary()}", flush=True)
        if cfg.caption_fallback == "long":
            # v12 text_to_3d_weighted: the yaml weights, else its documented default .35/.20/.10/.35
            w = cfg.caption_weights[:4] if cfg.caption_weights is not None else DEFAULT_TIER_WEIGHTS
            p = np.asarray(w, dtype=np.float64)
            self._tier_probs = p / p.sum()
        else:
            self._tier_probs = None
        self._batch_n_views: Optional[int] = None
        self.skips: collections.Counter = collections.Counter()
        self._win_n = 0
        self._win_bad = 0

    def __len__(self) -> int:
        return len(self.records)

    # ── batch-level parameters (called once per batch by the mixture, with the rank-synced generator) ──
    def _draw_n_views(self, rng) -> int:
        if self.cfg.im_view_probs:
            p = np.asarray(self.cfg.im_view_probs, dtype=np.float64)
            p = p / p.sum()
            return int(rng.choice(np.arange(2, 2 + p.size), p=p))
        return int(rng.integers(2, max(3, self.cfg.max_views + 1)))

    def set_batch_params(self, rng) -> None:
        if self.modality == "im":
            self._batch_n_views = self._draw_n_views(rng)

    def clear_batch_params(self) -> None:
        self._batch_n_views = None

    # ── per-item draws ──
    def caption_index(self, rng, n_caps: int) -> int:
        """Index into the asset's non-empty captions."""
        if self.cfg.caption_fallback == "long":
            ci = int(rng.choice(len(self._tier_probs), p=self._tier_probs))
            if ci >= n_caps:
                ci = 0
            return min(ci, n_caps - 1)
        if n_caps <= 0:
            return 0
        w = self.cfg.caption_weights
        if not w:
            return int(rng.integers(0, n_caps))
        p = np.asarray(w[:n_caps], dtype=np.float64)
        if p.size < n_caps:
            p = np.concatenate([p, np.zeros(n_caps - p.size)])
        if p.sum() <= 0:
            return int(rng.integers(0, n_caps))
        return int(rng.choice(n_caps, p=p / p.sum()))

    def n_views(self, rng) -> int:
        if self._batch_n_views is not None:
            return int(self._batch_n_views)
        return self._draw_n_views(rng)

    @staticmethod
    def pick_im_views(rng, n_avail: int, k: int) -> List[int]:
        n = min(n_avail, len(IM_VIEW_WEIGHTS))
        w = np.asarray(IM_VIEW_WEIGHTS[:n], dtype=np.float64)
        k = min(k, n)
        return sorted(int(v) for v in rng.choice(n, size=k, replace=False, p=w / w.sum()))

    # ── loading ──
    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.get(i, None)

    def get(self, i: int, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        n = len(self.records)
        if self.compat:
            for _ in range(8):
                try:
                    return self.load(i, rng)
                except Exception as e:
                    print(f"[{self.task_name}] load failed for {self.records.sha[i]}: {e!r} — resampling", flush=True)
                    i = int(np.random.default_rng().integers(0, n)) if n > 1 else i
            return self.load(i, rng)
        reasons: List[str] = []
        for _ in range(self.tries):
            try:
                item = self.load(i, rng)
                self._health(ok=True)
                return item
            except Exception as e:
                r = L.skip_reason(e)
                if r is None:
                    raise RuntimeError(f"[data] {self.task_name}: asset {self.records.sha[i]} failed with a "
                                       f"non-recoverable error: {e!r}") from e
                self.skips[r] += 1
                reasons.append(f"{self.records.sha[i][:12]}:{r}")
                self._health(ok=False, cap=(r == "voxel_cap"))
                g = rng if rng is not None else np.random.default_rng()
                i = int(g.integers(0, n)) if n > 1 else i
        raise RuntimeError(f"[data] {self.task_name}: {self.tries} consecutive samples failed: {reasons}")

    def _health(self, ok: bool, cap: bool = False) -> None:
        self._win_n += 1
        if not ok and not cap:
            self._win_bad += 1
        if self._win_n >= self.FAIL_WINDOW:
            rate = self._win_bad / self._win_n
            if rate > self.ABORT_RATE:
                raise RuntimeError(f"[data] {self.task_name}: {rate:.0%} of the last {self._win_n} loads failed "
                                   f"({dict(self.skips)}) — systematic, not per-asset")
            if rate > self.WARN_RATE:
                print(f"[data] WARNING {self.task_name}: {rate:.1%} load failures in the last {self._win_n} "
                      f"(totals {dict(self.skips)})", flush=True)
            self._win_n = self._win_bad = 0

    def load(self, i: int, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        rec = self.records.record(i)
        g = rng if rng is not None else np.random.default_rng()
        if self.cap > 0 and rec.shape and not self.compat:
            # cheap pre-check before any image / tokenizer work (v12 did this for real-caption text only)
            nvox = int(L.read_coords(rec.shape).shape[0])
            if nvox > self.cap:
                raise L.VoxelCapExceeded(f"{nvox} voxels > {self.cap} (pre-check)")
        elif self.cap > 0 and rec.shape and self.cfg.caption_fallback == "long" and self.modality == "t":
            nvox = int(L.read_coords(rec.shape).shape[0])
            if nvox > self.cap:
                raise L.VoxelCapExceeded(f"{nvox} voxels > {self.cap} (pre-check)")
        if self.modality == "t":
            prep = self._prep_text(rec, g)
        elif self.modality == "i1":
            prep = self._prep_i1(rec, g)
        else:
            prep = self._prep_im(rec, g)
        item: Dict[str, Any] = {"task": self.task_name, "modality": self.modality, "id": rec.sha, "prep": prep,
                                "ss": L.load_ss(rec.ss)}
        self._attach_targets(item, rec, g)
        return item

    def _prep_text(self, rec: Record, g) -> Dict[str, torch.Tensor]:
        caps = rec.captions
        if self.cfg.caption_fallback == "long":
            ci = self.caption_index(g, len(caps))
            if not caps:
                raise L.MissingField("captions")
            caption = caps[ci]
        else:
            ci = self.caption_index(g, len(caps)) if caps else 0
            if not caps and not self.compat:
                raise L.MissingField("captions")
            caption = caps[ci] if caps else ""
        return prep_text(caption, qwen_path=self.qwen_path, template=prompts.template_index(rec.sha, ci))

    def _prep_i1(self, rec: Record, g) -> Dict[str, torch.Tensor]:
        if not rec.renders_dir:
            raise L.MissingField("renders_dir")
        view = int(g.integers(0, max(1, rec.n_views)))
        path = L.render_path(rec.renders_dir, view)
        if path is None:
            cand = sorted(f for f in os.listdir(rec.renders_dir) if f.endswith((".webp", ".png", ".jpg")))
            if not cand:
                raise L.MissingFile(f"no render in {rec.renders_dir}")
            path = os.path.join(rec.renders_dir, cand[0])
            view = int(os.path.basename(path)[:3])
        img = Image.open(path)
        return prep_render(img, qwen_path=self.qwen_path, dino_size=self.cfg.dino_size, view=view)

    def _prep_im(self, rec: Record, g) -> Dict[str, torch.Tensor]:
        if not rec.renders_dir:
            raise L.MissingField("renders_dir")
        views = self.pick_im_views(g, rec.n_views, self.n_views(g))
        imgs, kept = [], []
        for v in views:
            p = L.render_path(rec.renders_dir, v)
            if p is None:
                continue
            imgs.append(Image.open(p))
            kept.append(int(v))
        if not imgs:
            raise L.MissingFile(f"no renders for views {views} in {rec.renders_dir}")
        if len(kept) < len(views) and (not self.compat or len(kept) < 2):
            # v12 went on with the surviving views (breaking the batch's view-count lock); a single survivor
            # cannot be encoded as multi-image at all
            raise L.MissingFile(f"views {sorted(set(views) - set(kept))} missing in {rec.renders_dir}")
        return prep_renders(imgs, qwen_path=self.qwen_path, views=kept, tok_per_view=self.cfg.im_tok_per_view,
                            dino_size=self.cfg.dino_size)

    def _align(self, coords, g) -> torch.Tensor:
        # compat: v12 drew the cloud from its own fresh generator
        g = None if self.compat else g
        pts = L.sample_align_points(np.asarray(coords), self.cfg.align_points, rng=g)
        return torch.from_numpy(pts)

    def _attach_targets(self, item: Dict[str, Any], rec: Record, rng) -> None:
        cfg = self.cfg
        if "shape" not in cfg.targets:                          # SS tower: no voxel cap (v12)
            if cfg.align_points:
                if not rec.shape:
                    raise L.MissingField("shape_latent_512 (ROAD align points)")
                item["align_points"] = self._align(L.read_coords(rec.shape), rng)
            return
        if not rec.shape:
            if not self.compat:
                raise L.MissingField(f"shape_latent_{cfg.slat_resolution}")
            if cfg.align_points:
                raise L.MissingField("shape_latent_512 (ROAD align points)")
            return
        coords, feats = L.load_slat_raw(rec.shape)
        if self.cap > 0 and coords.shape[0] > self.cap:
            raise L.VoxelCapExceeded(f"{coords.shape[0]} voxels > {self.cap}")
        st = L.norm_stats()
        shape_item = {"coords": coords, "feats": L.normalize(feats, st.shape_mean, st.shape_std)}
        item["shape"] = shape_item
        if cfg.align_points:
            item["align_points"] = self._align(coords.numpy(), rng)
        want_tex = "tex" in cfg.targets
        if rec.pbr and (want_tex or self.compat):
            item["tex"] = L.load_tex(rec.pbr, rec.shape, shape_raw=(coords, feats))
            item["tex_valid"] = True
        elif want_tex and cfg.tex_missing == "placeholder":
            item["tex"] = L.tex_placeholder(shape_item)
            item["tex_valid"] = False
        elif want_tex and cfg.tex_missing == "resample" and not self.compat:   # v12: the batch lost tex instead
            raise L.MissingField(f"pbr_latent_{cfg.slat_resolution}")
