"""One entry point for generation: separate towers (the v12 product path) or the unified model (refine / joint /
interleave). Inputs: renders / images / text -> prep -> encoder record -> latents -> mesh -> GLB.

    pipe = Pipeline.separate(ss="<ss ckpt>", shape="<shape ckpt>", tex="<tex ckpt>")
    pipe = Pipeline.unified("<unified ckpt>", mode="refine")
    lat = pipe.generate(pipe.prep_text("a red chair"), seed=0)
    pipe.export(lat, "out/")                  # full.glb, shape.obj, shape.glb
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import torch
from PIL import Image

from ..cond import prep as P
from ..cond.encoder import DINO_DEFAULT, QWEN_DEFAULT, CondEncoder
from ..cond.prompts import template_index
from ..models.towers import latent_stats, load_decoders, load_tower
from ..utils.paths import REPO_ROOT, get_paths, resolve_hf_snapshot
from .cascade import run_cascade, tex_on_shape
from .export import Latents, decode, export_glb, export_shape_obj
from .presets import preset_for


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(get_paths().runs, p)


class Pipeline:
    def __init__(self, kind: str, *, towers=None, sampler=None, device: str = "cuda"):
        self.kind, self.towers, self.sampler, self.device = kind, towers, sampler, device
        paths = get_paths()
        self.qwen = resolve_hf_snapshot(QWEN_DEFAULT, paths.hf_cache)
        self.encoder = CondEncoder(self.qwen, resolve_hf_snapshot(DINO_DEFAULT, paths.hf_cache), device=device)
        self.dec = load_decoders(paths.trellis2_ckpt, paths.ss_decoder, device)
        self.stats = latent_stats(str(REPO_ROOT / "third_party" / "TRELLIS.2")).to(device)

    # ── construction ──
    @classmethod
    def separate(cls, ss: str, shape: str, tex: str, device: str = "cuda") -> "Pipeline":
        ck = get_paths().trellis2_ckpt
        towers = {k: load_tower(k, _abs(p), trellis2_ckpt=ck, device=device, layout="released")
                  for k, p in (("ss", ss), ("shape", shape), ("tex", tex))}
        return cls("separate", towers=towers, device=device)

    @classmethod
    def unified(cls, ckpt: str, mode: str = "refine", options=None, device: str = "cuda") -> "Pipeline":
        from ..models.unified.loading import load_unified
        from .unified import UnifiedOptions
        bundle = load_unified(_abs(ckpt), trellis2_ckpt=get_paths().trellis2_ckpt, device=device)
        pipe = cls("unified", device=device)
        pipe.sampler = (bundle, mode, options or UnifiedOptions())
        return pipe

    # ── inputs (CPU prep, same framing as training) ──
    def prep_render(self, path: str, view: int = 0):
        return P.prep_render(Image.open(path), qwen_path=self.qwen, view=view)

    def prep_renders(self, paths: Sequence[str], views: Optional[Sequence[int]] = None):
        return P.prep_renders([Image.open(p) for p in paths], qwen_path=self.qwen, views=views)

    def prep_text(self, caption: str, sha: Optional[str] = None, caption_index: int = 0):
        t = template_index(sha, caption_index) if sha else 0
        return P.prep_text(caption, qwen_path=self.qwen, template=t)

    # ── generation ──
    @torch.no_grad()
    def generate(self, prep: Dict, seed: int = 0, k0: int = 7) -> Latents:
        record = self.encoder.encode([prep])[0]
        mod = prep["modality"]
        if self.kind == "separate":
            return run_cascade(self.towers, self.dec, self.stats, record, preset_for(mod), seed=seed)
        from .unified import UnifiedSampler
        bundle, mode, opts = self.sampler
        smp = UnifiedSampler(bundle, self.dec.ss, preset_for(mod, unified=True), opts)
        out, info = smp.run(record, mode=mode, seed=seed, k0=k0)
        if out is None:
            return Latents(torch.empty(0, 4, dtype=torch.int32), torch.empty(0, 32), torch.empty(0, 32), info)
        x_s, x_x, coords = out
        s = self.stats
        return Latents(coords, x_s.feats.float() * s.shape_std + s.shape_mean, x_x.float() * s.tex_std + s.tex_mean, info)

    @torch.no_grad()
    def texture_on_shape(self, prep: Dict, coords: torch.Tensor, shape_raw: torch.Tensor, seed: int = 0) -> torch.Tensor:
        """Controlled-texture track: the tex tower (separate) on a given geometry, e.g. the GT shape."""
        if self.kind != "separate":
            raise NotImplementedError("unified texture-on-GT-shape: use UnifiedSampler.tex_given_mesh")
        rec = self.encoder.encode([prep])[0]
        return tex_on_shape(self.towers["tex"], self.stats, rec, preset_for(prep["modality"]), coords, shape_raw, seed)

    # ── output ──
    def export(self, lat: Latents, out_dir: str, name: str = "full") -> Dict[str, str]:
        import trimesh
        os.makedirs(out_dir, exist_ok=True)
        mw = decode(self.dec.shape, self.dec.tex, lat)
        files = {"glb": export_glb(mw, os.path.join(out_dir, f"{name}.glb")),
                 "obj": export_shape_obj(mw, os.path.join(out_dir, "shape.obj"))}
        v = mw.vertices.detach().cpu().numpy()[:, [0, 2, 1]].copy()   # the suite's shape.glb convention: [x, z, -y]
        v[:, 2] *= -1
        trimesh.Trimesh(vertices=v, faces=mw.faces.detach().cpu().numpy(), process=False).export(
            os.path.join(out_dir, "shape.glb"))
        files["shape_glb"] = os.path.join(out_dir, "shape.glb")
        lat.info.update(n_vertices=int(mw.vertices.shape[0]), n_faces=int(mw.faces.shape[0]))
        return files
