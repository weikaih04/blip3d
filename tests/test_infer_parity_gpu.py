#!/usr/bin/env python
"""GPU parity of blip3d.infer.unified.UnifiedSampler against the v12 GeoTexSampler (old code, read-only), on the
S3 17k EMA model: same model object, same conditioning, same seed -> identical coords and latents, for refine / joint /
interleave (the v12 benchmark call: whole sampler under bf16 autocast, SS K/V cache on).

    CUDA_VISIBLE_DEVICES=k PYTHONNOUSERSITE=1 python tests/test_infer_parity_gpu.py [--mem-gb 26]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
sys.path.insert(0, REPO)
from blip3d.utils import backend  # noqa: E402

backend.setup("eval")
sys.path.append(OLD)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from blip3d.cond import prep as P  # noqa: E402
from blip3d.cond.encoder import DINO_DEFAULT, QWEN_DEFAULT, CondEncoder  # noqa: E402
from blip3d.cond.prompts import template_index  # noqa: E402
from blip3d.infer.presets import preset_for  # noqa: E402
from blip3d.infer.unified import UnifiedOptions, UnifiedSampler  # noqa: E402
from blip3d.models.towers import load_decoders  # noqa: E402
from blip3d.models.unified.loading import load_unified  # noqa: E402
from blip3d.utils.paths import get_paths, resolve_hf_snapshot  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-gb", type=float, default=26)
    ap.add_argument("--ckpt", default="keep/v12/s3_unify_4n/checkpoint-17000_ema")
    a = ap.parse_args()
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb / tot), 0)
    p = get_paths()
    qwen = resolve_hf_snapshot(QWEN_DEFAULT, p.hf_cache)
    enc = CondEncoder(qwen, resolve_hf_snapshot(DINO_DEFAULT, p.hf_cache))
    rows = [json.loads(l) for l in open(os.path.join(p.manifests, "val200_capT.jsonl"))][:2]
    recs = {}
    r = rows[0]
    v = 5 + (int(r["sha256"][:8], 16) + 3) % 7
    recs["i1"] = enc.encode([P.prep_render(Image.open(os.path.join(r["renders_dir"], f"{v:03d}.webp")),
                                           qwen_path=qwen, view=v)])[0]
    recs["t"] = enc.encode([P.prep_text(rows[1]["captions"][0], qwen_path=qwen,
                                        template=template_index(rows[1]["sha256"], 0))])[0]
    del enc
    torch.cuda.empty_cache()

    bundle = load_unified(os.path.join(p.runs, a.ckpt), trellis2_ckpt=p.trellis2_ckpt)
    ssdec = load_decoders(p.trellis2_ckpt, p.ss_decoder).ss
    import trellis2_blip3o.geotex_sampler as GS
    GS.JOINT_OPTS["cache_ss_kv"] = True
    old = GS.GeoTexSampler(bundle.model)
    base = {k: dict(getattr(GS, k)) for k in ("SS_PARAMS", "SHAPE_PARAMS", "TEX_PARAMS")}
    for mod in ("i1", "t"):
        pre = preset_for(mod, unified=True)
        for k, st in (("SS_PARAMS", pre.ss), ("SHAPE_PARAMS", pre.shape), ("TEX_PARAMS", pre.tex)):
            d = dict(base[k])
            d.update(steps=st.steps, guidance_strength=st.cfg, guidance_rescale=st.rescale,
                     guidance_interval=tuple(st.interval), rescale_t=st.rescale_t)
            setattr(GS, k, d)
        old.shape_p, old.tex_p = dict(GS.SHAPE_PARAMS), dict(GS.TEX_PARAMS)
        new = UnifiedSampler(bundle, ssdec, pre, UnifiedOptions())
        cs = new.conds(recs[mod])
        args = (ssdec, cs["ss"][0], cs["ss"][1], cs["geo"][0], cs["geo"][1], cs["tex"][0], cs["tex"][1])
        for mode in ("refine", "joint", "interleave"):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                if mode == "interleave":
                    xs_o, xt_o, c_o, _ = old.sample_three_tower_interleaved(*args, seed=0, alpha=32.0, k0=7, read=True,
                                                                            refine_tex=True)
                else:
                    xs_o, xt_o, c_o, _ = old.sample_three_tower(*args, seed=0, alpha=32.0,
                                                                refine_tex=(mode == "refine"))
            (xs_n, xt_n, c_n), info = new.run(recs[mod], mode=mode, seed=0, k0=7)
            fo = lambda z: (z.feats if hasattr(z, "feats") else z).float()
            same_c = torch.equal(c_o.cpu(), c_n.cpu())
            ok = same_c and torch.equal(fo(xs_o), fo(xs_n)) and torch.equal(fo(xt_o), fo(xt_n))
            d = "" if not same_c else f"shape |d| {(fo(xs_o) - fo(xs_n)).abs().max():.3g} tex |d| {(fo(xt_o) - fo(xt_n)).abs().max():.3g}"
            check(f"unified {mod} {mode}", ok, f"coords {'=' if same_c else '!='} n={c_n.shape[0]} {d}")
    print(f"\n{len(FAILS)} failures" + (f": {FAILS}" if FAILS else ""), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
