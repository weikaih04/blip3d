#!/usr/bin/env python
"""WilD3D runner: one case dir per manifest row with full.glb + shape.obj + shape.glb + status.json (an empty
structure writes error.json and no asset: a declared failure, never a stub). Resumable (finished cases are skipped).

    python tools/bench.py --manifest <eval47.jsonl> --output <dir> --mode image \
        --separate keep/v12/s2_ss/checkpoint-106000_ema keep/v12/s2_shape/checkpoint-106000_ema keep/v12/s2_tex/checkpoint-106000_ema
    python tools/bench.py ... --mode text --unified keep/v12/s3_unify_4n/checkpoint-17000_ema --sampler refine
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blip3d.utils import backend  # noqa: E402

backend.setup("eval")

import torch  # noqa: E402

from blip3d.infer.pipeline import Pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["image", "text"], required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--separate", nargs=3, metavar=("SS", "SHAPE", "TEX"))
    g.add_argument("--unified")
    ap.add_argument("--sampler", choices=["refine", "joint", "interleave"], default="refine")
    ap.add_argument("--k0", type=int, default=7)
    ap.add_argument("--fixed", action="store_true", help="unified: the U-02/04/06 fixes instead of v12 numerics")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--count", type=int, default=10 ** 9)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--model-name", default="blip3d")
    a = ap.parse_args()

    if a.separate:
        pipe = Pipeline.separate(*a.separate)
        desc = {"kind": "separate", "ckpts": a.separate}
    else:
        from blip3d.infer.unified import UnifiedOptions
        pipe = Pipeline.unified(a.unified, mode=a.sampler, options=UnifiedOptions.fixed() if a.fixed else UnifiedOptions())
        desc = {"kind": "unified", "ckpt": a.unified, "sampler": a.sampler, "k0": a.k0, "fixed": a.fixed}
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.manifest)][a.start_index:a.start_index + a.count]
    for rec in rows:
        case = out / rec["case_id"]
        case.mkdir(exist_ok=True)
        if (case / "status.json").is_file() or (case / "error.json").is_file():
            continue
        t0 = time.time()
        if a.mode == "image":
            view = int(rec["input_view_index"])
            src = os.path.join(rec["renders_dir"], f"{view:03d}.webp")
            prep, prompt = pipe.prep_render(src, view=view), None
        else:
            prompt = rec["prompt"]
            prep = pipe.prep_text(prompt, sha=rec.get("sha256"), caption_index=0)
        lat = pipe.generate(prep, seed=a.seed, k0=a.k0)
        status = {"model": a.model_name, "case_id": rec["case_id"], "sha256": rec.get("sha256"), "modality": a.mode,
                  "prompt": prompt, "seed": a.seed, "input_view_index": rec.get("input_view_index"), **desc,
                  "info": {k: v for k, v in lat.info.items() if k != "preset"}, "preset": lat.info.get("preset")}
        if lat.info.get("empty") or lat.coords.shape[0] == 0:
            status.update(error="empty sparse structure", seconds=time.time() - t0)
            json.dump(status, open(case / "error.json", "w"), indent=2, default=str)
            print(f"[bench] {rec['case_id']} EMPTY", flush=True)
            continue
        pipe.export(lat, str(case))
        status.update(seconds=time.time() - t0, runtime_seconds=time.time() - t0, n_vox=int(lat.coords.shape[0]),
                      n_vertices=lat.info.get("n_vertices"), n_faces=lat.info.get("n_faces"))
        json.dump(status, open(case / "status.json", "w"), indent=2, default=str)
        print(f"[bench] {rec['case_id']} {status['seconds']:.0f}s vox={status['n_vox']}", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
