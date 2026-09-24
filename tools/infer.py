#!/usr/bin/env python
"""Generate one asset from an image (or 2-4 images) or a text prompt.

    python tools/infer.py --text "a red wooden chair" --out out/chair
    python tools/infer.py --image a.png [b.png ...] --out out/x --unified keep/v12/s3_unify_4n/checkpoint-17000_ema
Default model: the three S2 finals (separate towers). Images are RGBA renders or photos with alpha (the DINO
framing crops to alpha; Qwen sees the full frame, as in training).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blip3d.utils import backend  # noqa: E402

backend.setup("eval")

from blip3d.infer.pipeline import Pipeline  # noqa: E402

S2 = ["keep/v12/s2_ss/checkpoint-106000_ema", "keep/v12/s2_shape/checkpoint-106000_ema",
      "keep/v12/s2_tex/checkpoint-106000_ema"]


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--image", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--separate", nargs=3, default=S2, metavar=("SS", "SHAPE", "TEX"))
    ap.add_argument("--unified")
    ap.add_argument("--mode", choices=["refine", "joint", "interleave"], default="refine")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pipe = Pipeline.unified(a.unified, mode=a.mode) if a.unified else Pipeline.separate(*a.separate)
    if a.text:
        prep = pipe.prep_text(a.text)
    elif len(a.image) == 1:
        prep = pipe.prep_render(a.image[0])
    else:
        prep = pipe.prep_renders(a.image)
    lat = pipe.generate(prep, seed=a.seed)
    if lat.info.get("empty"):
        raise SystemExit("empty sparse structure (no voxels): try another seed")
    print(pipe.export(lat, a.out))


if __name__ == "__main__":
    main()
