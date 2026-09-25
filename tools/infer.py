#!/usr/bin/env python
"""Generate one asset from an image (or 2-4 images) or a text prompt.

    python tools/infer.py --text "a red wooden chair" --out out/chair
    python tools/infer.py --image a.png [b.png ...] --out out/x --unified <unified ckpt>
Pass either --separate (three tower checkpoints) or --unified. Images are RGBA renders or photos with alpha (the DINO
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

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text")
    src.add_argument("--image", nargs="+")
    ap.add_argument("--out", required=True)
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--separate", nargs=3, metavar=("SS", "SHAPE", "TEX"))
    m.add_argument("--unified")
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
