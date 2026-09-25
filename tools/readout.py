#!/usr/bin/env python
"""SS readout of one checkpoint on the held-out val200_capT set (see blip3d/eval/readout.py).

    python tools/readout.py <ss ckpt> --n 24 [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blip3d.utils import backend  # noqa: E402

backend.setup("eval")

import numpy as np  # noqa: E402

from blip3d.cond.encoder import DINO_DEFAULT, QWEN_DEFAULT, CondEncoder  # noqa: E402
from blip3d.eval.readout import heldout_records, ss_readout  # noqa: E402
from blip3d.models.decoders import load_decoders
from blip3d.models.flows import load_tower  # noqa: E402
from blip3d.utils.paths import get_paths, resolve_hf_snapshot  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--manifest", default=None, help="default: <manifests>/val200_capT.jsonl")
    ap.add_argument("--raw", action="store_true", help="raw weights instead of the EMA")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    p = get_paths()
    ck = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(p.runs, a.ckpt)
    qwen = resolve_hf_snapshot(QWEN_DEFAULT, p.hf_cache)
    tower = load_tower("ss", ck, use_ema=not a.raw)
    dec = load_decoders(p.trellis2_ckpt, p.ss_decoder)
    enc = CondEncoder(qwen, resolve_hf_snapshot(DINO_DEFAULT, p.hf_cache))
    recs = heldout_records(a.manifest or os.path.join(p.manifests, "val200_capT.jsonl"), a.n)
    r = ss_readout(tower, dec.ss, enc, recs, qwen, log=lambda s: print(s, flush=True))
    for k, name in (("img", "image occ IoU"), ("txt", "text occ IoU"), ("sc", "text self-consistency")):
        x = np.array(r[k])
        print(f"{name:24s} mean {x.mean():.3f}  median {np.median(x):.3f}  (n={len(x)})")
    if a.out:
        json.dump({"ckpt": ck, "ema": not a.raw, **r}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
