#!/usr/bin/env python
"""Convert a v12 tower checkpoint (weights + DeepSpeed ZeRO-1 optimizer + trainer state) into a BLIP3D checkpoint that
tools/train.py can resume: parameter names mapped (train/ckpt.py from_v12), dead decoders dropped, steps renumbered
to GLOBAL steps (v12 restarts counted locally), EMA update count recorded.

    python tools/import_v12.py --src keep/v12/s2_ss/checkpoint-100000 --kind ss --global-step 100000 --ema-n 25000 \
        --dst blip3d-v12-cont-ss/checkpoint-100000

Paths are absolute or relative to paths.runs. Optimizer shards are rewritten (names inside ``param_slice_mappings``),
so universal-checkpoint conversion later sees BLIP3D names. The parameter ORDER is unchanged (flow -> connector ->
ROAD), which is what ZeRO-1's flat partitions rely on.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402

from blip3d.train.ckpt import from_v12  # noqa: E402
from blip3d.train.ema import STATE_FILE  # noqa: E402
from blip3d.utils.paths import get_paths  # noqa: E402


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(get_paths().runs, p)


def _rename_names(names, kind):
    return [next(iter(from_v12({n: torch.empty(0)}, kind))) for n in names]


def _rename_od(od, kind):
    out = OrderedDict()
    for k, v in od.items():
        m = from_v12({k: torch.empty(0)}, kind)
        if m:
            out[next(iter(m))] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--kind", required=True, choices=["ss", "shape", "tex"])
    ap.add_argument("--global-step", type=int, default=None, help="global step of this checkpoint (default: its own)")
    ap.add_argument("--ema-n", type=int, required=True, help="EMA update count behind ema.safetensors")
    ap.add_argument("--weights-only", action="store_true", help="skip the optimizer shards")
    a = ap.parse_args()
    src, dst = _abs(a.src), _abs(a.dst)
    ts = json.load(open(os.path.join(src, "trainer_state.json")))
    local = int(ts["global_step"])
    G = a.global_step if a.global_step is not None else local
    os.makedirs(dst, exist_ok=True)
    print(f"[import] {src} (local step {local}) -> {dst} (global step {G})", flush=True)

    sd = from_v12(load_file(os.path.join(src, "model.safetensors")), a.kind)
    save_file(sd, os.path.join(dst, "model.safetensors"))
    ema = from_v12(load_file(os.path.join(src, "ema.safetensors")), a.kind)
    save_file(ema, os.path.join(dst, "ema.safetensors"))
    json.dump({"n": int(a.ema_n), "decay": 0.9999, "warmup": True, "global_step": G, "imported_from": src},
              open(os.path.join(dst, STATE_FILE), "w"))
    print(f"[import] model {len(sd)} tensors, ema {len(ema)} tensors (n={a.ema_n})", flush=True)

    ts["global_step"] = G
    ts["imported_from"] = {"src": src, "local_step": local}
    json.dump(ts, open(os.path.join(dst, "trainer_state.json"), "w"), indent=2)
    for f in glob.glob(os.path.join(src, "rng_state_*.pth")):
        shutil.copy2(f, os.path.join(dst, os.path.basename(f)))
    if a.weights_only:
        return

    tag_src = open(os.path.join(src, "latest")).read().strip()
    d_src, d_dst = os.path.join(src, tag_src), os.path.join(dst, f"global_step{G}")
    os.makedirs(d_dst, exist_ok=True)
    ms = torch.load(os.path.join(d_src, "mp_rank_00_model_states.pt"), map_location="cpu", weights_only=False)
    ms["module"] = from_v12(ms["module"], a.kind)
    ms["buffer_names"] = [n for n in _rename_names(ms["buffer_names"], a.kind)]
    ms["param_shapes"] = [_rename_od(od, a.kind) for od in ms["param_shapes"]]
    sch = ms.get("lr_scheduler")
    if isinstance(sch, dict):
        sch["last_epoch"], sch["_step_count"] = G, G + 1
    ms["global_steps"] = G
    ms["global_samples"] = int(ms.get("global_samples", 0)) // max(local, 1) * G
    torch.save(ms, os.path.join(d_dst, "mp_rank_00_model_states.pt"))
    shards = sorted(glob.glob(os.path.join(d_src, "*_optim_states.pt")))
    for f in shards:
        o = torch.load(f, map_location="cpu", weights_only=False)
        osd = o["optimizer_state_dict"]
        if osd.get("param_slice_mappings"):
            osd["param_slice_mappings"] = [_rename_od(od, a.kind) for od in osd["param_slice_mappings"]]
        torch.save(o, os.path.join(d_dst, os.path.basename(f)))
    open(os.path.join(dst, "latest"), "w").write(f"global_step{G}")
    print(f"[import] DeepSpeed state: model_states + {len(shards)} optimizer shards renamed; latest=global_step{G}",
          flush=True)


if __name__ == "__main__":
    main()
