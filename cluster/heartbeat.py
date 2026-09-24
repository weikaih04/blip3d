#!/usr/bin/env python
"""GPU reservation heartbeat for the holds a run uses. The benchmark supervisor on each hold yields its GPUs only while
``now - mtime(<dir>/workload_<job>__<node>.reserved) < 1200 s`` AND the file reads ``<ISO-8601 Z> pid=<n>``; a
one-off touch expires after 20 min and the benchmark walks back onto the GPUs (it OOMed a training run once).

    python cluster/heartbeat.py --holds 2185 2177 --hours 240      # run it detached (tmux) on any hold
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.common import node_of  # noqa: E402

RESERVATION_DIR = os.environ.get("BLIP3D_RESERVATION_DIR",
                                  "/fsx/home/weikai.huang/adacodec_vlm/logs/molmo2_stage2_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holds", nargs="+", required=True)
    ap.add_argument("--hours", type=float, default=240)
    a = ap.parse_args()
    keys = [f"{j}__{node_of(j)}" for j in a.holds]
    end = time.time() + a.hours * 3600
    while time.time() < end:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for k in keys:
            with open(os.path.join(RESERVATION_DIR, f"workload_{k}.reserved"), "w") as f:
                f.write(f"{stamp} pid={os.getpid()}\n")
        time.sleep(60)
    print(f"[heartbeat] expired after {a.hours} h", flush=True)


if __name__ == "__main__":
    main()
