#!/usr/bin/env python
"""Launch a recipe on N holds: one detached tmux session per node running torchrun (8 GPUs each).

    python cluster/launch.py recipes/s2_ss.yaml --holds 2185 2177 2176 2149 [--tag s2ss] [--set k=v ...]

Logs: <runs>/cache_logs/<tag>_r<rank>.log. The number of holds must match ``compute.nodes`` unless overridden with
``--set compute.nodes=N`` (the effective batch stays ``compute.eff_bs``; GA adapts; a world-size change on resume needs
``--set compute.universal=true`` and a universal checkpoint, see cluster/failover.py).
"""
from __future__ import annotations

import argparse
import os
import random
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.common import PY, REPO_ROOT, node_of, srun, train_env, write_script  # noqa: E402
from blip3d.train.recipe import load_recipe  # noqa: E402
from blip3d.utils.paths import get_paths  # noqa: E402


def launch(recipe: str, holds, tag: str, overrides, port: int = 0) -> None:
    r = load_recipe(recipe, overrides)
    if len(holds) != r.compute.nodes:
        raise SystemExit(f"{len(holds)} holds for a {r.compute.nodes}-node recipe (use --set compute.nodes=...)")
    r.compute.grad_accum                                  # validates eff_bs divisibility before anything starts
    nodes = [node_of(j) for j in holds]
    port = port or 29600 + random.randint(0, 300)
    logs = os.path.join(get_paths().runs, "cache_logs")
    os.makedirs(logs, exist_ok=True)
    sets = " ".join(shlex.quote(o) for o in overrides)
    for rank, (job, node) in enumerate(zip(holds, nodes)):
        sess = f"{tag}_r{rank}"
        log = os.path.join(logs, f"{sess}.log")
        inner = (f"cd {REPO_ROOT} && mkdir -p /dev/shm/blip3d_tmp && env {train_env()} {PY} -m torch.distributed.run "
                 f"--nnodes {len(holds)} --node_rank {rank} --nproc_per_node 8 --master_addr {nodes[0]} "
                 f"--master_port {port} tools/train.py {shlex.quote(recipe)}" + (f" --set {sets}" if sets else "")
                 + f" 2>&1 | tee -a {log}")
        body = (f"env -u TMUX tmux has-session -t {sess} 2>/dev/null && {{ echo '{sess} already exists on '$(hostname); exit 0; }}\n"
                f"env -u TMUX tmux new-session -d -s {sess} {shlex.quote(inner)}\n"
                f"echo 'launched {sess} on '$(hostname)\n")
        print(srun(job, write_script(f"launch_{sess}.sh", body)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--holds", nargs="+", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    tag = a.tag or os.path.splitext(os.path.basename(a.recipe))[0]
    launch(a.recipe, a.holds, tag, a.set, a.port)


if __name__ == "__main__":
    main()
