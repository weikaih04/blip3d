#!/usr/bin/env python
"""Keep a run alive across hold loss (port of lc/ss300k_failover.sh, 2026-09-24).

If the run dies: relaunch on 4 holds if available, else on 2 with the SAME effective batch (GA doubles); 3 is never
used (256 is not divisible by 24 ranks). A world-size change resumes through a DeepSpeed universal checkpoint
(converted in place, ``--inject_missing_state``), so weights, AdamW, step and LR schedule carry over; the EMA shadow
and its count come from the checkpoint (train/ema.py). When 4 holds are back and a checkpoint is < 15 min old, it
switches back to 4 nodes. Run it detached on a hold that will outlive the others:

    python cluster/failover.py recipes/cont_ss.yaml --holds 2185 2177 2176 2149 --tag cont_ss
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.common import PY, node_of, running_holds, srun, write_script  # noqa: E402
from cluster.launch import launch  # noqa: E402
from cluster.stop import stop  # noqa: E402
from blip3d.train.recipe import load_recipe  # noqa: E402


def log(msg):
    print(f"[failover {time.strftime('%m-%d_%H:%M', time.gmtime())}] {msg}", flush=True)


def ckpts(out):
    return sorted((d for d in glob.glob(os.path.join(out, "checkpoint-*")) if d.split("-")[-1].isdigit()),
                  key=lambda d: int(d.split("-")[-1]))


def alive(job, tag) -> bool:
    body = f"env -u TMUX tmux has-session -t {tag}_r0 2>/dev/null && echo yes\n"
    try:
        return "yes" in srun(job, write_script(f"alive_{tag}.sh", body), timeout=60)
    except Exception:
        return False


def to_universal(ck, job) -> bool:
    if os.path.isfile(os.path.join(ck, "latest_universal")):
        return True
    tag = open(os.path.join(ck, "latest")).read().strip()
    log(f"converting {ck} to universal")
    body = (f"cd {ck} && {PY} -m deepspeed.checkpoint.ds_to_universal --input_folder {ck}/{tag} "
            f"--output_folder {ck}/{tag}_universal --num_extract_workers 16 --num_merge_workers 8 --inject_missing_state"
            f" && echo {tag}_universal > {ck}/latest_universal\n")
    srun(job, write_script(f"universal_{os.path.basename(ck)}.sh", body), timeout=7200)
    return os.path.isfile(os.path.join(ck, "latest_universal"))


def heartbeat(holds, tag, anchor):
    keys = " ".join(holds)
    body = (f"env -u TMUX tmux kill-session -t hb_{tag} 2>/dev/null\n"
            f"env -u TMUX tmux new-session -d -s hb_{tag} 'cd {os.getcwd()} && {PY} cluster/heartbeat.py --holds {keys}'\n")
    srun(anchor, write_script(f"hb_{tag}.sh", body))
    log(f"heartbeat for {keys}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--holds", nargs="+", required=True, help="the holds the run is currently on")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--poll", type=int, default=180)
    a = ap.parse_args()
    r = load_recipe(a.recipe, a.set)
    out = r.out_dir()
    state_f = os.path.join(out, "failover_state.json")
    st = json.load(open(state_f)) if os.path.isfile(state_f) else {"holds": a.holds}
    anchor = subprocess.run(["bash", "-c", "echo ${SLURM_JOB_ID:-}"], capture_output=True, text=True).stdout.strip()
    log(f"watching {st['holds']} (anchor hold {anchor or '?'})")
    while True:
        time.sleep(a.poll)
        for c in ckpts(out):                               # record the world size each checkpoint was written under
            f = os.path.join(c, ".world_nodes")
            if not os.path.isfile(f) and os.path.isfile(os.path.join(c, "ema.safetensors")):
                open(f, "w").write(str(len(st["holds"])))
        holds = running_holds()
        cur = st["holds"]
        if not alive(cur[0], a.tag):
            log(f"run not alive on {cur}")
            stop([j for j in cur if j in holds], a.tag)
            pool = [j for j in cur if j in holds] + [j for j in holds if j not in cur]
            n = 4 if len(pool) >= 4 else 2 if len(pool) >= 2 else 0
            if not n:
                log("fewer than 2 holds; waiting")
                continue
            new = pool[:n]
            cs = ckpts(out)
            sets = list(a.set) + [f"compute.nodes={n}"]
            wn = os.path.join(cs[-1], ".world_nodes") if cs else None
            last_n = open(wn).read().strip() if wn and os.path.isfile(wn) else None
            if last_n is not None and last_n != str(n):
                if not to_universal(cs[-1], new[0]):
                    log("universal conversion failed; not launching")
                    continue
                sets.append("compute.universal=true")
            heartbeat(new, a.tag, anchor or new[0])
            launch(a.recipe, new, a.tag, sets)
            st["holds"] = new
            json.dump(st, open(state_f, "w"))
            log(f"launched {n} nodes {new}")
            continue
        if len(cur) == 2 and len(holds) >= 4:
            cs = ckpts(out)
            if cs and time.time() - os.path.getmtime(os.path.join(cs[-1], "ema.safetensors")) < 900:
                log(f"4 holds available, fresh {os.path.basename(cs[-1])}: back to 4 nodes")
                stop(cur, a.tag)
                st["holds"] = cur + [j for j in holds if j not in cur][:2]
                json.dump(st, open(state_f, "w"))
                # next poll finds it not alive and relaunches on 4 through the universal path


if __name__ == "__main__":
    main()
