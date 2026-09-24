"""Slurm-hold helpers shared by launch / stop / heartbeat / failover. Holds are long-lived one-node allocations; work
enters them as ``srun --jobid=<hold> --overlap`` steps, detached in tmux on the node (a step dies after ~70 min).

Rules learned the hard way (kept here so no script re-learns them):
* never put ``pkill -f <pattern>`` on an srun command line — the pattern matches the step's own shell; write a file;
* never edit a running bash script in place — bash reads it lazily; write a new file and move it;
* ``scancel`` only ever a step (``<job>.<step>``), never the hold itself (node allocation is the user's call).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from typing import List, Optional

from blip3d.utils.paths import REPO_ROOT, get_paths

PY = os.environ.get("BLIP3D_PYTHON", "/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis/bin/python")
SCRIPTS = os.path.join(get_paths().runs, "cache_logs", "blip3d_scripts")


def node_of(job: str) -> str:
    out = subprocess.run(["squeue", "-j", str(job), "-h", "-o", "%N"], capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError(f"hold {job} is not running")
    return out


def running_holds(min_left_h: float = 1.0) -> List[str]:
    """Running jobs of this user with at least ``min_left_h`` hours left."""
    out = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-t", "R", "-h", "-o", "%i %L"],
                         capture_output=True, text=True).stdout.split("\n")
    keep = []
    for line in out:
        if not line.strip():
            continue
        j, left = line.split()
        if "-" in left:
            keep.append(j)
            continue
        parts = [int(x) for x in left.split(":")]
        secs = parts[-1] + 60 * (parts[-2] if len(parts) > 1 else 0) + 3600 * (parts[-3] if len(parts) > 2 else 0)
        if secs >= min_left_h * 3600:
            keep.append(j)
    return sorted(keep, key=int)


def write_script(name: str, body: str) -> str:
    """Scripts that srun executes live on /fsx (visible on every node), written atomically (new file + rename)."""
    os.makedirs(SCRIPTS, exist_ok=True)
    path = os.path.join(SCRIPTS, name)
    fd, tmp = tempfile.mkstemp(dir=SCRIPTS, prefix=".tmp_")
    with os.fdopen(fd, "w") as f:
        f.write("#!/bin/bash\n" + body)
    os.chmod(tmp, 0o755)
    os.replace(tmp, path)
    return path


def srun(job: str, script: str, *args: str, timeout: Optional[int] = 120) -> str:
    cmd = ["srun", f"--jobid={job}", "--overlap", "--ntasks=1", "--mem=0", "bash", script, *args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def train_env() -> str:
    p = get_paths()
    return " ".join(f"{k}={shlex.quote(v)}" for k, v in {
        "PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HF_HOME": p.hf_cache, "HF_HUB_OFFLINE": "1",
        "TORCHINDUCTOR_CACHE_DIR": "/dev/shm/blip3d_ind", "TRITON_CACHE_DIR": "/dev/shm/blip3d_tri",
        "TMPDIR": "/dev/shm/blip3d_tmp", "WANDB_DIR": os.path.join(p.runs, "wandb")}.items())
