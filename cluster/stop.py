#!/usr/bin/env python
"""Stop a launched run on its holds: close the tmux sessions and kill the torchrun trees (from a script FILE —
a pkill pattern on the srun command line would match the step's own shell)."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cluster.common import srun, write_script  # noqa: E402


def stop(holds, tag: str) -> None:
    for rank, job in enumerate(holds):
        sess = f"{tag}_r{rank}"
        body = (f"env -u TMUX tmux kill-session -t {sess} 2>/dev/null\n"
                f"pkill -f 'tools/train.py' ; sleep 5 ; pkill -9 -f 'tools/train.py'\n"
                f"echo 'stopped {sess} on '$(hostname)\n")
        print(srun(job, write_script(f"stop_{sess}.sh", body)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--holds", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    stop(a.holds, a.tag)
