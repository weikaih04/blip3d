"""Kernel / attention backend selection. Call :func:`setup` once, BEFORE anything imports ``trellis2``.

TRELLIS.2 reads its backends from the environment at import time (``trellis2/modules/attention/config.py``,
``trellis2/modules/sparse/config.py``), so they cannot be changed afterwards. v12 used:

* training   dense ``ATTN_BACKEND=flash_attn_3``; sparse attention follows it (FA3); fused modulate kernels on.
* evaluation ``ATTN_BACKEND=flash_attn`` (FA2) for dense and sparse attention — every v12 benchmark and demo
  entry point set this before importing trellis2 (ISSUES.md §8), so goldens are recorded with it.

Sparse convolution is ``flex_gemm`` in both.
"""
from __future__ import annotations

import os
import sys
import warnings
from typing import Literal, Optional

from .paths import REPO_ROOT

TRELLIS2_ROOT = REPO_ROOT / "third_party" / "TRELLIS.2"

_PURPOSE = {
    "train": {"ATTN_BACKEND": "flash_attn_3", "FUSED_MODULATE": "1", "SPARSE_CONV_BACKEND": "flex_gemm"},
    "eval": {"ATTN_BACKEND": "flash_attn", "FUSED_MODULATE": "1", "SPARSE_CONV_BACKEND": "flex_gemm"},
}
_done: Optional[str] = None


def setup(purpose: Literal["train", "eval"], attn: Optional[str] = None) -> None:
    """Put TRELLIS.2 on ``sys.path`` and fix the backends. Idempotent for the same arguments."""
    global _done
    key = f"{purpose}:{attn}"
    if _done is not None:
        if _done != key:
            raise RuntimeError(f"backend already set up as {_done!r}; cannot switch to {key!r} in the same process")
        return
    if "trellis2" in sys.modules:
        raise RuntimeError("trellis2 was imported before blip3d.utils.backend.setup(); its backends are already fixed")
    env = dict(_PURPOSE[purpose])
    if attn is not None:
        env["ATTN_BACKEND"] = attn
    for k, v in env.items():
        os.environ[k] = v
    if str(TRELLIS2_ROOT) not in sys.path:
        sys.path.insert(0, str(TRELLIS2_ROOT))
    # nvdiffrast JIT-compiles a CUDA extension on first import; point it at the conda env's headers.
    inc = os.path.join(sys.prefix, "targets", "x86_64-linux", "include")
    if os.path.isdir(inc):
        os.environ.setdefault("CUDA_HOME", sys.prefix)
        os.environ["CPATH"] = inc + ":" + os.environ.get("CPATH", "")
    _check_user_site()
    _done = key


def _check_user_site() -> None:
    """~/.local site-packages shadowed the env's numpy 1.26.4 with 2.2.6 for every v12 run; refuse silently mixing."""
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        import site
        user = site.getusersitepackages()
        if user in sys.path and os.path.isdir(os.path.join(user, "numpy")):
            warnings.warn(f"numpy in {user} shadows the environment's numpy; run with PYTHONNOUSERSITE=1",
                          RuntimeWarning, stacklevel=3)
