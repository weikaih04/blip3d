"""SS-tower readout (port of lc/ss_road_eval.py): 64³ occupancy IoU against the GT structure for the image arm and the
text arm, plus text self-consistency (the same caption, seeds 0 and 1). Released SS sampler, EMA weights.

Difference from v12, on purpose: the image arm uses the TRAINING framing (``prep_render``: Qwen sees the raw render,
DINO the alpha crop) and the view ``5 + (sha[:8] + 3) % 7``; v12's readout went through eval_fusion_v22, which
composited the render on white for Qwen. Text-arm numbers are comparable with v12's; image-arm numbers are not.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from ..cond import prep as P
from ..cond.assemble import cond_uncond
from ..cond.prompts import template_index
from ..infer.presets import IMAGE
from ..infer.stages import sample_ss_released


def readout_view(sha: str) -> int:
    return 5 + (int(sha[:8], 16) + 3) % 7


def _iou(a, b) -> float:
    return (a & b).sum().item() / max(1, (a | b).sum().item())


@torch.no_grad()
def ss_readout(tower, ss_decoder, encoder, records: List[Dict], qwen_path: str, log=print) -> Dict[str, List[float]]:
    occ = lambda z: (ss_decoder(z) > 0)[0, 0]
    out = {"img": [], "txt": [], "sc": [], "sha": []}
    for k, r in enumerate(records):
        sha = r["sha256"]
        gt = occ(torch.from_numpy(np.load(r["ss_latent_64"])["z"]).float().cuda()[None])
        v = readout_view(sha)
        path = os.path.join(r["renders_dir"], f"{v:03d}.webp")
        rec_i = encoder.encode([P.prep_render(Image.open(path), qwen_path=qwen_path, view=v)])[0]
        rec_t = encoder.encode([P.prep_text(r["captions"][0], qwen_path=qwen_path, template=template_index(sha, 0))])[0]
        ci, ui = cond_uncond(tower.connector, tower.views, rec_i)
        ct, ut = cond_uncond(tower.connector, tower.views, rec_t)
        oi = occ(sample_ss_released(tower.flow, ci, ui, IMAGE.ss, 0))
        ot0 = occ(sample_ss_released(tower.flow, ct, ut, IMAGE.ss, 0))
        ot1 = occ(sample_ss_released(tower.flow, ct, ut, IMAGE.ss, 1))
        out["img"].append(_iou(oi, gt)); out["txt"].append(_iou(ot0, gt)); out["sc"].append(_iou(ot0, ot1))
        out["sha"].append(sha)
        log(f"  [{k + 1}/{len(records)}] {sha[:10]}  img {out['img'][-1]:.3f}  txt {out['txt'][-1]:.3f}  "
            f"self {out['sc'][-1]:.3f}")
    return out


def heldout_records(manifest: str, n: int) -> List[Dict]:
    recs = []
    for line in open(manifest):
        r = json.loads(line)
        if r.get("captions") and os.path.exists(r.get("ss_latent_64", "")) and os.path.isdir(r.get("renders_dir", "")):
            recs.append(r)
        if len(recs) >= n:
            break
    return recs
