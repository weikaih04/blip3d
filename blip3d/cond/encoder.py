"""GPU half of the conditioning: frozen Qwen3-VL-2B and DINOv3 features for a batch of prep dicts.

Qwen: one forward over the right-padded batch; the conditioning is the last hidden layer (post final RMSNorm)
taken from the base model, so no LM-head logits are computed. Structural tokens are masked; text
records are compacted to the kept tokens. DINO: TRELLIS.2's DinoV3FeatureExtractor (final norm skipped,
un-affine layer_norm over 1029 tokens per 512² view) on ImageNet-normalised pixels.

Both encoders are identified by ``<hf id>@<revision>``; the identity is required and is written into checkpoints.
Records hold fp16 features and these fields:
  qwen (T, 2048) fp16, qwen_keep (T,) bool, [qwen_rc (T, 2), qwen_views (T,) long for image inputs],
  [dino (K*1029, 1024) fp16, dino_keep, dino_views (K*1029,) long], modality, views.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from . import prompts, stamp

QWEN_DEFAULT = "Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203"
DINO_DEFAULT = "camenduru/dinov3-vitl16-pretrain-lvd1689m@3c276edd87d6f6e569ff0c4400e086807d0f3881"
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


class CondEncoder:
    def __init__(self, qwen_path: str, dino_path: str, *, device: str = "cuda", dino_size: int = 512,
                 attn_implementation: str = "sdpa", full_model_hidden: bool = False):
        """``qwen_path`` / ``dino_path``: local snapshot dirs (see utils.paths.Paths.hf_snapshot).
        ``full_model_hidden=True`` reproduces the old call exactly (generation model, hidden_states[-1])."""
        from transformers import AutoModelForImageTextToText
        from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor
        from .prep import processor
        self.device, self.dino_size, self.full_model_hidden = device, dino_size, full_model_hidden
        self.qwen_path = qwen_path
        self.proc = processor(qwen_path)
        self.qwen = AutoModelForImageTextToText.from_pretrained(
            qwen_path, torch_dtype=torch.bfloat16, device_map={"": device},
            attn_implementation=attn_implementation).eval().requires_grad_(False)
        tok = self.proc.tokenizer
        self.structural = torch.tensor(sorted(prompts.structural_token_ids(tok)), dtype=torch.long, device=device)
        self.image_pad_id = prompts.image_pad_id(tok)
        self.pad_id = int(tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        self.merge = int(self.proc.image_processor.merge_size)
        self.dino = DinoV3FeatureExtractor(dino_path, image_size=dino_size)
        self.dino.model.eval().to(device).requires_grad_(False)
        self._mean = torch.tensor(DINO_MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(DINO_STD, device=device).view(1, 3, 1, 1)

    def _qwen_hidden(self, ids, am, **kw) -> torch.Tensor:
        if self.full_model_hidden:
            return self.qwen(input_ids=ids, attention_mask=am, output_hidden_states=True,
                             use_cache=False, **kw).hidden_states[-1]
        return self.qwen.model(input_ids=ids, attention_mask=am, use_cache=False, **kw).last_hidden_state

    @torch.no_grad()
    def encode(self, preps: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """One homogeneous batch (all i1, all im, or all t) -> one record per sample."""
        mods = {p["modality"] for p in preps}
        if len(mods) != 1:
            raise ValueError(f"encode() takes one modality per batch, got {sorted(mods)}")
        mod = mods.pop()
        dev, B = self.device, len(preps)
        lens = [int(p["input_ids"].shape[0]) for p in preps]
        L = max(lens)
        ids = torch.full((B, L), self.pad_id, dtype=torch.long)
        am = torch.zeros((B, L), dtype=torch.long)
        for i, p in enumerate(preps):           # RIGHT padding: left padding shifts mRoPE positions
            ids[i, :lens[i]] = p["input_ids"]
            am[i, :lens[i]] = 1
        ids, am = ids.to(dev), am.to(dev)
        kw, grids = {}, [None] * B
        if mod != "t":
            kw["pixel_values"] = torch.cat([p["pixel_values"] for p in preps]).to(dev)
            grids = [p["image_grid_thw"] for p in preps]
            kw["image_grid_thw"] = torch.cat(grids).to(dev)
        hid = self._qwen_hidden(ids, am, **kw)
        keep_all = am.bool() & ~torch.isin(ids, self.structural)

        feats = None
        if mod != "t":
            px = torch.cat([p["dino_px"] for p in preps]).to(dev)
            feats = self.dino.extract_features((px - self._mean) / self._std)
        out, o = [], 0
        for i, p in enumerate(preps):
            n = lens[i]
            h, k = hid[i, :n], keep_all[i, :n]
            if mod == "t":
                h, k = h[k], k[k]
            rec = {"qwen": h.to(torch.float16), "qwen_keep": k, "modality": mod}
            if mod != "t":
                g = grids[i].to(dev)
                rec["qwen_rc"] = stamp.image_token_rc(ids[i, :n], g, self.image_pad_id, self.merge)
                if mod == "im":
                    rec["qwen_views"] = stamp.image_token_views(ids[i, :n], g, self.image_pad_id, self.merge)
                kv = int(p["dino_px"].shape[0])
                f = feats[o:o + kv]
                o += kv
                nd = f.shape[1]
                rec["dino"] = f.reshape(-1, f.shape[-1]).to(torch.float16)
                rec["dino_keep"] = torch.ones(kv * nd, dtype=torch.bool, device=dev)
                rec["dino_views"] = torch.arange(kv, dtype=torch.long, device=dev).repeat_interleave(nd)
            if "views" in p:
                rec["views"] = p["views"]
            out.append(rec)
        return out
