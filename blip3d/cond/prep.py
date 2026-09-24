"""CPU half of the conditioning: image framing and tokenization. Safe to run in dataloader workers.

Qwen and DINO see different framings of the same render (ISSUES C-13, C-14):
* Qwen: the full, uncropped render converted to RGB (alpha dropped, not composited);
  multi-image views are downscaled to ``im_tok_per_view`` vision tokens each.
* DINO: the alpha-cropped render composited on black, resized to ``dino_size`` (LANCZOS), in [0, 1].

A prep dict carries: ``input_ids`` (T,), ``pixel_values`` (bf16) + ``image_grid_thw`` for image inputs,
``dino_px`` (3,S,S) or (K,3,S,S) fp32, ``modality`` in {"i1", "im", "t"}, and ``views`` for the record.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from . import prompts

MIN_IMG_PX = 64          # smallest side accepted for the DINO crop before the final resize
DINO_CAP_PX = 1024       # RGBA inputs are downscaled to this long side before cropping


@lru_cache(maxsize=4)
def processor(path: str):
    """Qwen3-VL processor for a local snapshot dir (one per process; not fork-safe to share)."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(path)


def px_per_token(proc) -> int:
    ip = proc.image_processor
    return (int(ip.patch_size) * int(ip.merge_size)) ** 2


def cap_image(img: Image.Image, max_px: int) -> Image.Image:
    """Downscale-only, aspect-preserving, so that w*h <= max_px."""
    w, h = img.size
    if w * h <= max_px:
        return img
    s = (max_px / float(w * h)) ** 0.5
    return img.resize((max(32, round(w * s)), max(32, round(h * s))), Image.LANCZOS)


def alpha_crop(img: Image.Image) -> Image.Image:
    """Square crop around the solid alpha region (alpha > 0.8*255; fallback alpha > 0), TRELLIS.2's rule."""
    a = np.array(img.getchannel("A"))
    ys, xs = np.where(a > 0.8 * 255)
    if ys.size == 0:
        ys, xs = np.where(a > 0)
    if ys.size == 0:
        return img
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0, 2) // 2
    cropped = img.crop((cx - half, cy - half, cx + half, cy + half))
    return cropped if min(cropped.size) > 0 else img


def composite_black(img: Image.Image) -> Image.Image:
    a = np.array(img).astype(np.float32) / 255.0
    return Image.fromarray((a[:, :, :3] * a[:, :, 3:4] * 255).astype(np.uint8))


def dino_view(img: Image.Image) -> Image.Image:
    """The DINO framing of one render (before the final resize)."""
    if img.mode == "RGBA":
        m = max(img.size)
        if m > DINO_CAP_PX:
            sc = DINO_CAP_PX / m
            img = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
        img = composite_black(alpha_crop(img))
    else:
        img = img.convert("RGB")
    w, h = img.size
    if w < MIN_IMG_PX or h < MIN_IMG_PX:
        s = MIN_IMG_PX / max(1, min(w, h))
        img = img.resize((max(MIN_IMG_PX, int(round(w * s))), max(MIN_IMG_PX, int(round(h * s)))), Image.LANCZOS)
    return img


def dino_pixels(img: Image.Image, size: int) -> torch.Tensor:
    x = img.resize((size, size), Image.LANCZOS).convert("RGB")
    return torch.from_numpy(np.array(x).astype(np.float32) / 255).permute(2, 0, 1)


def prep_image(qwen_img: Image.Image, dino_img: Image.Image, *, qwen_path: str, dino_size: int = 512,
               view: int = 0) -> Dict[str, torch.Tensor]:
    """Single image (i1). ``qwen_img`` is the full frame, ``dino_img`` the framed object (see dino_view)."""
    proc = processor(qwen_path)
    text = prompts.chat_text(proc, prompts.image_prompt(1))
    qi = proc(text=[text], images=[qwen_img.convert("RGB")], return_tensors="pt")
    return {"input_ids": qi["input_ids"][0], "pixel_values": qi["pixel_values"].to(torch.bfloat16),
            "image_grid_thw": qi["image_grid_thw"], "dino_px": dino_pixels(dino_img, dino_size)[None],
            "modality": "i1", "views": torch.tensor([int(view)])}


def prep_render(render: Image.Image, *, qwen_path: str, dino_size: int = 512, view: int = 0):
    """Single image from a dataset render: Qwen gets the raw frame, DINO the alpha crop."""
    return prep_image(render, dino_view(render), qwen_path=qwen_path, dino_size=dino_size, view=view)


def prep_images(qwen_imgs: Sequence[Image.Image], dino_imgs: Sequence[Image.Image], *, qwen_path: str,
                views: Optional[Sequence[int]] = None, tok_per_view: int = 256,
                dino_size: int = 512) -> Dict[str, torch.Tensor]:
    """Multi-image (im), 2-4 views. Views are ordinals within the sample, in the order given."""
    n = len(qwen_imgs)
    if not 2 <= n <= 4 or len(dino_imgs) != n:
        raise ValueError(f"multi-image conditioning takes 2-4 views (got {n}); use prep_image for one")
    proc = processor(qwen_path)
    max_px = tok_per_view * px_per_token(proc)
    text = prompts.chat_text(proc, prompts.image_prompt(n))
    qi = proc(text=[text], images=[cap_image(q.convert("RGB"), max_px) for q in qwen_imgs], return_tensors="pt")
    return {"input_ids": qi["input_ids"][0], "pixel_values": qi["pixel_values"].to(torch.bfloat16),
            "image_grid_thw": qi["image_grid_thw"],
            "dino_px": torch.stack([dino_pixels(d, dino_size) for d in dino_imgs]),
            "modality": "im", "views": torch.tensor(list(views) if views is not None else list(range(n)))}


def prep_renders(renders: Sequence[Image.Image], *, qwen_path: str, views: Optional[Sequence[int]] = None,
                 tok_per_view: int = 256, dino_size: int = 512):
    return prep_images(list(renders), [dino_view(r) for r in renders], qwen_path=qwen_path, views=views,
                       tok_per_view=tok_per_view, dino_size=dino_size)


def prep_text(caption: str, *, qwen_path: str, template: int) -> Dict[str, torch.Tensor]:
    """Text (t). No DINO segment at all. ``template`` is explicit: training derives it with
    ``prompts.template_index(sha, caption_index)``, interactive use passes 0."""
    proc = processor(qwen_path)
    ti = proc(text=[prompts.chat_text(proc, prompts.text_prompt(caption, template))], return_tensors="pt")
    return {"input_ids": ti["input_ids"][0], "modality": "t"}
