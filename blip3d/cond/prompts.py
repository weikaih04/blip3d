"""Prompts, templates and the structural-token filter for the Qwen3-VL conditioning encoder.

Everything here is a v12 contract (docs/ISSUES.md C-04, C-06, C-12): the exact strings, how a text template
is chosen, and which token ids are removed from the conditioning sequence.
"""
from __future__ import annotations

from typing import Set

IMAGE_PAD = "<|image_pad|>"
IMAGE_BLOCK = "<|vision_start|>" + IMAGE_PAD + "<|vision_end|>"
PREFIX = "[3D Gen] "
IMAGE_INSTRUCTION = "\nReconstruct this object in 3D."

# Text templates; the one used for a training caption is picked by (asset sha, caption index).
TEXT_TEMPLATES = ("Generate a 3D asset: {c}", "Create a 3D model of: {c}", "Make this in 3D: {c}")

# Structural tokens removed from the conditioning (C-04). "user"/"assistant" are removed anywhere in the
# sequence, including inside a caption — that is what v12 trained with.
_STRUCTURAL = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<think>", "</think>")
_ROLES = ("user", "assistant")


def image_prompt(n_images: int = 1) -> str:
    """Single image: one image block; multi-image: n adjacent blocks, same instruction."""
    if n_images < 1:
        raise ValueError("image_prompt needs at least one image")
    return PREFIX + IMAGE_BLOCK * n_images + IMAGE_INSTRUCTION


def template_index(sha: str, caption_index: int) -> int:
    """v12 training rule: ``(int(sha[:8], 16) + caption_index) % 3``. ``sha`` must be hex."""
    return (int(sha[:8], 16) + int(caption_index)) % len(TEXT_TEMPLATES)


def text_prompt(caption: str, template: int) -> str:
    return PREFIX + TEXT_TEMPLATES[template % len(TEXT_TEMPLATES)].format(c=caption)


def chat_text(processor, prompt: str) -> str:
    """One user turn, generation prompt appended, no system message."""
    return processor.apply_chat_template([{"role": "user", "content": prompt}],
                                         tokenize=False, add_generation_prompt=True)


def structural_token_ids(tokenizer) -> Set[int]:
    ids: Set[int] = set()
    for name in _STRUCTURAL:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    for role in _ROLES:
        ids.update(tokenizer(role, add_special_tokens=False).input_ids)
    return ids


def image_pad_id(tokenizer) -> int:
    tid = tokenizer.convert_tokens_to_ids(IMAGE_PAD)
    if not isinstance(tid, int) or tid < 0:
        raise RuntimeError(f"tokenizer has no {IMAGE_PAD} token")
    return tid
