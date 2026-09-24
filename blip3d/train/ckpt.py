"""Checkpoint format and the key map from v12 (BLIP3o-era) checkpoints. Old files are read as-is, never converted.

v12 prefixes (docs/scan/02 §6, 03 §5): towers `ss_flow.*`, `shape_slat_512.*`, `tex_slat_512.*`; connectors
`diffusion_connector.*` (S1/S2: the tower's own; S3: the tex lane), `geo_connector.*`, `ss_connector.*` (S3);
`dino_view_embed` (model-level view codes); ROAD `repa_projector.*`, `_road_step`; unified model `unified_geotex.*`.
EMA files hold only the trained keys and are overlaid on model.safetensors.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

CONNECTOR_KEY_MAP = {"cond_seg_embed": "seg_code", "cond_patch_pos": "patch_code"}


def load_state(ckpt_dir: str, use_ema: bool = True, require_ema: bool = True) -> Dict[str, torch.Tensor]:
    """model.safetensors with ema.safetensors overlaid (or an already-overlaid `*_ema` dir)."""
    from safetensors.torch import load_file
    sd = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    ema_path = os.path.join(ckpt_dir, "ema.safetensors")
    if use_ema:
        if os.path.isfile(ema_path):
            ema = load_file(ema_path)
            missing = [k for k in ema if k not in sd]
            if missing:
                raise KeyError(f"{ckpt_dir}: {len(missing)} EMA keys not in the model, e.g. {missing[:3]}")
            for k, v in ema.items():
                sd[k] = v.to(sd[k].dtype)
        elif require_ema and not ckpt_dir.rstrip("/").endswith("_ema"):
            raise FileNotFoundError(f"{ckpt_dir}: no ema.safetensors (pass require_ema=False to use raw weights)")
    return sd


def connector_state(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """Old `<prefix>.*` connector keys -> blip3d.cond.connector.Connector state dict."""
    p = prefix.rstrip(".") + "."
    out = {}
    for k, v in sd.items():
        if k.startswith(p):
            name = k[len(p):]
            out[CONNECTOR_KEY_MAP.get(name, name)] = v
    if not out:
        raise KeyError(f"no '{p}*' keys in the state dict")
    return out


def view_code_table(sd: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    return sd.get("dino_view_embed")
