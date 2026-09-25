"""Checkpoint format and the key map from v12 checkpoints. Old files are read as-is, never converted.

v12 prefixes: towers `ss_flow.*`, `shape_slat_512.*`, `tex_slat_512.*`; connectors
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


# ── v12 tower checkpoints -> Blip3DTower names ──
V12_FLOW_PREFIX = {"ss": "ss_flow.", "shape": "shape_slat_512.", "tex": "tex_slat_512."}
V12_DEAD_PREFIXES = ("trellis_decoders.",)   # stock frozen decoders saved into every v12 shape/tex checkpoint


def is_v12(sd: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith(("diffusion_connector.", "ss_flow.", "shape_slat_512.", "tex_slat_512.")) for k in sd)


def from_v12(sd: Dict[str, torch.Tensor], kind: str) -> Dict[str, torch.Tensor]:
    """Rename a v12 tower state dict (model or EMA) to Blip3DTower names. Keys of the other towers and the dead
    decoders are dropped; anything else unknown raises (so a new v12 key cannot vanish silently)."""
    fp = V12_FLOW_PREFIX[kind]
    out = {}
    for k, v in sd.items():
        if k.startswith(V12_DEAD_PREFIXES):
            continue
        if k.startswith(fp):
            n = k[len(fp):]
            out["flow." + (n[len("_orig_mod."):] if n.startswith("_orig_mod.") else n)] = v
        elif k.startswith(tuple(p for t, p in V12_FLOW_PREFIX.items() if t != kind)):
            continue
        elif k.startswith("diffusion_connector."):
            n = k[len("diffusion_connector."):]
            out["connector." + CONNECTOR_KEY_MAP.get(n, n)] = v
        elif k == "dino_view_embed":
            out["views.table"] = v
        elif k.startswith("repa_projector."):
            out["road.projector." + k[len("repa_projector."):]] = v
        elif k == "_road_step":
            out["road.step"] = v
        else:
            raise KeyError(f"unmapped v12 key {k!r}")
    return out


def to_v12(sd: Dict[str, torch.Tensor], kind: str) -> Dict[str, torch.Tensor]:
    """Inverse of :func:`from_v12` (so old evaluation code can read a BLIP3D tower checkpoint)."""
    inv_conn = {v: k for k, v in CONNECTOR_KEY_MAP.items()}
    out = {}
    for k, v in sd.items():
        if k.startswith("flow."):
            out[V12_FLOW_PREFIX[kind] + k[5:]] = v
        elif k.startswith("connector."):
            n = k[len("connector."):]
            out["diffusion_connector." + inv_conn.get(n, n)] = v
        elif k == "views.table":
            out["dino_view_embed"] = v
        elif k.startswith("road.projector."):
            out["repa_projector." + k[len("road.projector."):]] = v
        elif k == "road.step":
            out["_road_step"] = v
        else:
            out[k] = v
    return out


def tower_state(ckpt_dir: str, kind: str, *, use_ema: bool, require_ema: bool = True) -> Dict[str, torch.Tensor]:
    """A tower checkpoint (v12 or BLIP3D) in Blip3DTower names, EMA overlaid when asked."""
    sd = load_state(ckpt_dir, use_ema=use_ema, require_ema=require_ema)
    return from_v12(sd, kind) if is_v12(sd) else sd


def load_ema_file(ckpt_dir: str, kind: Optional[str] = None) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    ema = load_file(os.path.join(ckpt_dir, "ema.safetensors"))
    return from_v12(ema, kind) if (kind is not None and is_v12(ema)) else ema
