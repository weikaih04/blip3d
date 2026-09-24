"""Latents -> mesh -> GLB (ISSUES I-17..I-19). Decimation 200k, texture 2048, webp, remesh band 1; no pre-simplify (v12)."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import torch

log = logging.getLogger(__name__)
PBR_LAYOUT = {"base_color": slice(0, 3), "metallic": slice(3, 4), "roughness": slice(4, 5), "alpha": slice(5, 6)}


@dataclass
class Latents:
    coords: torch.Tensor          # (N, 4) int32, batch column 0
    shape_raw: torch.Tensor       # (N, 32) de-normalised shape latent
    tex_raw: torch.Tensor         # (N, 32) de-normalised pbr latent
    info: Dict = field(default_factory=dict)


@torch.no_grad()
def decode(shape_dec, tex_dec, lat: Latents):
    from trellis2.modules import sparse as sp
    from trellis2.representations import MeshWithVoxel
    shape_dec.set_resolution(512)
    meshes, subs = shape_dec(sp.SparseTensor(lat.shape_raw.float(), lat.coords.cuda()), return_subs=True)
    mesh = meshes[0]
    try:
        mesh.fill_holes()
    except Exception as e:                       # v12 swallowed this silently; keep going but say so
        log.warning("fill_holes failed: %s", e)
        lat.info["fill_holes_failed"] = repr(e)
    vox = tex_dec(sp.SparseTensor(lat.tex_raw.float(), lat.coords.cuda()), guide_subs=subs) * 0.5 + 0.5
    return MeshWithVoxel(mesh.vertices, mesh.faces, origin=[-0.5, -0.5, -0.5], voxel_size=1 / 512,
                         coords=vox.coords[:, 1:], attrs=vox.feats,
                         voxel_shape=torch.Size([*vox.shape, *vox.spatial_shape]), layout=PBR_LAYOUT)


def export_glb(mw, path: str, decimation: int = 200_000, texture: int = 2048) -> str:
    import o_voxel
    glb = o_voxel.postprocess.to_glb(
        vertices=mw.vertices, faces=mw.faces, attr_volume=mw.attrs, coords=mw.coords, attr_layout=mw.layout,
        voxel_size=mw.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], decimation_target=decimation,
        texture_size=texture, remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    glb.export(path, extension_webp=True)
    return path


def export_shape_obj(mw, path: str) -> str:
    """Decoder-native Y-up geometry (no axis swap), as the Shape-GT track expects."""
    import trimesh
    trimesh.Trimesh(vertices=mw.vertices.detach().cpu().numpy(), faces=mw.faces.detach().cpu().numpy(),
                    process=False).export(path)
    return path
