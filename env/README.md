# Environment

Reference environment: conda env `blip3o_trellis` on the xgen-mm HyperPod cluster (p5en, H200, CUDA 12.4).
`versions.txt` has the core versions; `requirements.lock` is `pip freeze` with `PYTHONNOUSERSITE=1`.

**Always run with `PYTHONNOUSERSITE=1`.** Without it, numpy 2.2.6 from `~/.local` shadows the env's
numpy 1.26.4 (see `user_site_shadowing.txt`); every v12 run since ~2026-09-01 silently used 2.2.6.

Compiled / local builds (`local_builds.txt`), built from source against torch 2.6.0+cu124:
- flash-attn 2.7.3, plus FlashAttention-3 built from `hopper/` (training uses FA3 for sparse attention)
- flex_gemm, cumesh, spconv-cu124 2.3.8, triton 3.2.0, utils3d
- nvdiffrast 0.4.0, nvdiffrec_render (render / GLB export)
- o_voxel: editable install from `third_party/TRELLIS.2/o-voxel`

Model assets (HF cache): Qwen/Qwen3-VL-2B-Instruct @ 89644892…, camenduru/dinov3-vitl16-pretrain-lvd1689m @ 3c276edd…,
microsoft/TRELLIS.2-4B, JeffreyXiang/TRELLIS-image-large (SS decoder), BAAI/Uni3D (ROAD teacher, uni3d-g),
ZhengPeng7/BiRefNet (background removal in the demo).
