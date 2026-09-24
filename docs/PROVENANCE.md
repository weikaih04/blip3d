# Provenance

BLIP3D is a clean re-implementation of the "v12" recipe developed in the BLIP3o fork:

- old repo: `github.com/weikaih04/BLIP3o`, branch `BLIP3o-NEXT`, extracted from commit `25c4751`
  (plus the working-tree changes listed in `docs/ISSUES.md` → "Things to do in the old repo before freezing");
  the old repo will be tagged `v12-final` and kept read-only.
- TRELLIS.2: `github.com/weikaih04/TRELLIS.2`, branch `blip3o-next` @ `2ed033d`
  (upstream microsoft/TRELLIS.2 @ 5565d24 + cond_mask plumbing, data-toolkit changes, fused Triton kernels), MIT.
- ROAD: vendored from H-EmbodVis/ROAD @ c847391 (license to be confirmed before release).

No BLIP3o code is imported at runtime. Old checkpoints are read through the key map in `blip3d/train/ckpt.py`.
The per-file mapping (new module ← old files) is filled in as each module is ported.

| BLIP3D module | old source |
|---|---|
| (filled in during the port) | |
