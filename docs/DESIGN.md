# BLIP3D — design (draft 1, 2026-09-24)

BLIP3D is the final name of our 3D generator. This repository replaces the `BLIP3o` fork
(`/fsx/home/weikai.huang/3dgen/model/BLIP3o`, branch `BLIP3o-NEXT`). It contains **only the v12 recipe**:
one image, several images, or a sentence in; a textured 3D asset out. The old repo is frozen as an
archive (tag `v12-final`) so every historical number stays reproducible.

Evidence for every statement below lives in `docs/scan/0{1..5}_*.md` (read-only audit of the old repo,
file:line references). Nothing has been ported yet; this document is for review.

---

## 1. Scope

**In**
- Conditioning: stock Qwen3-VL-2B (frozen feature extractor, last hidden layer) + DINOv3 @512, encoded live;
  xf2 connector; CondStamp (segment codes, 2D sin-cos patch code at each Qwen image token, hashed view codes).
  Inputs: single image (i1), 2–4 images (im), text (t).
- Models: the three TRELLIS.2-4B towers (SS 16³ dense, shape SLAT 32³, tex SLAT 32³), ROAD head on SS,
  and the unified three-tower model (union attention + SS lane, fused path only).
- Training: S1 connector-only → S2 per-tower full fine-tune → S3 unified → continuation; WSD schedule,
  EMA with persisted count, ZeRO-1 fp32 accumulation, voxel-balanced batching, tex placeholder, caption tiers.
- Inference: separate-tower cascade and unified modes `refine` / `joint` / `interleave`, frozen presets per
  input type, GLB export.
- Evaluation adapters for the WilD3D suite (which stays its own repo), SS readout, condition-sensitivity probe.

**Out** (archived, not ported)
- The whole BLIP3o AR image generator: `blip3o_qwen*`, `blip3o_arch`, `conversation`, `blip3o/train`,
  `blip3o/data`, TA-tok, top-level `train.py`, `inference.py`, `gradio/`, `trl/`, `tok/`, `experiment/`.
- v2.2 VLM, cond cache, MDS/streaming, packed readers, tap datasets, category RFS.
- Retired features: `pos_stamp`, vggt-REPA, plkv, codebook, dual-cond, cond-stream coupling, gated coupling,
  x-attn anneal, from-scratch MMDiT, elastic GC on the unified model, two-tower assembly.
- ~390 one-off scripts (probes, ablations, profiling, hold sbatch, `_tmp`).

## 2. Naming

- Repo / package `blip3d`. Classes `Blip3D` (separate towers), `Blip3DUnified`, `Blip3DCond`.
- No "vlm", "v22", "geotex", "trellis_native" in names. Qwen and DINO are *encoders*.
- Stages `s1`, `s2`, `s3`; towers `ss`, `shape`, `tex`; unified modes `refine`, `joint`, `interleave`
  (not numbers — the old code used two conflicting numberings).
- Runs `blip3d-v12-s2-ss`, `blip3d-v12-s3-unify`, …; **all step numbers are global**.
- Project / benchmark names stay WilD3DGen / WilD3D ("BLIP3D (v12) on WilD3D").

## 3. Repository layout

```
blip3d/
  cond/      qwen.py (HF Qwen3-VL, frozen) · dino.py · prompts.py (templates, hash pick, boilerplate filter)
             prep.py (CPU half, runs in workers) · stamp.py · connector.py (xf2) · assemble.py (cond/uncond, train+infer)
  models/    towers.py (build + load the three TRELLIS.2 flows) · road/ (vendored)
             unified/{ops,plan,ss_lane,model,loading}.py (the 1800-line file split, fused path only)
  losses/    flow.py (timestep laws, FM loss) · road.py · unified.py (row classes clean/lag/solo, corners, tex mask)
  data/      manifest · latents · views · task (modality t/i1/im) · mixture · balance · collate · dist
  train/     trainer.py · stages.py (freeze rules) · ema.py (persists n) · schedule.py (WSD) · clip.py
             ckpt.py (format, key mapping from old checkpoints, universal conversion, EMA overlay)
  infer/     presets.py · schedule.py (t grid, alpha warp, guided) · stages.py · cascade.py · unified.py · export.py
  eval/      suite.py (WilD3D case dirs) · readout.py (SS occ / ULIP / Uni3D) · probe.py (T4)
  utils/     paths.py (paths.yaml) · backend.py (attention/sparse backends, set before trellis2 import) · dist · log
recipes/     s1_connector · s2_{ss,shape,tex} · s3_unify · continue_ss.yaml · data/*.yaml · infer/*.yaml
tools/       train.py · infer.py · bench.py · readout.py · mk_ema.py (thin CLIs; never imported by blip3d/)
cluster/     launch.py (holds as arguments) · heartbeat.py · failover.py · monitor.py
third_party/ TRELLIS.2 fork (pinned submodule; carries the fused Triton kernels) · env/ (lock file, build notes)
tests/       golden/ (outputs recorded from the old repo) · unit tests
docs/        DESIGN.md · recipe.md (v12 contracts) · ckpt_format.md · scan/
```

Rules: dependencies flow `utils → cond/models → losses/data → train/infer → eval`; `tools/` and `cluster/`
only call the package. No file over ~600 lines. No absolute `/fsx` path in package code (`paths.yaml`).
Environment variables only select paths and backends; everything that changes numbers is a recipe field and
is written into the checkpoint.

## 4. Configuration

The old trainer has 120 `NativeArgs` fields plus ~30 environment knobs; the real per-stage recipe is 7 fields.
Audit of the v12 checkpoint configs: ~35 fields constant in every run (hard-coded), ~78 never exercised (dropped),
7 per stage (`tower`, `mixture`, `init`/`resume`, `trainable`, `lr`, `max_steps`, `decay_steps`) plus
compute (`nodes`, `per_gpu_bs`, `save_*`). Full proposal: scan/02 §5.

```yaml
# recipes/s2_ss.yaml
tower: ss
mixture: v12_ss
init: {from: s1_connector/ss_20000, use_ema: true}
trainable: all
lr: 2.0e-5
max_steps: 106000
decay_steps: 6000
compute: {nodes: 4, per_gpu_bs: 4, save_steps: 5000, keep: 6}
```

Data mixes (scan/04 §6): tasks route on modality (`t / i1 / im`), `targets: [ss, shape, tex]` +
`tex_missing: placeholder|skip` replace `ss_only / load_tex / tex_placeholder`; an `n_vox` manifest field
replaces the external voxel table.

Checkpoint `config.json` records `model: blip3d`, `version: v12`, stage, tower, the full recipe, and the
**encoder identity** (Qwen and DINO HF id + revision), which is verified on load.

## 5. Contracts that must survive the port bit-for-bit

Complete lists with file:line: scan/01 §2, scan/02 §4, scan/03 §4, scan/04 §5. The ones most likely to be lost:

- **Encoder**: stock Qwen3-VL snapshot `89644892…` (never a silent fallback); last hidden layer; boilerplate
  filter also drops the "user"/"assistant" ids anywhere; hidden stored as fp16 before the connector.
  Tokens: i1 1050 (1024 image), im 1056 at 256/view, DINO 1029/view.
- **Framing**: Qwen sees the raw render (RGB, no alpha compositing, no crop); DINO sees the alpha-cropped image
  on black at 512. `crop_to_object` has no effect in v12.
- **Prompts**: text template chosen by hash of (sha, caption index); real-text tiers fall back to the long
  caption, synth-text tiers renormalise.
- **Stamps**: use the bf16 values stored in the checkpoints (regenerated fp32 differs up to 1.65e-3); addition
  order connector → view → patch → segment; i1 DINO tokens get view row 0, Qwen tokens none.
- **CFG uncond**: Qwen half = `connector(0) + stamps`, DINO half = 0 with keys kept. Not "all zero".
- **Dropout**: whole-row 0.1 and DINO 0.3 in every stage; Qwen 0.1 **only in S3**. S1/S2 draw dropout inside the
  cascade loss (different RNG order from `build_unified_cond`).
- **Timesteps**: SS logitNormal(1,1), SLAT uniform, rounded to bf16; unified row classes clean 0.7 / lag 0.3 /
  solo 0, corners 0.1/0.2, `t_s ≤ t_x`.
- **EMA**: 0.9999 with (1+n)/(10+n) warm-up; overlay to bf16 for eval.
- **Latents**: SS un-normalised; shape/pbr normalised with TRELLIS 512 stats — hard-fail if the stats file is missing.
- **Inference**: time grid `t_seq(steps, rescale_t)`, guidance interval semantics, CFG rescale, alpha warp
  `t_x = αt/(1+(α−1)t)`, refine seed `seed+1`, 8192 voxel truncation reported, TRELLIS v1 SS decoder for occupancy.

## 6. Known defects: fix or keep?

The audit found places where v12 did something unintended. BLIP3D fixes them by default, with a
`compat_v12` switch that reproduces the old behaviour so every fix can be A/B'd.

| # | defect (old repo) | effect | BLIP3D |
|---|---|---|---|
| 1 | Unified eval ran the loop attention path; training used the fused path | published unify numbers measured a different model | fused only (runners already patched 09-24) |
| 2 | Timesteps drawn on the CPU RNG, identical on every rank | 8–16 distinct t per step instead of 256 | per-rank seeding — **decision pending** |
| 3 | EMA reset to raw weights on every resume, count not saved | "106k EMA" averaged only the last 31k–66k steps | persist shadow + n |
| 4 | Data order replays from step 0 on every resume (seed 0, `ignore_data_skip`) | the same batches seen twice | seed per resume / resumable sampler |
| 5 | Refine and tex-on-GT-shape omit the SS K/V that training always had | train/infer mismatch | pass SS K/V |
| 6 | CFG uncond gives the SS lane the positive condition; geo never saw an uncond in S3 | CFG direction off-distribution | match training drops |
| 7 | Cached geo pass drops the t-mixer term | small (weight ≈ −1.2e-3) | include |
| 8 | Unlimited view codes break when loaded from bf16 checkpoints (>16 views errors) | runtime error | store the mode explicitly |
| 9 | Checkpoint config says `Qwen3.5-2B` and "no stamps" | loaders mis-build connectors | record encoder id; build from state dict and cross-check |
| 10 | SS tower has no 8192 voxel cap; balancer never emits >11.8k-voxel assets | small | single rule applied at the manifest level |
| 11 | Every exception silently resampled (fails only after 9 tries) | hides real bugs | log + counter + fail fast on repeated errors |
| 12 | shape/tex checkpoints carry 3.06B unused decoder params (~6 GB) | disk/GPU waste | not saved, not loaded |
| 13 | numpy 2.2.6 from `~/.local` shadows the env's 1.26.4 (no `PYTHONNOUSERSITE`) | unpinned environment | lock file, `PYTHONNOUSERSITE=1` |

## 7. Old checkpoints

BLIP3D loads every checkpoint in `runs/keep/v12/` and the running SS-300k run **without converting files**:
a key map from `diffusion_connector.*`, `ss_flow.*`, `shape_slat_512.*`, `tex_slat_512.*`, `repa_projector.*`,
`unified_geotex.*`, `geo_connector.*`, `ss_connector.*`; connectors are built from the state dict
(xf2 + stamp buffers present or not), then checked against the config. `dino_view_embed` is rebuilt. EMA files
contain only trained keys and are overlaid. Details: scan/02 §6, scan/03 §5.

## 8. Environment

Python 3.10, torch 2.6.0+cu124, transformers 5.2.0, deepspeed 0.19.0, flash-attn 2.7.3 + FA3 (hopper),
triton 3.2.0, spconv 2.3.8, flex_gemm 1.0.0, cumesh, utils3d, nvdiffrast 0.4.0, o_voxel.
TRELLIS.2 comes from the fork `weikaih04/TRELLIS.2@blip3o-next` (3 commits on upstream, fused Triton kernels in
unpushed commit `2ed033d`) — push it and pin it as a submodule. Assets: Qwen3-VL-2B `89644892…`, DINOv3 mirror,
TRELLIS.2-4B, TRELLIS-image-large SS decoder, Uni3D-g, BiRefNet; ULIP-2 stays with the eval suite.
Attention backend: FA3 for training, sdpa for eval today — pick one per purpose explicitly (`utils/backend.py`).

## 9. Verification (goldens)

Before the old repo is frozen, record from it, for 4 fixed inputs (i1, im-3, im-4, text) on the S2 finals and
the S3 17k checkpoint:
- cond / uncond tensors per tower (train path and eval path),
- first-step velocities of each tower and of the unified model (fused),
- final latents per inference mode with a fixed seed,
- one training batch: loss per tower and unified; 20-step loss curve,
- sfv50 metrics on 5 cases per mode.

Each ported module must reproduce its goldens (bit-exact where the math is unchanged, `compat_v12=True`),
then the fixes in §6 are switched on and measured separately.

## 10. Migration order

0. Freeze the archive: commit the working tree (EMA-continuation, SS K/V cache, fused runners), force-add the
   untracked files that matter (`tests/`, the `_tmp` modules eval_suite imports, voxel-table builder), commit
   eval_suite's `text/` and `run_v12_*`, push the TRELLIS.2 fork, tag `v12-final`.
   **After the SS-300k run finishes** — its failover relaunches import the old working tree.
1. Record the goldens (§9).
2. Skeleton, environment lock, `paths.yaml`, backend setup.
3. `cond/` + `models/` (towers, unified, ROAD) → cond and velocity goldens.
4. `infer/` → latent goldens; switch the demo, the benchmark runners and the eval-suite adapters to it.
5. `eval/` → re-score 5 cases and match.
6. `data/` + `train/` + `losses/` → batch-loss and 20-step goldens; a recipe must reproduce the old launcher arguments.
7. ROAD, voxel balance, resume/EMA/universal-checkpoint test from a kept checkpoint.
8. `cluster/` + `tools/`.

## 11. Decisions (resolved 2026-09-24: "fix everything")

1. Defect #2 (per-rank timesteps): fixed by default (`compat.shared_t` restores v12); the running SS-300k already
   reseeds the CPU generator per rank.
2. Training framework: HF Trainer + DeepSpeed ZeRO-1 (parity with v12 checkpoints, optimizer shards importable).
3. The unified model is first-class (`recipes/s3_unify.yaml`, `Pipeline.unified`); its inference fixes are
   implemented behind `UnifiedOptions.fixed()` until measured.
4. TRELLIS.2 is a pinned submodule of the pushed fork (`blip3o-next` @ `2ed033d`).
5. Manifests stay outside the repo (`paths.manifests`).
6. `s2_ss` history is documented as the two-leg run it was (docs/STATUS.md, CHECKPOINTS.md); recipes describe the
   clean single run.
