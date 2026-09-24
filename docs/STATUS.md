# BLIP3D — port status (2026-09-24)

What is ported from the v12 codebase, and how each part is verified against it. "bit-exact" = identical tensors from
the same inputs and seeds, old code imported read-only from the BLIP3o repo.

| part | module | verified | how |
|---|---|---|---|
| CPU prep, prompts, CondStamp, connector, assembly (train + CFG) | `cond/` | bit-exact, CPU | `tests/test_cond_parity_cpu.py` (41 checks, S2/S3 connectors) |
| Encoder (Qwen3-VL-2B + DINOv3) | `cond/encoder.py` | bit-exact, GPU (i1, im, t; base-model hidden == v12 full-model hidden) | `tests/test_train_parity_gpu.py` part A |
| Towers, decoders, latent stats, strict loading (v12 + BLIP3D names) | `models/towers.py` | CPU | strict loads of the three S2 finals and an imported checkpoint |
| Unified model (fused path) + loading | `models/unified/` | CPU strict load of S3 17k | GPU goldens pending |
| Tower flow loss (S1/S2/cont) | `losses/flow.py`, `models/blip3d.py` | loss bit-exact for ss/shape/tex x i1/im/t on the S2 finals; connector grads within the old code's own run-to-run noise (non-deterministic backward kernels, ~1e-5) | `tests/test_train_parity_gpu.py` part B |
| ROAD | `models/road/`, `losses/road.py` | CPU bit-exact (36 checks); GPU: teacher, matcher (== scipy optimum), loss at the 999/1000 gate bit-exact; the real-flow backprop part needs > 26 GB | `tests/test_road_cpu.py`, `tests/test_road_gpu.py` |
| S3 unified loss | `losses/unified.py` | CPU bit-exact on real S3 connectors (26 checks); GPU pending (needs a free GPU, > 26 GB even in light mode) | `tests/test_unified_loss_cpu.py`, `tests/test_unified_loss_gpu.py` |
| Data: manifests, tasks, mixture, balance, collate | `data/` | bit-exact streams / items / batches, CPU (121 checks) | `tests/test_data_parity_cpu.py` |
| Trainer, EMA (persisted n), per-rank RNG, adaptive clip (persisted), prefetch, recipes | `train/`, `tools/train.py` | GPU smoke on the real data mix: s1_shape 2 steps, save, resume to 4 (EMA restored at n=2, timestep generator reseeded at step 2); s1_ss with ROAD 2 steps | DeepSpeed ZeRO-1, 1 GPU; multi-node not yet run |
| v12 checkpoint import (weights, EMA, DeepSpeed shards, global steps) | `tools/import_v12.py` | CPU: strict module load, parameter order identical (682), slice maps renamed on every rank | — |
| Inference: presets, schedule, cascade, unified refine/joint/interleave, export, Pipeline | `infer/` | unified refine / joint / interleave, image and text: coords and latents bit-identical to v12 GeoTexSampler on S3 17k EMA; cascade not yet compared | `tests/test_infer_parity_gpu.py` |
| Benchmark runner, infer CLI, SS readout | `tools/bench.py`, `tools/infer.py`, `tools/readout.py` | — | re-score 5 cases pending |
| Cluster: launch, stop, heartbeat, failover | `cluster/` | ported from the live lc scripts | — |

## Fixes switched on by default (v12 behaviour behind a flag)

| issue | fix | flag for v12 behaviour |
|---|---|---|
| T-01 identical timesteps on every rank | per-rank timestep generator | `compat.shared_t` |
| T-02 EMA reset on resume | shadow + n saved and restored | — (`ema.from/n` seeds a continuation) |
| T-04 data replay on resume | data seed from (seed, resume step) | `compat.data_replay` |
| T-06 clip buffer lost on resume | `gradclip_state.json` | — |
| U-05 geo lane never CFG-dropped in S3 | geo replays the tex drops | `compat.unified_v12_drops` |
| C-03 Qwen LM-head logits computed for nothing | base-model hidden (identical tensor) | `compat.full_model_hidden` |
| D-07/D-08/D-11/D-13 data | cap at load, batcher age cap, whitelisted resampling, no pbr for the shape tower | `compat_replay` in data |
| K-04/K-05 | recipe + encoder identity in every checkpoint, global step numbering | — |

Inference fixes U-02 (SS K/V in refine), U-03 (t-mixer in the cached geo pass), U-04 (uncond SS lane), U-06
(lag pairing on the trained grid) are implemented but OFF by default (`UnifiedOptions.fixed()` turns them on) until
they are measured, so the default reproduces the v12 benchmark numbers.
