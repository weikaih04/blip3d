# 02 — Training path of the separate-tower stages (v12 S1 / S2 / SS continuation)

Read-only audit, 2026-09-24. Repo `/fsx/home/weikai.huang/3dgen/model/BLIP3o` @ `25c4751` (branch BLIP3o-NEXT).
Vendored TRELLIS.2 fork: `/fsx/home/weikai.huang/3dgen/model/third_party_3d_gen/TRELLIS.2` @ `2ed033d`.
Env: transformers 5.2.0, deepspeed 0.19.0, torch 2.6.0+cu124, accelerate 1.7.0 (`blip3o_trellis`).

**Evidence used**
- Four real `config.json` files (s1_{ss,shape,tex}_20000, s2_{ss,shape,tex}/checkpoint-106000), plus s2_ss/checkpoint-60000 and s3_ss_v12s2cont/checkpoint-{45..70}000.
- `training_args.bin` from the same checkpoints, loaded on CPU.
- `rng_state_*.pth`.
- Key lists of `model.safetensors` and `ema.safetensors`.
- wandb `output.log` files for every v12 tower run.
- The launchers: `_lc/v12_s1b.sh`, `_lc/v12_s2.sh`, `_lc/v12_s2_4n.sh`, `lc/v12_ss_cont.sh`, and `scripts/train_native_split.sh`.

**Important:** `config.json` holds only the model config (110 keys). It does not hold `NativeArgs`. Only the training-only fields (flow_tune, ema_*, compile_*, elastic_*, init_from_checkpoint, mixture_config, …) are missing from it. I rebuilt those from the launchers and checked them against logs and checkpoint contents.

Labels: **[verified]** = checked against files or checkpoints. **[likely]** = strong static reading, not executed. **[spec]** = speculation.

---

## 0. Headline findings (read these first)

1. **Timestep `t` is the same on every rank.** [verified]
   - `TRELLIS2FlowMatchingLoss` calls upstream `sample_t`, which uses the **CPU** generator (`torch.rand` / `torch.randn` on CPU, TRELLIS.2 `trainers/flow_matching/flow_matching.py:130-140`; called from `trellis2_blip3o/loss.py:177`).
   - `RankRNGCallback` (train_native.py:212-252) reseeds only CUDA. HF `set_seed(42)` gives every rank the same CPU state.
   - `rng_state_{0..3}.pth['cpu']` are **bit-identical across ranks** in s1_tex_20000, s2_shape_106000 and s2_ss_106000. The CUDA states differ across ranks.
   - Effect: each optimizer step covers only `per_gpu_bs × GA` distinct t values (16, or 8 for SS at bs4×GA2), not 256.
   - Noise, the CFG drop and the DINO drop all use CUDA RNG, so they are per-rank.
   - BLIP3D must decide: stay bug-compatible, or sample t on CUDA / with a per-rank CPU seed.
2. **A resume resets the EMA shadow to the raw weights with n=0.** [verified]
   - `EMACallback.on_train_begin` (train_native.py:77-121) always clones the current params. HF calls it after `deepspeed_load_checkpoint` (transformers trainer.py:1571 then :1642).
   - The update count n is never saved.
   - v12 was hit by this:
     - S2 tex resumed at about 40k (wandb `run-20260908_232233-luvzfm4w`: first gradclip line at step 41000).
     - S2 shape resumed at about 45k (`run-20260908_232346-tolg1l6t`).
     - S2 SS restarted weights-only at 75k (4-node run).
   - So the final `*_106000_ema` files are EMAs over the last ~66k / ~61k / 31k steps (with warm-up), not over 106k.
   - Only `--ema_init_from/--ema_init_updates` avoids this, and only the SS continuation uses them.
3. **The SS continuation launcher is now unsafe to resume.** [verified]
   - `lc/v12_ss_cont.sh` hard-codes `--ema_init_from …/s3_ss_v12s2cont/checkpoint-25000 --ema_init_updates 25000`.
   - That directory has been rotated away. The run dir now holds only 45000–70000 (save_total_limit 6).
   - Any relaunch or resume of this job will crash in `load_file(.../checkpoint-25000/ema.safetensors)`. If the file still existed, it would silently reload the stale 25k EMA over the real one.
   - Per the node/launch memories this is the user's call. I flag it and change nothing.
4. **`GEOTEX_BALANCE_TABLE` was NOT used by the separate-tower v12 runs.** [verified]
   - None of the three tower launchers set it.
   - No "体素均衡已开启" line appears in any v12 tower wandb log.
   - It is set only by the S3 unify launchers (`lc/v12_s3.sh:28`, `lc/v12_s3_4n.sh:32`, with `voxels_v11.npz`).
   - The brief's "voxel load-balancing via env GEOTEX_BALANCE_TABLE" is wrong for S1/S2.
5. **Every resume or restart replays the data from the start.** [likely]
   - `mixture_seed=0` and `--ignore_data_skip True` are set.
   - Mixture seeds are `base_seed ^ worker ^ rank` with no step term (`data/mixture.py:166-173`).
   - So every resume, and S2 relative to S1, replays the same index stream from its beginning. Per-item augmentation (view and caption draws) may differ.
   - Update during this audit: `lc/v12_ss_cont.sh:22` was edited while I was working. It now passes `--mixture_seed ${MIX_SEED:-${EMA_N:-0}}`.
   - That fixes the replay only when `MIX_SEED` or `EMA_N` is exported. The default is still 0, because `EMA_N` has no default in that expression.
   - The current continuation run was launched with seed 0.
   - The `--ema_init_from` default still points at the deleted checkpoint-25000 (§0.3).
6. **Shape and tex checkpoints carry about 3.06B params of dead weight.**
   - Each holds `trellis_decoders.{shape_slat,tex_slat,sc_vae_decoder}` (8.76 GB vs 2.66 GB for SS).
   - These are stock frozen modules built by `builder.py:158-179` that training never runs.
7. **`config.json` records the wrong conditioning VLM.** `vlm_model="Qwen/Qwen3.5-2B"`, but the real conditioning VLM is stock Qwen3-VL-2B, taken from env `COND_VLM_CKPT` (`live_cond.py:54`). The checkpoint does not record which VLM was used.

---

## 1. Call graph of one training step (one tower, v12 settings)

Entry: `train_native.py:1027 main()`. Branches v12 takes:

- `--mixture_config` → train_native.py:1436-1463.
- `train_stages ∈ {ss, shape, tex}`, so the geotex block at 1198-1296 is skipped.
- `_apply_flow_freeze` :349.
- `init_from_checkpoint` :1310-1326 (S2 only).
- Elastic GC :1381-1397 (shape/tex only).
- Callbacks :1481-1489.
- Adaptive-clip lazy binding :1497-1507.
- Auto-resume :1508.

**Data (dataloader workers, CPU)**
1. `MixtureIterableDataset.__iter__` (`data/mixture.py:134`):
   - The task is chosen per batch with a rank-shared RNG (:166-167). The index stream is per rank (:171-173).
   - Batches are homogeneous (`granularity: batch`).
2. `ThreeDTask.__getitem__` (`data/tasks/threed.py:428`, resample loop :425-443):
   - `live_cond: true` → emits `_live_prep` built by `prep_i1` / `prep_im` / `prep_t` (threed.py:457-500 → `live_cond_batch.py:160/205/291`).
   - Targets come from `_attach_slat_targets` (:626):
     - `_load_ss` :300 (no normalization; SS has none).
     - `_load_shape` :308 (normalized with shape norm).
     - `_load_tex` :316 (pbr norm, plus `concat_cond` = GT shape latent in tex shape norm).
     - Voxel cap 8192 :641-647 (shape/tex only; the SS job returns early at :626-628, so **the SS tower has no voxel cap**).
     - ROAD `align_points` :596-624 (env `ROAD_ALIGN_POINTS`, 10000 points from the 32³ GT voxel centres, `road/points.py:14`).
3. `collate_vlm_3d` (`vlm_collate.py:151` live/cached branch → `{"_live_prep": [...]}`):
   - Targets stacked at :197-215 (`SLat.collate_fn`, `SLatPbr.collate_fn` → `target_tex_slat_512`, `tex_concat_cond`).
   - `align_points` stacked at :242-248.

**Trainer (GPU)**
4. `NativeTrainer.get_train_dataloader` (train_native.py:565) wraps the loader in `_LiveCondPrefetch` (:471). Batch N+1 is encoded on a side stream (:500-531).
5. `TrainCondEncoder.encode` (`live_cond_batch.py:498-580`):
   - Qwen3-VL-2B `hidden_states[-1]` over right-padded ids.
   - Boilerplate masked (`keep_all`). Text rows keep only content tokens (:549-555).
   - DINOv3 @512 (1029 tokens/view) for I1/IM (:537-542).
   - Emits `qwen_view_ids` (IM only), `qwen_img_rc` (I1+IM), `dino_view_ids` (0..K-1; **I1 gets ordinal 0**).
   - Output goes through `_pack_live_conds` (train_native.py:411-457).
6. `_prepare_inputs` (:603-612) moves SparseTensors to the device.
7. `compute_loss` (:624):
   - Enters each elastic controller's `record()` (:642-651), then HF `Trainer.compute_loss`.
   - Calls `model.forward` (`trellis_native_vlm.py:1038`).
8. `forward`:
   - Cached-cond branch :1157-1168 (bf16 cast; `cond_key_mask = cond_keep_mask`).
   - fuse_dino contract check :1172-1175.
   - ROAD contract :1187-1191.
   - Calls `flow_heads.compute_cascade_flow_loss` :1346-1402. `road_kit` from `_road_kit()` :998-1036 advances the `_road_step` buffer per micro-step.
9. `compute_cascade_flow_loss` (`flow_heads.py:96`):
   - **Image rows (DINO present) → fusion branch :203-263:**
     - CFG drop mask p=0.1 :209.
     - `connector(cond_hidden*keep, key_mask)` :212 (`connector.py:195`).
     - Qwen view code :222-226.
     - DINO view code :231-235.
     - Patch position `_add_patch_pos` :238 (:505-546).
     - Segment codes :239-241.
     - `dino_seg *= keep` :242.
     - DINO drop p=0.3 :248-251.
     - Concat `[DINO; Qwen]` :261-262.
   - **Text rows → plain branch :264-283:**
     - SS job (ROAD on): explicit drop mask :269-272.
     - Shape/tex jobs: `mask_drop` :274.
     - Then the Qwen segment code :278-279.
   - `sdpa_mask = _mask_or_none` :295 (`GEOTEX_XATTN_FLASH` unset → always masked sdpa, :1710-1712).
   - **SS:** `loss_fn_ss(ss_flow, …)` :315-319 → `loss.py:117`:
     - `sample_t` (CPU) :177.
     - `diffuse` :178 → `x_t=(1-t)x0+(σmin+(1-σmin)t)ε`.
     - `t_in=t*1000` :181.
     - bf16 autocast :193.
     - Dense forward :247.
     - `mse(v_pred, (1-σmin)ε - x0)` :248-250.
   - **ROAD (SS job only)** :345-384:
     - The hook on `ss_flow.blocks[9]` (depth 30//3 = 10; `trellis_native_vlm.py:735,749-754`) stashes the block output.
     - Uni3D-g teacher under bf16 autocast :332-343.
     - `road_alignment_losses` (`road/loss.py:65-94`).
   - **Shape:** :403-408 → sparse path `loss.py:231-245` (cond list per sample built with the key mask, FA3 varlen).
   - **Tex:** :451-458 with `concat_cond=tex_concat_cond`.
   - Combine :465-497: one active stage, so normalized weight 1.0. `total = flow_mse (+ 0.5·road_proj + 0.1·road_opt for SS)`.
10. Back in HF and DeepSpeed:
    - Backward happens under ZeRO-1 `BF16_Optimizer` (selected because bf16 + `grad_accum_dtype fp32` + stage 1; deepspeed `runtime/engine.py:1565-1567`). It keeps fp32 master weights and fp32 grad accumulation.
    - Clip uses `self.clip_grad` (`bf16_optimizer.py:311-316`), then AdamW (single param group, wd 0.01 on all params; HF `integrations/deepspeed.py:591,499`), then the WSD LR scheduler (HF `create_scheduler`, :517-528).
11. Callbacks after the optimizer step:
    - `AdaptiveGradClipCallback.on_step_end` (train_native.py:299-319) sets `opt.clip_grad = min(p95(last 1000 norms), 1.0)` for the next step.
    - `EMACallback.on_step_end` (:142-151) updates the sharded fp32 shadow from the **bf16 module params** (not the fp32 master).
    - `on_save` (:153-179) gathers over gloo and writes `ema.safetensors` into `checkpoint-N`.
12. Logging: `NativeTrainer.log` (:724-742) all-reduces the `per_stage/*` means over the window. Each forward's `.item()` calls force a GPU sync (`loss.py:268-269`, `flow_heads.py:383-384,472`).

---

## 2. NativeArgs classification (120 fields, train_native.py:745-990)

**Classes**
- **a** = varies across v12 stages. This is a recipe field.
- **b** = constant in every v12 run. Hard-code it.
- **c** = never exercised by the separate-tower v12 path. Drop it.

"v12" values come from `train_native_split.sh:137-155` plus the launcher `EXTRA_ARGS` (argparse: last occurrence wins, so `--flow_tune last20` at :144 is overridden). They are confirmed in config.json where the field exists.

| field | default | v12 value(s) | cls | note |
|---|---|---|---|---|
| vlm_model | Qwen/Qwen3.5-2B | same | c | Used only for `AutoProcessor.from_pretrained` (:1404), which the live branch never uses [likely]. The real VLM comes from env `COND_VLM_CKPT`. Replace with a `cond_vlm` path field. |
| data_path | overfit jsonl | unused | c | legacy dataset only (:1464-1479) |
| freeze_vlm | True | True | b→drop | no VLM is built (build_vlm False) |
| build_slat | False | ss:False, shape/tex:True | a→derive | derive from tower |
| ss_only | True | ss:True, else False | c (CLI) | CLI ignored under mixture (mixture.py:335-343); the yaml sets it per task |
| num_cond_views | 1 | – | c | legacy |
| random_cond_view | True | – | c | legacy |
| flow_weight | 1.0 | 1.0 | b | |
| detach_cond | False | False | b→drop | nothing upstream to detach |
| cond_max_length | 8192 | 10240 | b | max IM cond ≈ 4×1029 DINO + ~1060 Qwen ≈ 5.2k; the truncation guard (flow_heads.py:175-183) never fires [likely] |
| flow_tune | full | S1:`none`, S2/cont:`full` | **a** | the only freeze knob that matters |
| cond_fusion | none | none | c | depthwise/penultimate dead |
| fusion_layers | 0 | 0 | c | |
| slat_resolution | 512 | 512 | b | the yaml also sets 512 |
| task_mix | T:.2,I1:.4,IM:.4 | – | c | legacy; the mixture yaml's weights govern |
| max_views | 4 | – | c (CLI) | yaml `max_views: 4` |
| crop_to_object | False | – | c (CLI) | yaml `true` |
| min_aesthetic | None | – | c (CLI) | yaml 4.5 |
| max_slat_tokens | 8192 | – | c (CLI) | **the CLI value is ignored.** The effective 8192 is the threed.py:75 default. Hard-code in data. |
| ema_decay | 0.9999 | 0.9999 | b | |
| ema_warmup | True | True | b | `min(d,(1+n)/(10+n))` |
| ema_init_from | "" | "" / cont: s2cont/ckpt-25000 | **a** | continuation only |
| ema_init_updates | 0 | 0 / cont: 25000 | **a** | continuation only; n is not persisted anywhere |
| dino_align, _weight, _mode | F/.5/spatial | defaults | c | |
| dual_cond … dual_ss_checkpoint (9 fields incl. anchor_drop_prob, cfg_joint_drop_prob) | off | defaults | c | |
| distill_dino … distill_cfg_null_grad (7) | off | defaults | c | |
| target_tokens_per_view | 0 | 1024 | b | recorded in config. Its only consumer (`set_default_target_tokens_per_view`, :1298-1302) feeds the non-live collate. The live path hard-codes `I1_QWEN_CANVAS=1024` / IM 256 per view [likely]. |
| build_vlm | True | False | b→drop | |
| train_stages | all | ss / shape / tex | **a** | = tower |
| geotex_* (39 fields, :838-893) | – | defaults | c here | used only by `train_stages=geotex` (S3 unify, other audit). Exception: `geotex_joint_cond_drop` also belongs to S3. |
| adaptive_grad_clip | True | True | b | |
| fuse_dino | False | True | b | |
| dino_drop_prob | 0.1 | 0.3 | b | |
| qwen_drop_prob | 0.0 | 0.0 | b→drop | branch flow_heads.py:258-260 dead at 0 |
| view_embed_mode | hashed | hashed | b | fixed buffer; scale from env `VIEW_EMBED_SCALE` (0.2; checkpoint row L2 = 4.525) |
| dino_view_embed_max | 16 | 16 | b | |
| repa_root | "" | "" | c | vggt REPA dead |
| repa_coeff | 0.5 | 0.5 | c | not used by ROAD. `_repa_lambda()` still runs each step and returns base. |
| repa_depth | 0 | 0 | b | **used by ROAD**: 0 → auto tap `num_blocks//3` = 10 → `ss_flow.blocks[9]`. Hard-code 10. |
| repa_zdim | 2049 | – | c | vggt only |
| repa_min_quality | 0.5 | – | c | vggt only |
| repa_mode | vggt | ss:`road`, shape/tex:`vggt`(=off) | a→derive | really "road on for SS" |
| road_uni3d_ckpt | "" (auto-glob) | "" | b | resolved to `hf_cache/hub/models--BAAI--Uni3D/snapshots/3d8233b…/modelzoo/uni3d-g/model.pt` (log). Make it an explicit path. |
| road_lambda_proj | 0.5 | 0.5 | b | |
| road_lambda_opt | 0.1 | 0.1 | b | |
| road_opt_start_step | 500 | 1000 (SS) | b | unit is **micro**-steps (500 optimizer steps at GA2). Never mattered in S2 (see §4.8). |
| road_token_count | 512 | 512 | b | |
| road_matcher | gpu | gpu | b | cpu matcher dead |
| road_align_points | 10000 | 10000 | b | |
| road_gc_from | -1 | -1 | b→drop | SS never gradient-checkpointed in v12 |
| road_stages | ss | ss | b | shape tap dead |
| road_shape_depth | 0 | 0 | c | |
| cond_pos_stamp | False | False | c | |
| cond_adapter | xf2 | xf2 | b | the MLP connector is dead |
| adapter_blocks | 2 | 2 | b | |
| cond_seg_embed | False | True | b | |
| cond_patch_pos | off | sincos2d | b | fixed buffer (the NativeArgs comment still says dino_sig) |
| cond_patch_lattice | 32 | 32 | b | |
| init_from_checkpoint | "" | S1:none; S2: `runs/s3_<t>_v12s1b/checkpoint-20000_ema`; SS 4n: `s3_ss_v12s2/checkpoint-75000` (raw); cont: none (true resume) | **a** | |
| mixture_config | None | `configs/mix_v12_<tower>_texfixed.yaml` | **a** (= tower) | |
| mixture_seed | 0 | 0 in every run so far; the cont launcher now passes `${MIX_SEED:-${EMA_N:-0}}` | b→a | causes the data replay (§0.5). BLIP3D should derive it from (run, resume step). |
| mds_root, mds_cache_limit, mds_shards | None | – | c | |
| compile_ss_flow | True | False (COMPILE_SS=0) | b→drop | the compile path is dead and ROAD forbids it |
| compile_mode, compile_dynamic | default/True | – | c | |
| elastic_slat | True | True | b | only acts when `build_slat` (shape/tex) |
| elastic_target_ratio | 0.75 | 0.75 | b | |

**Totals:** a = 7 (flow_tune, ema_init_from, ema_init_updates, train_stages, init_from_checkpoint, mixture_config, plus derived build_slat/repa_mode). b = ~35. c = ~78, of which 39 are the geotex_* fields owned by S3.

**Model-config fields that are not CLI flags but are part of the recipe** (TrellisNativeVLMConfig defaults, trellis_native_vlm.py:52-183; confirmed identical in all config.json files):
- `mask_drop_prob=0.1` (the CFG p_uncond; matches TRELLIS `p_uncond 0.1`).
- `logitnorm_mean=1.0`, `logitnorm_std=1.0`.
- `flow_sigma_min=1e-5`.
- `flow_stage_weights="ss=1,shape=1,tex=1"`.
- `trellis_*_ckpt=None` → the `builder.py:59-95` defaults under `…/3dgen/model/checkpoints/TRELLIS.2-4B/ckpts/`.

**The config.json diff across all v12 tower checkpoints** (S1×3, S2×3, S2-SS@60k, cont@45–70k) varies only in 4 keys:
- `train_stages`
- `build_slat`
- `repa_mode` (road for SS / vggt otherwise)
- `road_opt_start_step` (1000 for SS / 500 default elsewhere, unused)

**TrainingArguments** (`training_args.bin` diff):

| field | S1 (×3) | S2 shape/tex | S2 SS 0→75k (2n) | S2 SS 4n (75k→106k) | SS cont |
|---|---|---|---|---|---|
| learning_rate | 1e-4 | 2e-5 | 2e-5 | 2e-5 | 2e-5 |
| max_steps | 26000 | 106000 | 106000 | 31000 | 75000 (resumed at 25000) |
| per_device_bs × GA × world | 8×2×16 | 8×2×16 | 4×4×16 | 4×2×32 | 4×2×32 |
| save_steps / total_limit | 2500/12 | 5000/6 | 5000/6 | 5000/6 | 5000/6 |

Constant everywhere:
- warmup_steps 100, WSD `{num_decay_steps 6000, cosine, min_lr_ratio 0}`.
- AdamW (DeepSpeed) β = (0.9, 0.95), eps 1e-8, wd 0.01, max_grad_norm 1.0.
- bf16, seed 42, logging_steps 5, dataloader_num_workers 10, ignore_data_skip True, `configs/deepspeed_zero1_fp32acc.json`.

SS runs at bs4 because bs8 OOMs (~135 GB) with ROAD plus live cond (launcher comment).

---

## 3. Dead branches and modules (for the separate-tower v12 path)

### train_native.py
- `_apply_flow_freeze` :364-406 (crossattn / selfattn / lastNN / dual-param rules). v12 returns at :357-358 (full) or :359-363 (none).
  - Note: the "connector always trainable" loop :396-398 is never reached in v12. The connector is trainable only because nothing froze it.
- `_LiveCondPrefetch` fallback: :575-578 and `_encode_live` :583-601 run only with `LIVE_COND_PREFETCH=0`. The profiling code at :588-600 is dead.
- `compute_loss` geotex anneal :634-641. The `unified_geotex` entry in the elastic loop at :643 is dead.
- `S3_TASK_DEBUG` :683-688.
- vggt REPA setup :1041-1052. `else` error :1095-1096.
- Geotex block :1198-1296. Geotex half-load checks :1327-1347.
- `target_tokens_per_view` hook :1298-1302 (no live consumer) [likely].
- torch.compile + accelerate patch :1352-1372.
- MDS data path :1410-1435. Legacy dataset :1464-1479.
- `EMACallback._decay_at` warmup=False :140. `init_from` :104-115 is live only in the continuation.
- `PeakMemCallback` and `WandbFineGrainedCallback`: diagnostics only.
- `_enforce_no_offload` :993: policy guard. Keep or drop.

### blip3o/model/language_model/trellis_native_vlm.py
- VLM build :297-304 and hidden-size derivation :315-321. v12 takes `build_vlm=False` :305-311.
- Geotex / MMDiT3D :337-490. Slat 1024 builders :498-500.
- MLP connector :530-534. `pos_stamp` :537-539. Learned view table :561-562.
- `dino_align` :566-571. `dual_cond` :577-627. distill DINOv3 :630-641.
- Generic SS GC :662-666 (`road_gc_from=-1`).
- vggt projector :742-745. ROAD GC :755-766. Shape ROAD tap :778-808.
- Depthwise routing :813-833. `COMPILE_SS` env compile :839-844.
- `gradient_checkpointing_*` :850-867. `encode_cond` :874-927. `_build_dino_anchor` :943-971.
- `_repa_lambda` :973-996 is called but inert (`REPA_COEFF_HI` unset, value unused by ROAD).
- LM-task path :1088-1118. Depthwise / VLM-encode cond :1130-1156.
- vggt contract :1181-1185. Geotex forward :1202-1270. Dual CFG :1275-1312. Distill teacher :1318-1333.
- Post-loss depthwise / dual / aligner / gate diagnostics :1404-1409, :1424-1434, :1449-1459.
- `build_trellis_decoders` :506 is **live but useless**: it adds 3.06B frozen params plus ~6 GB of GPU bf16 weights to every shape/tex job, and writes them into every checkpoint.

### trellis2_blip3o/flow_heads.py
- Cond truncation :175-183 (guard only). `detach_cond` :186-187.
- `pos_stamp` :227-229, :280-282. `qwen_drop` :258-260 (p=0).
- Distill branch :284-294. KD kwargs :297-308.
- vggt REPA :385-400. ROAD shape tap :415-448 plus `_road_shape_pool` :85-93.
- KD logs :474-482. REPA add :484-487. Shape-ROAD add :494-497.
- The whole unified path :549-1690: `build_unified_cond`, `sample_timestep_pairs/triples`, `_row/_voxel_balanced_mse`, `compute_unified_geotex_loss`. S3 only.

### trellis2_blip3o/loss.py
- KD machinery :33-70, :183-226, :256-265, :272-291.
- Live parts: `__init__` :91-115, and :165-182 plus :229-255 plus :267-270.

### blip3o/model/multimodal_decoder/builder.py
- 1024 variants :132-155.
- `build_trellis_decoders` :158-179 (training-dead).
- Live parts: :22-56 (`_uniform_bf16`) and :59-129.

### trellis2_blip3o/tr2_modules.py
- Live: `to_bf16_keep_complex` :33-57, `load_norm_stats` plus config paths :153-195.
- Inference-only: `ensure_complex_rope` :60-86 and builders :89-135.
- Dead: the `/weka/...` fallback :158-159.

### trellis2_blip3o/repa.py
The whole file is dead (vggt mode).

### trellis2_blip3o/road/
- Live: `align_mlp.py`, `loss.py`, `points.py` (runs in workers), `teacher.py`, `point_encoder.py`, `uni3d.py` (needs `timm`), `matcher_gpu.py` (nvcc JIT on first step, cached in `~/.cache/torch_extensions`).
- Dead: `matcher_cpu.py`, `pool.py` (shape tap only), `evaluation/`.

### Config and scripts
- `configs/deepspeed_*`: only `deepspeed_zero1_fp32acc.json` is live. `_universal` exists as an override hook in lc/v12_ss_cont.sh (`DEEPSPEED_CFG`); not confirmed that it was used. The other 13 are dead.
- `scripts/train_native_split.sh`:
  - Dead: `ROUND=fusion` :107, the MDS branches :116-123, `DEEPSPEED=none` DDP :84-88, the compile-on branch :98.
  - Overridden: `--flow_tune last20` :144 (by EXTRA_ARGS), `--save_steps 500 --save_total_limit 4` :153 (overridden).
- `scripts/v12/v12_ss_cont.sh` in the repo is an **older version**: it starts from the 106k EMA with a fresh optimizer and EMA. It is not what ran. The one that ran is `/fsx/data/weikai.huang/lc/v12_ss_cont.sh`.

---

## 4. Hidden contracts in training

### 4.1 Timestep sampling
- SS: `sigmoid(N(1,1))` (logitNormal mean 1, std 1). Shape/tex: `U[0,1)`.
- Both come from TRELLIS `sample_t` on the **CPU generator** (flow_matching.py:130-140), with identical draws across ranks (§0.1).
- `t` is cast to the target dtype, which is **bf16** here because targets are cast to `cond.dtype` (flow_heads.py:316,404,452; loss.py:177). Upstream TRELLIS uses fp32 targets under AMP.
- So `t` and `t_in = t*1000` (loss.py:181) are bf16-quantized: steps of 4 in t_in near 1000. `x_t` is also formed in bf16.
- **Inference must feed the t-embedder values on the same grid.** [spec: likely harmless, unmeasured]

### 4.2 Loss and masking
- Loss is a plain `F.mse_loss` over all elements:
  - dense SS: mean over B×8×16³;
  - sparse: mean over all voxels in the batch, so large assets weigh more (loss.py:245,250).
- No per-sample weighting and no loss masking.
- With one stage per job the weight is 1.0.
- SS adds `0.5·(1−cos(global))` plus, once `_road_step ≥ 1000`, `0.1·Hungarian`. This sits outside flow_weight (flow_heads.py:492-493).
- ROAD aligns CFG-dropped and DINO-dropped rows too. This deviation is deliberate (flow_heads.py:361-366).

### 4.3 CFG dropout location
There is one per-sample drop mask with p=0.1 (config `mask_drop_prob`). Where it lives depends on the row type.

**Image rows** — explicit mask at flow_heads.py:209.
- Qwen input is zeroed before the connector, so its output = `connector(0)`, key mask intact.
- **The Qwen view code, patch-pos code and Qwen segment code are added AFTER the drop and are NOT zeroed** (:222-241). The dropped Qwen segment is therefore `connector(0)+codes`.
- The DINO segment is zeroed (view and segment codes are added first, then `*keep` :242). Its keys stay visible.
- The DINO drop (p=0.3, :248-251) is independent and masks DINO keys off.
- **The inference null cond must reproduce exactly this structure.** Not verified in this audit; hand to the inference audit. [spec]

**Text rows.**
- SS: explicit mask :270-272.
- Shape/tex: `mask_drop` (CUDA bernoulli, :274).
- The Qwen segment code is added afterwards (:278-279) and also survives the drop.

Other details:
- `connector(0)` uses the key mask. Every token of an all-zero input is identical, so the mask is irrelevant: the maskless `null_cond_like` (flow_heads.py:38-49) gives the same result.
- I1 DINO tokens receive view code row 0 (live_cond_batch.py:574-575). I1 Qwen tokens get no view code, because `qwen_view_ids` is IM only.

### 4.4 Freezing rules per stage
- **S1** (`flow_tune none`, train_native.py:359-363): every param whose name contains `ss_flow`, `shape_slat_512` or `tex_slat_512` is frozen. The connector trains, and for SS so does `repa_projector`.
  - Verified from the EMA: S1 SS tracks 42 tensors (36 connector + 6 projector); S1 shape/tex track 36.
  - The flows stay in train mode (no dropout), and activations are kept for backprop to the connector (no GC).
- **S2 / cont** (`full`): everything built is trainable except `trellis_decoders.*` (frozen by `builder._freeze`).
  - EMA tracks 682 tensors for SS (640 flow + 36 connector + 6 projector) and 676 for shape/tex.
- Fixed buffers are not trainable and not in the EMA: `dino_view_embed`, `diffusion_connector.cond_seg_embed`, `diffusion_connector.cond_patch_pos`, `ss_flow.rope_phases` (complex64), `_road_step`.

### 4.5 EMA
- decay 0.9999, warm-up `min(d,(1+n)/(10+n))` (train_native.py:138-140). One update per **optimizer** step (HF `on_step_end`).
- The shadow is fp32, sharded by parameter index `i % world_size` (:92-95). It is updated from the **bf16 module weights**, not the fp32 master weights (:151).
- It is written only at save time as `ema.safetensors` (fp32; keys = trainable param names). Rank 0 gathers it over a gloo group (:102-103, :162-178).
- It resets on every resume (§0.2) and n is not saved.
- `--ema_init_from` loads the shadow strictly (every trainable key must be present) and sets n (:104-115).
- The `*_ema` directories are made **outside the repo** by `/fsx/home/weikai.huang/_lc/mk_ema_init.py`. It overlays EMA tensors on model.safetensors, cast to the model dtype (bf16). **BLIP3D must port it.**
- S2 was initialized from the S1 **checkpoint-20000_ema**: the end of the stable phase, not the decayed 26k.

### 4.6 WSD scheduler
- HF `warmup_stable_decay`: 100 warm-up steps, stable to `max_steps−6000`, then cosine to 0.
- S1 stable ends at 20k; S2 at 100k.
- The SS 4-node restart re-warmed from 0 for 100 steps at 75k. Its stable phase ended at local 25k (global 100k).
- The continuation resumed the scheduler at local 25k with `max_steps` 75000: stable to local 69k (global 144k), decay to 75k (global 150k).
- A true resume restores the scheduler. `init_from_checkpoint` does not.

### 4.7 Gradient clipping
- DeepSpeed gets `gradient_clipping=1.0` from `max_grad_norm` (HF `integrations/deepspeed.py:158`).
- `AdaptiveGradClipCallback` binds the engine lazily on the first `training_step` (train_native.py:1497-1507). The log line "no DeepSpeed engine attached" at train begin is **misleading**; later lines show thresholds are active (0.11–0.18 in S2).
- After 1000 finite norms, it sets `engine.optimizer.clip_grad = min(p95, 1.0)` (:312-317).
- Its buffer is **not checkpointed**. `BF16_Optimizer` does save and restore `clip_grad` (bf16_optimizer.py:479,524). So after a resume the last threshold stays in force until 1000 new norms have accumulated. A weights-only restart goes back to a flat 1.0.
- `assert all_groups_norm > 0` (bf16_optimizer.py:309) is a hard failure if all grads are zero.

### 4.8 `_road_step` gate
- `_road_step` is a persistent buffer that counts micro-steps (trellis_native_vlm.py:730-731, 1027-1028).
- It is carried through `init_from_checkpoint`, so it accumulates across stages. Value in s2_ss/checkpoint-106000: **402000**.
  - 20000×2 from S1 at GA2
  - + 75000×4 from S2 2-node at GA4
  - + 31000×2 from the 4-node run.
- So the token (opt) term was on from S2 step 0.

### 4.9 Elastic GC (shape/tex only)
- The class is swapped in place: `m.__class__ = ElasticSLatFlowModel` (train_native.py:1388-1389).
- One `LinearMemoryController(buffer_size=1000, update_every=500, target_ratio=0.75, max_mem_ratio_start=0.5)` per SLAT flow (:1390-1395). Upstream TRELLIS uses the same configuration.
- Controller state is not checkpointed.
- `record()` contexts are entered in `compute_loss` (:642-651).
- SS: no GC at all (`road_gc_from=-1`).

### 4.10 Compile and kernels (env contract)
- `COMPILE_SS=0` and `--compile_ss_flow False`: nothing is compiled. ROAD hooks require this.
- `ATTN_BACKEND=flash_attn_3` (train_native_split.sh:134):
  - Sparse attention uses FA3 varlen (`trellis2/modules/sparse/config.py:16-24`).
  - Dense SS cross-attention always gets a mask, so it is forced to sdpa (`modules/attention/full_attn.py:63-72`; `GEOTEX_XATTN_FLASH` unset).
- `FUSED_MODULATE=1` (fused Triton norm-modulate / gate-residual, `trellis2/modules/sparse/fused_modulate.py:17`).
- Both kernel speedups exist only in the vendored TRELLIS.2 fork (`5f97b27` cond_mask plumbing, `2ed033d` fused / segment-sum kernels). **BLIP3D must vendor these patches.**
- `_paths.py:26` defaults `ATTN_BACKEND=sdpa` if unset.
- Other env: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `HF_HUB_OFFLINE=1`, `TORCHINDUCTOR/TRITON_CACHE_DIR` on /dev/shm.
- Cond-path env:
  - `COND_VLM_CKPT` = Qwen3-VL-2B-Instruct snapshot `89644892…`
  - `VIEW_EMBED_SCALE` (default 0.2)
  - `GEOTEX_IM_TOK_PER_VIEW` (default 256)
  - `LIVE_COND_PREFETCH` (default on)
  - `LIVE_COND_SHM`
  - `ROAD_ALIGN_POINTS` (exported by train_native.py:1082).

### 4.11 dtype
- All flows are hard-cast to bf16 except complex `rope_phases` (builder.py:22-56 → tr2_modules.py:33-57). There are no fp32 master weights in the module.
- The fp32 master and fp32 grad accumulation come from DeepSpeed `BF16_Optimizer`. This is why the code stays on ZeRO-1 rather than DDP (builder.py:48-54).
- The fixed codes (seg / patch / view) are stored as bf16 in checkpoints: DeepSpeed casts buffers too.
- The loss forward runs under `torch.autocast(bf16)` (loss.py:193).

### 4.12 Normalization
- Normalization is read at runtime from the TRELLIS.2 `configs/gen/*.json` files (tr2_modules.py:161-195):
  - shape: `normalization`
  - tex target: `pbr_slat_normalization`
  - tex concat: `shape_slat_normalization`
  - SS: **none** (absent upstream)
- `load_norm_stats` **silently returns None if the file is missing** (:182-183). BLIP3D should hard-code the stats or fail hard.

### 4.13 Checkpoint content
Each `checkpoint-N` contains:
- `model.safetensors` (bf16 plus I64/C64)
- `ema.safetensors` (fp32, trainable only)
- `global_stepN/` (ZeRO-1 optimizer shards, per-rank)
- `rng_state_{rank}.pth`
- `trainer_state.json`, `training_args.bin`, `config.json`, `latest`, `zero_to_fp32.py`

Notes:
- The 4n SS run's directory is named `checkpoint-106000` but contains `global_step31000` (it was renamed into forks).
- HF auto-resume uses the regex `checkpoint-\d+`, so `_ema` dirs are ignored.
- The resume trigger is any `checkpoint-*` present in output_dir (train_native.py:1508).

### 4.14 Data-level (training-relevant)
- Mixture per tower: text 0.6 real + 0.1 synth, I1 0.15, IM 0.15 (IM views 2/3/4 with p = 0.15/0.25/0.60).
- `caption_weights` real = [.4,.12,.08,.4], synth = [.45,.15,.15,.25].
- `min_aesthetic` 4.5, `crop_to_object`.
- Tex tower uses pbr-only pools (`*_pbr_train_texfixed_ff.jsonl`), because a mixed batch drops tex supervision (vlm_collate.py:208-230).
- Manifests: `manifests/splits/{capT1800k,capT_synth,pool1800k}_train_texfixed_ff.jsonl` (shape/SS) and `{capT800k_pbr,capT_synth_pbr,pool800k_pbr}_train_texfixed_ff.jsonl` (tex).
- The launcher comments say "*_evalclean"; the yaml actually points at `*_texfixed_ff`. See 04_data.md.

---

## 5. Proposed per-stage recipes (BLIP3D yaml)

### Hard-coded base (not in any yaml)
- cond: stock Qwen3-VL-2B `hidden[-1]`, boilerplate masked, right padding; DINOv3 @512 (1029 tok/view); I1 Qwen canvas 1024 tokens, IM 256 tokens/view; `cond_max_length` 10240.
- Connector: xf2 (2 blocks, 8 heads, mlp 4.0, 2048→1024, out LayerNorm).
- Fixed codes:
  - segment codes (seed 1234, orthogonalised, L2 4.8)
  - patch-pos `sincos2d` lattice 32 × scale 0.2
  - hashed view codes (HASH_BASE 1000003, max 16, scale 0.2)
- CFG p 0.1 per sample, with the structure in §4.3. DINO drop 0.3. Qwen drop 0.
- Flow math: TRELLIS rectified flow, σmin 1e-5, `t_in = 1000t`, plain MSE. SS logitNormal(1,1); SLAT uniform.
- Optimizer: AdamW β (0.9, 0.95), eps 1e-8, wd 0.01 on all params. ZeRO-1 bf16 with fp32 grad accumulation. Adaptive clip p95 / 1000 / max 1.0.
- EMA 0.9999 with warm-up, **plus persisted n and shadow restore on resume** (new).
- WSD warm-up 100, cosine, min 0.
- Effective batch 256, seed 42, `log_every` 5.
- Voxel cap 8192 (shape/tex).
- ROAD constants: Uni3D-g, tap block 10/30, AlignMLP 1536→2048→2048→1024, 512 tokens, GPU matcher, 10k points, λ = 0.5 / 0.1, gate 1000 micro-steps.
- Elastic LinearMemoryController(1000, 500, 0.75, 0.5).
- Base weights: TRELLIS.2-4B `ss_flow_img_dit_1_3B_64_bf16`, `slat_flow_img2shape_dit_1_3B_512_bf16`, `slat_flow_imgshape2tex_dit_1_3B_512_bf16`.

### Per-stage yaml fields
```yaml
# s1_connector.yaml   (instantiate ×3: tower ∈ {ss, shape, tex})
tower: ${tower}                 # derives build_slat, road (ss only), elastic (shape/tex), data pools
mixture: v12_${tower}           # configs/mix_v12_${tower}_texfixed.yaml
init: null                      # stock TRELLIS.2 flows + fresh connector (+ fresh ROAD projector for ss)
trainable: connector            # flow_tune none (ss also trains road projector)
lr: 1.0e-4
max_steps: 26000                # stable ends 20000; S2 initialises from 20000 EMA
decay_steps: 6000
nodes: 2
per_gpu_bs: 8                   # GA derived = 256/(8*16)=2
save_steps: 2500
save_total_limit: 12

# s2_ss.yaml
tower: ss
mixture: v12_ss
init: {from: s1_connector/ss_20000, use_ema: true}
trainable: all
lr: 2.0e-5
max_steps: 106000
decay_steps: 6000
nodes: 4                        # historically 2n bs4 GA4 for 0-75k, then a 4n weights-only restart
per_gpu_bs: 4                   # bs8 OOMs with ROAD + live cond
save_steps: 5000
save_total_limit: 6

# s2_shape.yaml / s2_tex.yaml
tower: shape | tex
mixture: v12_shape | v12_tex
init: {from: s1_connector/<tower>_20000, use_ema: true}
trainable: all
lr: 2.0e-5
max_steps: 106000
decay_steps: 6000
nodes: 2
per_gpu_bs: 8
save_steps: 5000
save_total_limit: 6

# ss_continuation.yaml
tower: ss
mixture: v12_ss
resume: {from: <S2-SS ckpt at global 100k>, weights: true, optimizer: true, rng: true,
         ema: {from: same, n: 25000}}   # = --ema_init_from/--ema_init_updates
trainable: all
lr: 2.0e-5
max_steps: 75000                # local step count, resumed at 25000 → stable to 69k, cosine 69k→75k
decay_steps: 6000
nodes: 4
per_gpu_bs: 4
save_steps: 5000
save_total_limit: 6
```

Decisions left to the user:
- Keep the CPU-RNG-synced t (bug-compatible), or fix it (§0.1).
- Whether `s2_ss` should be recorded as the historical two-leg run (75k, then a 31k restart with EMA reset) or as a clean 106k run.

---

## 6. Checkpoint key names BLIP3D must map from

`model.safetensors`:

| prefix | #keys | dtype | in EMA | notes |
|---|---|---|---|---|
| `diffusion_connector.in_proj.{weight,bias}` | 2 | bf16 | yes | [1024,2048] |
| `diffusion_connector.blocks.{0,1}.{norm1,norm2,q_norm,k_norm}.{weight,bias}`, `.qkv.*`, `.proj.*`, `.mlp.{0,2}.*` | 32 | bf16 | yes | qkv [3072,1024], mlp [4096,1024]/[1024,4096], q/k_norm [128] |
| `diffusion_connector.out_norm.{weight,bias}` | 2 | bf16 | yes | |
| `diffusion_connector.cond_seg_embed` | 1 | bf16 | **no** (buffer) | [2,1024], row0 = DINO, row1 = Qwen |
| `diffusion_connector.cond_patch_pos` | 1 | bf16 | **no** | [1024,1024] = 32×32 lattice |
| `dino_view_embed` (top level, not under connector) | 1 | bf16 | **no** | [16,1024] hashed, L2 4.525. **Dropped on `init_from_checkpoint`** and regenerated from env (train_native.py:1320-1321). |
| `ss_flow.*` (SS job) | 641 | bf16 + `ss_flow.rope_phases` C64 | 640 (no rope) | TRELLIS SparseStructureFlowModel names (`blocks.i.{self_attn,cross_attn,mlp,…}`, `t_embedder.mlp.{0,2}`, `adaLN_modulation.1`, `input_layer`, `out_layer`) |
| `repa_projector.layers.{0,2,4}.{weight,bias}` (SS job) | 6 | bf16 | yes | ROAD AlignMLP 1536→2048→2048→1024 (attribute name kept from vggt REPA) |
| `_road_step` (SS job) | 1 | int64 | no | cumulative micro-step counter (402000 at the final) |
| `shape_slat_512.*` (shape job) | 640 | bf16 | yes | SLatFlowModel names |
| `tex_slat_512.*` (tex job) | 640 | bf16 | yes | |
| `trellis_decoders.shape_slat.*`, `.tex_slat.*`, `.sc_vae_decoder.*` (shape/tex jobs) | 640+640+292 | bf16 | no | **stock, frozen, dead. Drop on import.** |

`ema.safetensors`: fp32, exactly the trainable keys above:
- SS S2: 682 keys; S1 SS: 42.
- Shape/tex S2: 676 keys; S1 shape/tex: 36.

Model and EMA share the same names, so no remapping is needed between them. Assembly for inference takes one connector per tower (three copies of `diffusion_connector.*`). BLIP3D should namespace them, e.g. `ss.connector.*`, `shape.connector.*`, `tex.connector.*`. The flows map `ss_flow.→ss.flow.`, `shape_slat_512.→shape.flow.`, `tex_slat_512.→tex.flow.`.

S3 unify keys (`unified_geotex.*`, `geo_connector.*`, `ss_connector.*`) are not covered here.

---

## 7. Risks and open questions

1. **t synced across ranks** (§0.1). Fixing it changes the recipe. Needs a user decision.
2. **EMA windows differ per tower** because of the resume resets (§0.2). The "106k EMA" is not a 106k EMA. If BLIP3D re-trains with persisted EMA, the results will differ from v12, probably slightly.
3. **The SS continuation cannot be resumed as launched** (§0.3). Its data also restarts from the stream origin on any resume (§0.5).
4. **The inference null cond must match the training uncond structure** (§4.3): codes survive the drop, DINO keys stay visible, and I1 DINO gets view code 0. Verify in the inference audit. [spec]
5. **bf16 t and x_t quantization** (§4.1). Unmeasured. [spec: small]
6. **The cond VLM identity is not recorded in the checkpoint** (config says Qwen3.5-2B). `VIEW_EMBED_SCALE` is not recorded either, but it can be recovered from the stored buffers (`view_code.detect_view_code_mode`).
7. **`init_from_checkpoint` drops `dino_view_embed`** and rebuilds it from env. A different `VIEW_EMBED_SCALE` at S2 launch would silently shift every conditioning code.
8. **Weight decay 0.01 hits LayerNorm, bias and adaLN params.** This matches TRELLIS upstream (single group). Keep it for parity.
9. **The EMA tracks bf16 weights, not the fp32 master.** For LR 2e-5 × 1.3B bf16 params some updates round away in the module copy. The EMA inherits that quantization. [spec: minor]
10. **ROAD aligns CFG-dropped rows** (deliberate), and the gate unit is micro-steps (depends on GA).
11. **The SS tower has no voxel cap** and needs `shape_latent_512` on every row (align_points), so rows without it are resampled away. Shape and tex silently skip assets over 8192 voxels.
12. **Mixture silently ignores the CLI data flags** (`--max_slat_tokens`, `--ss_only`, `--slat_resolution`; mixture.py:335-343). BLIP3D should keep a single source of truth for these.
13. **Launch fragility:**
    - duplicate `--flow_tune` (last wins);
    - `EXTRA_ARGS` overriding `--save_steps`;
    - the in-repo `scripts/v12/v12_ss_cont.sh` has drifted from the launcher that actually ran;
    - S1 checkpoints are symlinks into `/fsx/home` run dirs, which are near quota;
    - `mk_ema_init.py` lives outside the repo.
14. **The vendored TRELLIS.2 patches are load-bearing** (cond_mask plumbing, fused kernels, fp32 segment-sum). Without them the dense cross-attention mask path does not exist.
15. **`load_norm_stats` silently returns None** when a config is missing (tr2_modules.py:182-183). This is a silent-corruption risk.
16. **The Qwen3.5-2B `AutoProcessor` is still loaded** (train_native.py:1404). HF-offline startup needs it cached even though the live path never uses it. [likely]
17. **Two training-dead frozen 1.3B flows plus the SC-VAE are kept on GPU** in every shape/tex job (~6 GB). Dropping them in BLIP3D frees memory and 6 GB per checkpoint.
