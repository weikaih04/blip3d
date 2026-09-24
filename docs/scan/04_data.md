# 04 — v12 data pipeline audit (for BLIP3D extraction)

Repo `/fsx/home/weikai.huang/3dgen/model/BLIP3o` @ BLIP3o-NEXT (clean). Read-only audit, 2026-09-24.
Unqualified paths are relative to `trellis2_blip3o/`. `TN` = `train_native.py`. Anything marked **[spec]** is inference, not verified.

Env versions: accelerate 1.7.0, transformers 5.2.0, torch 2.6.0+cu124.

## 0. v12 runs and the data config each one used

| stage | launcher | mix yaml | PER_GPU_BS × GPUs × GA | WORKERS | voxel balance |
|---|---|---|---|---|---|
| S1 (towers) | `_lc/v12_s1b.sh` → `scripts/train_native_split.sh <tower> s3` | `configs/mix_v12_{ss,shape,tex}_texfixed.yaml` | 8 × 16 × 2 | 10 | off |
| S2 (towers) | `_lc/v12_s2.sh`, `_lc/v12_s2_4n.sh` | same | 8 × 16/32 × GA | 10 | off |
| SS cont | `scripts/v12/v12_ss_cont.sh` | `mix_v12_ss_texfixed.yaml` | 4 × 32 × 2 | 10 | off |
| S3 unify | `scripts/v12/v12_s3.sh`, `v12_s3_4n.sh` → `scripts/train_native_geotex.sh` | `/fsx/data/weikai.huang/lc/configs/mix_v12_unify_texfixed.yaml` | 4 × 32 × 2 | 8 | `GEOTEX_BALANCE_TABLE=/fsx/home/weikai.huang/_lc/balance/voxels_v11.npz` |

All stages: `--ignore_data_skip True`, `--dataloader_num_workers $WORKERS`, `--mixture_seed` never set (so it is 0), `COND_VLM_CKPT=<stock Qwen3-VL-2B snapshot>`, `GEOTEX_IM_TOK_PER_VIEW` = 256 (default, and S3 sets it explicitly). S2 SS and SS-cont also use `--repa_mode road`, so the dataset emits `align_points` (see §2.9). S3 dropped ROAD.

The four yamls differ only in: `ss_only` (true for ss, false elsewhere), manifests (tex uses the pbr-only pools), and `tex_placeholder: true` (unify only). None of them sets `max_slat_tokens`, so the dataset default of 8192 applies. The CLI `--slat_resolution`/`--max_slat_tokens`/`--ss_only` flags are **ignored** by the mixture; `mixture.py:339-343` prints a warning about this.

---

## 1. Data flow (v12 = the `live_cond: true` path only)

```
yaml ──build_mixture (data/mixture.py:293)──► 4 task Datasets (map-style) ──► MixtureIterableDataset (IterableDataset)
  per worker: task draw ─► index cycler ─► [TargetBatcher, S3 only] ─► ds[j] = _ThreeDTaskBase.__getitem__ ─► _load_one
  → item dict {"_task","id","_live_prep",targets…}
  → DataLoader(batch_size=PER_GPU_BS) → MultiTaskCollator (mixture.py:242) → collate_vlm_3d cached/live branch (vlm_collate.py:151-250)
  → batch {"_task","_live_prep":[prep…], target_ss_latent, target_shape_slat_512, target_tex_slat_512, tex_concat_cond, tex_valid, [align_points]}
  → accelerate DataLoaderShard (IterableDatasetShard pass-through patch, rank_aware.py)
  → _LiveCondPrefetch (TN:471) pops _live_prep, TrainCondEncoder.encode (live_cond_batch.py:498) on a side stream,
    _pack_live_conds (TN:411) → cond_hidden, cond_keep_mask, dino_hidden, dino_keep_mask, dino_view_ids, qwen_view_ids, qwen_img_rc
  → NativeTrainer._prepare_inputs (TN:603) moves SparseTensors to the device → model.forward(_task=…, …)
```

### 1.1 Construction
- `TN:1404`: `processor = AutoProcessor.from_pretrained(--vlm_model)` (`Qwen/Qwen3.5-2B` in the launchers). It goes into `MultiTaskCollator`, but the **live branch never uses it**. Tokenization happens in the workers with the COND_VLM_CKPT processor (see below).
- `TN:1436-1463`: `build_mixture(cfg, processor, batch_size=per_device_bs, base_seed=mixture_seed)`. The run must pass `--max_steps` (`TN:1450`), and the code sets `accelerator_config.dispatch_batches=False` (`TN:1459-1463`).
- `TN:1034`: `remove_unused_columns=False`.
- `build_mixture` (`mixture.py:293-355`) imports `trellis2_blip3o.data.tasks`, which registers every task (`tasks/__init__.py:4-7`). For each task it calls `cls(**s.args)` with **only the yaml args**, then builds `MixtureIterableDataset` and `MultiTaskCollator`.
- DataLoader: HF `Trainer._get_dataloader` (transformers 5.2). For an IterableDataset it passes batch_size, collate_fn, num_workers, `pin_memory` (default True) and `persistent_workers` (default False). It passes **no** `prefetch_factor` (so torch uses 2), **no** `worker_init_fn`, and `drop_last` is irrelevant. The multiprocessing start method is fork (Linux default).

### 1.2 Per-item (`data/tasks/threed.py`, `data/tasks/threed_textonly.py`)
- `__getitem__` (`threed.py:428-444`) → `_load_one`.
- **T real** = `text_to_3d_weighted` → `WeightedTextTo3DDataset._load_one` (`threed_textonly.py:68-109`): voxel precheck → weighted t-key draw → `prep_t` → `_load_ss` → `_attach_slat_targets`.
- **T synth** = `text_to_3d` → `threed.py:494-504`: `_pick_caption_idx` → `prep_t`.
- **I1** = `image_to_3d` → `threed.py:460-471`: uniform view → `prep_i1`.
- **IM** = `multi_image_to_3d` → `threed.py:475-487`: batch-locked n → `pick_im_views` → `prep_im`.
- Item dict emitted by the live path: `{"_task", "id": sha, "_live_prep": {...}, "target_ss_latent": (8,16,16,16) f32, ["target_shape_slat_512_item": {"coords": (N,3) int32, "feats": (N,32) f32}], ["target_tex_slat_512_item": {"x_0": SparseTensor, "concat_cond": SparseTensor}], ["tex_valid": bool], ["align_points": (10000,3) f32]}`.
- `text_to_3d_weighted` stamps `_task="text_to_3d"` (`threed_textonly.py:105`), not its own registry name. That routes it to the shared collate and keeps the model's "text has no DINO" exemption (`blip3o/model/language_model/trellis_native_vlm.py:1172-1175`). Per-task loss logging therefore merges real and synth text (`TN:702`).
- `_live_prep` payloads (CPU half, `live_cond_batch.py`):
  - `prep_t` (`:291-309`): `{"input_ids", "modality": "t"}`.
  - `prep_i1` (`:160-189`): `{"input_ids" (1054,), "pixel_values" bf16 (4096,1536), "image_grid_thw" (3,), "dino_px" (3,512,512) f32 in [0,1], "view"}`.
  - `prep_im` (`:205-241`): `{"input_ids", "pixel_values", "image_grid_thw" (N,3), "dino_px" (N,3,512,512), "views", "modality": "im"}`.
  - The processor is `AutoProcessor(COND_VLM_CKPT)`, cached once per worker process (`_worker_proc`, `:112-123`). `V22_CKPT = os.environ.get("COND_VLM_CKPT", _V22_DEFAULT)` is read **at import time** (`live_cond.py:54`).

### 1.3 Collate (`vlm_collate.py:151-250`, the branch taken when `_live_prep` is in batch[0])
- `out = {"_live_prep": [prep…]}` (`:153-154`); `_task` from batch[0] (`:195`).
- `target_ss_latent = stack` (`:197-199`) → (B,8,16,16,16).
- `target_shape_slat_512 = SLat.collate_fn(...)["x_0"]` (`:200-202`). This prepends the batch index to coords. It is gated on **batch[0] only**.
- tex: only when **every** item has `target_tex_slat_512_item` (`:207-221`). Then `SLatPbr.collate_fn` → `target_tex_slat_512`, `tex_concat_cond` (via sparse_cat), plus `tex_valid` (B,) bool if any item carries it. A mixed batch drops tex and warns once (`:222-229`).
- `align_points` are stacked all-or-none (`:245-249`).
- Not reached in v12: the `cond_hidden`/`dino_hidden` cached branch and the whole processor branch (`:252-365`).

### 1.4 GPU half (outside data/, but part of the contract)
- `TrainCondEncoder.encode` (`live_cond_batch.py:497-579`) runs one Qwen forward with right padding and takes `hidden_states[-1]`. `keep = attention_mask & ~boiler_ids`. For T rows, hiddens are **compacted to keep-only** (`:549-553`); I1/IM rows keep every token plus the mask. IM gets `qwen_view_ids`, and I1/IM get `qwen_img_rc` (cell-centre normalised coords, -1 for text). DINO normalises with ImageNet mean/std, and `dino_view_ids = arange(K).repeat_interleave(N_d)`, i.e. **ordinals 0..K-1, not render camera ids**. The `views`/`view` fields are dropped by `_pack_live_conds`.
- `_pack_live_conds` (`TN:411-457`) pads with `collate_cached` (`vlm_cache.py:129-140`). `qwen_view_ids` and `qwen_img_rc` are padded with -1, `dino_view_ids` with 0.
- `_LiveCondPrefetch` (`TN:471-531`) encodes batch N+1 on a side stream before it yields batch N. `LIVE_COND_PREFETCH=0` switches to inline encoding (`TN:583-612`).

---

## 2. Per-task behaviour actually used by v12

### 2.1 Filters at construction (`threed.py:169-203`)
- The whole manifest is read with `json.loads` into a Python list of dicts, once per task (so pool1800k is loaded **twice**, for I1 and IM).
- `min_aesthetic` keeps records whose top-level `aesthetic_score` is **None** or ≥4.5 (`:180-186`). About 53% of capT1800k rows have no score, so they bypass the filter. Measured counts from `runs/cache_logs/v12s3_4n_r0.log`:
  - capT1800k: 1,469,558 → 1,446,027
  - pool1800k: 1,469,650 → 1,446,119
  - capT_synth: 481,995 → 481,995 (no scores)
- `require_caption` defaults to True for mode T (`:165-167,187-188`).
- The voxel cap is **not** applied at load. It is enforced lazily (§2.5).

### 2.2 View picking
- **I1** (`threed.py:464`): `rng.integers(0, rec.n_views)`, **uniform over all 16 views**, including the 3 below-ground views. If the file is missing, `prep_i1` falls back to the first sorted render in the dir (`live_cond_batch.py:168-174`). That is an `os.listdir` of a single asset dir, which is fine.
- **IM**: n is drawn **once per batch** from `task_rng` via `set_batch_params` (`threed.py:222-232`, called at `mixture.py:210-213`), with p = `im_view_probs` [0.15, 0.25, 0.6] for n = 2, 3, 4. Views come from `pick_im_views` (`live_cond_batch.py:192-202`): k distinct views, sorted, weighted by `IM_VIEW_WEIGHTS` (`:64-67`: views 0-2 at 0.15, 3-4 at 0.6, 5-11 at 1.0, 12-15 at 0.5), with a fresh per-sample rng. `prep_im` **silently drops missing view files** and raises only if none exist (`:219-227`), so a row can carry fewer views than the batch lock.
- Image preprocessing contract:
  - **Qwen, I1**: sees the **full uncropped render** with `.convert("RGB")`, i.e. alpha is dropped rather than composited (`:178`), about 1024 vision tokens.
  - **Qwen, IM**: each view is downscaled by `cap_image` to `256 × px_per_tok` pixels (`:217,231`), about 256 tokens per view.
  - **DINO, both modes**: `_dino_image` (`:134-157`) calls the dataset's own `_alpha_crop` (tight square bbox of alpha > 0.8·255, `threed.py:395-417`) and `_composite_black` (`threed.py:388-393`), then LANCZOS resize to **512** (`DINO_SIZE_I1`/`DINO_SIZE_IM`).
  - `crop_to_object` in the yaml only affects `_load_views`, which is dead in v12. The live DINO path **always** crops, and the live Qwen path **never** does.

### 2.3 Caption tier sampling and templates
- Caption order is fixed as `[long, medium, short, long+texture]`. Every sampled row had 4 non-empty captions; median word counts are 31/11/3/53 for real and 25/9/4/31 for synth.
- **text_to_3d_weighted (real)**: `_t_probs` come from yaml `caption_weights` = {t000: .4, t001: .12, t002: .08, t003: .4} (`threed_textonly.py:56-62`). The yaml wins over the cache `_meta` and over the docstring fallback (.35/.20/.10/.35). If the drawn tier is ≥ len(caps), it falls back to `_t_fallback.get(key, "t000")`. `_t_fallback` is **empty** in live mode (no cache root), so it falls back to **long** (`:99-104`).
- **text_to_3d (synth)**: `_pick_caption_idx` (`threed.py:238-252`) uses weights [.45, .15, .15, .25], **renormalised over the first n_caps**. Tail weights are 0 if n_caps > 4, and it is uniform if weights are unset. The two T classes therefore have **different fallback policies**. Both index into the **non-empty** caption list, so an empty middle caption would shift tier meanings. No empties were seen in the samples.
- Prompt template (`live_cond_batch.py:72,79-81`): `"[3D Gen] " + TXT_PROMPTS[(int(sha[:8],16) + cap_idx) % 3].format(c=caption)` with 3 templates. The choice is **deterministic per (asset, caption)** for cache parity, not random. It requires a hex sha. `apply_chat_template(..., add_generation_prompt=True)` (`:305-307`).
- I1 prompt: `"[3D Gen] <|vision_start|><|image_pad|><|vision_end|>\nReconstruct this object in 3D."` (`live_cond.py:55-56`). IM uses the same prompt with n image tokens (`live_cond_batch.py:75-76`). Both use add_generation_prompt=True. The processor branch of vlm_collate uses **False**, a divergence that only matters if that branch is revived.

### 2.4 Latent loading and normalization (`threed.py:300-337`)
- `ss_latent_64` npz key is `z`, falling back to `latent` (`:302`); shape (8,16,16,16) f32. SS normalization is **None**, because the TRELLIS SS config has no `normalization`, so it is identity (verified).
- `shape_latent_512` has keys `coords` (N,3) **uint8** (converted to int32) and `feats` (N,32) f32. Normalised with `shape_norm` from `slat_flow_img2shape_dit_1_3B_512_bf16.json` → `dataset.args.normalization` (32 channels).
- `pbr_latent_512` has the same keys. `_load_tex` loads pbr feats normalised by `pbr_slat_normalization`, **re-reads the shape npz** and normalises it by `shape_slat_normalization` (verified bit-identical to shape_norm), prepends a zero batch column, and **asserts coords are equal** (`:333-336`). A failure raises → resample; this was seen in the S3 log.
- Norm stats come from `load_norm_stats` (`tr2_modules.py:166-195`), which returns **None silently if the config file is missing**, disabling normalization. The config root is `tr2_modules.py:153-163` (derived from `_paths`, with a hardcoded `/weka/...` fallback).
- `ss_only=True` (ss tower) skips shape and pbr entirely except for the ROAD `align_points` coords read (`threed.py:628-630`).
- In the shape tower (no placeholder), about 25% of pool1800k/capT1800k rows lack `pbr_latent_512`. Every row that has one still loads it (`load_tex` defaults True; the base class always loads it), which is **wasted I/O**, and mixed batches drop tex. **[spec]** The shape stage ignores tex (per the comment at `vlm_collate.py:203-206`).

### 2.5 Voxel cap 8192 → resampling
- Base class (`threed.py:641-647`): after loading shape, `N > max_slat_tokens` raises ValueError.
- `text_to_3d_weighted` adds a **cheap precheck** (`threed_textonly.py:78-86`) that reads only `coords` before any other I/O.
- The cap is **not applied when ss_only**, so the ss tower trains on over-cap assets while shape, tex and unify do not.
- In `voxels_v11.npz`, 27,359 of 2,236,184 entries (1.2%) exceed 8192 (median 2104, p99 8586, max 32812). The S3 r0 log shows **6,561 "load failed" lines, almost all of them the voxel cap**.

### 2.6 Tex placeholder (unify only) (`threed.py:657-682`)
- If `tex_placeholder` is set, the item has shape, has no pbr, and `ss_only` is off, the dataset emits `x_0 = zeros(N,32)` on the shape's coords (batch column prepended), with `concat_cond` set to the **shape feats in shape_norm space** (claimed bit-identical to tex_shape_norm), and `tex_valid=False`. Real pbr rows get `tex_valid=True`.
- The model masks placeholder voxels out of the tex loss and forces t_x=1 (model side). The pool has about 75% pbr coverage.

### 2.7 Aesthetic filter
- Construction-time only (§2.1). The `vlm` sub-dict (`structural_score`, `category`, …) and `vlm_passed` are **not read** by v12 code.

### 2.8 Error, retry and resample (`threed.py:428-444`)
- `except Exception` covers **everything**: missing or corrupt npz (EOFError, BadZipFile), a missing render (FileNotFoundError from prep_im), a PIL UnidentifiedImageError, the coords-assert, and the voxel cap.
- After each of 8 guarded attempts it jumps to a **uniformly random index drawn from an unseeded `np.random.default_rng()`**. That bypasses the epoch permutation and the balancer's cost accounting. The 9th attempt is unguarded and surfaces the error.
- Non-cap failures seen in `v12s3_4n_r0.log`: EOFError ×8, missing renders ×7, UnidentifiedImageError ×1, coords mismatch ×2.

### 2.9 ROAD align points (S2 SS / SS-cont only)
- `ROAD_ALIGN_POINTS` env (`TN:1082`, default 10000) → `_attach_align_points` (`threed.py:597-624`). It uses the shape_latent_512 coords (`with np.load`) → `road/points.py:14-19` (voxel centres in [-0.5,0.5]^3, unseeded rng, with replacement if N < 10k).
- It is hard-pinned to the 512 grid (32³). It raises if the record lacks `shape_latent_512`.

---

## 3. Manifest schema

### 3.1 Fields read by v12 code
| field | used by | notes |
|---|---|---|
| `sha256` | id, voxel-table key, T template hash (`int(sha[:8],16)`), `text_to_3d_weighted` (**hard `rec["sha256"]`**) | 64-hex (TexVerse/SWH), 32-hex (Sketchfab uid), **20-hex (synth)**; all must be hex |
| `captions` | T tasks | list of 4, fixed tier order |
| `ss_latent_64` | all | required (`rec[...]` KeyError → resample) |
| `shape_latent_512` | shape/tex/unify, ROAD | present on every sampled row |
| `pbr_latent_512` | tex, unify | about 75% of 1800k pools, 100% of pbr pools and synth |
| `renders_dir` | I1/IM | absent on synth rows |
| `n_views` | I1/IM | 16 everywhere; default 16 if missing |
| `aesthetic_score` | filter | top-level; None → keep |

Present but unused in v12: `subset`, `shape_latent_1024` (partial coverage, a trap if `slat_resolution: 1024`), `cond_cache` (partial), `vlm`, `vlm_passed`, and synth `concept/branch/tier/gen_model`.

### 3.2 Manifests v12 needs (all under `manifests/splits/`, built 2026-09-06)
| file | size | rows (~) | used by |
|---|---|---|---|
| `capT1800k_train_texfixed_ff.jsonl` | 2.52 GB | 1,469,558 | ss/shape/unify T-real |
| `capT_synth_train_texfixed_ff.jsonl` | 0.46 GB | 481,995 | ss/shape/unify T-synth |
| `pool1800k_train_texfixed_ff.jsonl` | 1.61 GB | 1,469,650 | ss/shape/unify I1 + IM |
| `capT800k_pbr_train_texfixed_ff.jsonl` | 1.06 GB | ~553k (est.) | tex T-real |
| `capT_synth_pbr_train_texfixed_ff.jsonl` | 0.46 GB | ~485k (est.) | tex T-synth |
| `pool800k_pbr_train_texfixed_ff.jsonl` | 0.70 GB | ~554k (est.) | tex I1 + IM |

Plus `voxels_v11.npz` (arrays `sha` <U64, `vox` int32; 2.24M rows; 100% coverage on sampled rows including synth). It was built by `scripts/_tmp/build_voxel_table.py`, which **scans latent dirs**. BLIP3D should build it from the manifests instead.

`_ff` lineage (per memory, not re-verified): evalclean → texfixed (renders repointed to `renders_cond_refix`, new captions) → `_ff` (pbr repointed to `_flatfix/<Subset>/<sha>.npz`). The sampled `_flatfix` npz files have `feats`/`coords` with coords equal to shape.

### 3.3 Where paths are hardcoded
- **Manifests**: absolute paths in the yamls (`/fsx/home/.../manifests/splits/...`).
- **Asset paths** are absolute inside the manifests:
  - `/fsx/home/weikai.huang/3dgen/data/trellis2/<Subset>/{renders_cond|renders_cond_refix}/<sha>`
  - `…/ss_latents/ss_enc_conv3d_16l8_fp16_64/<sha>.npz`
  - `…/shape_latents/shape_enc_next_dc_f16c32_fp16_512/`
  - `…/pbr_latents/tex_enc_next_dc_f16c32_fp16_512/`
  - `…/_flatfix/<Subset>/`
  - `/fsx/home/weikai.huang/3dgen/data/coverage/synth/latent/<id>/{ss,shape,tex}.npz`
- **TRELLIS norm configs**: `tr2_modules.py:153-163`.
- **Default cond VLM**: `live_cond.py:48` (`_V22_DEFAULT`); v12 overrides it with the `COND_VLM_CKPT` env.
- **Balance table**: the env in the launcher.
- **Other**: `IM_VIEW_WEIGHTS` and `TXT_PROMPTS` are constants in `live_cond_batch.py:64-72`, and the ROAD point count is a CLI flag exported as an env var.

---

## 4. Keep / drop for everything under `data/` (plus the pieces of vlm_collate and live_cond_batch the data path touches)

"Referenced by v12" means reached by one of the four v12 yamls on the live_cond path.

| module :: symbol | v12? | verdict | evidence |
|---|---|---|---|
| `__init__.py` | yes | keep (rewrite) | imports tasks (incl. chat) for registration |
| `registry.py` register_task/get_task | yes | keep, or replace with a 3-entry dict | mixture.py:268,333 |
| `base.py` TaskDataset | no | **drop** | not subclassed anywhere (grep); convention only |
| `mixture.py` TaskSpec, _temperature_normalize, MixtureIterableDataset, MultiTaskCollator, Mixture, build_mixture, _mk_batcher | yes | **keep** (core) | TN:1437 |
| `mixture.py` granularity="item" branch and num_workers>1 warning | no | drop | all yamls use batch; the warning is stale per its own note (:98-103) |
| `rank_aware.py` install_iterable_shard_passthrough | yes | **keep** (critical) | mixture.py:35,41 |
| `balanced_assign.py` VoxelTable, TargetBatcher | S3 only | keep (optional feature) | mixture.py:157,364 |
| `balanced_assign.py` balanced_partition, rank_slice, TargetBatcher.drain_ready | no | drop | no references (grep) |
| `tasks/threed.py` _ThreeDTaskBase: manifest load/filters, _draw_n_views, set/clear_batch_params, _pick_caption_idx, _sample_n_views, _load_ss/_load_shape/_load_tex, _attach_slat_targets (incl. placeholder), _attach_align_points, __getitem__, live branch of _load_one | yes | **keep** (merge) | |
| `tasks/threed.py` `_alpha_crop`, `_composite_black`, `MIN_IMG_PX` | yes (called by live_cond_batch._dino_image :142) | **keep**, move to `views.py` | |
| `tasks/threed.py` cached_hidden_root branch (:128-158, :282-298, :472-578), `_im_combo_sizes`, GEOTEX_IM_VIEW_SUBSET, hardcoded 292-token qwen_view_ids layout, packed_root/_reader/_src packed path, raw-PIL path (:580-595) + `_load_views`, `collate_fn` kwargs max_tokens_single/token_budget/target_tokens_per_view | no | **drop** | v12 is live_cond everywhere |
| `tasks/threed.py` TextTo3DDataset/ImageTo3DDataset/MultiImageTo3DDataset | yes | keep (as modes of one class) | |
| `tasks/threed_textonly.py` WeightedTextTo3DDataset | yes (T-real) | **merge into T mode** (weights + explicit fallback policy); drop the t-key cache path, `load_tex`, `_t_fallback` meta, and the `_task` stamping hack | |
| `tasks/threed_textonly.py` voxel precheck | yes | keep the idea; better: `n_vox` field in the manifest, filtered at load | |
| `tasks/threed_balanced.py` CategoryBalancedTextTo3DDataset | no | **drop** | no v12 yaml names `text_to_3d_balanced` |
| `category_balance.py` (RFS) | no | **drop** | only threed_balanced imports it |
| `tasks/chat.py` vqa/grounding/text_sft | no | **drop** | only `configs/mix_full.yaml` uses it |
| `streaming_task.py` StreamingImageTo3D (MDS) | no | **drop** | only the `--mds_root/--mds_shards` path (TN:1409-1435) and mfu/ablation scripts |
| `packed_reader.py` PackedShardReader | no | **drop** | only via `packed_root` (unset) |
| `tap_task.py`, `tap_ss_task.py` | no | **drop** | only scripts/ablate_*.py |
| `vlm_collate.py` collate_vlm_3d live/cached branch (:151-250) | yes | **keep**, as `collate.py` (live only) | |
| `vlm_collate.py` processor branch (:252-365), upscale_image, set_default_target_tokens_per_view, MAX_TOKENS_SINGLE/TOKEN_BUDGET, repa_target stacking, dino_images emission | no (training) | drop from data/ | **[spec]** check whether the BLIP3D inference harness still needs NativeVLMCollator |
| `vlm_collate.py` cap_image, px_per_tok, boiler_ids | yes (prep_im, TrainCondEncoder) | keep (move to cond/) | |
| `vlm_cache.py` collate_cached | yes (_pack_live_conds) | keep as a tiny pad helper | |
| `vlm_cache.py` everything else | no | drop | the live branch still imports view_key etc. at threed.py:458, but only as dead imports |
| `live_cond_batch.py` prep_t/prep_i1/prep_im, pick_im_views, _dino_image, _worker_proc, txt_prompt/im_prompt, constants | yes | **keep** (the CPU half, data side) | |
| `live_cond_batch.py` TrainCondEncoder, stage_to_shm, and `TN` _pack_live_conds/_LiveCondPrefetch | yes | keep, in a `cond/` module (GPU side) | |
| `live_cond_batch.py` prep_i1_image/prep_im_images | demo only | keep in inference | |
| `dataset.py`, `dataset_native.py` (TR2NativeVLMDataset) | no | **drop** | imported unconditionally at TN:322 but used only in the legacy else-branch |

---

## 5. Hidden contracts

1. **Rank-synced task draw.** `task_seed = (base_seed*1_000_003) ^ (worker_id+1)` has no rank term (`mixture.py:166-167`), so the same worker id on every rank draws the same task sequence. Per-batch params (IM n_views) are drawn from the same `task_rng` (`:211-213`), so every rank gets the same n. This depends on:
   - identical `num_workers` on all ranks;
   - torch's in-order round-robin worker consumption (batch k comes from worker k mod W);
   - nothing per-rank ever consuming `task_rng`.

   If it breaks, ranks train different parameter sets → a DDP hang or silent gradient corruption (`rank_aware.py:34-44`). Debug hook: `S3_TASK_DEBUG=1` (TN:677-684).
2. **Per-rank index streams.** `within_seed = base*1_000_003 ^ rank*9176 ^ (worker_id+1)`, and per task `^ (i*7919+1)` (`:171-175`). I checked 64 ranks × 10 workers × 4 tasks: no seed collisions.
   - Each worker holds its **own** permutation of each task and burns the indices of batches owned by other workers (`:196-204`, `:239`). So a worker actually uses only a 1/W thinning of its own permutation.
   - Epochs are not a partition across workers or ranks: in effect this is sampling with replacement across (rank, worker).
   - The "consistent with shared task_rng" comment (`:193-195`) is about task alignment, not data coverage.
3. **Seeds and resume.**
   - `base_seed = --mixture_seed`, which defaults to 0 and is never set in v12. With `--ignore_data_skip True`, **every launch or resume replays the identical task sequence and index order from step 0**. That includes the S2 4-node continuation (ranks 0-15 re-see their original order) and S3 relaunches.
   - Per-sample randomness (view choice, caption tier, IM combo, resample target, ROAD points) uses **unseeded `np.random.default_rng()`**, so it is non-reproducible. That also means fork gives no duplicated-RNG problem.
   - The CUDA RNG is reseeded per rank by `RankRNGCallback` (TN:212-252); the CPU/data RNG is deliberately left alone.
4. **accelerate re-sharding fix.** `IterableDatasetShard.__iter__` is monkeypatched to pass through when `dataset._rank_sharded` (`rank_aware.py:67-94`; `mixture.py:109`). Without it, (N-1)/N of every load is discarded and the per-rank task draws are split apart. It is applied at import of mixture.py and streaming_task.py, so workers see it under fork and under spawn. It is pinned to accelerate 1.7.0 internals.
5. **`dispatch_batches=False`** (TN:1459-1463). Without it, rank-0 broadcast would fail on the `_task` str and on SparseTensors.
6. **Iterable vs map.** Tasks are map-style Datasets (random index access plus `records`). The mixture is the only IterableDataset. HF therefore gets no sampler, no `worker_init_fn` and no prefetch_factor (torch default 2). `--max_steps` is mandatory.
7. **Workers.**
   - 10 in S1/S2, 8 in S3; fork start method.
   - Each worker lazily builds its own AutoProcessor (COND_VLM_CKPT). `RANK` is read from env inside `__iter__` (`:138`).
   - **Memory [spec]**: records are Python dicts, about 3.9 KB each (measured on 20k rows), so capT1800k is about 5.6 GB, pool1800k about 3.6 GB (loaded twice) and synth about 1 GB, roughly 14 GB per rank. Copy-on-write pages are dirtied by refcounting across the W+1 processes per rank.
8. **Voxel balance format and cost** (S3 only).
   - Table: `np.load(path)` with `sha` (U64) and `vox` (int) arrays (`balanced_assign.py:20-33`). A missing sha falls back to the median.
   - Per-item cost = `table.median + vox(sha)`; the median stands in for the fixed SS cost (`mixture.py:227-228`).
   - Per-task target = `per_batch * (median + mean vox over that task's records)` (`:366-368`).
   - `TargetBatcher(buf_size=max(24, per_batch*GEOTEX_BALANCE_BUF[24]), seed=rank*7919+13)` (`:375-377`) gives 96 at bs4.
   - `_pick`: a greedy seed, then 400 random k-combos, keeping the one with minimum |sum - target| (`balanced_assign.py:96-118`). Each rank balances **independently toward the same constant**, with no communication.
   - **Finding.** I simulated it with the real table distribution: 6,000 batches, buf 96, k 4. Items with cost over about 13.9k (vox over about 11.8k, roughly 0.2% of assets) are **never popped**. After 6k batches, 32 of 96 buffer slots were permanently stuck. So "every sample consumed exactly once" (`balanced_assign.py:79-81`) is false for the tail.
     - The stuck items are all over the 8192 cap and would have been rejected anyway, so the only cost is a slowly shrinking effective buffer. For S3 that is about 13 stuck slots per worker-task by 17k steps **[spec, extrapolated]**.
     - Items with vox between 8193 and about 11.8k *are* popped, fail the cap, and are replaced by a uniform random index, which silently un-balances that slot.
9. **Batch homogeneity requirements** that the collate relies on:
   - one `_task` per batch (asserted, `mixture.py:261-267`);
   - shape gated on batch[0];
   - tex all-or-none, which the placeholder satisfies;
   - align_points all-or-none (raises);
   - T rows carry no DINO. `_pack_live_conds` decides "has dino" from conds[0], so a mixed-modality batch would silently drop or misalign segments.
10. **`_task` routing contract.** The model special-cases `_task == "text_to_3d"` for the fusion/DINO check. That is why `text_to_3d_weighted` lies about its name.
11. **The mixture drops CLI data flags.** `max_slat_tokens`, `slat_resolution` and `ss_only` come only from the yaml (`mixture.py:334-343`). Live-cond enablement is inferred from the datasets (`TN:460-468`).
12. **Norm-stats silent-None** (`tr2_modules.py:182-189`). A missing TRELLIS config means training runs on un-normalised latents with no error.

---

## 6. Proposed BLIP3D `data/` layout and mix yaml schema

```
blip3d/data/
  __init__.py      build_mixture(), public types
  manifest.py      Record schema (sha, captions[4], ss, shape, pbr, renders_dir, n_views, aesthetic, n_vox),
                   load_manifest(path, filters) → compact columnar store (numpy/pyarrow; no per-row dicts),
                   validation (hex sha, tier count, required fields per task type), loud counts
  latents.py       NormStats (fail loudly if config missing), load_ss/load_shape/load_pbr (one shape read shared
                   between shape and tex targets), tex placeholder, voxel cap
  views.py         alpha_crop, composite_black, min-px guard, pick_i1_view (uniform), pick_im_views (IM_VIEW_WEIGHTS)
  prompts.py       TXT_PROMPTS, txt_prompt(sha, ci), PROMPT_I1, im_prompt(n), CaptionSampler(weights, fallback)
  cond_prep.py     CPU half: prep_t / prep_i1 / prep_im, per-worker processor cache (from live_cond_batch.py)
  task.py          ThreeDTask(mode ∈ {t, i1, im}) — single class, emits {"task", "modality", "id", "_live_prep", targets…}
  mixture.py       MixtureIterableDataset (rank-synced task draw, per-rank cyclers, batch params, resample policy),
                   MultiTaskCollator
  balance.py       VoxelCost (from manifest n_vox, table fallback) + TargetBatcher (with an eviction/age cap)
  collate.py       collate_live (the live/cached branch of collate_vlm_3d only; tex all-or-none + tex_valid)
  dist.py          accelerate IterableDatasetShard pass-through patch (+ version assert)
blip3d/cond/       TrainCondEncoder, pack_live_conds, LiveCondPrefetch (GPU half; not data)
```

Design notes:
- Route on `modality` (t/i1/im) rather than on the task name. Then the task name is free for logging (t_real vs t_synth) and the `_task` stamping hack goes away.
- Replace `ss_only`/`load_tex`/`tex_placeholder` with `targets: [ss, shape, tex]` plus `tex_missing: placeholder|drop`. That also removes the wasted pbr reads in the shape tower.
- Put `n_vox` in the manifest. The cap then filters at load (equivalent in distribution to v12's uniform resample), and the balance table plus the precheck read go away. Keep `ss` tower semantics explicit: v12's ss tower did **not** apply the cap.
- Seed per-sample RNG from (base_seed, rank, worker, step, slot) if reproducibility is wanted. That is a behaviour change versus v12; flag it.

Mix yaml schema:

```yaml
version: 1
defaults:                       # merged into every task
  slat_resolution: 512
  min_aesthetic: 4.5            # None-score rows kept (v12 semantics)
  max_slat_tokens: 8192         # applied only when 'shape' or 'tex' in targets (v12 semantics)
  targets: [ss, shape, tex]     # ss tower: [ss]; shape tower: [ss, shape]; tex: [ss, shape, tex]
  tex_missing: placeholder      # placeholder (unify) | drop (towers; batch loses tex)
  cond: {mode: live, fuse_dino: true, dino_size: 512, im_qwen_tok_per_view: 256}
tasks:
  - name: t_real
    modality: t
    weight: 0.6
    manifest: /…/capT1800k_train_texfixed_ff.jsonl
    captions: {weights: [0.4, 0.12, 0.08, 0.4], fallback: long}      # order [long, medium, short, long+texture]
  - name: t_synth
    modality: t
    weight: 0.1
    manifest: /…/capT_synth_train_texfixed_ff.jsonl
    captions: {weights: [0.45, 0.15, 0.15, 0.25], fallback: renormalize}
  - name: i1
    modality: i1
    weight: 0.15
    manifest: /…/pool1800k_train_texfixed_ff.jsonl
    views: {sampler: uniform}
  - name: im
    modality: im
    weight: 0.15
    manifest: /…/pool1800k_train_texfixed_ff.jsonl
    views: {sampler: weighted, n_probs: {2: 0.15, 3: 0.25, 4: 0.6}, batch_locked: true}
sampling: {granularity: batch, temperature: 1.0, seed: 0}
balance: {enabled: true, cost: manifest_n_vox, table: null, buf_mult: 24}   # v12: S3 only
loader: {num_workers: 8, resample_tries: 8}
```

Note that `fallback: long` versus `renormalize` reproduces the current, divergent v12 behaviour of the two T classes.

---

## 7. Risks and open questions

1. **Broad `except Exception` resampling** (`threed.py:439`) hides systematic bugs until 9 consecutive failures. The S3 r0 log alone has 6,561 resamples, mostly the voxel cap. BLIP3D should resample only on a whitelist (IO errors, cap) and count failures per task, alarming above a rate threshold.
2. **No seed variation across relaunches** (`mixture_seed=0` + `ignore_data_skip`). Every relaunch re-trains the same early index order. BLIP3D should derive the seed from (run seed, resume step), or skip indices cheaply, since cyclers yield indices without loading.
3. **TargetBatcher tail starvation** (§5.8). Harmless today only because the stuck items are over the cap. Add an age-based forced pop, or pre-filter over-cap items.
4. **Host RAM** from dict-per-row manifests loaded 4× per rank, plus worker COW **[spec; unmeasured RSS]**. Switch to columnar storage.
5. **IM view lock can break**: `prep_im` drops missing files silently. **I1 samples the below-ground views 0-2 uniformly** while IM down-weights them. Both are the v12 contract; keep them as-is, but they are open design questions.
6. **The Qwen I1 image is `convert("RGB")` without alpha compositing** (`live_cond_batch.py:178`), unlike DINO (composited on black). It is inherited cache parity, and the demo must match it. Worth confirming what RGB the renders hold under alpha=0 **[spec]**.
7. **The shape tower loads pbr for no benefit.** About 10% of its batches (0.75^8) arrive with tex targets. **[spec]** The shape-stage model ignores them; verify in the model audit.
8. **`shape_latent_1024` is only partially populated.** `slat_resolution: 1024` would silently drop tex for most batches and change the shape pool. BLIP3D should validate this at load.
9. **The comment at `vlm_collate.py:224-228` is wrong** ("every manifest row has pbr_latent_512"). It is about 75% in the 1800k pools.
10. **Eval contamination**: per the yaml comment, the texfixed_ff pools derive from `*_evalclean`, but eval sets have geometry-level duplicates in the training pool (memory note 2026-09-15). This was **not re-verified here**, including whether texfixed_ff ⊆ evalclean by sha.
11. **The two T classes differ in caption fallback** (long versus renormalise). Decide deliberately in BLIP3D.
12. **accelerate monkeypatch is version-pinned** (1.7.0). Assert the version, or own the DataLoader wrapping.
13. **`COND_VLM_CKPT` is read at import time.** Workers inherit it via fork. If it is set after import, the workers and the GPU encoder could disagree on the processor. BLIP3D should pass the path explicitly through the config.
14. **The voxel table was built by walking latent dirs**, which is the opposite of the "don't walk the trees" rule. Rebuild it from the manifest `shape_latent_512` paths, or add `n_vox` at manifest build time.
15. **The synth sha is 20-hex, not a content hash.** It collides with nothing in the table today, but it is a different id namespace. Keep sha as an opaque string id and keep the hex requirement for the template hash.
