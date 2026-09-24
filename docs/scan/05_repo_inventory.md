# 05 — BLIP3o repo inventory, environment, tooling (for the BLIP3D extraction)

Scope: repo `/fsx/home/weikai.huang/3dgen/model/BLIP3o` (branch `BLIP3o-NEXT`, HEAD `25c4751`, 416 commits, remote `github.com/weikaih04/BLIP3o`), plus the out-of-repo tooling that v12 actually ran on. Read-only audit done 2026-09-24. Huge dirs (third_party, manifests, runs) were not walked. **[spec]** marks inference that I did not verify by running anything.

## 0. Findings that change the migration plan (read these first)

1. **The TRELLIS.2 that v12 uses is NOT the copy inside the repo.** `trellis2_blip3o/_paths.py:14-16` resolves `_WORKSPACE = dirname(REPO)` → `/fsx/home/weikai.huang/3dgen/model/third_party_3d_gen/TRELLIS.2`. That is a fork (`github.com/weikaih04/TRELLIS.2`, branch `blip3o-next`): pristine upstream `b189e69` + 3 commits (cond_mask/attn_mask plumbing for variable-length VLM cross-attn, data_toolkit changes plus an o-voxel build tweak, fused Triton norm-modulate/gate-residual `2ed033d`). 18 files, +587/−62. **HEAD `2ed033d` has not been pushed** (origin/blip3o-next = `9cfb2b3`), and there are dirty data_toolkit edits plus untracked `data_toolkit/datasets/`. `o_voxel` is pip-editable from this fork too. The in-repo `third_party_3d_gen/TRELLIS.2` (185 MB, stock upstream `75fbf01` plus a 2-file DINOv3 patch) is **dead weight**; nothing puts it on sys.path.
2. **The working tree is dirty with live-run code:** `train_native.py` (+33, EMA `ema_init_from/ema_init_updates` for the SS 300k continuation), `trellis2_blip3o/geotex_sampler.py` (+9), `trellis2_blip3o/unified_geotex.py` (+33/−8), `scripts/v12/monitor_sscont.sh`. These are byte-identical to `/fsx/data/weikai.huang/lc/patched/*`. Port from the working tree, not from HEAD, or commit first.
3. **Code that v12 needs is untracked or gitignored:**
   - `tests/` is in `.gitignore`. All 4 test files are untracked.
   - `scripts/_tmp/` is gitignored. 37 files are tracked (force-added earlier). **Untracked but load-bearing:** `scripts/_tmp/eval_v10_tasks.py` + `_tmp/_hf.py` (imported by eval_suite `text/run_geotex_v10_text.py:44` and `text/run_ss_swap_text.py:40`), `scripts/_tmp/build_voxel_table.py` (builds the `GEOTEX_BALANCE_TABLE` npz that S3 trains with), and the v12-era unit tests `test_cond_stamp.py`, `test_view_code.py`, `test_cond_extrapolation.py`, `test_road_*`, `test_xattn_*`, `test_balance.py`, `test_tex_mask*.py` and others.
   - v12 **S1/S2 launchers exist only in `/fsx/home/weikai.huang/_lc`** (`v12_s1.sh`, `v12_s1b.sh`, `launch_s1b.sh`, `v12_s2.sh`, `v12_s2_4n.sh`, `launch_s2.sh`). Only the S3 and SS-continuation launchers were snapshotted into `scripts/v12/`.
   - The v12 texfixed training manifests are produced by `/fsx/home/weikai.huang/_lc/texswap/mk_texfixed_manifests.py` (outside git).
   - The S3 mix and balance table come from outside the repo: `MIX=/fsx/data/weikai.huang/lc/configs/mix_v12_unify_texfixed.yaml` and `GEOTEX_BALANCE_TABLE=/fsx/home/weikai.huang/_lc/balance/voxels_v11.npz` (`scripts/v12/v12_s3_4n.sh`).
   - eval_suite: the whole `text/` dir and every `scripts/*v12*` runner are untracked in the eval_suite git.
4. **`scripts/` is a library, not just scripts.** `trellis2_blip3o/demo_pipeline.py:203-208,306-308`, `benchmarks/run_wild3dgen.py:73-119`, every v12 bench/readout runner and eval_suite runners import `scripts.eval_fusion_v22`, `scripts.eval_tex_v22`, `scripts.export_glb_v22`, `scripts.export_glb_fullchain` (which in turn import `scripts.eval_render` and `benchmarks.checkpoint`). Repo-wide counts: `eval_fusion_v22` is imported 101×, `export_glb_fullchain` 51×, `eval_tex_v22` 27×.
5. **The `blip3o` package cannot be dropped wholesale.** `train_native.py:323` imports `blip3o.model.language_model.trellis_native_vlm` (1468 lines: the actual 3D model wrapper), and `trellis_native_vlm.py:38` plus `unified_geotex.py:1565,1601,1689` import `blip3o.model.multimodal_decoder.builder` (TRELLIS.2 flow builders). Importing either triggers `blip3o/model/__init__.py`, which eagerly imports the **2D AR models** (`blip3o_qwen`, `_inference`, `_grpo`) → `blip3o_arch` → `multimodal_encoder.builder` → `ta_tok_encoder` → `tok/`. BLIP3D must cut that chain.
6. **Env hazard: numpy from the user site.** The `blip3o_trellis` env ships numpy 1.26.4, but `/fsx/home/weikai.huang/.local/lib/python3.10/site-packages/numpy` 2.2.6 (installed 2026-09-01, together with matplotlib 3.10.9) shadows it. No launcher sets `PYTHONNOUSERSITE=1`. Every run since about 09-01 has been on numpy 2.2.6. `requirements.txt` (torch 2.3.0 / transformers 4.51.3) and `requirements-freeze.txt` (says transformers 4.52.4) both disagree with the live env (transformers **5.2.0**).

---

## 1. Top-level map

Sizes are from `du -sh`. third_party and runs were not measured (third_party du timed out at 60 s; the caller quotes 13 GB / 117k files). "T" means git-tracked (count), "U" means untracked/ignored.

| path | size | git | purpose | verdict |
|---|---|---|---|---|
| `train_native.py` | 90 KB | T (dirty) | **the v12 trainer** (HF Trainer + DeepSpeed; EMA callback, geotex/unified init guards, ROAD wiring, live cond) | **port** → `blip3d/train/` (split it: args, EMA, init/guards, loop) |
| `trellis2_blip3o/` | 2.9 MB | T 58 files, ~15k lines | the 3D package (connector, flow_heads, unified_geotex, geotex_sampler, live_cond*, eval_cond, view_code, data/, road/) | **port** (selectively, see §7 and the other agents' reports) |
| `blip3o/` | 943 KB | T 19 | forked BLIP3o AR package. **3D-relevant:** `model/language_model/trellis_native_vlm.py` (1468 lines), `model/multimodal_decoder/builder.py` (179, TRELLIS.2 flow builders). **2D-only:** `blip3o_qwen*.py`, `blip3o_arch.py`, `conversation.py`, `train/`, `data/dataset.py` (hardcodes `/fsx/home/jiuhai.chen/...`, line 175), `multimodal_encoder/ta_tok_encoder.py` | port the 2 files into `blip3d/models/`; **archive** the rest |
| `tok/` | 359 KB | T 10 | TA-Tok image tokenizer (2D AR) | **archive** (reachable only through `blip3o/model/__init__`) |
| `trl/` | 6.8 MB | T 156 | vendored TRL (GRPO for 2D RL) | **archive**; imported by no 3D code |
| `train.py` | 15 KB | T | pre-native trainer (the "setup A/B/C" era, Sana/BLIP3o-NEXT-Pretrain) | archive |
| `inference.py` | 3 KB | T | upstream BLIP3o T2I inference (`/fsx/home/jiuhai.chen/...`) | delete/archive |
| `gradio/` | 986 KB | T 2 | 2 demo images from upstream | delete |
| `figure/` | 20 MB | T 4 | upstream BLIP3o-NEXT README images | delete |
| `experiment/` | 203 KB | T 11 | Qwen3.5 era cache/train yamls (`qwen35_trellis2_training/`) | archive |
| `build_index.py` | 10 KB | T | builds `ready_v1.jsonl` (the oldest manifest layer) | archive |
| `setup.py` | 138 B | T | `blip3o_next` package, `find_packages()`, installed editable in the env | replace with `pyproject.toml` for `blip3d` |
| `requirements.txt` | 344 B | T | **stale** (torch 2.3.0, transformers 4.51.3, deepspeed 0.14.4) | delete; regenerate |
| `requirements-freeze.txt` | 4.7 KB | T | freeze from 2026-06-19; says transformers 4.52.4, live env is 5.2.0 | delete; regenerate from the live env (§4) |
| `README.md` | 2.7 KB | T | WilD3DGen readme + IM_QWEN_TOK_PER_VIEW convention table | port the conventions section into `docs/` |
| `docs/` | 472 KB | T 22 | design docs (UNIFIED_GEOTEX_DIT_DESIGN, DINO_ALIGNMENT_DESIGN, TRAINING_HANDOFF, MDS_SETUP, RESULTS, …) plus upstream `docs/README.md` | archive; cherry-pick UNIFIED_GEOTEX_DIT_DESIGN and MDS_SETUP into `docs/history/` |
| `configs/` | 603 KB | T 45 | DeepSpeed jsons (15) and mixture yamls (27) plus `setup_{A,B,C}.json` | **port** `deepspeed_zero1_fp32acc.json`, `deepspeed_zero1_fp32acc_universal.json` (SS 300k failover), `mix_v12_{ss,shape,tex}_texfixed.yaml`; the rest → archive |
| `benchmarks/` | 2.3 MB | T 79 | external-benchmark adapters (T3Bench, Hi3DEval, Toys4K, TRELLIS.2 paper metrics, pixal3d, hunyuan), `watch_*.sh`, plus **`checkpoint.py`** (`load_state_dict`, `load_connector`, imported by the v12 eval chain) and `run_wild3dgen.py` | port `checkpoint.py` → `blip3d/utils/ckpt.py`; the rest is archive-only (the live benchmark is eval_suite) |
| `scripts/` | 12 MB | T 471 / 528 on disk | see §2 | split |
| `tests/` | 180 KB | **U (gitignored)** 4 | CPU unit tests (repa, road_align, road_wiring, unified_conventions) | port road_* and unified_conventions (§6) |
| `third_party/` | ~13 GB | U | `eval/` only: 3dgen-bench, ImageReward, clip, dinov2, hi3deval, lpips, t3bench, ulip, uni3d, plus `.venv`, `.deps*` overlays | **archive/ignore.** v12 does not use it (only `benchmarks/run_trellis2.py` and `asset_provenance.json` reference it); eval_suite has its own third_party (ULIP, Uni3D, LumiTex, HY3D-Bench …) |
| `third_party_3d_gen/TRELLIS.2` | 185 MB | U | stale stock clone (see §0.1) | **delete** from BLIP3D; the real dependency is the external fork |
| `checkpoints/uni3d-b/` | 171 MB | U | Uni3D-**b** `model.pt` (v12 ROAD uses Uni3D-**g** from the HF cache) | ignore |
| `data/benchmarks/` | 546 KB | U | pilot jsonls for the old benchmarks | archive |
| `manifests/` | 8.9 GB | U | `splits/` (66 files, incl. `*_texfixed_ff.jsonl` 0.38–2.5 GB each), `heldout_swh_eval.jsonl`, `v5_matclean_*` | **data, not code** → move to a data root referenced by `paths.yaml` |
| `pix/`, `pixfix/` | 85 MB / 32 MB | U | per-rank `pix_r*.jsonl` material-bug detector outputs (48 each) | data → archive with the verdict files |
| `runs/` | (checkpoints) | U | 148 entries. 7 v12 run dirs are symlinks into `/fsx/data/weikai.huang/runs`. v12 milestones are frozen in `/fsx/data/weikai.huang/runs/keep/v12/` (CHECKPOINTS.md) | never in the new repo |
| `wandb/` | 587 MB | U | 85 local run dirs | ignore |
| `.compile_cache/` | 219 MB | U | old inductor/triton cache. Launchers now use `/dev/shm` (`TORCHINDUCTOR_CACHE_DIR=/dev/shm/...`) | delete |
| `__pycache__/`, `tmp0xcu0sie/`, `tmpx5knxyaa/`, `torchinductor_weikai.huang/`, `blip3o_next.egg-info/`, `demo_*.log` | small | U | junk | delete |

Sibling worktrees of the same `.git`: `../BLIP3o-edit` (`s3-edit`), `../BLIP3o-rl` (`s3-rl`), `../BLIP3o-unionpre` (detached). The edit-pair data work lives on `s3-edit`. Freezing the archive must keep these branches.

---

## 2. `scripts/` classification

On disk: **528 files** excluding `__pycache__` (289 .py, 110 .sh, 109 .sbatch, 9 png, 5 md, 4 json, 1 yaml, 1 jsonl). 471 are tracked; 57 are untracked, mostly `_tmp/`. The classification was done with a name-pattern classifier, then corrected by hand using the import graph.

### (a) v12 entry points and modules imported by package or runner code — 24 files, **must port**

Imported-library modules (these must become package code in `blip3d/infer` or `blip3d/eval`, not tools):

| module | lines | imported by |
|---|---|---|
| `scripts/eval_fusion_v22.py` | 273 | `demo_pipeline.py:204,306`; `benchmarks/run_wild3dgen.py:73,116,119`; `export_glb_*`; `eval_tex_v22`; lc `run_sep_text.py:24`, `unify_render.py:16`, `coord_ab.py:23`, `ss_road_{eval,align_score,ulip_score}.py`; `_lc/v12_tower_probe.py:13`, `v12_t4_probe*.py`. Provides `load_flow_and_connector`, `build_cond`, `build_cond_live`, `sample_shape`, `cam_from_transforms`, `good_view_b`, `input_image` |
| `scripts/eval_tex_v22.py` | 210 | `demo_pipeline.py:205,307`; eval_suite `run_v12_image.py:30`, `run_s3_t50.py:40`; lc runners. Provides `load_tex_flow`, `sample_tex`, `render_textured`, `HDR` |
| `scripts/export_glb_v22.py` | 106 | `demo_pipeline.py:308`; eval_suite `run_v12_image.py:31`, `run_s3_t50.py:41`, `text/run_geotex_v10_text.py:53`; lc `run_unify_bench.py:47`, `run_sep_text.py:27`. Provides `build_mw`, `export_glb` (o_voxel) |
| `scripts/export_glb_fullchain.py` | 172 | `demo_pipeline.py:203`; eval_suite `text/run_ss_swap_text.py:48`, `run_geotex_v10_text.py:52`; lc runners. Provides `load_ss_flow`, `SSDEC` (the path of the TRELLIS-image-large SS decoder) |
| `scripts/eval_render.py` | 151 | `eval_fusion_v22.py:23` (`decode_render` only; its `ablate_shape.ArmModel` import is lazy at line 32 and not on the v12 path) |
| `scripts/_tmp/eval_v10_tasks.py` **[untracked]** | 461 | eval_suite `text/run_geotex_v10_text.py:44`, `text/run_ss_swap_text.py:40`; about 30 `_lc` probes |
| `scripts/_tmp/_hf.py` **[untracked]** | 34 | `eval_v10_tasks.py:338` (`highfreq`) |
| `scripts/_tmp/build_voxel_table.py` **[untracked]** | — | produces `voxels*.npz` for `GEOTEX_BALANCE_TABLE` (`data/mixture.py:154`); must be rebuilt whenever the data pool changes |
| `benchmarks/checkpoint.py` (not in scripts/, same role) | 107 | eval_fusion_v22, eval_tex_v22, export_glb_fullchain, `test_demo_v12.py`, `_lc/v12_tower_probe.py` |

Launch and runtime entry points:

- `train_native_split.sh` (S1/S2 per-tower launcher; hardcodes `--vlm_model Qwen/Qwen3.5-2B` at line 138, which is inert with `--build_vlm False`. The real cond VLM comes from `COND_VLM_CKPT`.)
- `train_native_geotex.sh` (S3 unify)
- `v12/v12_s3.sh`, `v12/v12_s3_4n.sh`, `v12/launch_s3.sh`, `v12/launch_s3_4n.sh`, `v12/v12_ss_cont.sh`, `v12/launch_ss_cont.sh`, `v12/mk_ema_unify17k.sh`, `v12/split_unified_ckpt.py`
- `v12/run_unify_bench.py`, `v12/run_sep_text.py`
- `v12/configs/mix_v12_unify_texfixed.yaml`, `v12/configs/text_sfv_v12_eval50.jsonl`
- `demo_app.py`, `run_demo_app.sh`

Plus, outside the repo (must be recovered, §3): `_lc/v12_s1.sh`, `v12_s1b.sh`, `launch_s1b.sh`, `v12_s2.sh`, `v12_s2_4n.sh`, `launch_s2.sh`.

Note: `scripts/v12/*` copies differ from the live `/fsx/data/weikai.huang/lc` copies for `run_unify_bench.py`, `hold_reservation.sh`, `launch_ss_cont.sh`, `v12_ss_cont.sh`, `monitor_sscont.sh`, `sscont_kill.sh`. The **lc copies are newer** (09-23/24). These are LC-only: `ss_to300k.sh`, `ss300k_failover.sh`, `cleanup_v12.py`, `timing_all.sh`, `timing_sep.py`, `demo_smoke_v12.py`, `run_demo_smoke.sh`, `check_birefnet.py`, `u17_*.sh`, `mk_ema_ss106k_wait.sh`, `restore_flow_heads_retry.sh`, `kill_dup_readouts.sh`.

### (b) Tooling worth porting — 112 files by the classifier; about 70 after dropping duplicates and docs

| bucket → BLIP3D home | files |
|---|---|
| **data build** → `tools/data/` | `data/mk_splits.py`, `data/mk_pool800k_pbr.py`, `data/mk_capT_from_store.py`, `data/build_cachelist_manifest.py`, `detect_material_bug.py`, `build_heldout_clean.py`, `check_eval_set.py`, `dedup_spotcheck.py`, `v12/dup_scan.py`, `v12/geo_dup_scan.py`, `v12/vox_cover_v12.py`, `v12/mk_text_manifest_eval50.py`, `_tmp/build_voxel_table.py` (listed in a). External: `_lc/texswap/mk_texfixed_manifests.py`. MDS path (`build_mds.py`, `pack_webdataset.py`, `persist_mds.sh`, `stage_shards_to_nvme.sh`, `pack_node_to_nvme.sh`): v12 trains from jsonl manifests with live cond, not MDS, so **archive unless MDS is revived** |
| **eval / readouts** → `blip3d/eval/` + `tools/eval/` | `v12/unify_render.py`, `v12/coord_ab.py`, `v12/ss_road_eval.py`, `v12/ss_road_align_score.py`, `v12/ss_road_ulip_score.py`, `v12/v12_ss_readout.sh`, `v12/v12_ss_readout_data.sh`, `v12/bench_*.sh` (7), `v12/bench_summary.py`, `v12/u17_img_table.py`, `v12/u17_sweep.sh`, `v12/clipt_control.py`, `v12/run_text_track_sfv.sh`, `eval_geotex_g1g2.py` (imported by `eval/*`, `data/mk_viz16.py`), `eval/run_trellis2_official.py`, `eval/sweep_inference.py`, `eval/xtex_on_official_shape.py`, `data/mk_viz16.py`, `metric_study.py`, `eval_gallery.py`, `verify_input_gt.py`. External: `_lc/v12_tower_probe.py`, `v12_t4_probe*.py` (the S2 per-tower probes) |
| **render / export / viewer** → `tools/viz/` | `render_normal_glbs.py`, `compress_stage_glbs.py`, `compress_trimodal_glbs.py`, `export_glbs.py`, `export_glb_baselines.py`, `build_viewer3d_page.py`, `publish_viewer3d.py`, `build_report_html.py`, `build_im_report_html.py`, `build_trimodal_viewer.py`, `demo_assets/` (9 png) |
| **cluster ops** → `cluster/` | `v12/hold_reservation.sh` (take the lc version), `v12/monitor_s3.sh`, `v12/monitor_sscont.sh`, `v12/stall_probe.sh`, `v12/qwrite.sh`, `v12/kill_orphan_vllm.sh`, `v12/gpu_knock.py`, `v12/sscont_kill.sh`, `v12/s3_swap_deadline.sh`, `v12/s3_swap_rank.sh`, `v12/run_s3_smoke.sh` |
| **tests to harvest** → `tests/` | `v12/test_v12_assembly.py`, `v12/test_demo_v12.py`, `v12/test_geotex_init_guard.py` (reads `/fsx/data/weikai.huang/lc/patched/train_native.py`; repoint it), `road_gpu_smoke.py`, `g0_unified_identity.py`, and 29 `_tmp/test_*.py` (list in §6) |
| **docs** → `docs/` | `v12/README.md`, `v12/docs/{CHECKPOINTS,v12_status_2026-09-21,report_v12_progress,report_5day_topics}.md` |

### (c) Archive — about 390 files

| category | count | representative names |
|---|---|---|
| profiling / perf / hold sbatch | 70 | `hold_*.sbatch` (17), `probe_*.sbatch/.py`, `profile_*`, `mfu*`, `true_mfu.py`, `flops_mfu.py`, `fa3_measure.py`, `ss_compile*.py`, `modulate_triton.py`, `normmod_triton.py`, `gateres_triton.py` (these kernels landed in the TRELLIS.2 fork `fused_modulate.py`), `verify_modulation_*`, `bench_bs_sweep.py`, `bench_joint_vs_cascade.py` |
| ablations | 39 | `ablate_*.{py,sh,sbatch}` (incl. `ablate_shape.py` ArmModel), `abl_*viz.sbatch`, `render_abl_*`, `plot_abl_curves.py`, `extract_taps_*`, `repa_verdict*.sbatch`, `kd_verdict_viz.sbatch` |
| probes / diag / verify / smoke | 80 | `diag_*`, `g0_*` (except g0_unified_identity), `g05_*`, `g2_multiseed.py`, `verify_mmdit_*`, `verify_plkv_slat.py`, `pixal_ab*.py`, `synth_backpair_*`, `t1_*`, `vggt_gate*`, `smoke_*`, `_flow16_ref.py`, `_tok16_stats.py`, `ss_tok_roundtrip.py`, `mv_visibility_analysis.py`, `diag_im_multiview.py` (hardcoded 64-token spans, see README) |
| launchers / evals of earlier versions (v2.2 cache era, fusion, textonly, dpos, MDS) | 138 | `launch_v22_*`, `s1_*`, `train_[ABC].sh`, `train_native_q{25vl,35,3vl}.sh`, `train_ss_*`, `train_connector_warmup.sh`, `training_{3,4}n.sbatch`, `build_vlm_cache*.py`, `build_dino_cache.py`, `run_cache_*`, `*textonly*.sbatch`, `ood*.sbatch`, `indist_*`, `fused_fullchain*`, `trimodal_fullchain_eval.py`, `im_fullchain_eval.py`, `geotex_s1.sbatch`, `zero{1,2}.json`, `enumerate_v{2,3}.py`, `build_v{2,3}_manifest.py`, `make_dino_pos.py` |
| `_tmp/` scratch (non-test, non-imported) | 62 | `eval_all8*.py`, `eval_g3t2.py`, `eval_g4t4.py`, `head2head.py`, `lc_parity*.py`, `prof_v8/v10.py`, `qwen_prof*.py`, `mk_cached.py`, `mk_clean.py`, `v10_smoke.sh`, `watch_run.sh` |
| misc | 3 | `eval/__init__.py`, `eval/eval_overfit10.py`, `viz_overfit.py` |

---

## 3. External tooling v12 depends on

### 3a. `/fsx/home/weikai.huang/_lc/` (1231 entries: 110 .sh, 170 .py, the rest logs, figures and caches)

Only these matter for v12:

| file(s) | role | BLIP3D home |
|---|---|---|
| `v12_s1.sh`, `v12_s1b.sh`, `launch_s1b.sh` | S1 connector-only launch (3 towers × 2 nodes, texfixed_ff) | `recipes/v12/s1_{ss,shape,tex}.yaml` + `cluster/launch.sh` |
| `v12_s2.sh`, `launch_s2.sh`, `v12_s2_4n.sh` | S2 full FT (LR 2e-5, WSD 100k + 6k, eff bs 256, ROAD on SS, `COND_VLM_CKPT`, `configs/deepspeed_zero1_fp32acc.json`); S2 SS 4-node continuation | `recipes/v12/s2_*.yaml` + `cluster/` |
| `fork_guard.sh`, `fork_guard_4n.sh`, `ss4n_ckpt_mapper.sh`, `migrate_at_ckpt.sh` | checkpoint fork protection against `save_total_limit`; step-axis mapping | `cluster/ckpt_guard.sh`. Better: a `keep_steps` recipe field in the trainer |
| `hold_reservation.sh` (older), `check_resv.sh`, `kill_gpu_holders.sh`, `kill_stale_train.sh` | heartbeat, reservation check, node cleanup | `cluster/` (use the lc heartbeat, which writes the `<ISO Z> pid=` content) |
| `v12_tower_probe.py`, `v12_t4_probe.py`, `v12_t4_probe_any.py`, `v12_probe_wait.sh`, `v12s2_probe_wait.sh`, `run_fullchain_staged.sh`, `run_val32_forks*.sh`, `eval50_final_chain.sh`, `eval_final_texfix.sh` | per-checkpoint readouts and figures during S1/S2 | `tools/eval/` (probe) + `cluster/` (waiters) |
| `run_{tex,ss,shape}cfg_sweep*.sh`, `run_texgi_sweep.sh`, `run_texsteps_sweep.sh`, `run_material_cf.sh`, `run_channels.sh` | sampler-knob sweeps that fixed the v12 inference defaults (e.g. TEX_GI 0.9–1.0) | archive. Record the chosen values in `recipes/v12/infer.yaml` |
| `balance/voxels_v11.npz`, `balance/voxels.npz` | voxel edge table for load balancing | data asset → `paths.yaml`; builder → `tools/data/build_voxel_table.py` |
| `texswap/` (mk_texfixed_manifests.py, feeder, calib, refix queue …) | texture-swap scan and the texfixed manifest build | `tools/data/texswap/` (the data agent owns the details) |
| `third_party/ROAD` (@c847391), `third_party/Uni3D` | originals that `tests/test_road_align.py` checks the vendored `trellis2_blip3o/road/` against | reference only; pin the commit in `docs/`. The test skips when the originals are absent |
| everything else (`bak_*`, `*.log`, `fig_*.png`, `fullchain_cache_*.pt`, `align_score_*.json`, v8–v11 probes) | history | stays in the archive |

### 3b. `/fsx/data/weikai.huang/lc/` (105 entries)

This is the live working directory for S3 and the SS continuation.

| file(s) | role | BLIP3D home |
|---|---|---|
| `v12_s3.sh`, `v12_s3_4n.sh`, `launch_s3*.sh` | S3 unify launch (identical to `scripts/v12`) | `recipes/v12/s3_unify.yaml` + `cluster/` |
| `v12_ss_cont.sh`, `launch_ss_cont.sh`, `ss_to300k.sh`, `ss300k_failover.sh` + `ss300k_state` | **running now**: SS continuation to 300k with 4↔2 node failover through a universal checkpoint (`deepspeed_zero1_fp32acc_universal.json`) and EMA re-init (the dirty `train_native.py` change) | `cluster/failover.sh` + the `ema_init_from` recipe field. Port after this run ends |
| `hold_reservation.sh`, `monitor_s3.sh`, `monitor_sscont.sh`, `stall_probe.sh`, `sscont_kill.sh`, `s3_swap_*.sh`, `gpu_knock.py`, `kill_orphan_vllm.sh`, `qwrite.sh` | heartbeat, monitoring, quota-safe writes | `cluster/` |
| `run_unify_bench.py`, `run_sep_text.py`, `bench_*.sh`, `bench_summary.py`, `u17_*.{sh,py}`, `run_text_track_sfv.sh`, `mk_text_manifest_eval50.py` | WilD3D sfv50 benchmark runners (image and text; unified vs separate) | `blip3d/eval/` (the runner) + `tools/bench/` (the shells) |
| `timing_all.sh`, `timing_sep.py` | inference-speed benchmark (SS K/V cache exactness check) | `tools/bench/` |
| `ss_road_*.py`, `coord_ab.py`, `unify_render.py`, `clipt_control.py`, `dup_scan.py`, `geo_dup_scan.py`, `vox_cover_v12.py` | readouts and analysis | `tools/eval/`, `tools/data/` |
| `split_unified_ckpt.py`, `mk_ema_*.sh`, `cleanup_v12.py` | checkpoint surgery and retention | `tools/ckpt/` |
| `test_v12_assembly.py`, `test_geotex_init_guard.py`, `test_demo_v12.py`, `demo_smoke_v12.py` | acceptance tests | `tests/gpu/` |
| `configs/mix_v12_unify_texfixed.yaml`, `configs/text_sfv_v12_eval50.jsonl` | S3 mixture; text eval set | `recipes/v12/`, `eval/manifests/` |
| `patched/` | a copy of the dirty working-tree files | delete once committed |
| `pristine_repo/` | a symlink farm onto the repo (including `.git`) | delete, or leave. It is not a copy |
| `third_party_3d_gen` → symlink to `/fsx/home/.../model/third_party_3d_gen` | — | — |
| `*.md` (status, reports), `history_user_msgs.txt`, logs, `fullchain_cache_*.pt`, `figs/`, `unify_scratch*/`, `rp3do/`, `bak_s3/` | records and scratch | archive |

### 3c. `/fsx/home/weikai.huang/3dgen/eval_suite` (own git, 75 tracked files; v12 work is mostly untracked)

v12 runners and what they import from BLIP3o:

| runner | tracked? | BLIP3o imports |
|---|---|---|
| `scripts/run_v12_image.py` | **U** | `trellis2_blip3o.demo_pipeline`, `live_cond_batch.{TrainCondEncoder,prep_i1}`, `eval_cond.cond_uncond`, `scripts.eval_tex_v22.sample_tex`, `scripts.export_glb_v22.{build_mw,export_glb}` |
| `scripts/run_v12_eval_inference_8gpu.sh`, `_2models.sh` | U | runs the above in `blip3o_trellis`; baselines run in the `trellis2`, `hy3dpaint` and `hi3deval` envs. Default ckpts are `/fsx/data/weikai.huang/runs/forks/s3_*_v12s2/checkpoint-{60000,80000}` |
| `scripts/run_v12_eval_scoring_2models.sh`, `_tail.sh`, `validate_v12_generation.py`, `validate_hy3d_e2e_v12.py`, `build_v12_eval_manifest.py`, `run_trellis2_e2e_2models.py`; `configs/sfv_val_v12_eval{50,200}.jsonl`, `texture_gt_v12_2models.json` | U | scoring only (hi3deval / trellis2 envs) |
| `scripts/run_s3_t50.py` | T | `demo_pipeline.Pipeline`, `live_cond.build_stage_cond` (pre-CondStamp path; **wrong for v12 weights**, see the `demo_pipeline.py:49-55` comment), `scripts.eval_tex_v22`, `scripts.export_glb_v22` |
| `text/run_s3_text.py`, `run_ss_swap_text.py`, `run_geotex_v10_text.py`, `run_text_track.sh`, `score_text_track.py`, `validate_text_track.py`, `uv_flip_probe.py`, `build_blind_sheet.py`, `build_text_leaderboard.py`, … | **U (the whole `text/` dir)** | `scripts._tmp.eval_v10_tasks` (untracked in BLIP3o), `geotex_sampler`, `occ_metrics`, `export_glb_fullchain.SSDEC` |
| `checkpoints/trellis2_e2e/ULIP-2-PointBERT-10k-xyzrgb.pt` | U | used by lc `ss_road_ulip_score.py:33` |

**Verdict:** scoring stays in eval_suite. The *model adapters* (`run_v12_image.py`, `text/run_s3_text.py` and the lc `run_unify_bench.py` / `run_sep_text.py`) should shrink to thin CLIs over a single `blip3d.infer.Pipeline` API. eval_suite then depends on the installed `blip3d` package instead of `sys.path.insert(REPO); import scripts.*`. **Commit eval_suite's untracked v12 and text files before the freeze.**

---

## 4. Environment

### conda env `blip3o_trellis` (`/fsx/home/weikai.huang/miniconda3/envs/blip3o_trellis`)

Probed CPU-only on the login node:

| package | version | note |
|---|---|---|
| python | 3.10.20 | |
| torch / torchvision | 2.6.0+cu124 / 0.21.0+cu124 | CUDA 12.4, cuDNN 9.13.01 |
| transformers | **5.2.0** | freeze says 4.52.4 and requirements.txt says 4.51.3 — both wrong. `train_native.py:749-750` comments say Qwen3-VL "needs 4.57.6" (stale). DINOv3 exposes `self.layer` in 5.2.0 (`modeling_dinov3_vit.py:479`), so the fork's `image_feature_extractor` works unpatched |
| deepspeed | 0.19.0 | ZeRO-1 fp32acc; universal ckpt for world-size change |
| accelerate / diffusers / peft / timm | 1.7.0 / 0.34.0 / 0.15.2 / 1.0.24 | timm is needed for the Uni3D-g point encoder (`road/uni3d.py`) |
| huggingface_hub / safetensors | 1.20.1 / 0.7.0 | |
| flash-attn 2 | 2.7.3 (wheel `cu12torch2.6cxx11abiFALSE`) | |
| flash-attn 3 | `flash_attn_3` 3.0.0 egg + `flash_attn_interface.py` (built from `hopper/`) | launchers default to `ATTN_BACKEND=flash_attn_3` (`train_native_split.sh:134`, `train_native_geotex.sh:156`). **`_paths.py:26` defaults to `sdpa`**, so eval/demo use sdpa unless the env var is set **[spec: small numeric drift between train and eval]** |
| triton | 3.2.0 | fused modulate kernels in the TRELLIS.2 fork (`FUSED_MODULATE=1`) |
| spconv | 2.3.8 (cu124) | TRELLIS.2 sparse conv backend option |
| flex_gemm | 1.0.0 | TRELLIS.2 sparse conv / grid_sample (triton) |
| o_voxel | 0.0.1, **editable** from `.../model/third_party_3d_gen/TRELLIS.2/o-voxel` | built C++/CUDA extension; GLB export |
| cumesh | 0.0.1 | mesh post-processing (TRELLIS.2) |
| nvdiffrast | 0.4.0 (+ `nvdiffrec_render`) | `_paths.py:29-33` sets `CUDA_HOME` / `CPATH` for its JIT |
| utils3d | 0.0.2 (no `__version__`) | |
| streaming (MosaicML) | 0.13.0 | only `data/streaming_task.py` (MDS path, not v12) |
| natten 0.17.5, open3d 0.19.0, trimesh 4.11.2, pymeshlab, rembg 2.0.69, wandb 0.27.0 | | natten likely unused **[spec]** |
| missing | xformers, kaolin, pytorch3d, igl, cubvh, qwen_vl_utils, liger_kernel | none needed |
| **numpy** | env 1.26.4 — **but 2.2.6 loaded from `~/.local`** | set `PYTHONNOUSERSITE=1` in all BLIP3D launchers, or pin one version |

Editable installs: `blip3o_next` → the repo; `o_voxel` → the TRELLIS.2 fork.

**Recommendation:** generate `env/blip3d.lock.txt` from `pip freeze` under `PYTHONNOUSERSITE=1`. Write `env/BUILD.md` for the compiled pieces (FA3 from hopper, flex_gemm, o-voxel, cumesh, nvdiffrast, spconv-cu124), each pinned to a commit or wheel. Delete both current requirements files.

### How trellis2 gets on sys.path

`trellis2_blip3o/_paths.py` (imported first by every entry point) inserts the repo root and `<dirname(repo)>/third_party_3d_gen/TRELLIS.2`. It also sets `ATTN_BACKEND=sdpa` and `CUDA_HOME` / `CPATH`, and defines `CHECKPOINTS_ROOT=<dirname(repo)>/checkpoints` and `DATA_ROOT=<repo>/data`. Everything is relative to the **parent** of the repo, so moving the repo breaks it silently: tr2_modules then falls through to a `/weka/...` path (`tr2_modules.py:158-159`).

**BLIP3D:** add the TRELLIS.2 fork as a pinned git submodule (`third_party/TRELLIS.2` @ a pushed commit; push `2ed033d` first) or pip-install it. Resolve `TRELLIS2_ROOT` and `TRELLIS2_CKPTS` from `paths.yaml` or env. No parent-dir arithmetic.

### HF and model assets (HF_HOME=`/fsx/home/weikai.huang/hf_cache`, offline)

| asset | location / snapshot | used by |
|---|---|---|
| Qwen3-VL-2B-Instruct (stock cond VLM for v12) | `hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203` | `COND_VLM_CKPT` (launchers, `demo_pipeline.py:42-45`) |
| DINOv3 ViT-L/16 | `models--camenduru--dinov3-vitl16-pretrain-lvd1689m` @ `3c276edd…` (non-gated mirror; `dino_align.py:34`) | live cond DINO 512 |
| TRELLIS.2-4B | **not** from the HF cache for training: `/fsx/home/weikai.huang/3dgen/model/checkpoints/TRELLIS.2-4B/ckpts/*` (real files, ~16 GB; `tr2_modules.py:18-23`, `multimodal_decoder/builder.py:135-140`). The HF cache also has `af44b45f…` | flows, decoders, encoders |
| TRELLIS.2 stage configs (normalization stats) | `<TRELLIS.2 fork>/configs/gen/*.json` (`tr2_modules.py:161-163`) | latent normalization |
| TRELLIS-image-large SS decoder `ss_dec_conv3d_16l8_fp16` | `/fsx/home/weikai.huang/.cache/huggingface/hub/models--microsoft--TRELLIS-image-large/snapshots/25e0d31f…` (**second HF cache**, `export_glb_fullchain.py:34`) | SS → occupancy decode |
| Uni3D-g | `hf_cache/hub/models--BAAI--Uni3D/snapshots/3d8233b7…/modelzoo/uni3d-g/model.pt` (glob in `road/teacher.py:41-44`) | ROAD teacher (S2 SS), SS readouts |
| BiRefNet | `models--ZhengPeng7--BiRefNet` @ `e2bf8e44…` (`live_cond.py:138,143`, `LIVE_REMBG`, `BIREFNET_MODEL`) | background removal, demo / eval |
| ULIP-2 PointBERT | `eval_suite/checkpoints/trellis2_e2e/ULIP-2-PointBERT-10k-xyzrgb.pt`; CLIP-bigG `models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k` | SS ULIP readout |

**BLIP3D:** a single `paths.yaml` (`hf_home`, `trellis2_root`, `trellis2_ckpts`, `trellis1_ss_dec`, `cond_vlm`, `dino`, `uni3d_g`, `birefnet`, `ulip`, `data_root`, `runs_root`, `balance_table`), plus `tools/fetch_assets.py` that verifies snapshot hashes. Merge the two HF caches.

---

## 5. Hardcoded absolute paths and env-var knobs in package code

Line numbers refer to the current **working tree** (train_native.py is dirty).

### 5a. Absolute and implicit paths

| file:line | value | proposal |
|---|---|---|
| `trellis2_blip3o/_paths.py:13-16,36-40` | repo-parent-relative `third_party_3d_gen/TRELLIS.2`, `checkpoints/`, `data/` | `paths.yaml: trellis2_root, trellis2_ckpts`; drop `DATA_ROOT` / `WORLD_EXPLORE_ROOT` |
| `trellis2_blip3o/_paths.py:26` | `ATTN_BACKEND` default `sdpa` | recipe field `attn_backend` (default `flash_attn_3`, the training default) |
| `trellis2_blip3o/tr2_modules.py:18-23` | `CHECKPOINTS_ROOT/TRELLIS.2-4B/ckpts/...` | `paths.trellis2_ckpts` |
| `trellis2_blip3o/tr2_modules.py:153-159` | TRELLIS repo root, fallback `/weka/oe-training-default/weikaih/world_explore/...` | `paths.trellis2_root`; **delete the /weka fallback** |
| `blip3o/model/multimodal_decoder/builder.py:135,140` | `CHECKPOINTS_ROOT/TRELLIS.2-4B/ckpts` | `paths.trellis2_ckpts` |
| `trellis2_blip3o/live_cond.py:48-49` | `_V22_DEFAULT=/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000` (the v2.2 VLM; default when `COND_VLM_CKPT` is unset) | **delete the default**; make `cond_vlm` a required recipe/checkpoint-config field (per memory: the cond VLM follows the checkpoint) |
| `trellis2_blip3o/demo_pipeline.py:35` | `FINALS=/fsx/data/weikai.huang/runs` | `paths.runs_root` / CLI |
| `trellis2_blip3o/demo_pipeline.py:42-45` | stock Qwen3-VL snapshot under `/fsx/home/weikai.huang/hf_cache/hub` | `paths.cond_vlm` / `ckpt.config.cond_vlm` |
| `trellis2_blip3o/demo_pipeline.py:64-66` | `DEMO_{SS,SHAPE,TEX}_CKPT` defaulting to the S2 106k finals | `recipes/v12/infer.yaml` |
| `trellis2_blip3o/pos_stamp.py:14` | `DPOS_NPZ=/fsx/home/.../BLIP3o/runs/cache_logs/dino_pos32.npz` | **delete the module** (pos_stamp is retired in v12) |
| `trellis2_blip3o/data/tap_ss_task.py:14,16` | `vlm_hidden_cache/v22_3dvlm_tok1024_mv1`, `/opt/dlami/nvme/ss_taps16k` | delete (tap-cache era) |
| `trellis2_blip3o/data/tap_task.py:25,27` | `/opt/dlami/nvme/vp1taps_v22`, `ready_v3_clean.jsonl` | delete |
| `trellis2_blip3o/data/tasks/threed_balanced.py:18` (docstring) | `/fsx/hyperpod/weikaih_cap/judge_scores.jsonl` | recipe field, if the task is kept |
| `trellis2_blip3o/live_cond_batch.py:335,338` | `/dev/shm/v22ckpt_live`, `/dev/shm/condvlm_<sha>` staging | `paths.shm_stage` (keep the behavior) |
| `trellis2_blip3o/road/teacher.py:41-44` | glob over `$HF_HOME`, `/fsx/home/weikai.huang/hf_cache`, `~/.cache/huggingface` | `paths.uni3d_g` (explicit file) |
| `trellis2_blip3o/dino_align.py:34` | `camenduru/dinov3-vitl16-pretrain-lvd1689m` | `paths.dino` / recipe `cond.dino_model` |
| `trellis2_blip3o/live_cond.py:143` | `ZhengPeng7/BiRefNet` (env `BIREFNET_MODEL`) | `paths.birefnet` |
| `trellis2_blip3o/unified_geotex.py:6` (docstring) | `/fsx/home/.../reference_repos/` | fine (provenance note) |
| `blip3o/data/dataset.py:175` | `/fsx/home/jiuhai.chen/soda/overfit.tar`, `/fsx/sfr/data/jiuhai/webdataset` | archive (2D-only) |
| `train_native.py:747` | `vlm_model="Qwen/Qwen3.5-2B"` (and launcher `train_native_split.sh:138`) | delete. It is inert with `build_vlm False`; replace with `cond.vlm` |
| `train_native.py:751` | `data_path="data/overfit/imgtext.jsonl"` | delete (the mixture config is the only data source) |
| `train_native.py:929` (comment) | `~/.cache/torch_extensions` (ROAD GPU matcher JIT) | `TORCH_EXTENSIONS_DIR` → `/dev/shm` in `cluster/env.sh` |
| `configs/mix_v12_*_texfixed.yaml` | absolute `/fsx/home/.../BLIP3o/manifests/splits/*_texfixed_ff.jsonl` | `${data_root}/splits/...` interpolation |
| imported scripts: `eval_fusion_v22.py:12,27,33,234`, `eval_tex_v22.py:11,36,42,170`, `export_glb_v22.py:8-10`, `export_glb_fullchain.py:11-12,34,102`, `eval_render.py:13,24,28` | `sys.path.insert(REPO)`, v2.2 cond cache roots, `val200.jsonl`, `ready_v4 heldout_eval.jsonl`, a DejaVu font path inside the conda env, `SSDEC` in the second HF cache, `/opt/dlami/nvme/vp1taps_v22` | drop the sys.path hacks (installed package); `paths.trellis1_ss_dec`; font via `matplotlib.font_manager`; drop the cached-cond paths (v12 uses live cond) |

### 5b. Env-var knobs read by package code

| env var | file:line | proposal |
|---|---|---|
| `COND_VLM_CKPT` | `live_cond.py:54`, `demo_pipeline.py:45` | **recipe/ckpt field `cond.vlm`**, saved into the checkpoint config (this silent mismatch has already happened; see memory) |
| `GEOTEX_IM_TOK_PER_VIEW` | `live_cond_batch.py:56` (S3 sets 256) | recipe `cond.im_qwen_tok_per_view`; also persist it in ckpt |
| `IM_QWEN_TOK_PER_VIEW` (a constant, 128) | `live_cond.py:65` | same field. README warns it must match the cache/ckpt |
| `DINO_SIZE_IM` | `live_cond.py:64` | recipe `cond.dino_size_im` (512 for v12) |
| `VIEW_EMBED_SCALE` | `view_code.py:19` (default 0.2) | recipe `cond.view_code_scale`; persist it |
| `GEOTEX_IM_VIEW_SUBSET` | `data/tasks/threed.py:560` | recipe `data.im_view_subset` |
| `IM_VIEWS` / `IM_VIEW_SUBSET` | `dataset_native.py:188` / `data/streaming_task.py:140` | delete (legacy datasets) |
| `GEOTEX_BALANCE_TABLE` / `GEOTEX_BALANCE_BUF` | `data/mixture.py:154` / `:375` | recipe `data.balance_table` (path via `paths.yaml`) |
| `GEOTEX_XATTN_FLASH` | `flow_heads.py:1710` | recipe `perf.xattn_flash` (default on; lossless per its test) |
| `COMPILE_SS` | `trellis_native_vlm.py:840` | recipe `perf.compile_ss` |
| `ROAD_ALIGN_POINTS` | `train_native.py:1082` (exported), `data/tasks/threed.py:609` (read in workers) | recipe `road.align_points`, passed explicitly to the dataset |
| `REPA_ROOT`, `REPA_MIN_QUALITY` | `train_native.py:1048-1049`, `data/streaming_task.py:78-79` | delete (REPA/VGGT path is not in v12) |
| `REPA_COEFF_HI`, `REPA_WARMUP_STEPS` | `trellis_native_vlm.py:983,985` | delete |
| `LIVE_COND_PREFETCH`, `LIVE_COND_PROF` | `train_native.py:575,588` | recipe `perf.live_cond_prefetch`; drop PROF |
| `LIVE_COND_SHM` | `live_cond_batch.py:326` | `paths.shm_stage` / bool |
| `LIVE_REMBG`, `BIREFNET_MODEL` | `live_cond.py:138,143` | infer config |
| `DEMO_SS_CKPT` / `DEMO_SHAPE_CKPT` / `DEMO_TEX_CKPT` | `demo_pipeline.py:64-66` | infer config |
| `DUAL_CKPT` | `dual_cond.py:185` | delete (dual_cond is not v12) |
| `S3_TASK_DEBUG` | `train_native.py:683` | `--debug` flag |
| `POOL_SCALE` | `blip3o/model/multimodal_encoder/ta_tok_encoder.py:71` | archive |
| `HF_HOME` / `HF_HUB_CACHE` | `road/teacher.py:41`, `demo_pipeline.py:42` | `paths.yaml` |
| `ATTN_BACKEND`, `CUDA_HOME`, `CPATH` (setdefault) | `_paths.py:26,32,33` | `cluster/env.sh` + recipe |
| TRELLIS.2 fork: `ATTN_BACKEND`, `SPARSE_ATTN_BACKEND`, `SPARSE_CONV_BACKEND`, `FUSED_MODULATE`, `TR2_SPARSE_DEBUG` | `trellis2/modules/{attention,sparse}/config.py`, `sparse/fused_modulate.py:17`, `sparse/basic.py:725` | set from the recipe in one `blip3d.utils.backend.configure()` before the trellis2 import |
| `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE` | various | fine (torchrun) |

Launcher-level knobs that are really recipe fields: from `train_native_split.sh`, `train_native_geotex.sh` (231 lines of env defaults), `v12_s2.sh` and `v12_s3_4n.sh`. They are `MIX`, `MAX_STEPS`, `PER_GPU_BS`, `EFF_BS`, `LR`, `WARMUP`, `DECAY_STEPS`, `DINO_DROP`, `QWEN_DROP`, `ELASTIC_RATIO`, `COUPLING`, `P_CORNER*`, `P_SOLO`, `P_LAG`, `SS_DUAL`, `UNFREEZE_GEO`, `GEO_LOSS_W`, `COND_ADAPTER`, `VIEW_EMBED`, `COND_SEG`, `PATCH_POS`, `SHAPE_INIT` / `TEX_INIT` / `SS_INIT`, `EMA`, `SAVE_*`, `WANDB_*`, `FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE`. All of these → `recipes/v12/*.yaml`, with a shell launcher that only does cluster plumbing.

---

## 6. Tests

### What exists

| file | tracked? | CPU/GPU | covers | v12 status |
|---|---|---|---|---|
| `tests/test_road_align.py` | U | CPU | vendored ROAD vs originals (`_lc/third_party/ROAD`; skips if absent), matcher, losses, FPS, Uni3D-g args | **valid** (ROAD on SS in S2) |
| `tests/test_road_wiring.py` | U | CPU | dataset emits align points → collate, lazy timm import, gate buffer persistence | **valid** |
| `tests/test_unified_conventions.py` | U | CPU tiny | x_t/v convention, norm-stats adapter, unified identity vs `SLatFlowModel`, unified cond, sparse diffuse | valid in concept; written for the 2-tower unify (Aug). Needs a 3-tower (SS lane) update **[spec]** |
| `tests/test_repa_units.py` | U | CPU | REPA/VGGT aux | **obsolete** (REPA not in v12) |
| `scripts/v12/test_v12_assembly.py` | T | GPU | 3-tower assembly on S2 finals: strict load, CondStamp buffers equal the ckpt, hashed view code equals the ckpt, zero gates, SS lane bit-exact, geo/tex lanes equal the specialists | **golden candidate** |
| `scripts/v12/test_geotex_init_guard.py` | T | CPU | init_from_checkpoint guard (ok / half-load / arch mismatch / new head) | valid; repoint from `lc/patched/train_native.py` to the package function |
| `scripts/v12/test_demo_v12.py` | T | GPU | in-memory prep equals benchmark prep; end-to-end I1/IM/T → GLB | **golden candidate** |
| `scripts/_tmp/test_cond_stamp.py`, `test_view_code.py`, `test_cond_extrapolation.py` | **U** | CPU | CondStamp params / loader round-trip / add semantics; view code determinism and orthogonality; 2–4 → 8/12 views | **core v12 conditioning tests** |
| `scripts/_tmp/test_road_pool_equiv.py`, `test_road_shape_hook_gc.py` | U | CPU/GPU | ROAD pooling equivalence; tap under GC | valid |
| `scripts/_tmp/test_xattn_flash.py`, `test_xattn_varlen.py` | U | GPU | flash cross-attn fast path is numerically equivalent | valid (`GEOTEX_XATTN_FLASH=1` in S3) |
| `scripts/_tmp/test_balance.py` | U | CPU | voxel load balancing; off equals bit-identical | valid |
| `scripts/_tmp/test_sync_batch.py`, `test_sync_ddp.py` | U | CPU/multi-GPU | log-scalar batching bit-exact; key-set equality across ranks | valid |
| `scripts/_tmp/test_tex_mask.py`, `test_tex_mask_deep.py`, `test_corner_hoist.py`, `test_three_tower_sampler.py`, `test_ema_shard.py`, `test_gc_unified.py`, `test_ss_dual.py`, `test_ss_gc.py`, `test_ss_grad_path.py`, `test_ssread_rowsel.py`, `test_solo_redundant.py`, `test_rank_rng.py`, `test_eval_cond.py`, `test_occ_metrics.py`, `test_gate_metrics.py`, `test_patchpos.py` | mixed (see §2) | — | unified-model invariants from v10/v11 that v12 S3 inherited | mostly valid; `test_patchpos` probably superseded by sincos2d **[spec]** |
| `scripts/_tmp/test_v10_{units,identity,assembly,ddp,sslane}.py`, `test_leak_scale.py` | mixed | — | v10 assembly | superseded by test_v12_assembly; archive |

`train_native.py:49` cites `tests/test_native_infer.py`, which does not exist.

### Golden tests BLIP3D needs (proposed)

1. **CPU unit (every commit):** CondStamp / view code / extrapolation; ROAD equivalence (the original-source check skips when the originals are absent); flow convention and normalization; the three-lane union-softmax mask builder; voxel balance off equals identity; EMA sharded equals whole; init-guard cases; recipe → args round-trip (every v12 launcher's env set maps to a recipe that yields identical `NativeArgs`).
2. **Checkpoint golden (GPU, nightly/manual):** load the frozen `keep/v12` S2 finals and the S3 17k EMA. On a fixed seed and fixed 4 inputs (1 image, 1 multi-view, 1 text, plus 1 with CFG), assert that cond tensors (post-CondStamp) and first-step velocities match **tensors saved from the old repo** (bit-exact, or ≤1e-3 under bf16), and that final SS occupancy and SLAT feats match within tolerance. This is the proof that the extraction did not change semantics. Record the golden tensors with the old repo **before** the freeze.
3. **Assembly golden:** the port of `test_v12_assembly` (SS lane bit-exact, geo/tex lanes equal the specialists).
4. **End-to-end smoke:** the port of `test_demo_v12` plus a 20-step 1-GPU training smoke on 8 cached samples (loss finite, EMA file written, ckpt reload equals the pre-save state).
5. **Train/eval parity:** `flow_heads.build_unified_cond` (train) versus `eval_cond.cond_uncond` (eval) produce the same stamped cond for the same sample (the `demo_pipeline.py:49-55` hazard).

---

## 7. BLIP3D skeleton: confirmation and adjustments

The proposed skeleton is confirmed, with these adjustments:

```
blip3d/
  pyproject.toml            # package blip3d; no blip3o/tok/trl
  paths.example.yaml        # all absolute paths (§4, §5a); env BLIP3D_PATHS points at the real one
  blip3d/
    cond/        live_cond, live_cond_batch (TrainCondEncoder, prep_*), eval_cond (cond_uncond),
                 view_code, CondStamp, connector (xf2 TRELLIS2TransformerAdapter), dino extractor wrapper,
                 vlm_collate
    models/      trellis_native_vlm (← blip3o/…/trellis_native_vlm.py), tr2_builders
                 (← blip3o/…/multimodal_decoder/builder.py + tr2_modules.py), flow_heads, unified_geotex
    losses/      flow-matching loss, road/ (vendored, verbatim + provenance)
    data/        mixture, registry, tasks/{threed,threed_textonly,threed_balanced}, balanced_assign,
                 rank_aware, category_balance  (drop tap_*, streaming_task unless MDS returns)
    train/       trainer (← train_native.py split into args/ema/init_guards/loop), callbacks
    infer/       Pipeline (← demo_pipeline) + sampler (← geotex_sampler), and the pieces now in
                 scripts/{eval_fusion_v22,eval_tex_v22,export_glb_v22,export_glb_fullchain,eval_render}
                 → infer/{ss.py,shape.py,tex.py,decode.py,export.py,render.py}
    eval/        occ_metrics, bench runner (← run_unify_bench/run_sep_text) as a library + thin CLI,
                 readouts (ss_road_*), probe (← _lc/v12_tower_probe)
    utils/       paths.py (yaml resolver), backend.py (ATTN/SPARSE backends before the trellis2 import),
                 ckpt.py (← benchmarks/checkpoint.py + split_unified_ckpt + EMA overlay)
  recipes/v12/   s1_{ss,shape,tex}.yaml, s2_{ss,shape,tex}.yaml, s2_ss_cont300k.yaml, s3_unify.yaml,
                 infer.yaml (per-path sampler settings as benchmarked), mixes/*.yaml, deepspeed/*.json
  third_party/   TRELLIS.2 (submodule @ pushed fork commit; o-voxel is built from it)
  tools/         data/ (mk_splits, mk_texfixed_manifests, build_voxel_table, material_bug, dedup/geo_dup,
                 eval-set builders), viz/ (viewer, GLB compress/normal renders, report html),
                 ckpt/ (ema, split, retention), bench/ (sfv50 shells, timing)
  cluster/       env.sh (PYTHONNOUSERSITE=1, HF offline, /dev/shm caches, NCCL/EFA), launch.sh
                 (recipe + node rank → torchrun), hold_reservation.sh, monitor.sh, failover.sh,
                 ckpt_guard.sh, qwrite.sh, kill helpers (written as script files, not pkill via srun)
  tests/         unit/ (CPU), gpu/ (assembly, golden tensors, e2e smoke), goldens/ (saved tensors, small)
  docs/          conventions (CondStamp, view codes, token budgets, cond-VLM-follows-ckpt),
                 v12 recipe card + CHECKPOINTS, history/ (design docs cherry-picked)
  env/           lock file + BUILD.md for compiled deps
```

Adjustments versus the proposal:
- Add **`third_party/`** (the TRELLIS.2 fork submodule) and **`env/`**.
- Add **`blip3d/utils/backend.py`**, because backend env vars must be set before `import trellis2`.
- The `scripts/*_v22.py` modules go into **`blip3d/infer`**, not `tools/`.
- **`blip3d/eval` holds only the model-side runner.** Scoring stays in eval_suite, which should `pip install -e blip3d` and call `blip3d.infer.Pipeline`.
- **Do not port** (unreachable from any v12 entry point, or legacy): `codebook_prep.py`, `lora.py`, `mmdit_ss.py`, `plkv_ss.py`, `data/tap_ss_task.py`. Probably also not `mmdit3d.py`, `mmdit_slat.py`, `plkv_slat.py`, `dual_cond.py`, `depth_fusion.py`, `repa.py`, `pos_stamp.py`, `dino_align.py` (the projection-head part), `dataset.py`, `dataset_native.py`, `vlm_cache.py`, `data/tap_task.py`, `data/streaming_task.py`, `data/packed_reader.py`. Several of these are *imported* (the train closure reaches them), but only through option branches that v12 disables **[spec: confirm against the training/conditioning agents' reports before deleting]**. Import-closure results (a static AST walk that includes function-level imports):
  - **Train closure** from `train_native.py` reaches every `trellis2_blip3o` module except `codebook_prep`, `demo_pipeline`, `eval_cond`, `lora`, `mmdit_slat`, `mmdit_ss`, `plkv_slat`, `plkv_ss`, `data.tap_ss_task`, `data.tap_task` and `road.evaluation`. It also pulls in the 2D `blip3o.*` and `tok.*` through `blip3o/model/__init__.py`.
  - **Eval closure** (demo_pipeline, the lc runners, the eval_suite runners, `benchmarks/run_wild3dgen.py`) adds `demo_pipeline`, `eval_cond`, `mmdit_slat`, `plkv_slat`, `data.tap_task` and `road.evaluation`, plus `scripts.{eval_fusion_v22, eval_tex_v22, export_glb_v22, export_glb_fullchain, eval_render, ablate_shape (lazy), _tmp.eval_v10_tasks, _tmp._hf}` and `benchmarks.checkpoint`.
  - **Never reached:** exactly `codebook_prep`, `lora`, `mmdit_ss`, `plkv_ss` and `data.tap_ss_task`.

### Migration order

0. **Before anything (archive hygiene):**
   - Commit the dirty working tree, or wait for the SS 300k run to finish and then commit.
   - `git add -f` the load-bearing untracked files: `tests/*`, `scripts/_tmp/{eval_v10_tasks,_hf,build_voxel_table,test_*}.py`, `_lc` S1/S2 launchers, `lc` newer copies, `texswap/mk_texfixed_manifests.py`.
   - Push the TRELLIS.2 fork `2ed033d` and commit its dirty data_toolkit edits.
   - Commit eval_suite `text/` and the v12 scripts.
   - Tag `v12-final` in BLIP3o, the TRELLIS.2 fork and eval_suite.
1. **Record goldens with the old repo** (§6.2): cond tensors, first-step velocities and final latents for the 4 fixed inputs × {S2 finals, S3 17k}.
2. **Skeleton + env:** pyproject, `paths.yaml` resolver, `backend.py`, TRELLIS.2 submodule, env lock with `PYTHONNOUSERSITE=1`.
3. **models/ + cond/** (port the `trellis_native_vlm` and builder files, cutting the `blip3o/model/__init__` chain); CPU unit tests.
4. **infer/** (demo_pipeline + the five `scripts/*` library modules + geotex_sampler); pass the inference goldens; port `test_demo_v12`.
5. **eval/** runner; switch eval_suite `run_v12_image.py` / `text/run_s3_text.py` to `blip3d`; re-score one small set (e.g. eval50 image) and compare against the recorded results.
6. **data/ + train/** (split train_native.py, recipes from the 6 launchers); 20-step smoke; recipe → args parity test against the old launchers' `NativeArgs`.
7. **losses/road** + balance table + ROAD tests; a short S2-SS resume from a keep checkpoint to confirm loss and grad-norm curves overlap for about 200 steps.
8. **cluster/ + tools/** (last; they are plumbing). Delete `patched/`, `pristine_repo/`, `.compile_cache/` and the in-repo stale TRELLIS.2 copy only after the user approves.
