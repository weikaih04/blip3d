# BLIP3D scan 03: the unified three-tower model, its S3 loss, and every inference path

Read-only audit, 2026-09-24. Repo `/fsx/home/weikai.huang/3dgen/model/BLIP3o` on branch `BLIP3o-NEXT`.
The working tree has uncommitted edits to `trellis2_blip3o/geotex_sampler.py` and `trellis2_blip3o/unified_geotex.py`
(the `JOINT_OPTS` / `precompute_ss_kv` additions) and to `train_native.py`. `lc/patched/geotex_sampler.py` is byte-identical to the repo copy.
Unless a path is given, file:line refers to the repo. `UG` = `trellis2_blip3o/unified_geotex.py`, `GS` = `trellis2_blip3o/geotex_sampler.py`,
`FH` = `trellis2_blip3o/flow_heads.py`, `TNV` = `blip3o/model/language_model/trellis_native_vlm.py`.
Findings not checked by running code are marked **[unverified]**. Anything I could not confirm is marked **[speculation]**.

---

## 0. Headline findings (read these first)

1. **The v12 unify benchmark runs a different attention convention from the one S3 trained with.** Training sets
   `unified_geotex.fused_attn = True` (TNV:416, from `geotex_fused=True`). `UnifiedGeoTexFlow.__init__` defaults it to `False` (UG:359).
   `run_unify_bench.py` and `unify_render.py` never set it (lc/run_unify_bench.py:78-81, lc/unify_render.py:105-112).
   So every joint-loop step at inference goes through the loop path `_run_block_pair_bidir` (UG:784-855).
   The loop path rotates the geo key that the tex lane reads (`k_s_tag = _rotate_pad_pair(k_s)`, UG:839). The tex query is rotated as well (UG:811),
   so the tex→geo stream tag cancels out. In the fused path the geo key stays unrotated (UG:758, 720), so tex→geo logits carry the ±π/2 tag.
   The G0 script says the two are "DIFFERENT models by design" and only checks agreement within rel < 0.35 (`scripts/g0_fused_attn.py:199-204`).
   Affected: the geometry trajectory in all three modes, and the texture in mode "joint" (no refine).
   Not affected: the refine texture, because `tex_forward_cached` → `_tex_block_inner` (UG:857-882) uses the trained absolute tag with geo keys unrotated.
   Older evaluations did set `fused_attn=True` (`scripts/eval_geotex_g1g2.py:117`, `scripts/bench_joint_vs_cascade.py:13`); the v12 runners do not.
   **BLIP3D should port only the fused path and use it for inference too.** Rerunning a few unify17k cases with fused on vs off is cheap and tells us whether the published numbers move.
2. **The refine / tex|mesh pass drops the SS segment that training always had.** In S3 every row carried `x_ss`: clean rows with t_ss=0, lag rows with t_ss>0 (FH:1214-1228).
   The fused tex lane always includes SS K/V (`_fusion_plan`: tex lane = `(True, True)`, UG:224). `sample_tex_given_mesh` → `tex_forward_cached` passes no `ss_kv` (UG:1493-1495; GS:299-328).
   Both mode-3 final textures and the `--texture-gt-shape` track (lc/run_unify_bench.py:145-151) therefore run tex|mesh with no SS keys in the union softmax.
   The texture is conditioned on a different attention set than the one it was trained on **[impact speculation]**.
3. **`precompute_geo_kv` leaves out the geo-side cross-t mixer.** Training (bidir) computes `t_emb_s = t_embedder(t_s) + cross_alpha_s * t_mixer_s(t_x)` (UG:1326).
   The cached geo pass uses `t_embedder(t_s)` only (UG:1456). In the 17k EMA, `cross_alpha_s` = −1.16e-3 and `cross_alpha` = −1.46e-2, so the term is small but not zero **[impact speculation]**.
   Strictly, because t_x changes every step, geo is not time-invariant at t_s=0. The one-pass K/V cache (the premise of GS:305-307) is exact only when cross_alpha_s = 0.
4. **The geo lane got no CFG-uncond training in S3, yet inference applies shape CFG 7.5.** `joint_cond_drop=False` in the v12 config, so `cond_s` is built with `mask_drop_prob=0.0` and no replayed drops (FH:1084-1105).
   The geo tower last learned its uncond in S2. At inference the uncond branch nulls `uncond_s` (GS:412). Separately, the SS lane in the uncond branch gets the positive `cond_ss` (GS:230-232 via `**kw`, GS:412-413).
   Training instead drops SS cond together with tex (`_ss_ext` replay, FH:1118-1120). Both are train/infer mismatches in the uncond branch **[impact speculation]**.
5. **The mode-1 (interleave) text benchmark rehearses a lag curve that training never saw.** Round 2 pairs nodes as `k0/S` with `S = SS_PARAMS["steps"]` (GS:514-529). The text launcher sets `SS_STEPS=24` (`scripts/v12/bench_unify17k_m12.sh`, ENVT).
   Training pairs on the 12-step grid (`_t_seq_len_probe`, FH:905-910). The training code also reads the mutable `SS_PARAMS`, so mutating it in a training process would silently change the lag schedule.
6. **The two image-benchmark paths do not treat voxels the same way.** The unify path randomly subsamples to 8192 voxels (GS:218-224). The separate cascade never truncates (demo_pipeline.py:349-352, lc/run_sep_text.py:69-73).
   Unify runs the all-bf16 training layout under autocast. Separate towers run the `from_pretrained` mixed layout (fp32 boundary layers).
   Refine noise uses `seed+1` (GS:425); cascade tex uses `seed` (demo_pipeline.py:390-392).
7. **`eval_suite/scripts/run_v12_image.py` is broken against today's `demo_pipeline`** **[unverified, by reading]**. It monkeypatches `DP.build_stage_cond` (run_v12_image.py:40-42) and calls `pipe.generate(None, ...)` (:65).
   The 2026-09-21 demo (d8d76dd) no longer calls `build_stage_cond`; it dereferences `pack.rec` and `pack.modality` (demo_pipeline.py:101, 319-324), so `pack=None` raises AttributeError.
   The sep106k image results were produced with the older demo (a copy is in `lc/pristine_repo/`).
8. **The unify run's `config.json` is wrong about its own connectors.** It says `cond_seg_embed=False`, `cond_patch_pos='off'`, `vlm_model='Qwen/Qwen3.5-2B'`, yet the connectors carry CondStamp buffers and the real cond VLM is stock Qwen3-VL-2B via `COND_VLM_CKPT`.
   `_connector_from_cfg(cfg)` (UG:1650-1673) / `load_run_connectors` (UG:1773-1783) would build stamp-less connectors and fail strict load. The runners build from the state dict instead (lc/run_unify_bench.py:62-76). BLIP3D must do the same (§5).

---

## 1. Unified model (`UnifiedGeoTexFlow`, UG:251-1537)

### 1.1 Structure as v12 S3 builds it

Assembly: `TNV:398-412` → `assemble_unified_tri(shape_init, tex_init, ss_init, all_trainable=True, cond_seg_embed=False, cond_patch_pos='off', cond_mode='cross_attn', coupling='union', bidirectional=True)` (UG:1582-1628). Then `fused_attn = True` (TNV:416).
Connectors come from `load_tri_connectors` (UG:1631-1647), and all three are trained (TNV:429-433 skip the freeze when `_ss_init`).

| part | what it is | params / keys (17k ckpt) |
|---|---|---|
| `geo_flow` | TRELLIS.2 shape SLAT-512 DiT (30 blocks, 1536 ch, 12 heads × 128, rope, share_mod). Trainable (all_trainable + UNFREEZE_GEO). | 640 tensors |
| `tex_flow` | TRELLIS.2 tex SLAT-512 DiT. Input = cat[x_tex(32); concat_cond(32)] (UG:1307). | 640 |
| `ss_flow` | TRELLIS.2 SS flow (dense 16³ = 4096 tokens, 30 blocks). Complex `rope_phases` buffer is persistent (C64 in ckpt). | 641 |
| `t_mixer`, `cross_alpha` | fresh TimestepEmbedder on t_s into the tex vec (MF v2), zero-init scalar gate (UG:391-400, used at UG:1334, 1490) | 4 + 1 |
| `t_mixer_s`, `cross_alpha_s` | symmetric geo-side mixer on t_x (bidir only; UG:370-377, used at UG:1326) | 4 + 1 |
| `ss_reads_gate` (30,2,12) | per-head zero-init gate for the SS←slat read (UG:493); 17k EMA mean abs 0.004, max 0.084 | 1 |
| `ss_phases_slat` | non-persistent re-rope table for the SS grid mapped c→2c+0.5 into the 32³ frame (UG:505-510) | not in ckpt |

Stream tags on the identity-pad rotation pair (the last two head dims; TRELLIS ropes 63 of 64 pairs): geo 0, tex π/2 (`TAG_TEX=1`), SS 3π/2 (`TAG_SS=3`) (UG:102-139).
Tags are absolute: a stream's own q and k are both rotated (UG:741-747).

### 1.2 Forward call graph (v12 training: bidir + union + fused + SS)

```
forward (UG:1270)                                   # memory controller unused (ELASTIC=False)
└─ _forward_impl (UG:1290-1428)
   ├─ x_x_in = sparse_cat[x_x, tex_concat_cond]  (1304-1307; coords asserted equal)
   ├─ geo: h_s = input_layer(x_s); t_emb_s = t_emb(t_s) + cross_alpha_s·t_mixer_s(t_x)   (1320-1327)
   ├─ tex: h_x = input_layer(x_x_in); t_emb_x = t_emb(t_x) + cross_alpha·t_mixer(t_s)     (1332-1336)
   ├─ corner_on = (t_s != 0) host list  (1368-1373)
   ├─ ss_on → ss_prologue (1070-1088)
   ├─ plan = _fusion_plan(layout, corner_host, n_ss=4096)  (182-231, 1388-1390)
   └─ for idx in 30 blocks:                                 (1391-1399)
        ├─ _run_ss_block(idx, want_kv=True, ss_read=_ss_reads_slat(...))  (1090-1147, 1168-1227)
        ├─ ss_kv = ss_kv_for_slat(k_pre_rope, v)             (1229-1238)  # re-rope into 32³
        └─ _run_block_pair_fused(idx, ..., plan, ss_kv)      (739-782)
             ├─ geo qkv, tex qkv (tex q/k tagged π/2); SS k tagged 3π/2
             ├─ _fused_joint_attn: tex lane K=[geo;tex;SS]; geo lane K=[geo;tex;SS] if t_s>0 else [geo]  (706-737)
             └─ each lane: residual + read-once cross_attn(cond) · xattn_scale(=1) + MLP (fused Triton kernels)
   ├─ out: layer_norm + out_layer for geo and tex (1419-1425)
   └─ returns (v_s, v_x, v_ss) when the SS lane ran, else (v_s, v_x)  (1426-1428)
```

Inference-only entries: `precompute_ss_kv` (UG:1430-1440), `precompute_geo_kv` (UG:1443-1477), `tex_forward_cached` (UG:1479-1498), `ss_forward` (UG:1159-1166).
`forward(..., ss_kv_cache=list)` skips the SS lane and the third output (UG:1382-1383, 1426).

### 1.3 Options: used by v12 vs dead

| option / branch | v12 value | status | location |
|---|---|---|---|
| `coupling="union"` | union | **LIVE** | `_union_attn` UG:570-611 (loop path), fused path UG:706-737 |
| `coupling="gated"` (`_gated_cross_attn`, `b_gates`, `c_gates`) | — | dead | UG:361-364, 378-386, 614-643, 830-831, 843, 874-875 |
| `bidirectional` | True | **LIVE** | UG:355, 370-377, 1365-1413 |
| `fused_attn` | True in training (TNV:416), **False at v12 inference** | **LIVE / mismatch** | UG:359, 1386; see §0.1 |
| `_run_block_pair_bidir` (loop path) | used at inference only by accident | port only as a reference test, or drop | UG:784-855 |
| `_run_block_pair` (one-way S1 path, frozen geo under no_grad) | — | dead | UG:673-703, 1414-1417 |
| `cond_mode="cross_attn"` | cross_attn | **LIVE** | read-once cross-attn in every block |
| `cond_mode="stream"` (cond_proj, cond_blocks, cond_gates_*, cond_reads_gate, `_cond_block_lane`, `_run_block_triple`, `_attn_qkv_norope`) | — | dead | UG:410-478, 884-1032, 1338-1364, 234-244 |
| `xattn_scale` anneal | 1.0 (anneal 0/0) | dead knob (constant 1) | UG:441; train_native.py:630-640 (stream-mode only) |
| `from_scratch` / `MMDiT3D` | False | dead (whole `trellis2_blip3o/mmdit3d.py`, 810 lines) | TNV:337-365; GS:242-250; FH:1081-1082 |
| `all_trainable` | True | **LIVE** (makes `unfreeze_geo` a no-op) | UG:337-344 |
| `unfreeze_geo`, `train()` override for frozen geo | UNFREEZE_GEO=True | redundant under all_trainable | UG:1240-1254 |
| `ss_flow` third tower | present | **LIVE** | UG:302-322, 332, 489-510 |
| `ss_reads_gate` / `_ss_reads_slat` (SS←slat, lag rows only) | trained; used at inference only by mode "interleave" | **LIVE (weak)**; queries are not roped and carry no stream tag, possibly on purpose **[speculation]** | UG:1168-1227 |
| `ss_reads_enabled` (plain attribute, inference switch) | False except interleave round 2 | LIVE | UG:494, GS:188-205, 512 |
| unified-model `cond_seg_embed` / `cond_patch_pos` (per-stream tables) | off (the stamps live on the connectors) | dead | UG:519-567; FH:1463-1482 |
| `test_identity_mask` (G0) | — | test-only | UG:386, 598-605 |
| elastic memory controller | ELASTIC=False | dead in v12; also drops `ss_kv_cache` (UG:1283-1286 does not forward it) | UG:1043-1067, 1277-1288 |
| static GC `_gc_on` | GC=1.0 | LIVE (training) | UG:646-671 |
| `stage1/2_trainable_parameters` | unused (all_trainable) | dead | UG:1501-1536 |
| `sample_mesh_only`, `sample_mesh_only_marginal`, `sample_joint(alpha=inf)` | — | dead for v12 (no mode uses them). mesh_only calls `geo_flow` directly and bypasses t_mixer_s and the SS segment. | GS:238-296, 359-363 |
| assembly / loaders: `assemble_unified`, `assemble_unified_from_run`, `_from_pretrained_pair`, `assemble_unified_inference`, `load_unified_inference`, `load_run_connectors`, `load_connectors` | two-tower / MLP-connector era | dead for v12 (`load_run_connectors` would fail on the unify ckpt, §0.8) | UG:1559-1579, 1676-1803 |

### 1.4 Proposed split of the 1803-line file (BLIP3D)

```
blip3d/models/unified/
  ops.py          ~150 lines  _block_mod_params, _attn_qkv, _attn_out, _dense_* helpers, _rot_pad / TAG_* (UG:77-179)
  plan.py          ~60        _fusion_plan (UG:182-231), with a unit test that reproduces per-sample concatenation (G0 [1])
  ss_lane.py      ~150        ss_prologue / _run_ss_block / ss_epilogue / ss_forward / ss_kv_for_slat / _ss_reads_slat / precompute_ss_kv
  model.py        ~350        UnifiedTriFlow: __init__ (parity asserts UG:286-322 kept verbatim), fused block, forward,
                              precompute_geo_kv (with the t_mixer_s term), tex_forward_cached (with optional ss_kv)
  loading.py      ~150        load_state (+EMA overlay), build_skeleton (NO specialist weight load), load_unified,
                              connector_from_state (§5)
tests/            G0-style: fused-vs-reference softmax, corner bit-exactness, tag antisymmetry, ss re-rope, strict-load key sets
```

Drop: the gated coupling, the cond stream, the one-way pair, the loop-bidir path (keep at most a test oracle), elastic GC, per-stream stamp tables on the model, and every two-tower assembly/loader.
Make the fused kernel path the only path, used by both training and inference.

---

## 2. Unified loss (`compute_unified_geotex_loss`, FH:979-1689; call site TNV:1216-1267)

### 2.1 v12 knob values (checkpoint-17000/config.json + launcher `lc/v12_s3_4n.sh`)

P_SOLO 0.0, P_LAG 0.3, SS_DUAL True, P_CORNER 0.1, P_CORNER2 0.2, k0 3..11, GEO_LOSS_W 1.0 (only active because UNFREEZE_GEO, TNV:1249-1252), ss_loss_w 1.0, mismatch_w 0,
joint_cond_drop False, mask_drop 0.1, dino_drop 0.3, qwen_drop 0.1, fuse_dino True, tex_placeholder on every task (`lc/configs/mix_v12_unify_texfixed.yaml`), flow_weight 1.0, LR 2e-5 WSD.
Run history: 5-node run to 15k, then a 4-node weights-only restart via `--init_from_checkpoint` (train_native.py:1205-1216; optimizer and EMA reset) for 17k more.

### 2.2 Conditioning per lane

- tex: `build_unified_cond(connector_tex, ...)` with the full dropout stack. Realized drops are recorded in `_sdrops` (FH:1066-1074).
- geo: `connector_geo` with `mask_drop_prob=0.0` and no ddrop/qdrop. `ext_drops` is set only when `joint_cond_drop` (FH:1084-1105). **In v12, geo never sees an uncond.**
- SS: replays the tex realized drops (FH:1118-1128). Returns the dense sdpa mask for the SS cross-attn.
- CondStamp: `_stamp` falls back to the connector's fixed `cond_seg_embed` / `cond_patch_pos` buffers when the unified model has none, which is the v12 case (FH:1054-1065).
- Uncond is **not** all zeros. A drop row gets `connector(cond_hidden*0)` + view code + patch pos + text-seg code on the qwen segment, and a zeroed dino segment (dino seg code included) with its keys kept (FH:601-632). Eval reproduces this by forcing `ext_drops` (eval_cond.py:125-139).

### 2.3 Row classes and timesteps (FH:788-902, `sample_timestep_triples`)

All primitives are drawn for every row in a fixed order and composed with `torch.where` (RNG discipline, FH:801-805). Invariant t_ss ≤ t_s ≤ t_x (asserted, FH:900-901).

| class | prob (v12) | t_ss | t_s | t_x | SS loss | slat losses |
|---|---|---|---|---|---|---|
| solo | 0.0 | logitNormal(1,1) | 1 | 1 | yes | masked (dead in v12) |
| lag | 0.3 | shift₅(u − k0/12) | shift₃(u), u ∈ (k0/12, 1], k0 ~ U{3..11} | t_s + (1−t_s)·U | yes (row-balanced) | yes; **only class where SS reads slat** (`ss_read_on=m_lag`, FH:1228) |
| clean | 0.7 | 0 (raw latent) | from `sample_timestep_pairs`, corners rescaled (see below) | same draw | interleaved term masked (unlearnable at t=0); SS_DUAL gives these rows a separate term | yes |

The clean class calls `sample_timestep_pairs` (FH:682-751) with p_c_eff = 0.1 × (1/0.7) = 0.143 and p_c2_eff = 0.2/0.7 = 0.286 (rescale at FH:853-861).
The draw is: t_s ~ U[0,1], then t_x ~ U[t_s, 1]. On the t_s=0 edge (prob 0.143) t_s is set to 0 and t_x is redrawn as U[0,1]. On the t_x=1 edge (0.286) t_x is set to 1.

- Tex-less rows (`tex_valid=False`, placeholder target) get t_x forced to 1 (FH:1158-1160) and their tex voxels are removed by row selection (FH:1251-1258).
  `t_x_sampled` keeps the scheduler's draw for logging (FH:1157).
- SS_DUAL: a second standalone `ss_forward` on the clean rows only, t2 = sigmoid(1 + N(0,1)), under explicit autocast (FH:1312-1343). Structurally isolated from the clean pass.
- Noising: `loss_fn_slat.diffuse` / `get_v` (official TRELLIS2FlowMatchingLoss, sigma_min 1e-5). The tex concat_cond is the **noised geo state x_ts** (FH:1210), in the same space as shape-norm.
  The SS target latent is **raw** (the SS config has no normalization).
- t is cast to the target dtype before ×1000 (FH:1193-1198, 1233-1234). If the targets arrive as bf16, the training timesteps are bf16-quantized, while inference passes fp32 t·1000 **[open question: target dtype not checked]**.

### 2.4 Loss terms

- tex: `_voxel_balanced_mse` over voxels in rows (tex_valid ∧ ¬solo) (FH:1264-1276). Global-count normalization with all_reduce, called unconditionally (FH:945-976).
- SS interleaved: `_row_balanced_mse(v_ss, target, ¬clean)` (FH:1285-1287). SS dual: `_row_balanced_mse` on the clean rows (FH:1342-1343).
- geo: voxel-balanced MSE on rows (t_s ≠ 0) ∧ ¬solo, weight 1.0 (FH:1534-1570). t_x=1 rows are **not** masked for tex (FH:1502-1506).
- Total = tex + ss + ss_dual + geo (× flow_weight 1).
- Diagnostics: `probe_every=200` is the default and is **live**. It runs two extra no-grad forwards at t=1 with rolled cond and logs `cond_sens_geo/tex` (FH:1634-1687). The probe omits the SS lane.

### 2.5 Dead in the loss for v12

from_scratch branch (FH:1081-1082), `joint_cond_drop` (FH:1085-1090, 1118), mismatch hinge (FH:1594-1616), the unified-model stamp logs (FH:1463-1482), the `ss_flow_present=False` branch (FH:1149-1153), and the solo class (p_solo=0; the code is general, keep it only if BLIP3D wants the knob).
**Keep:** `_row/_voxel_balanced_mse` (collective discipline), `sample_timestep_triples` (with the lag grid pinned to a constant, not read from the mutable `SS_PARAMS`), the tex placeholder masking, SS_DUAL, the geo corner mask, and the global row-count logs.

---

## 3. Inference inventory

### 3.1 Samplers

| impl | where | what | status / reference? |
|---|---|---|---|
| `_t_seq`, `_guided`, `_pred_to_xstart`, `_xstart_to_pred`, `warp_tx` | GS:67-124 | hand port of TRELLIS.2 FlowEuler + CFG + guidance-interval mixins | **reference** for the math (sparse-safe global std) |
| `FlowEulerGuidanceIntervalSampler` (official) | third_party TRELLIS.2 `pipelines/samplers/*`; used by demo_pipeline.py:216, 284; lc/run_sep_text.py:42, 68; ~30 scripts | dense SS sampling | reference for **separate** SS. Numerically equivalent to `_guided` on dense tensors (unbiased vs population std cancels in the ratio). |
| `GeoTexSampler.sample_three_tower` | GS:153-235 | SS alone → occ → coords (trunc 8192) → `_joint_on_coords` (+ refine) | **reference** for unify modes "joint" and "refine" |
| `GeoTexSampler._joint_on_coords` | GS:367-426 | α-warped joint loop, SS lent at t_ss=0, optional `cache_ss_kv`, refine = `sample_tex_given_mesh(seed+1)` | reference (but see §0.1, §0.4) |
| `GeoTexSampler.sample_tex_given_mesh` | GS:299-328 | cached geo K/V at t_s=0, tex steps; skips neg forward at cfg 1 | reference for refine / tex-on-GT-shape (but see §0.2, §0.3) |
| `sample_three_tower_interleaved` (monkeypatched onto the class) | GS:434-572 | round 1 = three_tower(refine=False); **reruns SS** to recover z (GS:474-492); round 2 = lag-paired SS with slat reads; final `_joint_on_coords` | reference for mode "interleave". Wasteful double SS; monkeypatch pattern should go. |
| `sample_mesh_only`, `sample_mesh_only_marginal`, `sample_joint` | GS:238-365 | two-tower modes | dead for v12 |
| `sample_shape` | scripts/eval_fusion_v22.py:120-152 | separate shape tower, released params via `SHAPE_PARAMS` | **reference** for the cascade shape stage |
| `sample_tex` | scripts/eval_tex_v22.py:60-89 | separate tex tower, x_in = cat[x; shape_z] | **reference** for the cascade tex stage |
| `sample_ss` | scripts/export_glb_fullchain.py:55-67 | legacy uniform 25 steps, cfg 3, no interval | **dead / wrong**; do not port |
| `eval_render.sample_slat` | scripts/eval_render.py:40 | legacy 25/3.0 | dead |
| `Pipeline._sample_ss` / `generate` | trellis2_blip3o/demo_pipeline.py:278-409 | resident separate cascade with per-modality presets | reference for **presets** (SS/tex text switch) and the product API shape |

### 3.2 Loaders

| impl | where | notes |
|---|---|---|
| `load_state_dict(ckpt, use_ema)` | benchmarks/checkpoint.py:17-28 | model.safetensors + EMA overlay. **Reference.** |
| `load_connector(ckpt, state)` | benchmarks/checkpoint.py:39-106 | reads **only** `diffusion_connector.*` (:57-61); infers xf2/mlp and the stamp flags from keys (:65-82); maps patch_pos 'off' + key present to 'zero' (:75-76; the buffer is overwritten anyway); strict load |
| `load_flow_and_connector` (shape) | scripts/eval_fusion_v22.py:41-55 | `from_pretrained(DEFAULT_SHAPE_SLAT)` layout, `strict=False`, **missing keys only printed** (:47) |
| `load_tex_flow` | scripts/eval_tex_v22.py:45-57 | same pattern, `strict=False`, missing only printed (:50) |
| `load_ss_flow` | scripts/export_glb_fullchain.py:38-52 | strips `_orig_mod.`, asserts missing==0 (:48) |
| `SSDEC` | scripts/export_glb_fullchain.py:34-35 | **TRELLIS v1** `ss_dec_conv3d_16l8_fp16` at a hard-coded `~/.cache/huggingface` path |
| `assemble_unified_tri` | UG:1582-1628 | training (all-bf16) builders + **loads all three specialist weights**, strict |
| unify load in runners | lc/run_unify_bench.py:77-81; lc/unify_render.py:105-112 | builds via `assemble_unified_tri(skeleton ckpts)` and then strictly overwrites with `unified_geotex.*`. The specialist weight load is wasted I/O. Reads model.safetensors **without** EMA overlay; trusts the `_ema` dir. |
| connector-from-state (unify) | lc/run_unify_bench.py:62-76; lc/unify_render.py:44-63 | **reference** for unify connectors (§5) |
| `load_tri_connectors` / `_connector_from_cfg` | UG:1631-1673 | config-driven; right for the S2 tower dirs, wrong for the unify dir |
| decoders + norms | tr2_modules.py:116-125, 161-163, 166-195 | `build_sc_vae_{shape,tex}_decoder_frozen`, `load_norm_stats` (**returns None silently if the config file is missing**) |
| cond encoder | live_cond_batch.py:363-399; eval_cond.py:75-140 | `TrainCondEncoder(vlm_path)` + `cond_uncond` = **reference** conditioning (§4.9) |

### 3.3 Exporters / decode

| impl | where | notes |
|---|---|---|
| `build_mw` | scripts/export_glb_v22.py:31-43 | **reference**. shape_dec.set_resolution(512), fill_holes (silent except), tex_dec(guide_subs)·0.5+0.5, MeshWithVoxel origin −0.5, voxel 1/512, PBR layout |
| `export_glb` | scripts/export_glb_v22.py:46-54 | **reference**. `o_voxel.postprocess.to_glb(decimation 200000, texture 2048, remesh=True, band 1, project 0)`, webp. Same params as eval_suite/scripts/run_trellis2.py:81-95. The official benchmarks/run_trellis2.py:205-220 also calls `mesh.simplify(16_777_216)` first; ours does not. |
| `render_textured` | scripts/eval_tex_v22.py:92-128 | duplicate of build_mw + render; logs fill_holes failures (build_mw swallows them) |
| `voxel_cubes_glb` / `_voxel_cubes_glb` | export_glb_fullchain.py:70-85 / demo_pipeline.py:135-152 | duplicates |
| `shape_mesh_glb` / `_shape_mesh_glb` / `shape_glb` | export_glb_fullchain.py:88-97 / demo_pipeline.py:155-164 / lc/run_unify_bench.py:94-96 | three copies of the [x, z, −y] swap |
| `_shape_mesh_obj` / inline trimesh OBJ | demo_pipeline.py:167-176 / lc/run_unify_bench.py:143 | decoder-native Y-up, no swap |
| other `to_glb` callers | scripts/viewer3d_s3_eval.py:189, scripts/export_glbs.py:33, benchmarks/run_pixal3d.py:79-92 | legacy / baselines |

### 3.4 Runners (v12)

| runner | model | notes |
|---|---|---|
| lc/run_unify_bench.py (≠ `scripts/v12/run_unify_bench.py`; the lc copy adds `--cache-ss-kv`, `--save-latents`, `--timing-only`) | unify 17k | `--sampler refine|joint|interleave`, env overrides SS_*/SHAPE_*/TEX_*/TEX_GI/SS_GI (:50-58). **Reference unify runner.** |
| lc/unify_render.py (== scripts/v12 copy) | unify | two-phase (encode, then sample+render); same sampler calls |
| lc/run_sep_text.py (== scripts/v12 copy) | separate v12 S2 | text track; official SS sampler + `sample_shape`/`sample_tex`; **reference separate runner** |
| eval_suite/scripts/run_v12_image.py | separate v12 S2 via demo_pipeline | **broken vs current demo** (§0.7) |
| trellis2_blip3o/demo_pipeline.py | separate v12 S2 | resident; per-modality presets. Default `vlm=V22_CKPT`, which resolves to `COND_VLM_CKPT` at import (live_cond.py:54). |
| eval_suite/text/run_geotex_v10_text.py, run_ss_swap_text.py, scripts/_tmp/eval_v10*.py | v10 | dead for v12 |

### 3.5 Duplicates to collapse in BLIP3D

1. SS sampling: the official sampler (demo, sep_text) plus the hand loop (GS:191-203, and again at GS:477-490). Collapse to one `sample_ss` built on `_guided`.
2. The SS noise → occupancy → coords chain, written 4× (demo_pipeline.py:286/349-352, run_sep_text.py:69-73, occ_metrics.py:71-99, export_glb_fullchain.py:137-150). Collapse to one `occ_to_coords`.
3. Shape/tex sampling: `sample_shape`/`sample_tex` vs `sample_mesh_only`/`sample_tex_given_mesh`. Same Euler/CFG core, different forward. Collapse to one `euler_cfg(forward_fn, x0, preset, seed)` driver.
4. Connector construction: 4 variants (`load_connector`, `_connector_from_cfg`, `load_connectors`, the runner inline). Collapse to a single `connector_from_state` (§5).
5. Mesh/GLB helpers: 3× shape GLB, 2× voxel cubes, build_mw vs render_textured.
6. Preset tables: `SS_SAMPLER`/`SS_TEXT_PARAMS`/`TEX_TEXT_PARAMS` (demo) vs env overrides (runners) vs module globals `GS.*_PARAMS`. Collapse to frozen preset objects (§6).

---

## 4. Hidden contracts

1. **Time grid.** `t_seq = r·u/(1+(r−1)u)` on u = linspace(1, 0, steps+1) (GS:67-71 = flow_euler.py `sample`). The model is called at the first `steps` nodes only; the final 0 is never an input. t goes in as `t·1000` float (GS:197, 260).
   Update: `x ← x − (t − t_prev)·v`.
2. **Guidance interval.** Inclusive `lo ≤ t ≤ hi` on the node value t. Outside it: the conditional prediction alone (strength 1), not a blend (GS:91-92 = guidance_interval_mixin.py).
   In the joint loop, tex guidance is evaluated at the **warped** tx (GS:415-417). TEX steps/rescale_t are **ignored** there because tex follows the shape grid (GS:386-387). They only affect refine and the separate tex tower.
3. **CFG rescale.** `x0 = r·x0_cfg·std(x0_pos)/std(x0_cfg) + (1−r)·x0_cfg`, std = **one scalar over all voxels and channels** (population std) (GS:93-108). Plain per-row std on an (N,C) tensor is a different operator (documented bug class, GS:97-105).
   Dense SS: equivalent to official (the unbiased/population factor cancels).
4. **`_guided` shortcut.** When strength == 1 the negative branch is irrelevant; `sample_tex_given_mesh` skips the uncond forward (GS:323-324). The joint loop always runs both.
5. **Alpha warp.** The shape stream leads on its own grid; tex time = f_α(t_s) = α·t/(1+(α−1)t), α=32 (GS:112-124, 386-387). α=∞ means the literal cascade (dead for v12).
   The last joint step covers about 90% of the tex trajectory, which is why refine exists (GS:344-354).
6. **Seeds.** SS noise `randn(1,8,16³)` from `Generator(seed)` (GS:185, 193-194). Truncation randperm uses the **same** generator (GS:222).
   Joint noise: a new `Generator(seed)` yields x_s then x_x (GS:380-384). Refine uses **seed+1** (GS:425). Interleave draws round-2 noises from the first generator after round 1 (GS:458, 517-520).
   The separate cascade uses `seed` for SS, shape and tex alike (demo_pipeline.py:280, 359-360, 390-392; run_sep_text.py:66, 75-76).
7. **Voxel truncation.** Unify only: n > 8192 → random subset (sorted) and `info.truncated=True` (GS:218-224, 558-560). Training resamples assets over `max_slat_tokens`. Separate cascade: no cap.
   An empty SS gives `info.empty` and callers write `error.json`, never a stub asset (lc/run_unify_bench.py:128-130).
8. **Occupancy decode.** SS latent is **raw** (no normalization; `load_norm_stats` returns None, occ_metrics.py:74-79). Occupancy = `ssdec(z) > 0` (64³ logits).
   Coords = `max_pool3d(occ64.float(), 2, 2) > 0.5` → argwhere → prepend batch column 0 as int32 (occ_metrics.py:86-99; demo_pipeline.py:349-352). The SS decoder is TRELLIS **v1** image-large (export_glb_fullchain.py:34-35).
9. **Conditioning.** Env must be set **before import**:
   - `COND_VLM_CKPT` = stock Qwen3-VL-2B snapshot 8964489…. `V22_CKPT` resolves at `live_cond` import (live_cond.py:54); demo_pipeline.py:42-45 setdefaults it.
   - `GEOTEX_IM_TOK_PER_VIEW=256` (live_cond_batch.py:56).
   - `VIEW_EMBED_SCALE` (0.2), used only when a hashed table is extended past its 16 stored rows (view_code.py:17-28, eval_cond.py:104-112).

   Cond = `cond_uncond(connector, TrainCondEncoder record, dino_view_embed)` (eval_cond.py:75-140). It returns key-selected (1,T,C) tensors: cond = no drops; uncond = forced CFG drop (§2.2).
   CondStamp buffers are read off the connector. The connector's `forward` never applies them; the caller does (demo_pipeline.py:48-57, FH:614-632).
   Image path = fusion branch cat[dino(+view code+seg0); qwen(+view code+patch pos+seg1)]. Text path = plain branch, qwen + seg1.
   `dino_view_embed` is the unify ckpt's top-level buffer (16×1024 hashed), shared by all three connectors.
10. **Normalization stats.** shape = `SHAPE_SLAT_CONFIG 'normalization'`; tex = `TEX_SLAT_CONFIG 'pbr_slat_normalization'`; tex concat = `TEX_SLAT_CONFIG 'shape_slat_normalization'`.
    That last set is bit-identical to shape normalization, guarded by `tests/test_unified_conventions.py` (UG:404-408). The unify path feeds the shape-norm latent directly as concat.
    raw = norm·std + mean before decode (lc/run_unify_bench.py:131; demo_pipeline.py:361, 397).
11. **Model-side time/cond conventions.** t=0 clean, t=1 noise, x_t = (1−t)x0 + (σ_min + (1−σ_min)t)ε, v = (1−σ_min)ε − x0, σ_min = 1e-5 (UG:47-49; GS:64-81).
    Corner mask `t_s != 0` in flow units (UG:1368). The SS lane in joint/refine is fed the **generated** z at t_ss=0 (GS:230-232).
12. **Dtype layouts.** Unify: training builders `_uniform_bf16` (all bf16, complex rope preserved; builder.py:22-56, 105-129) + `torch.autocast(bf16)` in the runner (lc/run_unify_bench.py:113), fp32 noise states.
    Separate towers: `from_pretrained` mixed layout (fp32 input/out layers) (UG:1718-1736 docstring; eval_fusion_v22.py:43). `_joint_on_coords` enables autocast itself only when the SS lane is present (GS:400-402).
13. **EMA overlay.** "Product weights = EMA." `load_state_dict` / `_load_prefixed_state` overlay `ema.safetensors` onto model (checkpoint.py:25-27; UG:1550-1552).
    `*_ema` dirs are pre-folded by `_lc/mk_ema_init.py`, which **casts EMA fp32 → bf16** onto the model dtype. EMA holds no buffers (stamps, rope_phases, dino_view_embed).
    run_unify_bench reads model.safetensors only, so pointing it at a raw dir silently gives raw weights while status.json still says "EMA-overlaid".
14. **GLB / UV / axes.** `to_glb` output is glTF (UV origin top-left). The eval-suite UV bug was in the **reader** (`prepare_trellis2_e2e.load_pbr_mesh`, fixed 2026-09-02), not the exporter. Any new GLB consumer must run `text/uv_flip_probe.py`.
    `shape.obj` is decoder-native Y-up. `shape.glb` / voxel GLBs apply [x, z, −y] to match to_glb's swap.
    PBR voxel layout: base_color 0:3, metallic 3, roughness 4, alpha 5. The tex decoder output is mapped ·0.5+0.5.
    Flat-colour assets have darker GT by toolkit convention (memory: flat-colour-factor-convention).
15. **Mutable global presets.** `GeoTexSampler.__init__` copies SHAPE/TEX params at construction (GS:137-138). SS params are read **live** at every call (GS:191, 477). `sample_shape` / `sample_tex` read `SHAPE_PARAMS` / `TEX_PARAMS` live.
    The demo swaps `TEX_PARAMS` under a lock (demo_pipeline.py:384-395). The training lag grid reads `SS_PARAMS` too (FH:905-910).

---

## 5. Checkpoint keys and building connectors from the state dict

### 5.1 Unify run (`runs/keep/v12/s3_unify_4n/checkpoint-17000` → `geotex_s1_v12s3_4n/checkpoint-17000`; the eval dir is `forks/geotex_s1_v12s3_4n/checkpoint-17000_ema`)

model.safetensors holds 2047 tensors: 2046 BF16 + 1 C64. ema.safetensors holds 2039, all F32.

```
unified_geotex.geo_flow.*   640      unified_geotex.tex_flow.*   640
unified_geotex.ss_flow.*    641      (incl. ss_flow.rope_phases [4096,64] C64 — model only, not EMA)
unified_geotex.t_mixer.mlp.{0,2}.{weight,bias}        [1536,256],[1536],[1536,1536],[1536]
unified_geotex.t_mixer_s.mlp.{0,2}.{weight,bias}
unified_geotex.cross_alpha [1]   unified_geotex.cross_alpha_s [1]   unified_geotex.ss_reads_gate [30,2,12]
geo_connector.*  38   diffusion_connector.* 38 (= TEX)   ss_connector.* 38
   xf2: in_proj.{weight [1024,2048],bias}; blocks.{0,1}.{norm1,qkv,q_norm,k_norm,proj,norm2,mlp.0,mlp.2}.*; out_norm.*
        cond_seg_embed [2,1024] (buffer)  cond_patch_pos [1024,1024] (buffer, sincos2d 32×32)   — model only, not EMA
dino_view_embed [16,1024] (hashed buffer) — model only
```

Absent, as expected for union / cross_attn / stamp-off: `b_gates`, `c_gates`, `cond_proj`, `cond_blocks`, `cond_gates_*`, `cond_reads_gate`, `unified_geotex.cond_seg_embed`, `unified_geotex.cond_patch_pos`.

### 5.2 Separate S2 towers (`runs/keep/v12/s2_{shape,tex,ss}/checkpoint-106000[_ema]`)

`shape_slat_512.*` / `tex_slat_512.*` (640) or `ss_flow.*` (641, may carry an `_orig_mod.` infix, which is stripped), `diffusion_connector.*` (38, xf2 + stamps), `dino_view_embed`.
Ignored extras: `trellis_decoders.*` (1572, shape/tex ckpts), `repa_projector.*` and `_road_step` (SS ckpt, ROAD aux).
Tower configs say `cond_seg_embed=True, cond_patch_pos=sincos2d`. Here the config is correct.

### 5.3 Connector-from-state-dict logic BLIP3D must support (generalize lc/run_unify_bench.py:62-76)

```python
def connector_from_state(sd: dict, prefix: str, adapter_blocks: int = 2) -> nn.Module:
    s = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}   # assert non-empty
    if "in_proj.weight" in s:                     # xf2 TRELLIS2TransformerAdapter
        cond_dim, vlm_dim = s["in_proj.weight"].shape
        n_blocks = 1 + max(int(k.split(".")[1]) for k in s if k.startswith("blocks."))   # infer, don't trust config
        P = isqrt(s["cond_patch_pos"].shape[0]) if "cond_patch_pos" in s else 32
        conn = TRELLIS2TransformerAdapter(vlm_dim, cond_dim, n_blocks=n_blocks,
                   seg_embed="cond_seg_embed" in s,
                   patch_pos="sincos2d" if "cond_patch_pos" in s else "off",   # buffer value comes from ckpt
                   patch_lattice=P)
    elif "fc1.weight" in s:                       # legacy MLP TRELLIS2Connector
        cond_dim, vlm_dim = s["fc1.weight"].shape; conn = TRELLIS2Connector(vlm_dim, cond_dim)
    if any(k.startswith("pos_stamp.") for k in s): attach DinoPosStamp   # legacy (checkpoint.py:88-92)
    conn.load_state_dict(s, strict=True)
```

Prefix maps:

| checkpoint kind | ss | geo/shape | tex |
|---|---|---|---|
| unify | `ss_connector.` | `geo_connector.` | `diffusion_connector.` |
| tower | the tower's own `diffusion_connector.` | same | same |

Also required:
- `patch_pos='sincos2d_learn'` makes `cond_patch_pos` an nn.Parameter (connector.py:178-183). The name in the state dict is the same; it only matters for training.
- Model keys: strip `unified_geotex.` and load strictly. Build the skeleton **without** loading specialist weights, e.g. from TRELLIS.2 json configs + `_uniform_bf16`.
  The complex `rope_phases` must survive dtype casts (the `to_bf16_keep_complex` assert at UG:311-314).
- `dino_view_embed`: take it from the ckpt; extend via `ensure_view_codes` if more than 16 views.
- Record the cond-VLM path in BLIP3D checkpoint metadata. The HF config's `vlm_model` field is not authoritative (§0.8).

---

## 6. Proposed `blip3d/infer/` API

```python
# presets.py — frozen, explicit; no module-global mutation
@dataclass(frozen=True)
class StageParams: steps: int; cfg: float; rescale: float; interval: tuple[float, float]; rescale_t: float
RELEASED_SS    = StageParams(12, 7.5, 0.7, (0.6, 1.0), 5.0)
RELEASED_SHAPE = StageParams(12, 7.5, 0.5, (0.6, 1.0), 3.0)
RELEASED_TEX   = StageParams(12, 1.0, 0.0, (0.6, 0.9), 3.0)
@dataclass(frozen=True)
class Preset: ss: StageParams; shape: StageParams; tex: StageParams; alpha: float = 32.0; max_voxels: int | None = 8192
IMAGE = Preset(RELEASED_SS, RELEASED_SHAPE, RELEASED_TEX)
TEXT  = Preset(replace(RELEASED_SS, steps=24, cfg=10.0), RELEASED_SHAPE,
               StageParams(24, 7.5, 0.5, (0.9, 1.0), 5.0))
def preset_for(modality: Literal["image", "images", "text"]) -> Preset
TRAIN_LAG_GRID = dict(steps=12, ss_rescale_t=5.0, slat_rescale_t=3.0)   # used by training AND interleave; never derived from a preset

# schedule.py
def t_seq(steps: int, rescale_t: float) -> list[float]
def warp(t: float, alpha: float) -> float
def guided(v_pos, v_neg, x_t, t: float, p: StageParams) -> Tensor        # scalar global std rescale
def euler_cfg(forward: Callable[[Tensor, float, bool], Tensor], x: Tensor, p: StageParams) -> Tensor

# cond.py
class CondEncoder:  # wraps TrainCondEncoder; vlm_path required (no env import-order magic); tok_per_view=256
    def encode_image(self, img: Image, remove_bg=True) -> CondRecord
    def encode_images(self, imgs: list[Image]) -> CondRecord
    def encode_text(self, text: str, template: int = 0) -> CondRecord
def cond_uncond(connector, rec: CondRecord, view_codes: Tensor | None) -> tuple[Tensor, Tensor]

# loading.py
def load_state(ckpt_dir: str, use_ema: bool = True, require_ema: bool = True) -> dict[str, Tensor]
def connector_from_state(sd, prefix: str) -> nn.Module                       # §5.3
@dataclass
class Tower: flow: nn.Module; connector: nn.Module; view_codes: Tensor | None
def load_tower(ckpt_dir: str, kind: Literal["ss", "shape", "tex"], use_ema=True) -> Tower   # strict, asserts no missing
@dataclass
class UnifiedBundle: model: UnifiedTriFlow; conns: dict[str, nn.Module]; view_codes: Tensor
def load_unified(ckpt_dir: str, use_ema=True) -> UnifiedBundle                # skeleton from configs; fused=True
@dataclass
class Decoders: ss_dec; shape_dec; tex_dec; shape_norm; tex_norm   # hard-fail if a norm config is missing
def load_decoders() -> Decoders

# stages.py (shared by cascade and unified)
def sample_ss(forward_ss, c, u, p: StageParams, seed: int) -> Tensor                 # (1,8,16,16,16) raw
def occ_to_coords(ss_dec, z, max_voxels: int | None, gen) -> tuple[Tensor, dict]     # (N,4) int32 + info(empty/truncated)

# cascade.py
def sample_shape(tower: Tower, c, u, coords, p, seed) -> Tensor          # norm feats (N,32)
def sample_tex(tower: Tower, c, u, coords, shape_norm, p, seed) -> Tensor
def run_cascade(towers: dict[str, Tower], dec: Decoders, rec: CondRecord, preset: Preset, seed=0,
                max_voxels: int | None = None) -> Latents

# unified.py
class UnifiedSampler:
    def __init__(self, b: UnifiedBundle, preset: Preset, cache_ss_kv: bool = True)
    def ss_alone(self, conds, seed) -> Tensor
    def joint(self, coords, conds, z_ss, seed) -> tuple[Tensor, Tensor]          # α-warped, SS K/V cached
    def tex_given_mesh(self, coords, shape_norm, conds, seed, z_ss: Tensor | None = None) -> Tensor  # passes SS K/V when z_ss given
    def interleave(self, ss_dec, conds, seed, k0: int = 7) -> ...                # returns z from round 1; no SS rerun
def run_unified(b, dec, rec, mode: Literal["refine", "joint", "interleave"], preset: Preset, seed=0,
                k0=7) -> Latents        # refine == mode 3, joint == mode 2, interleave == mode 1

# export.py
@dataclass
class Latents: coords; shape_norm; tex_norm; info: dict
def decode(dec: Decoders, lat: Latents) -> MeshWithVoxel
def export_glb(mw, path, decimation=200_000, texture=2048) -> str
def export_shape_obj(mw, path) -> str;  def export_shape_glb(mw, path) -> str;  def export_voxels_glb(occ64, path) -> str
def generate(model, rec, *, mode="cascade"|"refine"|"joint"|"interleave", preset=None, seed=0, out_dir) -> Result
```

Name modes by meaning, not number. The repo's own numbering conflicts: GS docstrings call joint ①, tex|mesh ②, mesh ③, three-tower ④ (GS:152, 237, 298, 330), while the bench calls refine 3, joint 2, interleave 1 (lc/run_unify_bench.py:24).
Keep an explicit `bench_compat=True` flag that reproduces today's numbers (loop path, no SS in refine, seed conventions) so the fixes in §0 can be A/B'd.

---

## 7. Risks and open questions

1. **(High)** Fused vs loop tag convention at inference (§0.1). Do the unify17k numbers (image ≈ separate; text CLIP-T 26.6 vs 27.0) change on the trained fused path? Not measured.
2. **(High)** Refine / texture_gt_shape runs without the SS segment (§0.2). Should BLIP3D's `tex_given_mesh` take `z_ss` (generated, or GT SS latent for the controlled-texture track)? Needs a small A/B.
3. **(Med)** `precompute_geo_kv` drops `cross_alpha_s·t_mixer_s(t_x)` (§0.3). The fix is exact but costs a geo pass per tex step when t_x varies. Alternatively, confirm the term is negligible (|cross_alpha_s| ≈ 1e-3; the t_mixer_s output scale was not measured).
4. **(Med)** Uncond-branch mismatches: geo never sees uncond in S3, and the SS lane uses positive cond in the uncond branch (§0.4). Decide before any further unify training: turn on `joint_cond_drop`, or make the inference uncond branch match training.
5. **(Med)** Mode-1 text used SS_STEPS=24 in the lag pairing (§0.5). BLIP3D should pin the lag grid to `TRAIN_LAG_GRID`.
6. **(Med)** 8192 random truncation (unify only) punches holes and biases unify-vs-separate comparisons. How often did it fire on sfv50? (`status.json` has `truncated`; not tallied here.)
7. **(Med)** `run_v12_image.py` is broken against the current demo (§0.7). Any rerun of the separate image track needs a fix or the `lc/pristine_repo` demo.
8. **(Med)** Separate-tower loaders use `strict=False` for shape/tex with only a printed missing count (eval_fusion_v22.py:47, eval_tex_v22.py:50). BLIP3D must assert.
9. **(Low)** Timestep dtype in training (cast to target dtype; bf16-quantized if targets are bf16) vs fp32 at inference **[open question]**.
10. **(Low)** SS←slat read (mode 1) uses un-roped SS queries against roped slat keys, with no stream tag (UG:1213-1220). The gate is tiny (mean 4e-3). Mode 1 is the only consumer. Worth porting only if mode 1 is kept.
11. **(Low)** Unify inference layout (all-bf16 + autocast) ≠ separate-tower layout (mixed fp32 boundaries). A confound in unify-vs-separate deltas **[speculation on size]**.
12. **(Low)** The unify config.json is misleading (§0.8). The EMA fold casts to bf16. The unify runner does not verify EMA.
13. **(Low)** Uncommitted repo edits (`cache_ss_kv`, `precompute_ss_kv`) back the lc runner's `--cache-ss-kv`. BLIP3D should treat them as spec, not as committed history.
    The claim that `cache_ss_kv` is exact holds by construction (SS lane at t_ss=0 with no slat read, and cond_ss is the same object in both branches) **[not executed here]**.
14. **(Low)** `forward()` with a memory controller silently drops `ss_kv_cache` (UG:1283-1286). Dead in v12; don't port the controller.
15. **(Info)** `load_norm_stats` returning None on a missing config is only correct for SS. The hard-coded `SSDEC` path lives under `~/.cache`. `demo_pipeline` default checkpoints are `/fsx/data/weikai.huang/runs/...` literals (demo_pipeline.py:35, 64-66).
