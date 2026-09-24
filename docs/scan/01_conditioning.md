# BLIP3D port audit, part 1: the conditioning path (v12)

Repo: `/fsx/home/weikai.huang/3dgen/model/BLIP3o` @ `BLIP3o-NEXT` (clean). Read-only audit, 2026-09-24.
All paths below are relative to the repo root unless absolute. `L123` = line 123.

**How I checked this.** I read every file listed in the brief. On CPU I also ran four checks that only read files:
(a) the stock Qwen3-VL-2B processor on synthetic 1024² / 512² images, to get exact token layouts and boilerplate masks;
(b) the v12 checkpoint headers and configs (`/fsx/data/weikai.huang/runs/keep/v12/*`);
(c) the CondStamp and view-code buffers, regenerated from code and compared with the checkpoints;
(d) the transformers 5.2.0 source, to see what `hidden_states[-1]` actually returns.
Anything I did not verify is marked **[unverified]**.

---

## 0. Corrections to the brief's "ground truth" (verified)

| Brief says | Actual (evidence) |
|---|---|
| CFG dropout: qwen 0.1 | Only S3 (unify) has `qwen_drop_prob=0.1`. **S1 and all three S2 towers used `qwen_drop_prob=0.0`.** Evidence: `config.json` of `keep/v12/{s1_connector/shape_20000_ema, s2_{shape,tex,ss}/checkpoint-106000_ema}`, and `_lc/v12_s2.sh` / `train_native_split.sh`, which never pass `--qwen_drop_prob` (the default is 0.0 at `train_native.py:898`). The whole-row drop (0.1) and the DINO drop (0.3) are the same in every stage. |
| cond assembly in training = `build_unified_cond` | That is true for **S3 only** (`train_stages=geotex` → `compute_unified_geotex_loss` → `build_unified_cond`, `flow_heads.py:1066/1098/1121`). **S1 and S2 (the three towers) trained through `compute_cascade_flow_loss`'s own fusion and plain branches** (`flow_heads.py:203-283`), called from `trellis_native_vlm.py:1346`. That code is parallel to `build_unified_cond`, not shared with it. On the deterministic path the two are equivalent (the drift guard is `tests/test_unified_conventions.py::test_unified_cond_and_sampler`), but the RNG draws differ. |
| "uncond = keys kept, values zeroed" | This holds for the **DINO segment only**. For the Qwen segment the CFG drop zeroes the connector *input*: `connector(hidden*0)` (a non-zero constant), and then the view code, patch-position code and Qwen segment code are **still added** (`flow_heads.py:602-631`; only `dino_seg * keep` at `L632`). So the uncond Qwen segment is `connector(0) + stamps`, not zero. The comments at `L623-625` / `L236-237` ("a CFG-dropped row stays all-zero") are wrong for the Qwen half. |
| hashed view codes, "unlimited views" | This works only while the table is fp32. The checkpoints store `dino_view_embed` in **bf16**, and `view_code.detect_view_code_mode` (`view_code.py:40-48`, `atol=1e-4`) then classifies it as `"learned"`. As a result `ensure_view_codes(table, n>16)` **raises** (verified on `s3_unify_4n/checkpoint-17000_ema`). Effective limit: 16 views when loading from a checkpoint. |
| "Qwen image tokens get a fixed 2D sin-cos code" | Correct, and the checkpoints agree: `cond_patch_pos=sincos2d`, `cond_patch_lattice=32`. Several comments still say `dino_sig` (`train_native.py:947`, `trellis_native_vlm.py:150`); they are stale. |

---

## 1. Data flow (v12 training and eval)

```
                           ┌──────────────────── CPU: dataloader worker ────────────────────┐
manifest row (jsonl) ─────►│ threed.py _ThreeDTaskBase._load_one  L446                       │
  sha256, renders_dir,     │   rng = np.random.default_rng()   (UNSEEDED, per item)  L448    │
  n_views, captions[]      │   caps = [c for c in captions if c]                      L450   │
                           │   ── T (text_to_3d, synth pool, weight .10) ──                   │
                           │     ci = _pick_caption_idx(rng,len(caps))  L238-252  (renorm)   │
                           │     prep_t(sha, ci, caps[ci])                 L496-503           │
                           │   ── T (text_to_3d_weighted, capT1800k, weight .60) ──           │
                           │     threed_textonly.py L91-109: key~{t000..t003} by yaml        │
                           │     caption_weights; ci>=len(caps) → t000; prep_t(sha,ci,cap)   │
                           │   ── I1 (image_to_3d, weight .15) ──                            │
                           │     view = rng.integers(0, n_views)  (uniform 0..15)   L464     │
                           │     prep_i1(renders_dir, view)                         L467     │
                           │   ── IM (multi_image_to_3d, weight .15) ──                      │
                           │     n = batch-locked draw from im_view_probs [.15,.25,.6]       │
                           │         (mixture.py L208-213 → threed.py L222-233)              │
                           │     views = pick_im_views(rng, n_views, n)  lcb L192-202        │
                           │     prep_im(renders_dir, views)             lcb L205-241        │
                           └──────────────┬──────────────────────────────────────────────────┘
      prep dict ("_live_prep"): input_ids, pixel_values(bf16), image_grid_thw, dino_px(fp32 [0,1]),
                                view/views, modality                      (lcb = live_cond_batch.py)
                                          │
  vlm_collate.collate_vlm_3d L151-154: batch → {"_live_prep": [dicts], targets...}
                                          │
                           ┌──────────── GPU: main process (train_native.py) ─────────────┐
                           │ _LiveCondPrefetch._launch L500-519 (side stream, batch N+1)   │
                           │   or NativeTrainer._prepare_inputs L603-612 → _encode_live    │
                           │ TrainCondEncoder.encode  lcb L497-579                         │
                           │   Qwen3-VL fwd: right-pad ids, output_hidden_states → [-1]    │
                           │     (L512-532); keep = am & ~isin(ids, boiler) (L533)         │
                           │     T: compact h[k] (L549-553)                                │
                           │     IM: qwen_view_ids (L556-557);  I1/IM: qwen_img_rc (L563-4)│
                           │   DINOv3: (px-mean)/std → extract_features  (L535-542)        │
                           │     dino_view_ids = arange(K).repeat_interleave(Nd) (L572-3)  │
                           │   rec: cond_hidden fp16, cond_keep_mask, [dino_hidden fp16,   │
                           │        dino_keep_mask(all 1), dino_view_ids, qwen_view_ids,   │
                           │        qwen_img_rc]                                           │
                           │ _pack_live_conds L411-457: pad (hidden 0 / mask False,        │
                           │   dino_view_ids pad 0, qwen_view_ids pad -1, rc pad -1)       │
                           └──────────────┬────────────────────────────────────────────────┘
                                          │
  TrellisNativeVLM.forward (trellis_native_vlm.py):
     cond_hidden/dino_hidden .to(connector dtype, bf16)  L1162-1168; key mask = keep mask L1164-1166
     fuse_dino guard (exempts _task=="text_to_3d")         L1172-1175
     ├─ S1/S2 towers: compute_cascade_flow_loss  L1346-1386  → flow_heads.py L203-283
     └─ S3 unify:     compute_unified_geotex_loss L1216-1236 → flow_heads.py L1054-1128
                         tex lane  build_unified_cond(drops drawn, drops_out recorded) L1066
                         geo lane  build_unified_cond(no drops; joint_cond_drop=False)  L1098
                         ss  lane  build_unified_cond(ext_drops = tex lane's realised drops) L1121
                         stamps read off each connector (_stamp)  L1057-1065
                                          │
  build_unified_cond (flow_heads.py L549-679), fusion branch (I1/IM):
     cond_q = connector(cond_hidden * keep_row, key_mask=qwen keep)            L601-602
     cond_q += view_code[qwen_view_ids]·[qv>=0]      (IM only)                 L603-607
     cond_q = _add_patch_pos(cond_q, cond_patch_pos, qwen_img_rc)             L614 (fn L505-546)
     dino_seg = dino_hidden(bf16) + view_code[dino_view_ids]                   L611-613
     dino_seg += seg[0];  cond_q += seg[1]                                     L630-631
     dino_seg *= keep_row          (CFG: DINO values → 0, keys kept)          L632
     ddrop: DINO keys masked; qdrop (& ~ddrop): Qwen keys masked              L633-648
     cond = cat([dino_seg ; cond_q]); key = cat([dmask ; qmask])              L651-652
  plain branch (T):  cond = connector(mask_drop(hidden)) + patch_pos(no-op) + seg[1]   L653-677
                                          │
  Eval / bench / demo: eval_cond.cond_uncond L75-140 → build_unified_cond with ext_drops forced
     cond   = (drop=0, ddrop, qdrop as requested), uncond = (drop=1, same ddrop/qdrop); rows compacted
     by key mask (L130). Record comes from TrainCondEncoder + prep_i1 / prep_t (run_unify_bench.py
     L86-109, run_sep_text.py L43-62) or prep_i1_image / prep_im_images (demo_pipeline.py L241-270).
```

Token counts with the **stock** Qwen3-VL-2B processor. I measured these; the docstrings' 1054/292 are v2.2-era numbers.

| mode | Qwen seq | image_pad | kept (non-boiler) | DINO tokens |
|---|---|---|---|---|
| I1 (1024² render) | 1050 | 1024 at idx [10,1034) | 1043 | 1029 (1 CLS + 4 reg + 32×32) |
| IM n=4 @256 tok/view (each view capped to 512²) | 1056 | 4×256, blocks at 10/268/526/784 | 1043 | 4×1029 = 4116 |
| IM n=2 | 540 | 2×256 | 531 | 2058 |
| T ("a red chair") | 23 before compaction | 0 | 18 (stored compacted) | — (no DINO segment) |

---

## 2. Hidden contracts (each one changes numbers if ported differently)

### 2.1 Prompts and templates
1. **Single-image (I1) prompt**: `"[3D Gen] <|vision_start|><|image_pad|><|vision_end|>\nReconstruct this object in 3D."` (`live_cond.py:55-56`). It goes through `apply_chat_template([{"role":"user","content":...}], tokenize=False, add_generation_prompt=True)` (`live_cond_batch.py:120-122`). There is no system message. The stock Qwen3-VL template adds no default system prompt and no `<think>` block; I verified the rendered text is `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`.
2. **Multi-image (IM) prompt**: `"[3D Gen] " + IMG_TOKEN*n + "\nReconstruct this object in 3D."` (`live_cond_batch.py:75-76`). The images are adjacent with no separator text.
3. **Text (T) prompt**: `"[3D Gen] " + TXT_PROMPTS[(int(sha[:8],16) + cap_idx) % 3].format(c=caption)` (`live_cond_batch.py:72, 79-81, 303-304`). `TXT_PROMPTS = ["Generate a 3D asset: {c}", "Create a 3D model of: {c}", "Make this in 3D: {c}"]`.
   - The template is **a deterministic hash of (sha, caption index)**, not a random draw. Each (asset, caption tier) always gets the same template.
   - `cap_idx` indexes the **empties-filtered** caption list (`threed.py:450`, `threed_textonly.py:89`).
   - Eval calls `prep_t(sha, 0, prompt)` (`run_unify_bench.py:107`), so the template depends on the eval case's sha.
   - The demo passes `sha="0"*8` plus an explicit `template` (`demo_pipeline.py:267`, default 0 = "Generate a 3D asset:").
4. **Caption-tier selection has two different semantics**, one per T task in the mix (`scripts/v12/configs/mix_v12_unify_texfixed.yaml`; the S2 mixes `configs/mix_v12_*_texfixed.yaml` use identical weights):
   - `text_to_3d_weighted` (weight .6, `caption_weights [.4,.12,.08,.4]`): a key is drawn over 4 fixed tiers. If the asset lacks that tier, it falls back to **t000 (long)**, then `min(ci, len-1)` (`threed_textonly.py:56-62, 91, 99-104`). A 3-caption asset therefore gets long at 0.8.
   - `text_to_3d` (weight .1, synth, `[.45,.15,.15,.25]`): the weights are **renormalised** over the captions that exist (`threed.py:238-252`).
5. Both T tasks stamp `_task="text_to_3d"` (`threed_textonly.py:105`). The model's fuse_dino guard exempts exactly that string (`trellis_native_vlm.py:1172-1175`).

### 2.2 Tokens, masks, layers
6. **Boilerplate ids** (`vlm_collate.py:84-110`, `include_system=False`) = ids of `<|im_start|> <|im_end|> <|vision_start|> <|vision_end|> <think> </think>` plus the **token ids of "user" and "assistant"** (Qwen3-VL: 151644, 151645, 151652, 151653, 151667, 151668, 872, 77091).
   - These ids are removed **anywhere** in the sequence, including inside a caption if it tokenizes to bare id 872/77091 (for example "user" without a leading space).
   - The `\n` tokens after the role names and the `[3D Gen]` tag are **kept**.
   - The mask is `am & ~isin(ids, boiler)` (`live_cond_batch.py:383-384, 533`).
7. **Keep-mask layout differs by mode.** I1/IM keep the **full sequence** (1050/1056 tokens) with a boolean keep mask, so boilerplate positions exist but are masked. T is **compacted** to kept tokens only (`live_cond_batch.py:549-553`). This matters in two places:
   - The xf2 connector attends with `key_mask`, so masked tokens are not keys inside the connector (`connector.py:195-203`, called at `flow_heads.py:212/602`).
   - Eval compacts the final cond by the key mask (`eval_cond.py:130`); training passes the mask to cross-attention instead (`_mask_or_none`, `flow_heads.py:1692-1712`) or builds per-row lists (`L1048-1050`).
8. **Hidden layer**: `self.model(..., output_hidden_states=True, use_cache=False).hidden_states[-1]` on the full `AutoModelForImageTextToText` (`live_cond_batch.py:376-377, 531-532`). In transformers 5.2.0, `capture_outputs(tie_last_hidden_states=True)` replaces `hidden_states[-1]` with `last_hidden_state`, which is the output of the **final RMSNorm** (`transformers/utils/output_capturing.py:258-268`; `models/qwen3_vl/modeling_qwen3_vl.py:936`). So the cond is the **post-final-norm** last layer. Deepstack visual injection into LLM layers 0-2 happens inside HF, so it must stay HF.
9. **Image token id**: resolved by name, `<|image_pad|>` = 151655 (`live_cond_batch.py:431-442`). **Pad id**: `tok.pad_token_id` = 151643 `<|endoftext|>` (`L392-396`). Padding is **right** padding (`L503-516`); left padding would shift the mRoPE positions.
10. **Per-view token budget for IM**: `IM_QWEN_TOK_PER_VIEW = int(env GEOTEX_IM_TOK_PER_VIEW or 256)` (`live_cond_batch.py:56`).
    - Each view is `cap_image(img, 256 * px_per_tok)` with `px_per_tok = (16*2)^2 = 1024`: downscale-only LANCZOS, `max(32, round(w*s))` (`vlm_collate.py:52-70`). A 1024² render becomes exactly 512², giving a 16×16 merged grid = 256 tokens.
    - S3 exports the env var explicitly (`v12_s3.sh:13`); S1/S2 use the default.
    - **Do not confuse this with `live_cond.IM_QWEN_TOK_PER_VIEW = 128`** (`live_cond.py:65`), which only the dead `LiveCondEncoder` reads.
11. **I1 Qwen image**: the **full uncropped** render passed through `.convert("RGB")` (`live_cond_batch.py:176-178`). PIL drops alpha without compositing, so the RGB under alpha=0 (≈(1,1,1)) stays and anti-aliased edges are not attenuated. The image is 1024², so there is no resize; the processor applies mean=std=0.5 (the snapshot's `preprocessor_config.json`, `Qwen2VLImageProcessorFast`). `pixel_values` are cast to **bf16 on CPU** (`L186`).

### 2.3 Image framing and DINO
12. **DINO framing**: `_dino_image` (`live_cond_batch.py:134-157`) takes the **same raw RGBA** and applies these steps in order:
    1. cap the long side at 1024 (LANCZOS);
    2. `_alpha_crop` (`threed.py:396-417`): alpha threshold `> 0.8*255` with fallback to `> 0`; centre `(x0+x1)/2.0`; `half = max(w,h,2)//2`; float box. PIL `crop` rounds the box with Python `round` (**banker's rounding**, `PIL/Image.py` 11.3.0 `_crop`), and out-of-bounds areas become zeros;
    3. `_composite_black` = `RGB*alpha` in float, then `uint8` via truncation (`threed.py:389-393`);
    4. upscale to at least 64 px if tiny (`MIN_IMG_PX=64`, `threed.py:346`);
    5. then, in the prep functions, resize to 512² LANCZOS, `/255`, CHW, fp32 (`live_cond_batch.py:182-183, 233-235`).

    The live path **ignores the yaml `crop_to_object` flag**: it always crops RGBA input. RGB (non-RGBA) input only gets `.convert("RGB")`, with no crop.
13. **DINO normalisation** happens on the GPU: ImageNet mean `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)` (`live_cond_batch.py:103-104, 389-390, 541`). Then `DinoV3FeatureExtractor.extract_features` (TRELLIS.2 `trellis2/modules/image_feature_extractor.py:81-92`) runs the embeddings, RoPE and all layers, **skips the model's own final norm**, and returns **unaffine `F.layer_norm`** over all 1029 tokens.
    - Model: `camenduru/dinov3-vitl16-pretrain-lvd1689m` (`dino_align.py:34`), run in **fp32**.
    - Sizes: `DINO_SIZE_I1=512` (`live_cond.py:63`) and `DINO_SIZE_IM=512` (hard-coded at `live_cond_batch.py:62`). The same name in `live_cond.py:64` reads env `DINO_SIZE_IM`; that copy is used only by the dead encoder.
14. **Dtype chain**:
    - Qwen: bf16 → stored **fp16** (`live_cond_batch.py:554`) → cast to the connector dtype, bf16 in training (`trellis_native_vlm.py:1162-1163`), or fp32 in eval scripts (`run_unify_bench.py:88-92`, connectors `.float()` at `L75`).
    - DINO: fp32 → **fp16** (`L570`) → bf16 in training / fp32 in eval.
    - bf16→fp16→bf16 is lossless except for fp16 subnormals (|x| < 6.1e-5). No overflow is possible: the final RMSNorm output is bounded by `sqrt(2048)·max|w| ≈ 45·12.1 ≈ 549`, measured from the snapshot's `model.language_model.norm.weight`. fp32→fp16→bf16 is double rounding. **For bit parity, keep the fp16 intermediate.**

### 2.4 Positions, views and CondStamp
15. **`qwen_img_rc`** (`live_cond_batch.py:444-476`): for each image_pad token, a normalised **cell centre** `((r+0.5)/gh, (c+0.5)/gw)` with `gh,gw = grid_thw[1:]//merge_size`. Raster order within each image, restarting at every image (no per-view offset). Non-image tokens get -1, and padding is also -1 (`train_native.py:447-456`).
    - I1 samples a 32×32 grid (identity on the 32-lattice).
    - IM samples a 16×16 grid, which bilinearly averages 2×2 cells.
    - It is emitted for I1 and IM (`L563-564`), not T.
16. **`_add_patch_pos`** (`flow_heads.py:505-546`): bilinear sampling with `align_corners=False`: `y = (r*P - 0.5).clamp(0,P-1)`, floor/ceil clamped. The table and weights are cast to **`cond_q.dtype`** (bf16 in training), and the result is added only where `rc[...,0] >= 0`.
17. **Patch table** `sincos2d` (`connector.py:61-77`): `Q=C//4`, `omega = 1/10000^(arange(Q)/Q)`, `pos = (arange(P)+0.5)`, row half = [sin,cos](row·ω), column half = [sin,cos](col·ω), times `VIEW_EMBED_SCALE` (env, default 0.2). Row L2 = 0.2·√512 = 4.5255, and P=32. **The scale is read from the env when the connector is constructed**; the checkpoint buffer wins on strict load.
18. **Segment codes** (`connector.py:168-172`):
    ```
    g = torch.Generator().manual_seed(1234)
    a = randn(C, g); b = randn(C, g)
    b = b - (b@a)/(a@a)*a
    a, b = unit(a)*4.8, unit(b)*4.8
    ```
    Row 0 = DINO, row 1 = **every** Qwen token (image and text). This is fixed, not learned, even though the comments at `flow_heads.py:623, 1467` say "zero-init". The parameter doc at `flow_heads.py:553` calls the rows "[image-segment, text-segment]", which is misleading: Qwen image tokens get row 1.
19. **View codes** (`view_code.py:22-29`): row v = `randn(C, Generator.manual_seed(1_000_003 + v), float64)`, unit-normalised, times `0.2·√(C/2)` (L2 4.5255), then `.float()`. The table has 16 rows (`dino_view_embed_max=16`, `train_native.py:902`) and is a model-level buffer shared by all towers. Regeneration is dropped on warm start (`train_native.py:1320-1321`).
    - **View-id assignment**: `dino_view_ids = arange(K)` per sample (`live_cond_batch.py:572-573`), ordinal within the sample, **not** the camera index. IM views are sorted ascending by camera id (`pick_im_views`), so ordinal 0 is the lowest camera id.
    - **I1's DINO segment gets view code row 0** (its dino_view_ids are zeros), while **I1's Qwen segment gets no view code** (`qwen_view_ids` is emitted only for IM, `L556-557`).
    - IM Qwen image tokens get the same row as their DINO view (`_qwen_view_ids`, `L478-495`). The IM Qwen `<|vision_*|>` / text tokens get -1, so no code.
20. **Stored precision**: every stamp buffer in the checkpoints is **bf16**. I verified `ckpt == bf16(regenerated)` exactly for `dino_view_embed`, `{diffusion,geo,ss}_connector.cond_seg_embed` and `cond_patch_pos` (max diff 0.0; the fp32 regenerated view codes differ by 1.65e-3). **BLIP3D must use the bf16-rounded values**: load them from the checkpoint, or regenerate and round through bf16.
21. **Order of additions in the fusion branch (bf16, so the order matters for bit parity)**:
    - Qwen: connector → + view code (IM) → [+ pos_stamp, off] → + patch_pos → + seg[1].
    - DINO: to bf16 → + view code → + seg[0] → × keep.

    `flow_heads.py:601-632`, and identically in the cascade copy `L212-242`.

### 2.5 Dropout and CFG
22. **Dropout semantics** (`flow_heads.py:596-652`; cascade copy `L208-262`):
    - `drop` (p = `mask_drop_prob` 0.1): Qwen hidden is zeroed **before** the connector, stamps are still added, DINO values are zeroed, and **all keys stay**.
    - `ddrop` (p = 0.3, DINO): DINO keys are masked out.
    - `qdrop` (p = 0.1 in S3 only, 0 in S1/S2): Qwen keys are masked, and it is **mutually exclusive with ddrop** (`& ~ddrop`).
    - All three are independent per row, so a row can be both CFG-dropped and modality-dropped.
    - T rows only get `drop` (`mask_drop`, torch.bernoulli; `L26-35, 667`); the uncond is `connector(0) + seg[1]`.
    - `dino_drop_prob` is gated by `self.training` but `qwen_drop_prob` is not (`trellis_native_vlm.py:1234-1236, 1377-1379`). Eval bypasses both anyway.
23. **S3 per-lane drop contract** (`flow_heads.py:1066-1128`, `geotex_joint_cond_drop=False` per the S3 config):
    - The tex lane draws its own drops.
    - The SS lane **replays the tex lane's realised drops** (`L1118-1120`).
    - The **geo (shape) lane is never dropped** (`mask_drop_prob=0.0`, `ext_drops=None`, `L1098-1105`), even though geo is unfrozen with its own loss in S3. The shape lane's CFG uncond therefore comes only from S2 training.
24. **Eval CFG** (`eval_cond.py:125-139`): the cond uses `ext_drops=(0, dd, qd)` and the uncond uses `(1, dd, qd)`, both in the same modality regime. Rows are compacted by key mask, and `cond_max_length=10240` (`L77`, matching the checkpoint configs). The view table is auto-extended (`L105-112`), which breaks past 16 with bf16 tables (see §0).
25. **Truncation**: `cond_max_length` (10240) applies to the **Qwen segment only**, before the DINO concat (`flow_heads.py:583-588`). It never fires in v12: the largest Qwen segment is 1056 tokens.

### 2.6 Views, sampling, batching
26. **View picking**:
    - I1 in training is **uniform over all `n_views` (0-15)**, including the below-ground views 000-002 (`threed.py:464`).
    - IM uses `IM_VIEW_WEIGHTS = [.15]*3 + [.6]*2 + [1]*7 + [.5]*4` over the first `min(n_avail,16)` views: distinct, sorted, `np.random.Generator.choice(replace=False, p)` (`live_cond_batch.py:64-67, 192-202`).
    - IM n ∈ {2,3,4} with p = [.15,.25,.6], locked per batch by the mixture's seeded `task_rng` (`threed.py:222-233`, `mixture.py:208-213`).
    - If a view file is missing: I1 falls back to the first sorted render and takes the view id from its filename (`L168-174`); IM silently drops that view (`L220-227`).
    - Eval I1 view: the manifest's `input_view_index` (`run_unify_bench.py:105`) or `good_view(sha) = 5 + (int(sha[:8],16)+3) % 7` (`eval_cond.py:65-68`).
27. **Item RNG is unseeded** (`np.random.default_rng()` at `threed.py:448`), so view, caption and IM view choices are not reproducible across resumes. The batch-level n_views is seeded.
28. **Batches are single-modality** (`sampling.granularity: batch`). `encode` keys on `preps[0]` (`live_cond_batch.py:521, 536`) and `_pack_live_conds` keys on `conds[0]` (`train_native.py:428-447`), so mixing modalities in a batch would silently drop keys.

---

## 3. Keep / drop table

Legend: **KEEP** = v12 reference, port faithfully. **PORT-PART** = keep a subset. **DROP** = legacy or dead for v12 (evidence given).

| symbol (file:line) | verdict | evidence / notes |
|---|---|---|
| `live_cond_batch.TrainCondEncoder` (`L363-585`) | **KEEP (reference encoder)** | Used in training (`train_native.py:555-556`), eval (`eval_cond.py:40-47`), the bench (`run_unify_bench.py:86`), text eval (`run_sep_text.py:43`) and the demo (`demo_pipeline.py:211`). `warmup` (`L401-425`) is keep-optional; `encode_paths` (`L581-585`) is tests only. |
| `prep_i1`, `prep_im`, `prep_t`, `pick_im_views`, `_dino_image`, `_view_path`, `_worker_proc`, `im_prompt`, `txt_prompt`, `TXT_PROMPTS` (`L72-310`) | **KEEP** | Called from `threed.py:465-500`, `threed_textonly.py:98-106` and `eval_cond.py:54-61`. |
| `prep_i1_image`, `prep_im_images` (`L244-288`) | **KEEP (inference API)** | Demo only (`demo_pipeline.py:253, 260`) and `test_demo_v12.py`. |
| `stage_to_shm` (`L313-359`) | PORT-PART | Infra: the /dev/shm copy of the VLM. Keep as an optional utility with the hardcoded `/dev/shm/v22ckpt_live` special case removed. |
| `live_cond.V22_CKPT`, `IMG_TOKEN`, `PROMPT_I1`, `DINO_SIZE_I1`, `I1_QWEN_CANVAS` (`L48-74`) | PORT-PART | Constants imported by `live_cond_batch.py:39`. Move them into the encoder module; **drop the v2.2 default path**. |
| `live_cond.object_frame`, `FramedImage`, `_square_box`, `_birefnet_mask`, `_get_birefnet`, `_border_bg_mask`, `_largest_component`, `RENDER_BG`, `OBJECT_EXTENT` (`L77-247`) | KEEP for inference/demo only | `demo_pipeline.py:47, 252, 259`. Not used in training. |
| `live_cond.LiveCondEncoder`, `CondPack`, `build_stage_cond`, `assert_matches_cache`, `CAPTION_PROMPTS`/narrator (`L250-610`) | **DROP** | No callers outside `live_cond.py` (grep). `build_stage_cond` predates CondStamp, so it would feed v12 weights unstamped conds (`demo_pipeline.py:49-54` says so). |
| `live_cond.IM_QWEN_TOK_PER_VIEW=128`, `live_cond.DINO_SIZE_IM` (env), `PROMPT_IM` | **DROP** | Only `LiveCondEncoder` reads them. They conflict with the v12 values. |
| `eval_cond.cond_uncond` (`L75-140`) | **KEEP (reference inference assembler)** | Bench, demo and text eval all use it. |
| `eval_cond.encode`, `encoder`, `good_view` (`L40-68`) | KEEP (thin helpers) | |
| `eval_cond.cond_uncond_from_tensors` (`L143-165`) | DROP | Only cache-era scripts use it (`scripts/{trimodal_fullchain_eval, eval_im_vs_i1_ss, im_fullchain_eval, t1_multiimage_infer, ss_backface_recovery, synth_backpair_eval, diag_im_multiview, eval/eval_overfit10}.py`). |
| `flow_heads.build_unified_cond`, `_add_patch_pos`, `mask_drop`, `_mask_or_none` (`L26-35, 505-679, 1692-1712`) | **KEEP (reference training assembler)** | S3 and eval. `ext_drops` / `drops_out` are needed for lane replay and eval. |
| `flow_heads.compute_cascade_flow_loss` fusion/plain branches (`L203-283`) | PORT by delegating to `build_unified_cond` | S1/S2 trained on this copy. Equivalent on the deterministic path; the RNG draw order differs. The KD/teacher branch (`L284-308`) and `null_cond_like` (`L38-49`) are **DROP** (KD archived; `assert teacher_cond is None` at `L207`). The `pos_stamp` hooks (`L227-229, 280-282, 608-610, 675-677`) are **DROP** (`cond_pos_stamp=False` in every v12 config). |
| `connector.TRELLIS2TransformerAdapter`, `_AdapterBlock`, `_patch_pos_table("sincos2d")` (`L56-203`) | **KEEP** | The v12 xf2 connector: in_proj (xavier, zero bias) → 2 pre-norm QK-norm blocks, 8 heads, MLP ×4 tanh-GELU → out LayerNorm. |
| `_patch_pos_table` `dino_sig` / `zero` / `*_learn` branches | DROP | Not v12. `dino_sig` pulls in `pos_stamp.DPOS_NPZ`. |
| `connector.TRELLIS2Connector` (MLP, `L28-53`) | DROP for v12 (keep only to load v10/v11) | All v12 checkpoints have `cond_adapter=xf2`. |
| `view_code.hashed_view_codes`, `view_embed_scale`, `ensure_view_codes`, `detect_view_code_mode` (`L10-69`) | KEEP, with a fix for the bf16 detection bug | `sincos_view_codes` is DROP (A/B arm, not v12). |
| `pos_stamp.py` (entire file) | **DROP** | `cond_pos_stamp=False` in all v12 configs. The only live import is the lazy `dino_sig` branch. |
| `vlm_collate.boiler_ids`, `cap_image`, `px_per_tok` (`L52-110`) | **KEEP** | Used by the encoder. |
| `vlm_collate.collate_vlm_3d` live branch (`L151-154`) plus the target stacking in `L155-250` | PORT-PART | The target stacking belongs to the data/targets audit. |
| `vlm_collate.collate_vlm_3d` raw-processor path (`L252-365`), `upscale_image`, `set_default_target_tokens_per_view`, `MAX_TOKENS_SINGLE`/`TOKEN_BUDGET`/`DINO_IMAGE_SIZE` | **DROP** | v12 runs `build_vlm=False` with `live_cond: true`. `--target_tokens_per_view 1024` (`train_native_split.sh`) is inert on the live path. |
| `vlm_cache.collate_cached` (`vlm_cache.py:129-140`) | KEEP (the padding helper `_pack_live_conds` uses) | The rest of `vlm_cache` (`load_entry`, keys, meta) is cache-era: DROP. |
| `train_native._pack_live_conds`, `_LiveCondPrefetch`, `_mixture_uses_live_cond`, `NativeTrainer._live_encoder`/`get_train_dataloader`/`_encode_live`/`_prepare_inputs` (`L411-612`) | **KEEP** | |
| `threed._ThreeDTaskBase` live branches, `_pick_caption_idx`, `_draw_n_views`, `set/clear_batch_params`, `_sample_n_views`, `_alpha_crop`, `_composite_black`, `MIN_IMG_PX` | **KEEP** | |
| `threed` cache branches (`L128-158, 282-298, 472-474, 488-493, 505-578`), `_load_views` (`L348-386`, raw path), raw `images` path (`L580-595`), `collate_fn` (`L688-701`) | **DROP** | `cached_hidden_root` / `packed_root` are unused in v12 mixes. `_load_views` is only the raw-processor path. `_dino_image` reuses only `_alpha_crop` and `_composite_black`. |
| `threed_textonly.WeightedTextTo3DDataset` | KEEP, simplified (live branch plus the voxel precheck `L78-86`) | The cache branch (`L110-129`) and `_meta.json` weight lookup (`L45-55, 63`) are DROP. **But its "fall back to t000" semantics must be kept** (§2 item 4). |
| `trellis_native_vlm.encode_cond` / the `self.vlm` build (`L298-321, 874-940`), depthwise routing, `_LM_TASKS` path | **DROP** | `build_vlm=False`, `cond_fusion=none` in every v12 config. |
| `trellis_native_vlm` `dino_view_embed` buffer build (`L547-562`), connector build (`L514-539`), forward cond handling (`L1157-1195`), both loss calls | KEEP (the cond part) | |
| `unified_geotex._connector_from_cfg`, `load_tri_connectors` (`L1631-1673`) | KEEP (a fixed version) | `load_run_connectors` (`L1773-1783`) is **broken for the v12 S3 final** (§6 item 4). |
| `scripts/build_vlm_cache_v22.py`, `build_dino_cache.py`, `build_vlm_cache.py`, `t1_build_perview_conds.py`, `synth_backpair_conds.py`; the `build_cond_*` helpers in `scripts/{trimodal_fullchain_eval, eval_im_vs_i1_ss, im_fullchain_eval, t1_multiimage_infer, eval_fusion_v22, render_textonly_showcase, synth_backpair_eval, eval/eval_overfit10}.py` | **DROP** | Cond-cache era (v2.2 VLM, 64 tok/view, DINO 320, hardcoded 10/76/142/208 layout). |
| `scripts/_tmp/lc_parity.py`, `tests/test_unified_conventions.py`, `scripts/v12/test_v12_assembly.py`, `scripts/v12/test_demo_v12.py` | **PORT as tests** | These are the parity and assembly guards. Re-point them at BLIP3D. |

---

## 4. Duplications to collapse (reference implementation in bold)

1. **Encoders (3)**: **`TrainCondEncoder`** (`live_cond_batch.py:363`); `LiveCondEncoder` (`live_cond.py:280`, dead: per-image set lookup for the boiler mask, `.float()` output, no `qwen_img_rc`); `TrellisNativeVLM.encode_cond` (dead). The cache builders in `scripts/build_vlm_cache_v22.py` and `build_dino_cache.py` are a fourth, offline copy.
2. **Assemblers (4 live, plus N in scripts)**:
   - **`build_unified_cond`** (`flow_heads.py:549`) is the reference.
   - `compute_cascade_flow_loss` fusion/plain (`L203-283`) is the parallel copy S1/S2 trained on.
   - `eval_cond.cond_uncond` is a thin, correct wrapper; keep it as the inference API.
   - `live_cond.build_stage_cond` is stale: no CondStamp, and its CFG convention differs.
   - The scripts' `build_cond_*` are cache-era.

   BLIP3D should have one `assemble_cond(..., drops)` and call it from training (all stages) and inference.
3. **Prompt constants**: `TXT_PROMPTS` is duplicated (`live_cond.py:58` and `live_cond_batch.py:72`). `PROMPT_IM` (`live_cond.py:57`) overlaps `im_prompt()` (`live_cond_batch.py:75`).
4. **Budget constants with the same name but different values**:
   - `IM_QWEN_TOK_PER_VIEW`: 128 (`live_cond.py:65`) vs env default 256 (`live_cond_batch.py:56`).
   - `DINO_SIZE_IM`: env default 512 (`live_cond.py:64`) vs hard 512 (`live_cond_batch.py:62`).
   - `DINO_IMAGE_SIZE=512` (`vlm_collate.py:38`, dead path).
5. **`_resolve_image_token_id`**: duplicated (`live_cond.py:302` and `live_cond_batch.py:431`).
6. **`_qwen_view_ids`**: three variants. **`live_cond_batch.py:478`** (derived from grids) is the reference; `live_cond.py:340` has an equal-split fallback; `threed.py:525-531` hardcodes the 292-token cache layout.
7. **Alpha-crop / square box**: **`threed._alpha_crop`** (`L396`: official `size//2`, 0.8 alpha threshold, float box) is the reference. `live_cond._square_box` (`L186`: `>0` mask, `max(…,1)/2.0`, `int()` truncation) is a different rule used only for demo photos. `_alpha_crop` should live next to the encoder, not in the dataset.
8. **DINO image preparation (4 paths)**:
   - **`_dino_image` + resize in `prep_*`** is the reference.
   - `threed._load_views` (raw path).
   - `vlm_collate` `dino_images` (uses **BILINEAR**, `L330-331`, dead).
   - `DinoV3FeatureExtractor.__call__` list branch (LANCZOS, then `.cuda()` internally).
9. **Boiler masking**: the `torch.isin` tensor version (**`live_cond_batch.py:383, 533`**, `vlm_collate.py:307-309`) vs the Python set loop (`live_cond.py:326`).
10. **View-code tables**: `dino_view_embed` is used for both segments. There is one model-level buffer, but S3 checkpoints also carry per-lane stamps on three connectors. Collapse to one "CondStamp" object holding `{seg (2,C), patch_pos (P²,C), view (V,C)}`, stored bf16.
11. **Connector loaders**: `_connector_from_cfg` (config-driven, breaks on S3) vs `run_unify_bench.py:66-76` (infers from state-dict keys; this one works). The state-dict-driven version should be the reference.
12. **Caption-tier choosers**: `_pick_caption_idx` (renormalise) vs `WeightedTextTo3DDataset` (fall back to t000). Collapse into one function with an explicit `fallback` policy, and keep both behaviours selectable to reproduce v12.

---

## 5. Naming to change for BLIP3D, and hardcoded paths

**Names**
- `V22_CKPT`, `_V22_DEFAULT` (`live_cond.py:48-54`). The name says v2.2 finetune, but v12 uses stock Qwen3-VL-2B. Rename to e.g. `COND_VLM_PATH`, make it **required** (no default), and record it in the checkpoint config.
- `"/dev/shm/v22ckpt_live"` and `condvlm_<sha1>` (`live_cond_batch.py:334-338`).
- `vlm_model="Qwen/Qwen3.5-2B"` and `vlm_hidden_size` in every v12 `config.json`: wrong or misleading. `--vlm_model Qwen/Qwen3.5-2B` is hardcoded in `scripts/train_native_split.sh:139`. The comments `connector.py:33` and `trellis_native_vlm.py:58` also say Qwen3.5.
- `build_vlm_cache_v22` references throughout the docstrings (`live_cond.py:5-24`, `live_cond_batch.py:9-25, 80, 196, 208, 550`). The token counts in them (1054 / 292 / 405) are wrong for stock Qwen3-VL (1050 / 1056 / 1029).
- `dino_view_embed`: used for **both** DINO and Qwen segments. Rename to `view_code`. `dino_view_embed_max` → `max_views`.
- `diffusion_connector` means the **tex** lane in S3 checkpoints; `geo_connector` = shape and `ss_connector` = SS. In S1/S2 checkpoints `diffusion_connector` is that tower's own connector. Rename per lane (`cond_adapter.{ss,shape,tex}`).
- `TRELLIS2TransformerAdapter` / `xf2` / `cond_adapter` → e.g. `CondAdapter(n_blocks=2)`. `TRELLIS2Connector` (MLP) is legacy.
- `cond_seg_embed` / `cond_patch_pos`, which live on the connector but are applied by the caller; `pos_stamp`, `IMG_SPAN_FULL/KEPT` (dead).
- `live_cond`, `_live_prep`, `LiveCondEncoder` vs `TrainCondEncoder`: once there is one encoder, call it `CondEncoder` and `cond_prep`.
- `GEOTEX_IM_TOK_PER_VIEW`, `VIEW_EMBED_SCALE`, `COND_VLM_CKPT`, `LIVE_COND_SHM`, `LIVE_COND_PREFETCH`, `LIVE_COND_PROF`, `GEOTEX_XATTN_FLASH` (env-driven numerics). Move the first two into config and record them in the checkpoint.
- Task names: `text_to_3d_weighted`, and the `_task == "text_to_3d"` string exemption (`trellis_native_vlm.py:1173`).
- Package names `trellis2_blip3o`, `blip3o/model/language_model/trellis_native_vlm.py`, class `TrellisNativeVLM`.

**Hardcoded paths**
- `live_cond.py:48-49` `/fsx/home/weikai.huang/3dgen/vlm3d_runs/stage1_v22/v0-20260703-051346/checkpoint-1000`
- `pos_stamp.py:14` `/fsx/home/.../BLIP3o/runs/cache_logs/dino_pos32.npz` (reached from `connector.py:82` only for `dino_sig`)
- `demo_pipeline.py:40-45` stock Qwen3-VL snapshot (`…/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`); also every v12 launcher (`_lc/v12_s{1,2}.sh`, `scripts/v12/v12_s3.sh:14`, `v12_ss_cont.sh:13`)
- `dino_align.py:34` `camenduru/dinov3-vitl16-pretrain-lvd1689m` (HF id; needs `HF_HOME`)
- `live_cond.py:143` BiRefNet `ZhengPeng7/BiRefNet` (demo only)
- `live_cond_batch.py:335, 338` `/dev/shm/...`
- Mix yamls: absolute manifest paths `/fsx/home/.../manifests/splits/*_texfixed_ff.jsonl`
- `demo_pipeline.py` `DEMO_{SS,SHAPE,TEX}_CKPT` defaults under `/fsx/data/weikai.huang/runs/...`

---

## 6. Risks and open questions

1. **Which hidden state.** In the installed transformers 5.2.0, `hidden_states[-1]` equals `last_hidden_state`, the post-final-RMSNorm output. I verified this by reading the source (`output_capturing.py:258-268` with `tie_last_hidden_states=True`; `modeling_qwen3_vl.py:936`); **[unverified numerically]**. It depends on the transformers version: a version with `tie_last_hidden_states=False` semantics would return the pre-norm state. BLIP3D should call the base model (`model.model(...)`, or `logits_to_keep=1`) and use `last_hidden_state`. The current code loads the **full generation model** (`AutoModelForImageTextToText`, `live_cond_batch.py:376`) and, with default `logits_to_keep`, computes full-vocab logits: 151,936 × L × B in bf16, about 2.5 GB at bs8×1050. It also keeps all 29 hidden states (`output_hidden_states=True`). Port with a one-batch parity test (cosine 1.0 / max-abs diff 0) against `TrainCondEncoder.encode`. Also pin `attn_implementation`: it is currently the default, and bf16 batched-vs-bs1 noise is acknowledged at `L503-507`.
2. **Silent cond-VLM fallback.** If `COND_VLM_CKPT` is unset, every default silently loads the **v2.2 finetune** (`live_cond.py:54`). The `prep_*` defaults (`vlm_path=V22_CKPT`) are bound **at import time**, so the env var must be set before the import. The checkpoint `config.json` does not record which VLM was used. `run_unify_bench.py:33` requires the env var; `eval_cond.encoder()` and `TrainCondEncoder()` do not.
3. **bf16 stamp tables.** Loading from a checkpoint yields bf16 codes, which `detect_view_code_mode` misclassifies as "learned", so `ensure_view_codes` raises for more than 16 views (verified). Also, `connector.cond_patch_pos` / `cond_seg_embed` get cast to the model dtype, and at eval to fp32, but the *values* are bf16-rounded. Regenerating fresh fp32 codes in BLIP3D would shift the view codes by up to 1.65e-3 and the others similarly. Fix `detect` (compare after bf16 rounding, or store the mode and scale explicitly) and keep the bf16 values.
4. **`load_run_connectors` breaks on the v12 S3 checkpoint.** The S3 `config.json` says `cond_seg_embed=False, cond_patch_pos=off`, because S3 inherited stamped connectors from S2. `_connector_from_cfg` (`unified_geotex.py:1650-1673`) therefore builds unstamped adapters, and strict load fails on the unexpected `cond_seg_embed`/`cond_patch_pos` keys. It also never loads `ss_connector`. `run_unify_bench.py:64-76` works around this by inferring from the state dict. **[unverified by execution]**; the conclusion is from reading the code plus the checkpoint keys and config.
5. **Uncond is not zero on the Qwen side.** See §0 and §2 item 22. A "cleanup" that zeroes the whole uncond, or skips the stamps on dropped rows, would change the unconditional the model learned. The comments in `flow_heads.py` claim otherwise.
6. **Per-stage dropout differences.** S1/S2 used qwen_drop 0.0; S3 used 0.1. The S3 geo lane is never dropped (joint_cond_drop False), while the SS lane replays the tex lane's drops. If BLIP3D unifies the recipe, decide explicitly whether to reproduce these asymmetries. For resuming or porting S3 they are part of the contract.
7. **I1 view distribution.** Training draws I1 views uniformly over 0-15, including the three below-ground views weighted 0.15 in IM. Eval uses views 5-11 or the manifest's `input_view_index`. This is intentional or an oversight, **[unclear]**; either way it is part of what v12 learned.
8. **Train/infer framing gap (demo).** In training, Qwen gets the raw `.convert("RGB")` render and DINO gets `_alpha_crop` + `RGB*alpha`. The demo gets `object_frame`: segmentation, a `_square_box` rule, object at 0.52 extent, RGB on black (`demo_pipeline.py:252-253`). Also, the demo's IM path accepts **1 image** and emits an IM record with `qwen_view_ids`, a configuration never trained (n ∈ {2,3,4}).
9. **Non-determinism.** Per-item `np.random.default_rng()` is unseeded (`threed.py:448`, `threed_textonly.py:70`), so the conditioning inputs (view, caption tier, IM combo) are not replayable on resume. The batch-level n_views is seeded.
10. **Processor coupling.** The code relies on HF `Qwen2VLImageProcessorFast` with `min_pixels=65536`, `max_pixels=16777216` and patch 16 / merge 2 from the snapshot, plus the raster order of image_pad tokens within the merged grid (`_qwen_img_rc`). A processor change (slow vs fast, a different `max_pixels`, a version bump) changes token counts and `rc`. Pin the processor config in BLIP3D and assert I1 = 1024 image tokens and IM = 256 per view.
11. **fp16 intermediate.** It is harmless in magnitude (Qwen post-norm is bounded at about 549), but it is part of bit parity: fp32 DINO → fp16 → bf16 is double rounding. Decide whether BLIP3D keeps it (exact v12 parity) or drops it (a tiny, documented deviation).
12. **Stale documentation to avoid porting**:
    - "1054 tok", "292 tok", "DINO@320" (`live_cond*.py`).
    - "zero-init" segment code (`flow_heads.py:623-629, 1464-1467`).
    - "--cond_patch_pos dino_sig" (`train_native.py:947`), and `trellis_native_vlm.py:150`'s option list, which lacks `sincos2d`.
    - `view_embed_mode` comment "sincos default" (`train_native.py:899`), while the actual default is `hashed`.
    - `threed.py:109` "None = uniform (legacy)": fine, but see §2 item 4 for the second semantics.
13. **DINO parity against the old cache** (cos 0.9937, `live_cond_batch.py:83-95`) no longer matters for v12, because v12 trained live end-to-end. The live path *is* the contract; don't try to match the cache.
14. **Resolved.** Apart from the manifest paths, the S1 mixes `configs/mix_v12_{ss,shape,tex}.yaml` are identical to the S2 `_texfixed` mixes (verified by diff). Task weights, caption_weights and im_view_probs are therefore constant from S1 through S3; S3 only adds `tex_placeholder` and changes the pools.
