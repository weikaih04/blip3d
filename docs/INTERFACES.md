# BLIP3D — module interfaces (training side)

These are the contracts between `data/`, `cond/`, `losses/`, `models/` and `train/`. Each is checked by a test.

## 1. A training batch (output of `blip3d.data.collate`, CPU, pinned by the DataLoader)

| key | type | when |
|---|---|---|
| `task` | str, the yaml task name (`t_real`, `t_synth`, `i1`, `im`) | always; logging only |
| `modality` | `"t"`, `"i1"` or `"im"` | always; one modality per batch (asserted) |
| `ids` | list[str] asset ids | always |
| `prep` | list[dict], one CPU prep payload per row (`blip3d.cond.prep.prep_render / prep_renders / prep_text`) | always |
| `ss` | float32 (B, 8, 16, 16, 16), raw (SS has no normalization) | `ss` in targets |
| `shape` | `SparseTensor`, feats (N, 32) shape-normalized, coords (N, 4) int32 `[b, x, y, z]` | `shape` in targets |
| `tex` | `SparseTensor`, pbr-normalized, same coords as `shape` | `tex` in targets and every row has it (placeholder rows included) |
| `tex_concat` | `SparseTensor`, shape latent in tex's shape-normalization (bit-identical to `shape`), same coords | with `tex` |
| `tex_valid` | bool (B,) | `tex_missing: placeholder` |
| `align_points` | float32 (B, 10000, 3), voxel centres in [-0.5, 0.5]^3 | ROAD on (SS tower) |

Rows of one batch never mix modalities. Placeholder tex rows have zero feats and `tex_valid=False`.

## 2. Conditioning on the GPU (`blip3d.cond`)

`CondEncoder.encode(batch["prep"]) -> list[record]` → `assemble.collate(records) -> cond batch` →
`assemble.assemble(connector, views, cond_batch, drops, dtype=torch.bfloat16) -> (tokens (B,T,1024), key_mask (B,T))`.
`train.prefetch.LiveCondPrefetch` runs the encoder for batch N+1 on a side stream while batch N trains, and replaces
`prep` with `cond` (the collated records).

Drops are drawn with `assemble.draw_drops` in the v12 RNG order. S1/S2 towers draw them inside the tower loss,
S3 inside the unified loss (tex lane draws, SS lane replays, geo lane has none unless `joint_cond_drop`).

## 3. Losses

- `losses.flow`: `sample_t(kind, B, generator)`, `diffuse(x0, t, noise)`, `velocity_target`, `tower_loss(tower_module, batch, cond)`.
  Timesteps: SS logitNormal(1, 1), SLAT uniform; drawn from a per-rank generator (v12 drew them from a CPU
  generator identical on every rank, `compat_v12.t_rng="shared_cpu"` reproduces that).
- `losses.road`: ROAD alignment on the SS tower (Uni3D-g teacher, block-10 tap, AlignMLP, cosine + Hungarian).
- `losses.unified`: row classes (clean / lag / solo), `sample_timestep_pairs/triples`, row/voxel-balanced MSE,
  tex placeholder masking, SS dual term, geo corner mask, the cond-sensitivity probe.

Every loss returns `(total, logs: dict[str, float tensor])`; logs are averaged across ranks by the trainer.

## 4. Models

- `models.blip3d.Blip3DTower(tower, connector, flow, road=None)`: registration order connector → flow → road
  projector, the same parameter order as the v12 module, so ZeRO optimizer shards map by position.
- `models.unified.Blip3DUnified` plus three connectors (`geo`, `tex`, `ss`) for S3.

## 5. Ownership while porting (2026-09-24)

`data/` + `recipes/data/` + `tests/test_data_*`: data port. `models/road/` + `losses/road.py` + `tests/test_road_*`: ROAD
port. `losses/unified.py` + `tests/test_unified_loss_*`: unified-loss port. Everything else under `train/`,
`losses/flow.py`, `models/blip3d.py`, `recipes/*.yaml`, `tools/`, `cluster/`: trainer port.
