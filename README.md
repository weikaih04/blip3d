# BLIP3D

One model for image-, multi-image- and text-to-3D generation: a frozen Qwen3-VL-2B + DINOv3 conditioning
stack feeding the TRELLIS.2 sparse-structure / shape / texture flows, trained with the v12 recipe
(connector warm-up → per-tower fine-tune → unified three-tower model).

Design: [`docs/DESIGN.md`](docs/DESIGN.md) · module contracts: [`docs/INTERFACES.md`](docs/INTERFACES.md) ·
issue ledger (what was fixed, what is kept for parity): [`docs/ISSUES.md`](docs/ISSUES.md) ·
verification status: [`docs/STATUS.md`](docs/STATUS.md) · audit of the previous codebase: [`docs/scan/`](docs/scan).

```
blip3d/cond     Qwen3-VL + DINOv3 encoder, CPU prep, CondStamp codes, xf2 connector, cond/uncond assembly
blip3d/models   TRELLIS.2 towers (towers.py), Blip3DTower (blip3d.py), unified model (unified/), ROAD (road/)
blip3d/losses   flow matching per tower, ROAD, the S3 unified loss
blip3d/data     manifests, latents, tasks (t / i1 / im), mixture, voxel balance, collate
blip3d/train    trainer (HF + DeepSpeed ZeRO-1), EMA (persisted), callbacks, recipes, builders, checkpoint maps
blip3d/infer    presets, schedules, cascade, unified sampler (refine / joint / interleave), export, Pipeline
recipes/        s1_* · s2_* · cont_ss · s3_unify (+ data/ mixes, deepspeed/ configs)
tools/          train.py · bench.py · import_v12.py
cluster/        launch · stop · heartbeat · failover (Slurm holds)
tests/          CPU and GPU parity tests against the previous codebase
```

## Setup

```bash
cp paths.example.yaml paths.yaml        # data, manifests, runs, HF cache, TRELLIS.2 checkpoints
git submodule update --init             # third_party/TRELLIS.2 (fork with the fused kernels)
export PYTHONNOUSERSITE=1               # the env is pinned (env/); ~/.local must not shadow it
```

## Train

```bash
# one node per hold, 8 GPUs each; resumes automatically from the latest checkpoint of the run
python cluster/launch.py recipes/s2_ss.yaml --holds <job> <job> <job> <job>
python cluster/failover.py recipes/s2_ss.yaml --holds <jobs...> --tag s2_ss      # optional watcher (4 <-> 2 nodes)
# direct (inside an allocation)
torchrun --nnodes 4 --nproc_per_node 8 ... tools/train.py recipes/s2_ss.yaml --set compute.per_gpu_bs=4
```

Stages: `s1_{ss,shape,tex}` (connector only, 26k) → `s2_{ss,shape,tex}` (full, 106k, WSD 6k) → `s3_unify`
(32k). `cont_ss` continues the SS tower to 306k. A v12 checkpoint is resumed after a one-time conversion:

```bash
python tools/import_v12.py --src keep/v12/s2_ss/checkpoint-100000 --kind ss --global-step 100000 --ema-n 25000 \
       --dst blip3d-v12-cont-ss/checkpoint-100000
```

Every checkpoint carries `ema.safetensors` + `ema_state.json` (update count), `gradclip_state.json` and
`blip3d_recipe.json` (recipe and encoder identities).

## Generate

```python
from blip3d.utils import backend; backend.setup("eval")
from blip3d.infer.pipeline import Pipeline
pipe = Pipeline.separate("keep/v12/s2_ss/checkpoint-106000_ema", "keep/v12/s2_shape/checkpoint-106000_ema",
                         "keep/v12/s2_tex/checkpoint-106000_ema")
lat = pipe.generate(pipe.prep_text("a red wooden chair"), seed=0)
pipe.export(lat, "out/chair")          # full.glb, shape.obj, shape.glb
```

`Pipeline.unified("keep/v12/s3_unify_4n/checkpoint-17000_ema", mode="refine")` samples the unified model.
Benchmarks (WilD3D case directories): `tools/bench.py --manifest ... --output ... --mode image|text --separate ...`.
