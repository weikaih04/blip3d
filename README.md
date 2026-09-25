# BLIP3D

One model for image-, multi-image- and text-to-3D generation: a frozen Qwen3-VL-2B + DINOv3 conditioning stack
feeding the TRELLIS.2 sparse-structure / shape / texture flows, trained in three stages (connector warm-up,
per-tower fine-tuning, unified three-tower model).

```
blip3d/cond     Qwen3-VL + DINOv3 encoder, preprocessing, position / segment / view codes, connector, CFG assembly
blip3d/models   TRELLIS.2 towers, Blip3DTower, the unified three-tower model, ROAD alignment head
blip3d/losses   flow matching per tower, ROAD, the unified-model loss
blip3d/data     manifests, latents, tasks (text / single image / multi image), mixture, voxel balance, collate
blip3d/train    trainer (HF Trainer + DeepSpeed ZeRO-1), EMA, callbacks, configs, model builders
blip3d/infer    sampler presets, cascade and unified samplers (refine / joint / interleave), export, Pipeline
blip3d/eval     sparse-structure readout
configs/        train/ (s1_*, s2_*, s3_unify) · data/ · deepspeed/ · infer/ · paths.example.yaml
tools/          train.py · infer.py · bench.py · readout.py
```

## Installation

```bash
git clone --recursive <this repo> && cd blip3d                # third_party/TRELLIS.2 is a submodule
conda create -n blip3d python=3.10 && conda activate blip3d
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .                                              # core dependencies (requirements.txt)
```

Compiled extensions, built against torch 2.6 / CUDA 12.4 as in TRELLIS.2:
flash-attn 2.7.3 (FlashAttention-3 from its `hopper/` directory for training), flex_gemm, cumesh, spconv-cu124,
nvdiffrast, and `o_voxel` (`pip install -e third_party/TRELLIS.2/o-voxel`).

```bash
cp configs/paths.example.yaml configs/paths.yaml             # data, manifests, checkpoints, HF cache
```

## Train

```bash
torchrun --nnodes N --nproc_per_node 8 --node_rank R --master_addr HOST --master_port PORT \
    tools/train.py configs/train/s2_ss.yaml [--set compute.per_gpu_bs=4 ...]
```

Stages: `s1_{ss,shape,tex}` (connector only) → `s2_{ss,shape,tex}` (full fine-tune, warmup-stable-decay) →
`s3_unify` (unified model). Training resumes automatically from the latest checkpoint in the output directory.
Every checkpoint carries `ema.safetensors` with its update count, the gradient-clip state and `blip3d_config.json`
(training config and encoder identities).

## Generate

```bash
python tools/infer.py --text "a red wooden chair" --out out/chair
python tools/infer.py --image photo.png --out out/x
```

```python
from blip3d.utils import backend; backend.setup("eval")
from blip3d.infer.pipeline import Pipeline
pipe = Pipeline.separate(ss_ckpt, shape_ckpt, tex_ckpt)        # or Pipeline.unified(ckpt, mode="refine")
lat = pipe.generate(pipe.prep_text("a red wooden chair"), seed=0)
pipe.export(lat, "out/chair")                                  # full.glb, shape.obj, shape.glb
```
