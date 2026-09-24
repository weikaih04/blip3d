# BLIP3D

One model for image-, multi-image- and text-to-3D generation: a frozen Qwen3-VL-2B + DINOv3 conditioning
stack feeding TRELLIS.2 sparse-structure / shape / texture flows, trained with the v12 recipe
(connector warm-up → per-tower fine-tune → unified three-tower model).

**Status: under construction.** The design is in [`docs/DESIGN.md`](docs/DESIGN.md), the audit of the
previous codebase in [`docs/scan/`](docs/scan), provenance in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

```
blip3d/     cond · models · losses · data · train · infer · eval · utils
recipes/    per-stage training recipes, data mixes, inference presets
tools/      command-line entry points
cluster/    Slurm launch, GPU reservation heartbeat, failover, monitoring
tests/      unit tests and goldens recorded from the previous codebase
env/        pinned environment
```

Run everything with `PYTHONNOUSERSITE=1` (see `env/README.md`).
