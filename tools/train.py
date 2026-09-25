#!/usr/bin/env python
"""Train one BLIP3D config (launch with torchrun on every node).

    torchrun --nnodes N --nproc_per_node 8 ... tools/train.py configs/train/s2_ss.yaml [--set compute.per_gpu_bs=4 ...]

Resume is automatic: the config's ``resume`` checkpoint if set, else the latest ``checkpoint-*`` in the output dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blip3d.utils import backend  # noqa: E402

backend.setup("train")

import torch  # noqa: E402
from transformers import TrainerCallback, TrainingArguments  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402

from blip3d.cond.encoder import DINO_DEFAULT, QWEN_DEFAULT, CondEncoder  # noqa: E402
from blip3d.train.callbacks import AdaptiveGradClipCallback, RankRNGCallback  # noqa: E402
from blip3d.train.ckpt import from_v12, is_v12  # noqa: E402
from blip3d.train.ema import EMACallback  # noqa: E402
from blip3d.train.config import load_config  # noqa: E402
from blip3d.train.trainer import Blip3DTrainer  # noqa: E402
from blip3d.utils.paths import REPO_ROOT, get_paths, resolve_hf_snapshot  # noqa: E402


class ConfigStamp(TrainerCallback):
    """Every checkpoint records its training config and the encoder identities."""

    def __init__(self, config_json: str, encoders: dict):
        self.blob = {"model": "blip3d", "version": "v12", "config": json.loads(config_json), "encoders": encoders}

    def on_save(self, args, state, control, **kw):
        if state.is_world_process_zero:
            d = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.isdir(d):
                json.dump(self.blob, open(os.path.join(d, "blip3d_config.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, key.sub=value")
    ap.add_argument("--dry-run", action="store_true", help="build model and data, run no step")
    a = ap.parse_args()

    r = load_config(a.config, a.set)
    if os.environ.get("BLIP3D_MEM_GB"):        # smoke tests beside another job on the same GPU
        lr_ = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(lr_)
        tot = torch.cuda.get_device_properties(lr_).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["BLIP3D_MEM_GB"]) / tot), lr_)
    paths = get_paths()
    out = r.out_dir()
    os.makedirs(out, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    qwen = resolve_hf_snapshot(QWEN_DEFAULT, paths.hf_cache)
    dino = resolve_hf_snapshot(DINO_DEFAULT, paths.hf_cache)

    resume = r.resolve(r.resume) if r.resume else get_last_checkpoint(out) if os.path.isdir(out) else None
    if resume and r.resume and get_last_checkpoint(out):
        resume = get_last_checkpoint(out)        # the run has progressed past the config's starting checkpoint
    resume_step = json.load(open(os.path.join(resume, "trainer_state.json")))["global_step"] if resume else 0

    if r.stage == "s3":
        from blip3d.train.build_unified import build_unified_trainable
        model = build_unified_trainable(r, paths.trellis2_ckpt)
        key_map = None
    else:
        from blip3d.train.build import build_tower
        model = build_tower(r, paths.trellis2_ckpt)
        key_map = (lambda sd: from_v12(sd, r.tower) if is_v12(sd) else sd)

    from blip3d.data import build_mixture
    mix = build_mixture(r.data_path(), per_gpu_bs=r.compute.per_gpu_bs, seed=r.seed, resume_step=resume_step,
                        compat_replay=r.compat.data_replay, align_points=r.road, qwen_path=qwen)
    if rank == 0:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[blip3d] {r.name}: stage {r.stage} tower {r.tower} | trainable {n_tr / 1e6:.1f}M | "
              f"GA {r.compute.grad_accum} | resume {resume} (step {resume_step}) | out {out}", flush=True)
        with open(os.path.join(out, "blip3d_config.json"), "w") as f:
            f.write(r.to_json())
    if a.dry_run:
        return

    ds_cfg = REPO_ROOT / "configs" / "deepspeed" / ("zero1_fp32acc_universal.json" if r.compute.universal
                                                     else "zero1_fp32acc.json")
    args = TrainingArguments(
        output_dir=out, run_name=r.name, max_steps=r.max_steps, seed=r.seed, bf16=True,
        per_device_train_batch_size=r.compute.per_gpu_bs, gradient_accumulation_steps=r.compute.grad_accum,
        learning_rate=r.lr, warmup_steps=r.warmup_steps, weight_decay=0.01, adam_beta1=0.9, adam_beta2=0.95,
        adam_epsilon=1e-8, max_grad_norm=1.0, lr_scheduler_type="warmup_stable_decay",
        lr_scheduler_kwargs={"num_decay_steps": r.decay_steps, "decay_type": "cosine", "min_lr_ratio": 0.0},
        logging_steps=5, save_steps=r.compute.save_steps, save_total_limit=r.compute.keep, report_to=r.report_to,
        deepspeed=str(ds_cfg), dataloader_num_workers=mix.num_workers, ignore_data_skip=True,
        remove_unused_columns=False, accelerator_config={"dispatch_batches": False})

    clip = AdaptiveGradClipCallback(resume_from=resume)
    ema = EMACallback(decay=float(r.ema.get("decay", 0.9999)), warmup=bool(r.ema.get("warmup", True)),
                      resume_from=resume, init_from=r.resolve(r.ema.get("from")), init_n=r.ema.get("n"),
                      key_map=key_map)
    enc = {"qwen": QWEN_DEFAULT, "dino": DINO_DEFAULT}
    trainer = Blip3DTrainer(
        model=model, args=args, train_dataset=mix.dataset, data_collator=mix.collate_fn,
        callbacks=[ema, RankRNGCallback(shared_t=r.compat.shared_t), clip, ConfigStamp(r.to_json(), enc)],
        encoder_factory=lambda: CondEncoder(qwen, dino, device=str(args.device),
                                            full_model_hidden=r.compat.full_model_hidden),
        clip=clip, task_names=mix.task_names)
    trainer.train(resume_from_checkpoint=resume)


if __name__ == "__main__":
    main()
