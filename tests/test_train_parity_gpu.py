#!/usr/bin/env python
"""GPU parity of the BLIP3D training path against the v12 code (old repo imported read-only).

A. Encoder: v12 TrainCondEncoder vs blip3d CondEncoder on the same prep payloads (i1, im, t batches from the tex mix),
   with ``full_model_hidden`` True (the v12 call) and False (base-model hidden; must be identical, ISSUES C-03).
B. Tower loss: v12 ``compute_cascade_flow_loss`` vs ``Blip3DTower.forward`` for ss / shape / tex on the S2 finals,
   same flow module, same seeds, shared-CPU timesteps (v12). Loss must be bit-identical and the connector gradients
   identical. ROAD is compared separately (tests/test_road_gpu.py).

    CUDA_VISIBLE_DEVICES=k PYTHONNOUSERSITE=1 python tests/test_train_parity_gpu.py --mem-gb 24
``--mem-gb`` caps this process (safe to run beside a training job that leaves that much free).
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
sys.path.insert(0, REPO)
from blip3d.utils import backend  # noqa: E402

backend.setup("train")
sys.path.append(OLD)

import torch  # noqa: E402

from blip3d.cond.assemble import collate as cond_collate  # noqa: E402
from blip3d.cond.connector import Connector  # noqa: E402
from blip3d.cond.encoder import DINO_DEFAULT, QWEN_DEFAULT, CondEncoder  # noqa: E402
from blip3d.cond.stamp import ViewCodes  # noqa: E402
from blip3d.models.blip3d import Blip3DTower  # noqa: E402
from blip3d.models.towers import released_flow, to_bf16_keep_complex  # noqa: E402
from blip3d.train.ckpt import tower_state, to_v12  # noqa: E402
from blip3d.utils.paths import REPO_ROOT, get_paths, resolve_hf_snapshot  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def get_batches(qwen):
    from blip3d.data import build_mixture
    mix = build_mixture(str(REPO_ROOT / "recipes/data/v12_tex.yaml"), per_gpu_bs=2, seed=0, qwen_path=qwen)
    dl = torch.utils.data.DataLoader(mix.dataset, batch_size=2, collate_fn=mix.collate_fn, num_workers=0)
    got = {}
    for b in dl:
        got.setdefault(b["modality"], b)
        if len(got) == 3:
            return got
    raise RuntimeError("mixture ended early")


def encoder_parity(batches, qwen, dino, dev):
    from trellis2_blip3o.live_cond_batch import TrainCondEncoder
    old = TrainCondEncoder(vlm_path=qwen, device=dev)
    ref = {m: [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in r.items()} for r in old.encode(b["prep"])]
           for m, b in batches.items()}
    del old
    torch.cuda.empty_cache()
    out = {}
    for fmh in (True, False):
        enc = CondEncoder(qwen, dino, device=dev, full_model_hidden=fmh)
        for m, b in batches.items():
            recs = enc.encode(b["prep"])
            for i, (r, o) in enumerate(zip(recs, ref[m])):
                tag = f"enc fmh={fmh} {m}[{i}]"
                check(f"{tag} qwen", eq(r["qwen"].cpu(), o["cond_hidden"]),
                      f"maxdiff {(r['qwen'].float().cpu() - o['cond_hidden'].float()).abs().max():.3g}"
                      if r["qwen"].shape == o["cond_hidden"].shape else f"{tuple(r['qwen'].shape)} vs {tuple(o['cond_hidden'].shape)}")
                check(f"{tag} keep", eq(r["qwen_keep"].cpu(), o["cond_keep_mask"]))
                if m != "t":
                    check(f"{tag} rc", eq(r["qwen_rc"].cpu().float(), o["qwen_img_rc"].float()))
                    check(f"{tag} dino", eq(r["dino"].cpu(), o["dino_hidden"]))
                    check(f"{tag} dino_views", eq(r["dino_views"].cpu(), o["dino_view_ids"]))
                if m == "im":
                    check(f"{tag} qwen_views", eq(r["qwen_views"].cpu(), o["qwen_view_ids"]))
            if not fmh:
                out[m] = recs
        if fmh:
            del enc
            torch.cuda.empty_cache()
    return out, enc


def old_inputs(cb):
    """blip3d collated cond records -> the v12 compute_cascade_flow_loss keyword inputs (bf16 as the v12 model cast)."""
    kw = {"cond_hidden": cb["qwen"].to(torch.bfloat16), "cond_key_mask": cb["qwen_keep"].bool(),
          "qwen_img_rc": cb.get("qwen_rc")}
    if "dino" in cb:
        kw.update(dino_hidden=cb["dino"].to(torch.bfloat16), dino_key_mask=cb["dino_keep"].bool(),
                  dino_view_ids=cb["dino_views"], qwen_view_ids=cb.get("qwen_views"))
    return kw


def tower_parity(kind, batches, recs, paths, dev):
    from trellis2_blip3o.connector import TRELLIS2TransformerAdapter
    from trellis2_blip3o.flow_heads import compute_cascade_flow_loss
    from trellis2_blip3o.loss import TRELLIS2FlowMatchingLoss
    ck = os.path.join(paths.runs, f"keep/v12/s2_{kind}/checkpoint-106000")
    sd = tower_state(ck, kind, use_ema=False)
    flow = to_bf16_keep_complex(released_flow(kind, paths.trellis2_ckpt))
    flow.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("flow.")}, strict=True)
    flow = flow.to(dev).train().requires_grad_(False)
    new_conn = Connector()
    new_conn.load_state_dict({k[10:]: v for k, v in sd.items() if k.startswith("connector.")}, strict=True)
    old_conn = TRELLIS2TransformerAdapter(2048, 1024, n_blocks=2, seg_embed=True, patch_pos="sincos2d", patch_lattice=32)
    old_sd = {k[len("diffusion_connector."):]: v for k, v in to_v12({k: v for k, v in sd.items()
                                                                     if k.startswith("connector.")}, kind).items()}
    old_conn.load_state_dict(old_sd, strict=True)
    new_conn, old_conn = new_conn.to(dev, torch.bfloat16).train(), old_conn.to(dev, torch.bfloat16).train()
    table = sd["views.table"].to(dev)
    tower = Blip3DTower(kind, flow, new_conn, ViewCodes(table=table)).to(dev).train()
    lf_ss = TRELLIS2FlowMatchingLoss("logitNormal", 1.0, 1.0, 1e-5)
    lf_slat = TRELLIS2FlowMatchingLoss("uniform")
    for m, b in batches.items():
        cb = cond_collate([{k: v for k, v in r.items()} for r in recs[m]], device=dev)
        tb = {k: (v.to(dev) if (torch.is_tensor(v) or hasattr(v, "feats")) else v) for k, v in b.items()}
        tb["cond"] = cb
        tgt = {"ss": dict(target_ss_latent=tb["ss"]),
               "shape": dict(target_ss_latent=None, target_shape_slat_512=tb["shape"]),
               "tex": dict(target_ss_latent=None, target_tex_slat_512=tb["tex"], tex_concat_cond=tb["tex_concat"])}[kind]

        def run_old():
            torch.manual_seed(1234); torch.cuda.manual_seed(1234)
            lo, _ = compute_cascade_flow_loss(
                connector=old_conn, ss_flow=flow if kind == "ss" else None,
                shape_slat=flow if kind == "shape" else None, tex_slat=flow if kind == "tex" else None,
                loss_fn_ss=lf_ss, loss_fn_slat=lf_slat, cond_max_length=10240, mask_drop_prob=0.1,
                dino_drop_prob=0.3 if "dino" in cb else 0.0, qwen_drop_prob=0.0, dino_view_embed=table,
                cond_seg_embed=old_conn.cond_seg_embed, cond_patch_pos=old_conn.cond_patch_pos, **old_inputs(cb), **tgt)
            lo.backward()
            g = {n: p.grad.clone() for n, p in old_conn.named_parameters()}
            old_conn.zero_grad(set_to_none=True)
            return lo, g

        lo, g_old = run_old()
        _, g_old2 = run_old()            # the old code against itself: the backward-kernel noise floor
        floor = max((g_old[n].float() - g_old2[n].float()).abs().max().item() for n in g_old)
        torch.manual_seed(1234); torch.cuda.manual_seed(1234)
        ln, logs = tower(tb)
        ln.backward()
        g_new = {n: p.grad.clone() for n, p in new_conn.named_parameters()}
        new_conn.zero_grad(set_to_none=True)
        check(f"loss {kind} {m}", torch.equal(lo.detach(), ln.detach()), f"old {lo.item():.6f} new {ln.item():.6f}")
        gd = max((g_old[n].float() - g_new[n].float()).abs().max().item() for n in g_new)
        rel = max(((g_old[n].float() - g_new[n].float()).norm() / g_old[n].float().norm().clamp_min(1e-30)).item()
                  for n in g_new)
        check(f"grads {kind} {m}", gd <= 2 * floor or gd == 0.0,
              f"max |dg| new-old {gd:.3g} vs old-old {floor:.3g}; max rel {rel:.2e} over {len(g_new)} tensors")
    del flow, tower, old_conn
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-gb", type=float, default=24)
    ap.add_argument("--kinds", nargs="*", default=["ss", "shape", "tex"])
    ap.add_argument("--skip-encoder", action="store_true")
    a = ap.parse_args()
    dev = "cuda:0"
    tot = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb / tot), 0)
    paths = get_paths()
    qwen = resolve_hf_snapshot(QWEN_DEFAULT, paths.hf_cache)
    dino = resolve_hf_snapshot(DINO_DEFAULT, paths.hf_cache)
    batches = get_batches(qwen)
    print({m: (len(b["ids"]), b["task"]) for m, b in batches.items()}, flush=True)
    recs, enc = encoder_parity(batches, qwen, dino, dev)
    del enc
    torch.cuda.empty_cache()
    for k in a.kinds:
        tower_parity(k, batches, recs, paths, dev)
    print(f"\n{len(FAILS)} failures" + (f": {FAILS}" if FAILS else ""), flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
