"""blip3d.data against the v12 data path (BLIP3o/trellis2_blip3o/data), CPU only.

Run (compute node, not the login node):
    PYTHONNOUSERSITE=1 <blip3o_trellis python> tests/test_data_parity_cpu.py [--slices DIR]

What is compared, with ``compat_replay=True`` on the new side:
  1. manifest filtering        same records in the same order (sha, captions, paths, n_views)       per task, 4 towers
  2. task / index / n_views    the mixture streams, several seeds x ranks x workers, balance on and off
  3. items                     prep payloads (input_ids, pixel_values, grid, dino pixels, views), ss / shape / tex /
                               placeholder / align points, with v12's unseeded generators replaced by a shared
                               deterministic sequence on both sides
  4. collate                   every batch tensor
And for the default (fixed) semantics:
  5. seed = seed + resume_step reproduces the old stream at base seed resume_step
  6. reproducible per-item draws; whitelisted failures resample, anything else raises; a DataLoader with workers
     yields batches with exactly the documented keys
  7. the balancer's age cap releases items v12 left stuck; the accelerate pass-through yields every item
Manifests are 2000-row slices of the real v12 manifests (created on first run).
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLD = Path("/fsx/home/weikai.huang/3dgen/model/BLIP3o")
OLD_YAML = {"ss": OLD / "configs/mix_v12_ss_texfixed.yaml", "shape": OLD / "configs/mix_v12_shape_texfixed.yaml",
            "tex": OLD / "configs/mix_v12_tex_texfixed.yaml",
            "unify": Path("/fsx/data/weikai.huang/lc/configs/mix_v12_unify_texfixed.yaml")}
NEW_YAML = {k: REPO / f"recipes/data/v12_{k}.yaml" for k in OLD_YAML}
BALANCE_TABLE = "/fsx/home/weikai.huang/_lc/balance/voxels_v11.npz"
OLD2NEW = {"text_to_3d_weighted": "t_real", "text_to_3d": "t_synth", "image_to_3d": "i1", "multi_image_to_3d": "im"}

sys.path.insert(0, str(REPO))
from blip3d.utils import backend  # noqa: E402

backend.setup("train")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail and not cond else ""), flush=True)
    return bool(cond)


def eq(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    return a == b


# ─────────────────────────── setup ───────────────────────────
def make_slices(dst: Path, rows: int = 2000) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    src = OLD / "manifests/splits"
    names = set()
    for y in OLD_YAML.values():
        for t in yaml.safe_load(open(y))["tasks"]:
            names.add(Path(t["args"]["manifest"]).name)
    for n in sorted(names):
        out = dst / n
        if not out.is_file():
            with open(out, "w") as f:
                subprocess.run(["head", "-n", str(rows), str(src / n)], stdout=f, check=True)
    return dst


def setup_paths(slices: Path) -> str:
    """paths.yaml copy whose ``manifests`` points at the slices; returns the Qwen snapshot dir."""
    d = yaml.safe_load(open(REPO / "paths.yaml"))
    d["manifests"] = str(slices)
    f = Path(tempfile.mkdtemp()) / "paths.yaml"
    yaml.safe_dump(d, open(f, "w"))
    os.environ["BLIP3D_PATHS"] = str(f)
    from blip3d.utils.paths import get_paths
    get_paths.cache_clear()
    p = get_paths()
    return p.hf_snapshot(p.encoders["qwen"])


def old_modules(qwen: str):
    os.environ["COND_VLM_CKPT"] = qwen
    os.environ.pop("GEOTEX_BALANCE_TABLE", None)
    sys.path.insert(1, str(OLD))
    import trellis2_blip3o.data.mixture as om
    from trellis2_blip3o.data import registry
    import trellis2_blip3o.data.tasks  # noqa: F401
    import trellis2_blip3o.vlm_collate as oc
    import trellis2_blip3o.live_cond_batch as olc
    return om, registry, oc, olc


def old_datasets(tower: str, slices: Path, registry):
    cfg = yaml.safe_load(open(OLD_YAML[tower]))
    out, weights = [], []
    for t in cfg["tasks"]:
        args = dict(t["args"])
        args["manifest"] = str(slices / Path(args["manifest"]).name)
        cls = registry.get_task(t["name"])
        ds = cls(**args)
        ds._probe_name = OLD2NEW[t["name"]]
        out.append(ds)
        weights.append(float(t["weight"]))
    return out, weights


def new_mixture(tower: str, qwen: str, *, seed=0, resume_step=0, compat=True, bs=4):
    from blip3d.data import build_mixture
    from blip3d.data import manifest as M
    M.clear_cache()
    return build_mixture(str(NEW_YAML[tower]), per_gpu_bs=bs, seed=seed, resume_step=resume_step,
                         compat_replay=compat, align_points=(tower == "ss"), qwen_path=qwen)


class OldProbe:
    """Old task wrapped so the mixture yields (name, index, batch n_views) instead of loading."""

    def __init__(self, ds):
        self.ds, self.records, self.task_name = ds, ds.records, ds._probe_name

    def __len__(self):
        return len(self.ds)

    def set_batch_params(self, rng):
        self.ds.set_batch_params(rng)

    def clear_batch_params(self):
        self.ds.clear_batch_params()

    def __getitem__(self, j):
        return (self.task_name, int(j), self.ds._batch_n_views)


def probe_new(tasks):
    for t in tasks:
        t.get = types.MethodType(lambda self, j, rng=None: (self.task_name, int(j), self._batch_n_views), t)


class FakeWorker:
    def __init__(self, wid, n):
        self.id, self.num_workers = wid, n


def take(ds, n, module, wid=0, nw=1):
    module.get_worker_info = (lambda: FakeWorker(wid, nw)) if nw > 1 else (lambda: None)
    it = iter(ds)
    return [next(it) for _ in range(n)]


class SeqRng:
    """Stand-in for np.random.default_rng: unseeded calls return generators seeded 1, 2, 3, ... in call order."""

    def __init__(self, orig, start):
        self.orig, self.k = orig, start

    def __call__(self, seed=None, *a, **kw):
        if seed is None:
            self.k += 1
            return self.orig(self.k)
        return self.orig(seed, *a, **kw)


# ─────────────────────────── 1. manifest filtering ───────────────────────────
def test_manifests(slices, registry, qwen):
    for tower in OLD_YAML:
        olds, _ = old_datasets(tower, slices, registry)
        mix = new_mixture(tower, qwen)
        for od, nt in zip(olds, mix.dataset.tasks):
            recs = od.records
            m = nt.records
            ok = len(recs) == len(m)
            bad = ""
            if ok:
                for i, r in enumerate(recs):
                    nr = m.record(i)
                    want = (r.get("sha256"), [c for c in (r.get("captions") or []) if c], r.get("ss_latent_64"),
                            r.get("shape_latent_512"), r.get("pbr_latent_512"), r.get("renders_dir"),
                            int(r.get("n_views", 16)))
                    got = (nr.sha, nr.captions, nr.ss, nr.shape, nr.pbr, nr.renders_dir, nr.n_views)
                    if want != got:
                        ok, bad = False, f"row {i}: {want} != {got}"
                        break
            check(f"manifest {tower}/{nt.task_name}: {len(recs)} records identical", ok, bad or f"{len(recs)} vs {len(m)}")


# ─────────────────────────── 2. streams ───────────────────────────
def test_streams(slices, om, registry, qwen):
    import blip3d.data.mixture as nm
    for tower in ("ss", "unify"):
        olds, w = old_datasets(tower, slices, registry)
        for balance in (False, True):
            if balance and tower != "unify":
                continue
            for seed in (0, 7, 150000):
                for rank in (0, 3):
                    for wid, nw in ((0, 1), (0, 4), (3, 4)):
                        os.environ["RANK"] = str(rank)
                        if balance:
                            os.environ["GEOTEX_BALANCE_TABLE"] = BALANCE_TABLE
                        else:
                            os.environ.pop("GEOTEX_BALANCE_TABLE", None)
                        ods = om.MixtureIterableDataset([OldProbe(d) for d in olds], w, batch_size=4, base_seed=seed)
                        a = take(ods, 400, om, wid, nw)
                        mix = new_mixture(tower, qwen, seed=seed, compat=True)
                        if not balance:
                            mix.dataset.balance = None
                        probe_new(mix.dataset.tasks)
                        b = take(mix.dataset, 400, nm, wid, nw)
                        n_im = sum(1 for x in a if x[0] == "im")
                        check(f"stream {tower} bal={balance} seed={seed} rank={rank} worker={wid}/{nw} "
                              f"(400 items, {n_im} multi-image)", a == b,
                              next((f"item {i}: {x} vs {y}" for i, (x, y) in enumerate(zip(a, b)) if x != y), ""))
    os.environ.pop("GEOTEX_BALANCE_TABLE", None)
    os.environ["RANK"] = "0"
    # 5. fixed semantics: base seed = seed + resume_step
    olds, w = old_datasets("ss", slices, registry)
    ods = om.MixtureIterableDataset([OldProbe(d) for d in olds], w, batch_size=4, base_seed=25000)
    a = take(ods, 400, om)
    mix = new_mixture("ss", qwen, seed=0, resume_step=25000, compat=False)
    probe_new(mix.dataset.tasks)
    b = take(mix.dataset, 400, nm)
    check("fixed: seed 0 + resume_step 25000 = v12 stream at base seed 25000", a == b)
    mix2 = new_mixture("ss", qwen, seed=0, resume_step=0, compat=False)
    probe_new(mix2.dataset.tasks)
    c = take(mix2.dataset, 400, nm)
    check("fixed: a different resume step gives a different stream", a != c)
    check("compat_replay ignores resume_step (v12 replay)",
          new_mixture("ss", qwen, seed=0, resume_step=25000, compat=True).dataset.base_seed == 0)


# ─────────────────────────── 3/4. items and collate ───────────────────────────
def find_rows(task, pred, k):
    out = []
    for i in range(len(task.records)):
        if pred(task.records.record(i)):
            out.append(i)
            if len(out) == k:
                break
    return out


def over_cap(rec):
    from blip3d.data.latents import read_coords
    return bool(rec.shape) and read_coords(rec.shape).shape[0] > 8192


def compare_prep(op, np_, modality):
    ok = eq(op["input_ids"], np_["input_ids"])
    if modality == "i1":
        ok &= eq(op["pixel_values"], np_["pixel_values"])
        ok &= eq(op["image_grid_thw"].reshape(1, 3), np_["image_grid_thw"])
        ok &= eq(op["dino_px"][None], np_["dino_px"])
        ok &= int(op["view"]) == int(np_["views"][0])
    elif modality == "im":
        for k in ("pixel_values", "image_grid_thw", "dino_px", "views"):
            ok &= eq(op[k], np_[k])
    else:
        ok &= op.get("modality") == np_.get("modality") == "t"
    return ok


def compare_item(o, n, modality):
    from trellis2.modules.sparse import SparseTensor  # noqa: F401
    ok = o["id"] == n["id"]
    ok &= compare_prep(o["_live_prep"], n["prep"], modality)
    ok &= eq(o["target_ss_latent"], n["ss"])
    ok &= ("target_shape_slat_512_item" in o) == ("shape" in n)
    if "shape" in n:
        ok &= eq(o["target_shape_slat_512_item"]["coords"], n["shape"]["coords"])
        ok &= eq(o["target_shape_slat_512_item"]["feats"], n["shape"]["feats"])
    ok &= ("target_tex_slat_512_item" in o) == ("tex" in n)
    if "tex" in n:
        for k in ("x_0", "concat_cond"):
            ok &= eq(o["target_tex_slat_512_item"][k].coords, n["tex"][k].coords)
            ok &= eq(o["target_tex_slat_512_item"][k].feats, n["tex"][k].feats)
        ok &= o.get("tex_valid") == n.get("tex_valid")
    ok &= ("align_points" in o) == ("align_points" in n)
    if "align_points" in n:
        ok &= eq(o["align_points"], n["align_points"])
    return bool(ok)


def test_items_and_collate(slices, registry, oc, qwen):
    from blip3d.data.collate import collate
    from blip3d.data.latents import VoxelCapExceeded
    orig = np.random.default_rng
    for tower in OLD_YAML:
        os.environ["ROAD_ALIGN_POINTS"] = "10000" if tower == "ss" else "0"
        olds, _ = old_datasets(tower, slices, registry)
        mix = new_mixture(tower, qwen, compat=True)
        for od, nt in zip(olds, mix.dataset.tasks):
            rows = list(range(0, 1600, 400))
            if tower == "unify":      # rows without pbr exercise the placeholder
                rows = find_rows(nt, lambda r: r.pbr is None, 2) + rows[:2]
            if nt.modality == "im":
                nt._batch_n_views = od._batch_n_views = 3
            items_o, items_n, ok_all = [], [], True
            for k, i in enumerate(rows):
                np.random.default_rng = SeqRng(orig, 1000 * k)
                try:
                    o = od._load_one(i)
                    np.random.default_rng = SeqRng(orig, 1000 * k)
                    n = nt.load(i, None)
                finally:
                    np.random.default_rng = orig
                good = compare_item(o, n, nt.modality)
                ok_all &= good
                if not good:
                    print(f"   mismatch {tower}/{nt.task_name} row {i}")
                items_o.append(o)
                items_n.append(n)
            nt._batch_n_views = od._batch_n_views = None
            check(f"items {tower}/{nt.task_name}: {len(rows)} rows (prep + targets) identical", ok_all)
            bo = oc.collate_vlm_3d(items_o, None)
            bn = collate(items_n, strict_shape=False)
            ok = eq(bo["target_ss_latent"], bn["ss"]) and bn["ids"] == [x["id"] for x in items_o]
            pairs = [("target_shape_slat_512", "shape"), ("target_tex_slat_512", "tex"), ("tex_concat_cond", "tex_concat")]
            for ko, kn in pairs:
                ok &= (ko in bo) == (kn in bn)
                if kn in bn:
                    ok &= eq(bo[ko].coords, bn[kn].coords) and eq(bo[ko].feats, bn[kn].feats)
                    ok &= bo[ko].shape == bn[kn].shape
            ok &= ("tex_valid" in bo) == ("tex_valid" in bn)
            if "tex_valid" in bn:
                ok &= eq(bo["tex_valid"], bn["tex_valid"])
            ok &= ("align_points" in bo) == ("align_points" in bn)
            if "align_points" in bn:
                ok &= eq(bo["align_points"], bn["align_points"])
            check(f"collate {tower}/{nt.task_name}: batch tensors identical "
                  f"(keys {sorted(k for k in bn if k not in ('prep', 'ids'))})", ok)
        if tower == "shape":
            nt = mix.dataset.tasks[2]
            od = olds[2]
            rows = find_rows(nt, over_cap, 1)
            if rows:
                try:
                    od._load_one(rows[0])
                    old_raised = False
                except ValueError as e:
                    old_raised = "max_slat_tokens" in str(e)
                try:
                    nt.load(rows[0], None)
                    new_raised = False
                except VoxelCapExceeded:
                    new_raised = True
                check("voxel cap: an over-cap asset is rejected by both", old_raised and new_raised)
    os.environ["ROAD_ALIGN_POINTS"] = "0"


def test_draw_parity(slices, registry, olc):
    olds, _ = old_datasets("unify", slices, registry)
    from blip3d.data.task import ThreeDTask
    ok = True
    for s in range(300):
        for n in (1, 2, 3, 4):
            ok &= olds[1]._pick_caption_idx(np.random.default_rng(s), n) == \
                ThreeDTask.caption_index(types.SimpleNamespace(cfg=types.SimpleNamespace(
                    caption_fallback="renormalize", caption_weights=[0.45, 0.15, 0.15, 0.25])),
                    np.random.default_rng(s), n)
        for n_avail, k in ((16, 2), (16, 3), (16, 4), (12, 4)):
            ok &= olc.pick_im_views(np.random.default_rng(s), n_avail, k) == \
                ThreeDTask.pick_im_views(np.random.default_rng(s), n_avail, k)
    check("draws: caption tier (renormalize) and multi-image views, 300 seeds", ok)


# ─────────────────────────── 6/7. fixed semantics, loader, balancer, accelerate ───────────────────────────
def test_fixed_semantics(qwen):
    from torch.utils.data import DataLoader
    from blip3d.data.latents import MissingFile
    import blip3d.data.mixture as nm
    nm.get_worker_info = __import__("torch.utils.data", fromlist=["get_worker_info"]).get_worker_info
    for tower in ("ss", "shape", "tex", "unify"):
        mix = new_mixture(tower, qwen, compat=False, seed=3)
        dl = DataLoader(mix.dataset, batch_size=4, collate_fn=mix.collate_fn, num_workers=2)
        it = iter(dl)
        batches = [next(it) for _ in range(6)]
        want = {"task", "modality", "ids", "prep", "ss"}
        if tower == "ss":
            want |= {"align_points"}
        if tower in ("shape", "tex", "unify"):
            want |= {"shape"}
        if tower in ("tex", "unify"):
            want |= {"tex", "tex_concat", "tex_valid"}
        ok = all(set(b) == want for b in batches)
        ok &= all(b["ss"].shape == (4, 8, 16, 16, 16) and b["ss"].dtype == torch.float32 for b in batches)
        ok &= all(len(b["prep"]) == 4 and b["task"] in mix.task_names for b in batches)
        if tower == "ss":
            ok &= all(b["align_points"].shape == (4, 10000, 3) for b in batches)
        if "shape" in want:
            ok &= all(b["shape"].coords.shape[1] == 4 and b["shape"].coords.dtype == torch.int32 for b in batches)
        check(f"fixed loader {tower}: 6 batches via 2 workers carry exactly {sorted(want)}", ok,
              str([sorted(b) for b in batches][:2]))
        # reproducible
        mix2 = new_mixture(tower, qwen, compat=False, seed=3)
        dl2 = DataLoader(mix2.dataset, batch_size=4, collate_fn=mix2.collate_fn, num_workers=2)
        it2 = iter(dl2)
        b2 = [next(it2) for _ in range(6)]
        same = all(x["ids"] == y["ids"] and all(eq(p["input_ids"], q["input_ids"]) for p, q in zip(x["prep"], y["prep"]))
                   for x, y in zip(batches, b2))
        check(f"fixed loader {tower}: identical batches on a rebuild (seeded per-item draws)", same)
    # whitelist: IO-type failures resample and are counted; anything else raises
    mix = new_mixture("ss", qwen, compat=False)
    t = mix.dataset.tasks[2]
    orig_load = t.load
    calls = {"n": 0}

    def flaky(self, i, rng=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise MissingFile("simulated")
        return orig_load(i, rng)
    t.load = types.MethodType(flaky, t)
    item = t.get(0, np.random.default_rng(0))
    check("fixed: whitelisted failure resampled and counted", item["id"] and t.skips["missing_file"] == 2)

    def buggy(self, i, rng=None):
        raise TypeError("simulated bug")
    t.load = types.MethodType(buggy, t)
    try:
        t.get(0, np.random.default_rng(0))
        raised = False
    except RuntimeError as e:
        raised = isinstance(e.__cause__, TypeError)
    check("fixed: a non-whitelisted error fails immediately", raised)


def test_balancer_and_accelerate():
    from blip3d.data.balance import TargetBatcher
    rng = np.random.default_rng(0)
    costs = list(rng.integers(1000, 3000, 2000).astype(float))
    for j in (5, 50, 500):
        costs[j] = 40000.0          # can never fit a batch near the target
    def run(max_age):
        b = TargetBatcher(4 * 4000.0, 4, buf_size=96, seed=13, max_age=max_age)
        popped, j = [], 0
        for _ in range(450):
            while not b.ready() and j < len(costs):
                b.push(costs[j], j)
                j += 1
            popped += b.pop_batch()
        return popped, [e[1] for e in b._buf]
    p_old, buf_old = run(None)
    p_new, buf_new = run(64)
    check("balancer v12 (no age cap): the over-cost items stay stuck", {5, 50, 500} <= set(buf_old))
    check("balancer age cap: the over-cost items are released, nothing popped twice",
          {5, 50, 500} <= set(p_new) and len(p_new) == len(set(p_new)))
    from accelerate.data_loader import IterableDatasetShard
    from blip3d.data.dist import install_shard_passthrough
    install_shard_passthrough()

    class Ds(torch.utils.data.IterableDataset):
        _rank_sharded = True

        def __iter__(self):
            return iter(range(64))
    got = list(IterableDatasetShard(Ds(), batch_size=4, num_processes=8, process_index=3))
    check("accelerate pass-through: a rank-sharded dataset is not re-sliced", got == list(range(64)))


def test_norm_stats():
    from blip3d.data.latents import norm_stats
    sys.path.insert(1, str(OLD))
    from trellis2_blip3o.tr2_modules import load_norm_stats, SHAPE_SLAT_CONFIG_PATH, TEX_SLAT_CONFIG_PATH
    st = norm_stats()
    a = load_norm_stats(SHAPE_SLAT_CONFIG_PATH, "normalization")
    b = load_norm_stats(TEX_SLAT_CONFIG_PATH, "pbr_slat_normalization")
    c = load_norm_stats(TEX_SLAT_CONFIG_PATH, "shape_slat_normalization")
    ok = eq(a["mean"], st.shape_mean) and eq(a["std"], st.shape_std) and eq(b["mean"], st.tex_mean) \
        and eq(b["std"], st.tex_std) and eq(c["mean"], st.tex_shape_mean) and eq(c["std"], st.tex_shape_std)
    check("latent normalization statistics identical (shape, pbr, tex-shape)", ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", default=str(Path(tempfile.gettempdir()) / "blip3d_data_slices"))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    slices = make_slices(Path(args.slices))
    qwen = setup_paths(slices)
    om, registry, oc, olc = old_modules(qwen)
    tests = {
        "norm": lambda: test_norm_stats(),
        "manifests": lambda: test_manifests(slices, registry, qwen),
        "draws": lambda: test_draw_parity(slices, registry, olc),
        "streams": lambda: test_streams(slices, om, registry, qwen),
        "items": lambda: test_items_and_collate(slices, registry, oc, qwen),
        "fixed": lambda: test_fixed_semantics(qwen),
        "balancer": lambda: test_balancer_and_accelerate(),
    }
    for name, fn in tests.items():
        if args.only and name not in args.only.split(","):
            continue
        try:
            fn()
        except Exception:
            traceback.print_exc()
            check(f"{name}: ran without an exception", False)
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
