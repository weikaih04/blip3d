"""CPU parity of blip3d.cond against the v12 code (old repo, read-only). Run: PYTHONNOUSERSITE=1 python tests/test_cond_parity_cpu.py
Covers prompts, CPU prep (tokens, pixels, DINO framing), stamp tables vs every v12 checkpoint, the connector, token
rc/views, and assembly (training path with drops, eval CFG pair) in fp32 and bf16. The GPU encoder parity is separate."""
import json
import os
import sys
import types

import torch

OLD = "/fsx/home/weikai.huang/3dgen/model/BLIP3o"
NEW = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QWEN = "/fsx/home/weikai.huang/hf_cache/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
KEEP = "/fsx/data/weikai.huang/runs/keep/v12"
os.environ["COND_VLM_CKPT"] = QWEN
sys.path.insert(0, NEW)
from blip3d.utils import backend  # noqa: E402
backend.setup("eval")
sys.path.insert(1, OLD)
from PIL import Image  # noqa: E402
from blip3d.cond import prompts, prep, stamp, assemble as A  # noqa: E402
from blip3d.cond.connector import Connector  # noqa: E402
from blip3d.train.ckpt import load_state, connector_state  # noqa: E402
import trellis2_blip3o.live_cond_batch as LCB  # noqa: E402
import trellis2_blip3o.live_cond as LC  # noqa: E402
from trellis2_blip3o.vlm_collate import boiler_ids  # noqa: E402
from trellis2_blip3o.connector import TRELLIS2TransformerAdapter  # noqa: E402
from trellis2_blip3o.flow_heads import build_unified_cond  # noqa: E402
import trellis2_blip3o.eval_cond as EC  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL.append(name)


def eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


# ── prompts ──
check("image prompt i1", prompts.image_prompt(1) == LC.PROMPT_I1)
check("image prompt im3", prompts.image_prompt(3) == LCB.im_prompt(3))
sha = "0977de4829574a2ba5668b89a7f48227"
for ci in range(4):
    check(f"text prompt ci={ci}", prompts.text_prompt("a red chair", prompts.template_index(sha, ci)) == LCB.txt_prompt(sha, ci, "a red chair"))
proc = prep.processor(QWEN)
check("structural ids", prompts.structural_token_ids(proc.tokenizer) == boiler_ids(proc.tokenizer))

# ── prep on real renders ──
recs = [json.loads(l) for l in open("/fsx/home/weikai.huang/3dgen/eval_suite/configs/sfv_val_v12_eval50.jsonl")][:3]
for r in recs:
    rd, v = r["renders_dir"], int(r["input_view_index"])
    old = LCB.prep_i1(rd, v, vlm_path=QWEN)
    new = prep.prep_render(Image.open(os.path.join(rd, f"{v:03d}.webp")), qwen_path=QWEN, view=v)
    check(f"prep i1 {r['case_id']}", eq(old["input_ids"], new["input_ids"]) and eq(old["pixel_values"], new["pixel_values"])
          and eq(old["image_grid_thw"], new["image_grid_thw"][0]) and eq(old["dino_px"], new["dino_px"][0]),
          f"T={new['input_ids'].shape[0]}")
    for views in ([5, 8, 11], [3, 6, 9, 12]):
        old = LCB.prep_im(rd, views, vlm_path=QWEN)
        imgs = [Image.open(os.path.join(rd, f"{x:03d}.webp")) for x in views]
        new = prep.prep_renders(imgs, qwen_path=QWEN, views=views)
        check(f"prep im{len(views)} {r['case_id']}", eq(old["input_ids"], new["input_ids"]) and eq(old["pixel_values"], new["pixel_values"])
              and eq(old["image_grid_thw"], new["image_grid_thw"]) and eq(old["dino_px"], new["dino_px"]),
              f"T={new['input_ids'].shape[0]}")
old = LCB.prep_t(sha, 2, "a small wooden chair with a red cushion", vlm_path=QWEN)
new = prep.prep_text("a small wooden chair with a red cushion", qwen_path=QWEN, template=prompts.template_index(sha, 2))
check("prep t", eq(old["input_ids"], new["input_ids"]))

# ── stamp tables vs every v12 checkpoint ──
from safetensors import safe_open  # noqa: E402
seg, pat, vw = stamp.round_bf16(stamp.segment_codes(1024)), stamp.round_bf16(stamp.patch_codes(32, 1024, 0.2)), stamp.round_bf16(stamp.view_codes(16, 1024, 0.2))
for ck in ["s2_ss/checkpoint-106000", "s2_shape/checkpoint-106000", "s2_tex/checkpoint-106000", "s3_unify_4n/checkpoint-17000"]:
    f = safe_open(os.path.join(KEEP, ck, "model.safetensors"), "pt")
    ks = list(f.keys())
    ok = True
    for k in ks:
        if k.endswith("cond_seg_embed"):
            ok &= eq(f.get_tensor(k), seg)
        elif k.endswith("cond_patch_pos"):
            ok &= eq(f.get_tensor(k), pat)
        elif k == "dino_view_embed":
            ok &= eq(f.get_tensor(k), vw)
    check(f"stamps == ckpt {ck}", ok, f"({sum(k.endswith(('cond_seg_embed','cond_patch_pos')) or k=='dino_view_embed' for k in ks)} tables)")
vc = stamp.ViewCodes(16, 1024, 0.2, table=vw)
check("view codes extend 16->24 == regenerated", eq(vc.rows(24), stamp.round_bf16(stamp.view_codes(24, 1024, 0.2))))

# ── connector: load the S2 shape connector into both implementations ──
sd = load_state(os.path.join(KEEP, "s2_shape/checkpoint-106000"))
old_c = TRELLIS2TransformerAdapter(2048, 1024, n_blocks=2, seg_embed=True, patch_pos="sincos2d", patch_lattice=32)
old_c.load_state_dict({k[len("diffusion_connector."):]: v for k, v in sd.items() if k.startswith("diffusion_connector.")}, strict=True)
new_c = Connector()
new_c.load_state_dict(connector_state(sd, "diffusion_connector"), strict=True)
old_c, new_c = old_c.float().eval(), new_c.float().eval()
g = torch.Generator().manual_seed(0)
h = torch.randn(2, 40, 2048, generator=g)
km = torch.ones(2, 40, dtype=torch.bool); km[1, 30:] = False
with torch.no_grad():
    check("connector forward fp32", eq(old_c(h, key_mask=km), new_c(h, key_mask=km)))
views = stamp.ViewCodes(16, 1024, 0.2, table=sd["dino_view_embed"])

# ── token rc / views ──
fake = types.SimpleNamespace(image_token_id=prompts.image_pad_id(proc.tokenizer), proc=proc)
for views_ in ([5, 8, 11], None):
    p = prep.prep_renders([Image.open(os.path.join(recs[0]["renders_dir"], f"{x:03d}.webp")) for x in views_], qwen_path=QWEN) if views_ else prep.prep_render(Image.open(os.path.join(recs[0]["renders_dir"], "008.webp")), qwen_path=QWEN)
    ids, gr = p["input_ids"], p["image_grid_thw"]
    o_rc = LCB.TrainCondEncoder._qwen_img_rc(fake, ids, gr if gr.ndim == 2 else gr[None])
    n_rc = stamp.image_token_rc(ids, gr, fake.image_token_id, 2)
    check(f"qwen_rc {'im3' if views_ else 'i1'}", eq(o_rc, n_rc))
    if views_:
        check("qwen_views im3", eq(LCB.TrainCondEncoder._qwen_view_ids(fake, ids, gr), stamp.image_token_views(ids, gr, fake.image_token_id, 2)))


# ── assembly: synthetic records for i1 / im / t, training path (drops) + eval CFG pair ──
def rec(mod, T=50, K=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = {"qwen": torch.randn(T, 2048, generator=g).half(), "qwen_keep": torch.rand(T, generator=g) > 0.2, "modality": mod}
    if mod != "t":
        rc = torch.full((T, 2), -1.0); rc[5:5 + 16] = torch.rand(16, 2, generator=g); r["qwen_rc"] = rc
        r["dino"] = torch.randn(K * 30, 1024, generator=g).half(); r["dino_keep"] = torch.ones(K * 30, dtype=torch.bool)
        r["dino_views"] = torch.arange(K).repeat_interleave(30)
        if mod == "im":
            qv = torch.full((T,), -1, dtype=torch.long)
            for k in range(K):
                qv[5 + 4 * k:9 + 4 * k] = k
            r["qwen_views"] = qv
    return r


def old_batch(rs):
    b = A.collate(rs, device="cpu")
    ob = {"cond_hidden": b["qwen"], "cond_key_mask": b["qwen_keep"]}
    if "dino" in b:
        ob.update(dino_hidden=b["dino"], dino_key_mask=b["dino_keep"], dino_view_ids=b["dino_views"])
    if "qwen_rc" in b:
        ob["qwen_img_rc"] = b["qwen_rc"]
    if "qwen_views" in b:
        ob["qwen_view_ids"] = b["qwen_views"]
    return b, ob


for mod, K in (("i1", 1), ("im", 3), ("t", 0)):
    rs = [rec(mod, K=K, seed=s) for s in range(4)]
    for dt in (torch.float32, torch.bfloat16):
        oc, nc = old_c.to(dt), new_c.to(dt)
        b, ob = old_batch(rs)
        ob = {k: (v.to(dt) if k in ("cond_hidden", "dino_hidden") else v) for k, v in ob.items()}
        for p in ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)):
            torch.manual_seed(123)
            with torch.no_grad():
                c_old, m_old, _, _ = build_unified_cond(oc, ob["cond_hidden"], ob["cond_key_mask"], oc.cond_seg_embed, oc.cond_patch_pos,
                                                        ob.get("qwen_img_rc"), mask_drop_prob=p[0], dino_hidden=ob.get("dino_hidden"),
                                                        dino_key_mask=ob.get("dino_key_mask"), dino_drop_prob=p[1], qwen_drop_prob=p[2],
                                                        dino_view_ids=ob.get("dino_view_ids"), qwen_view_ids=ob.get("qwen_view_ids"),
                                                        dino_view_embed=views.table if mod != "t" else None, cond_max_length=10240)
            torch.manual_seed(123)
            d = A.draw_drops(4, has_dino=mod != "t", p_cfg=p[0], p_dino=p[1], p_qwen=p[2], device="cpu", dtype=dt)
            with torch.no_grad():
                c_new, m_new = A.assemble(nc, views, b, d, dt)
            check(f"assemble {mod} {str(dt)[6:]} drops={p[0]}", eq(c_old, c_new) and eq(m_old, m_new))
    oc, nc = old_c.float(), new_c.float()
    r0 = {"cond_hidden": rs[0]["qwen"], "cond_keep_mask": rs[0]["qwen_keep"]}
    if mod != "t":
        r0.update(dino_hidden=rs[0]["dino"], dino_keep_mask=rs[0]["dino_keep"], dino_view_ids=rs[0]["dino_views"], qwen_img_rc=rs[0]["qwen_rc"])
    if mod == "im":
        r0["qwen_view_ids"] = rs[0]["qwen_views"]
    co, uo = EC.cond_uncond(oc, r0, dino_view_embed=views.table if mod != "t" else None, device="cpu")
    cn, un = A.cond_uncond(nc, views, rs[0], device="cpu")
    check(f"cond_uncond {mod}", eq(co, cn) and eq(uo, un), f"T={cn.shape[1]}")

print(f"\n{len(FAIL)} failure(s)" + (": " + ", ".join(FAIL) if FAIL else ""))
sys.exit(1 if FAIL else 0)
