import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F, numpy as np
from PIL import Image, ImageDraw
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from Qwen_image_edit_plus import generate, SpotEditConfig

REPO = os.environ.get("QWEN_REPO", "Qwen/Qwen-Image-Edit-2509")
TAG = REPO.rsplit("-", 1)[-1]
PROMPT = os.environ.get("PROMPT", "add a blue knitted scarf around the dog's neck")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "42,123,0").split(",")]
STEPS = 40
R8 = [8, 16, 24, 32]
AMP = 8                       # heatmap amplification
ALPHA = 0.30                  # mask overlay opacity (high-transparency wash)

print(f"[load] {REPO}  seeds={SEEDS}", flush=True)
pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
img = load_image("asset/dog.jpg").resize((1024, 1024))

def base_gen(seed):
    torch.manual_seed(seed)
    return pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0).images[0]

def spot_gen(seed):
    aux = {}
    cfg = SpotEditConfig(initial_steps=4, reuse_mode="velocity", threshold=0.15,
                         reset_steps=R8, dilation_radius=2)
    torch.manual_seed(seed)
    out = generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg, aux=aux).images[0]
    return out, aux

def mask_to_img(aux, H, W):
    m = aux["reuse_mask"].reshape(aux["H_lat"], aux["W_lat"]).float()  # 1 = non-edited / reused
    return F.interpolate(m[None, None], size=(H, W), mode="nearest")[0, 0].numpy()

def overlay(out_img, M, color=(0, 220, 0)):
    # tint the NON-edited (reused) region with a high-transparency colour wash;
    # the edited region (scarf) stays true-colour so it visually pops.
    a = np.asarray(out_img).astype(np.float32)
    tint = np.array(color, np.float32)[None, None, :]
    w = (M[..., None] > 0.5).astype(np.float32) * ALPHA
    return Image.fromarray((a * (1 - w) + tint * w).clip(0, 255).astype(np.uint8))

def change_vs_orig(out_img, orig_arr, M):
    d = np.abs(np.asarray(out_img).astype(np.float32) - orig_arr).mean(2)  # per-pixel change vs ORIGINAL input
    ne = M > 0.5
    mean_ne = float(d[ne].mean())                 # mean change inside non-edited region (ideal -> 0)
    mean_ed = float(d[~ne].mean())                # mean change inside edited region (the scarf)
    hm = Image.fromarray(np.clip(d * AMP, 0, 255).astype(np.uint8)).convert("RGB")
    return mean_ne, mean_ed, hm

rows = []
for sd in SEEDS:
    b = base_gen(sd)
    s, aux = spot_gen(sd)
    H, W = s.size[1], s.size[0]
    orig_arr = np.asarray(img.resize((W, H))).astype(np.float32)
    M = mask_to_img(aux, H, W)
    frac = float((M > 0.5).mean())                # fraction of image marked non-edited
    s_ne, s_ed, s_hm = change_vs_orig(s, orig_arr, M)
    b_ne, b_ed, b_hm = change_vs_orig(b, orig_arr, M)
    ov = overlay(s, M)
    print(f"[seed {sd}] non-edit frac={frac:.2f} | SPOT non-edit Δ={s_ne:.2f} (scarf Δ={s_ed:.1f}) | "
          f"BASE non-edit Δ={b_ne:.2f} (scarf Δ={b_ed:.1f})", flush=True)
    rows.append((sd, img.resize((W, H)), s, ov, s_hm, b_hm, s_ne, b_ne, frac))

# montage: orig | spot | spot+non-edit mask | spot Δvs orig x8 | base Δvs orig x8
th, gap, lh = 300, 6, 26
cols = 5
W0 = th*cols + gap*(cols+1)
H0 = (th+lh)*len(rows) + gap*(len(rows)+1)
c = Image.new("RGB", (W0, H0), (20, 20, 20)); dr = ImageDraw.Draw(c)
for r, (sd, orig, s, ov, s_hm, b_hm, s_ne, b_ne, frac) in enumerate(rows):
    y = gap + r*(th+lh+gap)
    panels = [
        (f"s{sd} ORIGINAL", orig),
        ("spot (velocity)", s),
        (f"non-edit mask {frac:.0%} (green)", ov),
        (f"spot Δ vs orig x8  ne={s_ne:.2f}", s_hm),
        (f"base Δ vs orig x8  ne={b_ne:.2f}", b_hm),
    ]
    for cc, (lab, im) in enumerate(panels):
        x = gap + cc*(th+gap)
        dr.text((x+3, y+2), lab, fill=(255, 255, 255))
        dr.text((x+3, y+13), "", fill=(255, 255, 255))
        c.paste(im.resize((th, th)), (x, y+lh))
out = f"nonedit_viz_{TAG}.png"
c.save(out)
print(f"[done {TAG}] saved {out} {c.size}", flush=True)
