import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from PIL import Image, ImageDraw
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from Qwen_image_edit_plus import generate, SpotEditConfig

REPO = os.environ.get("QWEN_REPO", "Qwen/Qwen-Image-Edit-2509")
TAG = REPO.rsplit("-", 1)[-1]  # 2509 / 2511
PROMPT = os.environ.get("PROMPT", "add a blue knitted scarf around the dog's neck")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,42,123,7").split(",")]
STEPS = 40
R8 = [8, 16, 24, 32]

print(f"[load] {REPO}  seeds={SEEDS}", flush=True)
pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
img = load_image("asset/dog.jpg").resize((1024, 1024))

def base_gen(seed):
    torch.manual_seed(seed)
    return pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0).images[0]

def spot_gen(seed, mode):
    cfg = SpotEditConfig(initial_steps=4, reuse_mode=mode, threshold=0.15,
                         reset_steps=R8, dilation_radius=2)
    torch.manual_seed(seed)
    return generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg).images[0]

def diffmap(x, b, amp=8):
    d = np.abs(np.asarray(x).astype(np.float32) - np.asarray(b).astype(np.float32)).mean(2)
    return float(d.max()), float(d.mean()), Image.fromarray(np.clip(d*amp,0,255).astype(np.uint8)).convert("RGB")

rows = []
for sd in SEEDS:
    t0 = time.time()
    b = base_gen(sd); ov = spot_gen(sd, "overwrite"); ve = spot_gen(sd, "velocity")
    omx, omn, od = diffmap(ov, b); vmx, vmn, vd = diffmap(ve, b)
    print(f"[seed {sd}] overwrite max={omx:.0f}/mean={omn:.2f}  velocity max={vmx:.0f}/mean={vmn:.2f}  ({time.time()-t0:.0f}s)", flush=True)
    rows.append((sd, b, ov, ve, od, vd, omn, vmn))

# montage: rows = seeds, cols = base | overwrite | velocity | overwrite Δ×8 | velocity Δ×8
th, gap, lh = 300, 6, 16
cols = 5
W = th*cols + gap*(cols+1)
H = (th+lh)*len(rows) + gap*(len(rows)+1)
c = Image.new("RGB", (W, H), (20, 20, 20)); dr = ImageDraw.Draw(c)
for r, (sd, b, ov, ve, od, vd, omn, vmn) in enumerate(rows):
    y = gap + r*(th+lh+gap)
    panels = [(f"s{sd} base", b), (f"overwrite |Δ|{omn:.1f}", ov),
              (f"velocity |Δ|{vmn:.1f}", ve), ("ovr Δx8", od), ("vel Δx8", vd)]
    for cc, (lab, im) in enumerate(panels):
        x = gap + cc*(th+gap)
        dr.text((x+3, y+2), lab, fill=(255, 255, 255))
        c.paste(im.resize((th, th)), (x, y+lh))
out = f"seam_seeds_{TAG}.png"
c.save(out)
print(f"[done {TAG}] saved {out} {c.size}", flush=True)
