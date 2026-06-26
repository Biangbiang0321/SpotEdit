import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from PIL import Image, ImageDraw
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from Qwen_image_edit_plus import generate, SpotEditConfig

REPO = os.environ.get("QWEN_REPO", "Qwen/Qwen-Image-Edit-2509")
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,7,42,123").split(",")]
PROMPT = "add a red knitted scarf around the dog's neck"
STEPS = 40

print(f"[load] {REPO}", flush=True)
pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
img = load_image("asset/dog.jpg").resize((1024, 1024))
cfg = SpotEditConfig(threshold=0.15, initial_steps=4, reset_steps=[8, 16, 24, 32], dilation_radius=2)

rows = []
for s in SEEDS:
    torch.manual_seed(s); torch.cuda.synchronize(); t0 = time.time()
    base = pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0).images[0]
    torch.cuda.synchronize(); tb = time.time() - t0
    torch.manual_seed(s); torch.cuda.synchronize(); t0 = time.time()
    spot = generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg).images[0]
    torch.cuda.synchronize(); ts = time.time() - t0
    d = float(np.abs(np.asarray(spot).astype(float) - np.asarray(base).astype(float)).mean())
    print(f"[seed {s:4d}] base {tb:.1f}s  spot {ts:.1f}s  speedup {tb/ts:.2f}x  mean|Δ|={d:.2f}/255", flush=True)
    rows.append((s, base, spot, tb / ts))

# montage: one row per seed, columns [Base | SpotEdit]
th, gap, lh = 340, 6, 22
W = th * 2 + gap * 3
H = (th + lh + gap) * len(rows) + gap
c = Image.new("RGB", (W, H), (25, 25, 25)); draw = ImageDraw.Draw(c)
for r, (s, base, spot, sp) in enumerate(rows):
    y = gap + r * (th + lh + gap)
    draw.text((gap + 2, y + 4), f"seed {s} — Base", fill=(255, 255, 255))
    draw.text((gap * 2 + th + 2, y + 4), f"seed {s} — SpotEdit ({sp:.2f}x)", fill=(255, 255, 255))
    c.paste(base.resize((th, th)), (gap, y + lh))
    c.paste(spot.resize((th, th)), (gap * 2 + th, y + lh))
c.save("seeds_2509.png")
print(f"[done] saved seeds_2509.png {c.size}", flush=True)
