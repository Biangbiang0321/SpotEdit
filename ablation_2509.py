import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from PIL import Image, ImageDraw
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from Qwen_image_edit_plus import generate, SpotEditConfig

REPO = os.environ.get("QWEN_REPO", "Qwen/Qwen-Image-Edit-2509")
PROMPT = os.environ.get("PROMPT", "add a blue knitted scarf around the dog's neck")
STEPS = 40
R8 = [8, 16, 24, 32]

print(f"[load] {REPO}", flush=True)
pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
img = load_image("asset/dog.jpg").resize((1024, 1024))

# base: single-forward (no CFG) to match SpotEdit compute
torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
base = pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0).images[0]
torch.cuda.synchronize(); tb = time.time() - t0
print(f"[base] {tb:.1f}s", flush=True)

def C(**kw):
    return SpotEditConfig(initial_steps=4, **kw)

configs = [
    ("overwrite",     C(reuse_mode="overwrite", threshold=0.15, reset_steps=R8, dilation_radius=2)),
    ("feather",       C(reuse_mode="feather",   threshold=0.15, reset_steps=R8, dilation_radius=2)),
    ("velocity",      C(reuse_mode="velocity",  threshold=0.15, reset_steps=R8, dilation_radius=2)),
    ("vel thr=0.10",  C(reuse_mode="velocity",  threshold=0.10, reset_steps=R8, dilation_radius=2)),
    ("vel thr=0.25",  C(reuse_mode="velocity",  threshold=0.25, reset_steps=R8, dilation_radius=2)),
    ("vel reset=4",   C(reuse_mode="velocity",  threshold=0.15, reset_steps=list(range(8, 40, 4)), dilation_radius=2)),
    ("vel reset=16",  C(reuse_mode="velocity",  threshold=0.15, reset_steps=[16, 32], dilation_radius=2)),
    ("vel dil=0",     C(reuse_mode="velocity",  threshold=0.15, reset_steps=R8, dilation_radius=0)),
    ("vel dil=4",     C(reuse_mode="velocity",  threshold=0.15, reset_steps=R8, dilation_radius=4)),
]

panels = [("base (ref)", base)]
for label, cfg in configs:
    torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
    out = generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg).images[0]
    torch.cuda.synchronize(); ts = time.time() - t0
    d = float(np.abs(np.asarray(out).astype(float) - np.asarray(base).astype(float)).mean())
    print(f"[{label:14s}] {ts:5.1f}s  speedup {tb/ts:.2f}x  mean|Δ|={d:.1f}", flush=True)
    panels.append((f"{label}|{tb/ts:.2f}x d={d:.0f}", out))

# montage
th, gap, lh, cols = 300, 6, 30, 5
rows = math.ceil(len(panels) / cols)
W = th * cols + gap * (cols + 1)
H = (th + lh + gap) * rows + gap
c = Image.new("RGB", (W, H), (20, 20, 20)); dr = ImageDraw.Draw(c)
for i, (label, im) in enumerate(panels):
    r, cc = divmod(i, cols); x = gap + cc * (th + gap); y = gap + r * (th + lh + gap)
    parts = label.split("|")
    for j, line in enumerate(parts):
        dr.text((x + 3, y + 2 + j * 14), line, fill=(255, 255, 255))
    c.paste(im.resize((th, th)), (x, y + lh))
c.save("ablation_2509.png")
print(f"[done] saved ablation_2509.png {c.size}", flush=True)
