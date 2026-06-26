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

torch.manual_seed(42)
base = pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0).images[0]
print("[base] done", flush=True)

def gen(mode):
    cfg = SpotEditConfig(initial_steps=4, reuse_mode=mode, threshold=0.15,
                         reset_steps=R8, dilation_radius=2)
    torch.manual_seed(42)
    return generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg).images[0]

ov = gen("overwrite"); print("[overwrite] done", flush=True)
ve = gen("velocity");  print("[velocity] done", flush=True)

base.save("seam_base.png"); ov.save("seam_overwrite.png"); ve.save("seam_velocity.png")

b = np.asarray(base).astype(np.float32)
def diffmap(x, amp=8):
    d = np.abs(np.asarray(x).astype(np.float32) - b).mean(2)
    print(f"   diff max={d.max():.0f} mean={d.mean():.2f} (amp x{amp})", flush=True)
    return Image.fromarray(np.clip(d * amp, 0, 255).astype(np.uint8)).convert("RGB")

print("overwrite vs base:"); dov = diffmap(ov)
print("velocity  vs base:"); dve = diffmap(ve)

# 2-row montage: full images / amplified diff maps
th, gap, lh = 380, 6, 18
labels = [("base", base, None), ("overwrite", ov, dov), ("velocity", ve, dve)]
cols = 3
W = th * cols + gap * (cols + 1)
H = (th + lh) * 2 + gap * 3
c = Image.new("RGB", (W, H), (20, 20, 20)); dr = ImageDraw.Draw(c)
for i, (name, full, dm) in enumerate(labels):
    x = gap + i * (th + gap)
    dr.text((x + 3, 2), name, fill=(255, 255, 255))
    c.paste(full.resize((th, th)), (x, lh))
    y2 = lh + th + gap
    dr.text((x + 3, y2 + 2), "" if dm is None else f"|Δ|x8 vs base", fill=(255, 200, 120))
    if dm is not None:
        c.paste(dm.resize((th, th)), (x, y2 + lh))
c.save("seam_check_2509.png")
print(f"[done] saved seam_check_2509.png {c.size}", flush=True)
