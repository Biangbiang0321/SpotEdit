import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from FLUX2 import generate, SpotEditConfig

REPO = os.environ.get("FLUX2_REPO", "black-forest-labs/FLUX.2-klein-4B")
PROMPT = os.environ.get("PROMPT", "add a blue knitted scarf around the dog's neck")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
GUID = float(os.environ.get("GUID", "4.0"))

print(f"[load] {REPO}", flush=True)
t0 = time.time()
pipe = Flux2Pipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
print(f"[load] done in {time.time()-t0:.0f}s", flush=True)

img = load_image("asset/dog.jpg").resize((1024, 1024))

# base (no SpotEdit)
torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
base = pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=GUID).images[0]
torch.cuda.synchronize(); tb = time.time() - t0
base.save("flux2_base.png")
print(f"[base]     {tb:.1f}s ({STEPS} steps) -> flux2_base.png", flush=True)

# SpotEdit
cfg = SpotEditConfig(threshold=0.2, initial_steps=4, reset_steps=list(range(8, STEPS, RES)), dilation_radius=2)
print(f"[spotedit] config: reset_steps={cfg.reset_steps} threshold={cfg.threshold}", flush=True)
torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
out = generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, guidance_scale=GUID, config=cfg).images[0]
torch.cuda.synchronize(); ts = time.time() - t0
out.save("flux2_spotedit.png")
print(f"[spotedit] {ts:.1f}s -> flux2_spotedit.png", flush=True)

a = np.asarray(base).astype(np.float32); b = np.asarray(out).astype(np.float32)
print(f"[result] speedup={tb/ts:.2f}x | mean|Δpx|={np.abs(a-b).mean():.2f}/255 "
      f"| range=[{b.min():.0f},{b.max():.0f}] finite={np.isfinite(b).all()}", flush=True)
print("[done]", flush=True)
