import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, numpy as np
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image
from Qwen_image_edit_plus import generate, SpotEditConfig

REPO = os.environ.get("QWEN_REPO", "Qwen/Qwen-Image-Edit-2511")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
PROMPT = os.environ.get("PROMPT", "add a red knitted scarf around the dog's neck")
# reset cadence: every RES steps after the initial block (refreshes frozen KV -> less drift).
# pass arg2 to control it; smaller = better quality, less speedup.
RES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
RESET = list(range(8, STEPS, RES))
STRENGTH = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0  # <1.0 = SDEdit init from noised source
EVERY = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False  # recompute reuse mask every step
TAG = f"s{STEPS}r{RES}str{STRENGTH}e{int(EVERY)}"

print(f"[load] {REPO} ...", flush=True)
t0 = time.time()
pipe = QwenImageEditPlusPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to("cuda")
print(f"[load] done in {time.time()-t0:.0f}s", flush=True)

img = load_image("asset/dog.jpg").resize((1024, 1024))

# ---- baseline: stock pipeline, single forward (true_cfg_scale=1.0) to match SpotEdit compute ----
torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
base = pipe(image=img, prompt=PROMPT, num_inference_steps=STEPS, true_cfg_scale=1.0).images[0]
torch.cuda.synchronize(); t_base = time.time() - t0
base.save(f"smoke_base_{TAG}.png")
print(f"[base]    {t_base:.1f}s ({STEPS} steps) -> smoke_base_{TAG}.png", flush=True)

# ---- SpotEdit ----
cfg = SpotEditConfig(threshold=0.15, initial_steps=4, reset_steps=RESET, dilation_radius=2,
                     source_init_strength=STRENGTH, select_every_step=EVERY)
print(f"[spotedit] config: steps={STEPS} reset_steps={RESET} threshold=0.15 "
      f"reuse_mode={cfg.reuse_mode} source_init_strength={STRENGTH} select_every_step={EVERY}", flush=True)
torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
out = generate(pipe, image=img, prompt=PROMPT, num_inference_steps=STEPS, config=cfg).images[0]
torch.cuda.synchronize(); t_spot = time.time() - t0
out.save(f"smoke_spotedit_{TAG}.png")
print(f"[spotedit] {t_spot:.1f}s -> smoke_spotedit_{TAG}.png", flush=True)

a = np.asarray(base).astype(np.float32); b = np.asarray(out).astype(np.float32)
print(f"[result] speedup={t_base/t_spot:.2f}x | mean|Δpx|={np.abs(a-b).mean():.2f}/255 "
      f"| spotedit range=[{b.min():.0f},{b.max():.0f}] finite={np.isfinite(b).all()}", flush=True)
print("[done]", flush=True)
