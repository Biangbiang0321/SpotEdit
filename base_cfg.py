import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from diffusers import QwenImageEditPlusPipeline
from diffusers.utils import load_image

pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16).to("cuda")
img = load_image("asset/dog.jpg").resize((1024, 1024))
PROMPT = "add a red knitted scarf around the dog's neck"

# proper base inference: full true-CFG (recommended for 2511), NO SpotEdit
torch.manual_seed(42); torch.cuda.synchronize(); t0 = time.time()
out = pipe(image=img, prompt=PROMPT, negative_prompt=" ",
           num_inference_steps=40, true_cfg_scale=4.0).images[0]
torch.cuda.synchronize(); dt = time.time() - t0
out.save("smoke_base_cfg.png")
print(f"[base+CFG] {dt:.1f}s (40 steps, true_cfg=4.0) -> smoke_base_cfg.png", flush=True)
print("[done]", flush=True)
