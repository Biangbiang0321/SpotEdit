"""SpotEdit ComfyUI nodes (diffusers-backed, Qwen-Image-Edit family).

Route-1 integration: the node owns a diffusers pipeline internally, so it does not
compose with ComfyUI's native model loaders yet -- but it brings the full SpotEdit
feature set (judge + velocity write-back, sliced / hybrid / full compute) to ComfyUI
with a single node. Native (MODEL-patch) integration is planned.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_PIPE_CACHE = {}

MODELS = ["Qwen/Qwen-Image-Edit", "Qwen/Qwen-Image-Edit-2509", "Qwen/Qwen-Image-Edit-2511"]

# Recommended scheduler for Lightning few-step LoRAs (shift=3, from the Lightning repo)
LIGHTNING_SCHEDULER = {
    "base_image_seq_len": 256, "base_shift": math.log(3), "invert_sigmas": False,
    "max_image_seq_len": 8192, "max_shift": math.log(3), "num_train_timesteps": 1000,
    "shift": 1.0, "shift_terminal": None, "stochastic_sampling": False,
    "time_shift_type": "exponential", "use_beta_sigmas": False,
    "use_dynamic_shifting": True, "use_exponential_sigmas": False, "use_karras_sigmas": False,
}


def _letterbox(pil: Image.Image, size: int = 1024) -> Image.Image:
    """Aspect-preserving letterbox onto a black square canvas (repo guideline)."""
    w, h = pil.size
    s = size / max(w, h)
    nw, nh = round(w * s), round(h * s)
    canvas = Image.new('RGB', (size, size), (0, 0, 0))
    canvas.paste(pil.resize((nw, nh)), ((size - nw) // 2, (size - nh) // 2))
    return canvas


def _auto_schedule(steps: int) -> dict:
    """Scale the 50-step default judge schedule down to few-step models."""
    if steps <= 6:
        return dict(initial_steps=1, reset_steps=[])
    if steps <= 12:
        return dict(initial_steps=1, reset_steps=[3, 5])
    return dict(initial_steps=max(1, round(4 * steps / 50)),
                reset_steps=sorted({round(x * steps / 50) for x in (13, 22, 31)}))


def _get_pipeline(model: str, fp8_checkpoint: str, lora_path: str,
                  lightning_scheduler: bool, cpu_offload: bool):
    from diffusers import FlowMatchEulerDiscreteScheduler

    key = (model, fp8_checkpoint, lora_path, lightning_scheduler, cpu_offload)
    if key in _PIPE_CACHE:
        return _PIPE_CACHE[key]
    _PIPE_CACHE.clear()  # keep at most one pipeline resident
    torch.cuda.empty_cache()

    kwargs = {"torch_dtype": torch.bfloat16}
    if fp8_checkpoint:
        from .fp8_loader import load_fp8_transformer
        kwargs["transformer"] = load_fp8_transformer(fp8_checkpoint, model)
    if lightning_scheduler:
        kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(LIGHTNING_SCHEDULER)

    if model == "Qwen/Qwen-Image-Edit":
        from diffusers import QwenImageEditPipeline
        pipe = QwenImageEditPipeline.from_pretrained(model, **kwargs)
    else:
        from diffusers import QwenImageEditPlusPipeline
        pipe = QwenImageEditPlusPipeline.from_pretrained(model, **kwargs)

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to('cuda')
    if lora_path:
        p = Path(lora_path)
        pipe.load_lora_weights(str(p.parent), weight_name=p.name)

    _PIPE_CACHE[key] = pipe
    return pipe


class SpotEditQwenEdit:
    """Instruction-based editing with SpotEdit region control on Qwen-Image-Edit / 2509 / 2511."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "add a blue scarf"}),
                "model": (MODELS, {"default": "Qwen/Qwen-Image-Edit-2511"}),
                "mode": (["quality (full)", "balanced (hybrid)", "speed (sliced)"],
                         {"default": "balanced (hybrid)"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 125, "min": 0, "max": 2**31 - 1}),
                "lightning": ("BOOLEAN", {"default": True,
                                          "tooltip": "use the Lightning shift=3 scheduler and true_cfg_scale=1.0"}),
            },
            "optional": {
                "lora_path": ("STRING", {"default": "", "tooltip":
                              "path to a Lightning LoRA .safetensors (leave empty for fp8-merged checkpoints)"}),
                "fp8_checkpoint": ("STRING", {"default": "", "tooltip":
                                   "path to a Comfy scaled-fp8 single-file checkpoint; dequantized into the transformer"}),
                "true_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "letterbox": ("BOOLEAN", {"default": True}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "regen_mask")
    FUNCTION = "edit"
    CATEGORY = "SpotEdit"

    def edit(self, image, prompt, model, mode, steps, seed, lightning,
             lora_path="", fp8_checkpoint="", true_cfg_scale=1.0,
             letterbox=True, cpu_offload=False):
        # ComfyUI IMAGE: [B, H, W, C] float 0..1
        arr = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if letterbox:
            pil = _letterbox(pil, 1024)

        pipe = _get_pipeline(model, fp8_checkpoint, lora_path, lightning, cpu_offload)

        if model == "Qwen/Qwen-Image-Edit":
            from Qwen_image_edit import generate, SpotEditConfig
            img_arg = pil
        else:
            from Qwen_image_edit_plus import generate, SpotEditConfig
            img_arg = [pil]

        cfg_kw = _auto_schedule(steps)
        if mode.startswith("quality"):
            cfg_kw["compute_mode"] = "full"
        elif mode.startswith("balanced"):
            cfg_kw["compute_mode"] = "sliced"
            cfg_kw["full_last_steps"] = max(1, steps // 4)
        else:
            cfg_kw["compute_mode"] = "sliced"

        aux = {}
        res = generate(pipe, image=img_arg, prompt=prompt, config=SpotEditConfig(**cfg_kw),
                       num_inference_steps=steps, true_cfg_scale=true_cfg_scale,
                       generator=torch.manual_seed(seed), aux=aux)
        out = res.images[0]

        out_t = torch.from_numpy(np.asarray(out, np.float32) / 255.0)[None]  # [1,H,W,C]
        m2 = aux["reuse_mask"].numpy().reshape(aux["H_lat"], aux["W_lat"])
        regen = np.kron(~m2, np.ones((16, 16), dtype=np.float32))            # 1 = regenerated
        mask_t = torch.from_numpy(regen)[None]                               # [1,H,W]
        return (out_t, mask_t)


NODE_CLASS_MAPPINGS = {"SpotEditQwenEdit": SpotEditQwenEdit}
NODE_DISPLAY_NAME_MAPPINGS = {"SpotEditQwenEdit": "SpotEdit Qwen Image Edit"}
