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


def _comfy_mask_to_reuse(mask: torch.Tensor) -> torch.Tensor:
    """ComfyUI MASK [B,H,W] (1 = painted = REGENERATE) -> flat latent-grid reuse mask.
    A 16x16-pixel token is marked regenerate if any of its pixels are painted."""
    m = mask[0].float()
    if m.shape != (1024, 1024):
        m = torch.nn.functional.interpolate(m[None, None], size=(1024, 1024), mode='nearest')[0, 0]
    regen = torch.nn.functional.max_pool2d(m[None, None], 16)[0, 0] > 0.5   # [64, 64]
    return (~regen).reshape(-1).cpu()


def _blue_overlay(img: Image.Image, reuse_flat: torch.Tensor, h_lat: int, w_lat: int) -> Image.Image:
    arr = np.asarray(img, np.float32)
    regen = np.kron(~reuse_flat.numpy().reshape(h_lat, w_lat), np.ones((16, 16), dtype=bool))
    blue = np.array([60, 110, 255], np.float32)
    arr[regen] = 0.55 * arr[regen] + 0.45 * blue
    return Image.fromarray(arr.astype(np.uint8))


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
                "manual_mask": ("MASK", {"tooltip":
                                "painted regions (white) are REGENERATED, the rest is kept. "
                                "Paint on the Judge Preview output so it lines up with the token grid."}),
                "mask_policy": (["replace", "union", "intersect"], {"default": "replace", "tooltip":
                                "how the manual mask meets the judge: replace = use it as-is; union = "
                                "regenerate where either says so; intersect = only where both agree"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "regen_mask")
    FUNCTION = "edit"
    CATEGORY = "SpotEdit"

    def edit(self, image, prompt, model, mode, steps, seed, lightning,
             lora_path="", fp8_checkpoint="", true_cfg_scale=1.0,
             letterbox=True, cpu_offload=False, manual_mask=None, mask_policy="replace"):
        # ComfyUI IMAGE: [B, H, W, C] float 0..1
        arr = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if letterbox:
            pil = _letterbox(pil, 1024)

        pipe = _get_pipeline(model, fp8_checkpoint, lora_path, lightning, cpu_offload)

        if model == "Qwen/Qwen-Image-Edit":
            from .spot_qwen_edit import generate, SpotEditConfig
            img_arg = pil
        else:
            from .spot_qwen_edit_plus import generate, SpotEditConfig
            img_arg = [pil]

        cfg_kw = _auto_schedule(steps)
        if mode.startswith("quality"):
            cfg_kw["compute_mode"] = "full"
        elif mode.startswith("balanced"):
            cfg_kw["compute_mode"] = "sliced"
            cfg_kw["full_last_steps"] = max(1, steps // 4)
        else:
            cfg_kw["compute_mode"] = "sliced"

        if manual_mask is not None:
            cfg_kw["manual_reuse_mask"] = _comfy_mask_to_reuse(manual_mask)
            # the UI speaks in regenerate-regions; the library combines REUSE masks,
            # so union/intersect swap when translated
            cfg_kw["manual_mask_policy"] = {"replace": "replace",
                                            "union": "intersect",
                                            "intersect": "union"}[mask_policy]

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


class SpotEditJudgePreview:
    """Run only the first steps + judge, then return the decoded x0 draft, a blue
    regenerate-region overlay, and the judge's mask — so you can inspect and repaint
    the region before the real run (feed your mask into SpotEditQwenEdit.manual_mask)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "add a blue scarf"}),
                "model": (MODELS, {"default": "Qwen/Qwen-Image-Edit-2511"}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100,
                                  "tooltip": "total steps of the FINAL run (keeps the noise schedule identical)"}),
                "seed": ("INT", {"default": 125, "min": 0, "max": 2**31 - 1}),
                "lightning": ("BOOLEAN", {"default": True}),
                "judge_step": ("INT", {"default": 1, "min": 1, "max": 4,
                                       "tooltip": "which step the judge fires at (1 = after the first step; "
                                                  "later judges see a cleaner draft but cost more)"}),
            },
            "optional": {
                "lora_path": ("STRING", {"default": ""}),
                "fp8_checkpoint": ("STRING", {"default": ""}),
                "true_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "letterbox": ("BOOLEAN", {"default": True}),
                "cpu_offload": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("x0_preview", "overlay", "regen_mask")
    FUNCTION = "preview"
    CATEGORY = "SpotEdit"

    def preview(self, image, prompt, model, steps, seed, lightning, judge_step=1,
                lora_path="", fp8_checkpoint="", true_cfg_scale=1.0,
                letterbox=True, cpu_offload=False):
        arr = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        if letterbox:
            pil = _letterbox(pil, 1024)

        pipe = _get_pipeline(model, fp8_checkpoint, lora_path, lightning, cpu_offload)

        if model == "Qwen/Qwen-Image-Edit":
            from .spot_qwen_edit import generate, SpotEditConfig
            img_arg = pil
        else:
            from .spot_qwen_edit_plus import generate, SpotEditConfig
            img_arg = [pil]

        aux = {}
        res = generate(pipe, image=img_arg, prompt=prompt,
                       config=SpotEditConfig(initial_steps=judge_step, reset_steps=[],
                                             preview_after_judge=True),
                       num_inference_steps=steps, true_cfg_scale=true_cfg_scale,
                       generator=torch.manual_seed(seed), aux=aux)
        draft = res.images[0]

        reuse = aux["reuse_mask"]
        h_lat, w_lat = aux["H_lat"], aux["W_lat"]
        overlay = _blue_overlay(draft, reuse, h_lat, w_lat)

        draft_t = torch.from_numpy(np.asarray(draft, np.float32) / 255.0)[None]
        overlay_t = torch.from_numpy(np.asarray(overlay, np.float32) / 255.0)[None]
        regen = np.kron(~reuse.numpy().reshape(h_lat, w_lat), np.ones((16, 16), dtype=np.float32))
        return (draft_t, overlay_t, torch.from_numpy(regen)[None])


class SpotEditGridMask:
    """Click a token grid over the draft to pick which 16x16 blocks regenerate.

    Outputs a MASK (white = selected = REGENERATE) to feed SpotEditQwenEdit.manual_mask
    (mask_policy=replace). The clickable grid lives in the JS widget (web/spotedit_gridmask.js);
    this node parses the widget's `cells` bit-string and rasterises it to a full-res mask.
    An optional `init_mask` (e.g. the Judge Preview regen_mask) seeds the grid on first run."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "the draft to click on (e.g. Judge Preview x0/overlay)"}),
                "cols": ("INT", {"default": 64, "min": 1, "max": 256}),
                "rows": ("INT", {"default": 64, "min": 1, "max": 256}),
                "cells": ("STRING", {"default": "", "tooltip": "driven by the grid widget; row-major 0/1"}),
            },
            "optional": {
                "init_mask": ("MASK", {"tooltip": "seed the grid from this mask on the first run"}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "build"
    CATEGORY = "SpotEdit"
    OUTPUT_NODE = True

    def build(self, image, cols, rows, cells, init_mask=None):
        import os
        import uuid
        import folder_paths

        _, H, W, _ = image.shape

        # the judge suggestion, always recomputed from init_mask (independent of the
        # user's current selection) so the JS widget can "reset to judge" any time.
        judge_cells = ""
        if init_mask is not None:
            m = init_mask
            if m.dim() == 2:
                m = m[None]
            pooled = torch.nn.functional.adaptive_max_pool2d(m[:1].float().unsqueeze(1), (rows, cols))
            jg = pooled[0, 0].cpu().numpy() > 0.5
            judge_cells = "".join("1" if v else "0" for v in jg.reshape(-1))

        grid = None
        if cells and len(cells) == rows * cols:
            grid = (np.frombuffer(cells.encode("ascii"), dtype=np.uint8) == ord("1")).reshape(rows, cols)
        if grid is None:
            # no explicit click selection yet -> default to the judge suggestion
            if judge_cells:
                grid = (np.frombuffer(judge_cells.encode("ascii"), dtype=np.uint8) == ord("1")).reshape(rows, cols)
            else:
                grid = np.zeros((rows, cols), dtype=bool)

        cell_h, cell_w = max(1, H // rows), max(1, W // cols)
        up = np.kron(grid.astype(np.float32), np.ones((cell_h, cell_w), np.float32))
        mask = np.zeros((H, W), np.float32)
        hh, ww = min(H, up.shape[0]), min(W, up.shape[1])
        mask[:hh, :ww] = up[:hh, :ww]
        mask_t = torch.from_numpy(mask)[None]

        # save the background so the JS grid widget can render the draft under the cells
        arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        fname = f"spotedit_grid_{uuid.uuid4().hex[:12]}.png"
        Image.fromarray(arr).save(os.path.join(tmp, fname))
        ui = {
            "images": [{"filename": fname, "subfolder": "", "type": "temp"}],
            # seed_cells: adopted on the first run when the grid is empty.
            # judge_cells: always the current judge suggestion, for the "reset to judge" button.
            "spotedit_grid": [{"rows": rows, "cols": cols,
                               "seed_cells": judge_cells, "judge_cells": judge_cells}],
        }
        return {"ui": ui, "result": (mask_t,)}


NODE_CLASS_MAPPINGS = {
    "SpotEditQwenEdit": SpotEditQwenEdit,
    "SpotEditJudgePreview": SpotEditJudgePreview,
    "SpotEditGridMask": SpotEditGridMask,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SpotEditQwenEdit": "SpotEdit Qwen Image Edit",
    "SpotEditJudgePreview": "SpotEdit Judge Preview",
    "SpotEditGridMask": "SpotEdit Grid Mask",
}
