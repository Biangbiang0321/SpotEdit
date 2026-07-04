# SpotEdit ComfyUI node (experimental)

A single all-in-one node that brings SpotEdit's region-controlled editing to ComfyUI
on the Qwen-Image-Edit family (base / 2509 / 2511), including Lightning few-step LoRAs
and Comfy-style scaled-fp8 checkpoints.

> Status: experimental, diffusers-backed (route 1). The node owns its own pipeline, so it
> does not yet compose with ComfyUI's native model loaders / samplers. A native MODEL-patch
> integration is planned.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Biangbiang0321/SpotEdit.git
pip install -r SpotEdit/requirements.txt   # diffusers, peft, accelerate, ...
```

Restart ComfyUI; the node appears as **SpotEdit Qwen Image Edit** (category `SpotEdit`).

## Node

**Inputs**
- `image` / `prompt` — source image and edit instruction.
- `model` — `Qwen/Qwen-Image-Edit`, `...-2509` or `...-2511` (HF id; downloaded on first use unless cached).
- `mode`
  - `quality (full)` — the transformer processes every token; the judged mask only pins
    non-edited regions to the source (velocity write-back). Best fidelity, ~baseline cost.
  - `balanced (hybrid)` — sliced compute with the last `steps//4` steps run in full mode.
    In our tests this removes sliced-mode boundary artifacts at a fraction of full cost.
  - `speed (sliced)` — classic SpotEdit: the transformer only sees non-reused tokens.
- `steps` / `seed` / `lightning` — with `lightning` on, the shift=3 FlowMatch scheduler is used
  (pair with 4/8-step Lightning weights and `true_cfg_scale=1.0`).
- `lora_path` *(optional)* — a Lightning LoRA `.safetensors` (e.g.
  `Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors`). Not needed for fp8-merged files.
- `fp8_checkpoint` *(optional)* — a Comfy scaled-fp8 single-file checkpoint (e.g.
  `qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_4steps_v1.0.safetensors`). The weights are
  dequantized into the diffusers transformer (bf16 compute), so results match the quantized
  checkpoint while the rest of the pipeline still comes from the HF `model` id.

**Outputs**
- `image` — the edited image.
- `regen_mask` — the judged regenerated-region mask (1 = regenerated), ready for previews
  or downstream compositing.

## Notes

- Judge schedule is scaled automatically from the step count (`initial_steps=1` and no resets
  at ≤6 steps, `[3,5]` at ≤12, proportional otherwise).
- The input is letterboxed to 1024×1024 by default (repo guideline; disable with `letterbox=false`).
- VRAM: the pipeline is bf16 (~40 GB transformer + text encoder). Use `cpu_offload=true` on
  smaller cards; expect slower runs.
- Only the Qwen family is wired up for now; FLUX-Kontext / FLUX.2 [klein] can follow the same
  pattern once requested.
