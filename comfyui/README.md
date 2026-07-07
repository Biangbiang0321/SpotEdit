# SpotEdit ComfyUI node (experimental)

A single all-in-one node that brings SpotEdit's region-controlled editing to ComfyUI
on the Qwen-Image-Edit family (base / 2509 / 2511), including Lightning few-step LoRAs
and Comfy-style scaled-fp8 checkpoints.

> Status: experimental, diffusers-backed (route 1). The node owns its own pipeline, so it
> does not yet compose with ComfyUI's native model loaders / samplers. A native MODEL-patch
> integration is planned.

> Architecture: everything ComfyUI-specific lives in this folder. The extra sampling features
> used by the nodes (`compute_mode`, `full_last_steps`, manual mask, judge preview) live in
> self-contained samplers here — `spot_qwen_edit.py` / `spot_qwen_edit_plus.py` — which reuse
> the unchanged judge/attention helpers from the backbone packages. The backbone files
> (`Qwen_image_edit*/`, `FLUX*/`) are kept identical to `main`.

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

## Interactive region control (Judge Preview)

`SpotEdit Judge Preview` lets you see and override the judge's decision *before* the real run:

1. Feed your image + prompt into **SpotEdit Judge Preview**. It runs only the first
   `judge_step` steps plus one judge pass (a couple of seconds with Lightning) and returns:
   - `x0_preview` — the model's decoded x0 draft at that step,
   - `overlay` — the same draft with the proposed regenerate-region tinted blue,
   - `regen_mask` — the judge's mask as a MASK.
2. Decide the region yourself — two ways:
   - **SpotEdit Grid Mask** (recommended): feed it the `x0_preview` draft and (optionally) the
     `regen_mask` as `init_mask`. It draws the draft under a clickable 16×16 token grid seeded
     with the judge's suggestion; **click / drag cells** to toggle regenerate (red) vs keep, and
     use `reset to judge` / `clear grid` / `invert grid`. Its `mask` output is the selection.
   - Or paint a mask on the preview (Copy → Clipspace → paste into a `LoadImage` node → *Open in
     MaskEditor*), or edit `regen_mask` with the standard mask nodes (`GrowMask`, `MaskComposite`,
     `InvertMask`, ...). **White = regenerate, black = keep.**
3. Connect your mask to `SpotEditQwenEdit.manual_mask` and pick a `mask_policy`:
   - `replace` — your mask is the final decision (the judge is overridden),
   - `union` — regenerate wherever *either* you or the judge says so,
   - `intersect` — regenerate only where *both* agree.

Keep `seed`/`steps`/model settings identical between the preview and the final run so the
draft you annotated matches the trajectory of the real generation. Masks are consumed on the
16×16-pixel token grid (a token is regenerated if any of its pixels are painted).

## Example workflows

Ready-to-load graphs are in [`workflows/`](workflows/) — drag the `.json` onto the ComfyUI canvas:
- `spotedit_qwen_edit.json` — one-shot edit + a red mask-overlay preview.
- `spotedit_interactive.json` — Judge Preview → **Grid Mask** (click the region) → main edit,
  with the mask overlaid on both the draft and the final result.

## Notes

- Judge schedule is scaled automatically from the step count (`initial_steps=1` and no resets
  at ≤6 steps, `[3,5]` at ≤12, proportional otherwise).
- The input is letterboxed to 1024×1024 by default (repo guideline; disable with `letterbox=false`).
- VRAM: the pipeline is bf16 (~40 GB transformer + text encoder). Use `cpu_offload=true` on
  smaller cards; expect slower runs.
- Only the Qwen family is wired up for now; FLUX-Kontext / FLUX.2 [klein] can follow the same
  pattern once requested.
