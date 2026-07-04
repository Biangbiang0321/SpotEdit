"""Load a Comfy-style scaled-fp8 Qwen-Image-Edit checkpoint into a diffusers transformer.

Two quantization layouts are supported (both store per-tensor scales next to
float8_e4m3fn weights, with diffusers-compatible key names):
  - lightx2v 2511 style:  '<name>_scale'
  - comfy scaled_fp8 style: '<module>.scale_weight' plus a 'scaled_fp8' marker tensor

The weights are dequantized to bf16 (w = w_fp8 * scale), so this trades the fp8
memory saving for compatibility -- numerically it matches running the quantized
checkpoint with high-precision compute.
"""
import torch
from safetensors.torch import load_file
from diffusers import QwenImageTransformer2DModel


def load_fp8_transformer(path: str, model_id: str) -> QwenImageTransformer2DModel:
    sd = load_file(str(path))
    new_sd = {}
    for k, v in sd.items():
        if k.endswith('_scale') or k.endswith('.scale_weight') or k == 'scaled_fp8':
            continue
        if v.dtype == torch.float8_e4m3fn:
            sk = k + '_scale'
            if sk not in sd:
                sk = k.rsplit('.', 1)[0] + '.scale_weight'
            scale = sd[sk].to(torch.float32)
            if scale.ndim == 1 and v.ndim == 2 and scale.shape[0] == v.shape[0]:
                scale = scale[:, None]
            v = (v.to(torch.float32) * scale).to(torch.bfloat16)
        else:
            v = v.to(torch.bfloat16)
        new_sd[k] = v
    cfg = QwenImageTransformer2DModel.load_config(model_id, subfolder='transformer')
    tf = QwenImageTransformer2DModel.from_config(cfg).to(torch.bfloat16)
    missing, unexpected = tf.load_state_dict(new_sd, strict=False)
    if missing:
        raise RuntimeError(f'fp8 checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}')
    return tf
