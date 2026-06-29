"""SpotEdit generate() for FLUX.2 [klein] (Flux2KleinPipeline, Qwen3 text encoder).

klein shares the Flux2Transformer2DModel + AutoencoderKLFlux2 with FLUX.2-dev, so it
reuses the same attention processors (flux2SpotAttn) and token-LPIPS (FLUX2LPIPS). The
differences vs dev: Qwen3 encode_prompt, guidance=None in the transformer call, true-CFG
for guidance (skipped here -- SpotEdit runs single-forward), and an extra (h, w) arg on
_unpack_latents_with_ids. Requires diffusers >= 0.37 (has Flux2KleinPipeline).
Reused (non-edited) tokens use the flow-velocity-to-source trick (no end-of-run seam).
"""
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union

import PIL.Image
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline,
    compute_empirical_mu,
    retrieve_timesteps,
)
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2ParallelSelfAttention,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
logger = logging.get_logger(__name__)

from .flux2SpotAttn import Flux2SpotAttnProcessor, Flux2ParallelSpotAttnProcessor
from .flux2_spot_ultis import SpotEditConfig, SpotSelect, dilate_uncached_mask


@torch.no_grad()
def generate(
    self: Flux2KleinPipeline,
    image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 4.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple = (9, 18, 27),
    config: SpotEditConfig = SpotEditConfig(),
):
    self.check_inputs(
        prompt=prompt, height=height, width=width, prompt_embeds=prompt_embeds,
        callback_on_step_end_tensor_inputs=["latents"], guidance_scale=guidance_scale,
    )
    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]
    device = self._execution_device

    # 3. text (Qwen3); SpotEdit runs single-forward, so no CFG/negative pass
    prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt, prompt_embeds=prompt_embeds, device=device,
        num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
    )

    # 4. images
    if image is not None and not isinstance(image, list):
        image = [image]
    condition_images = None
    if image is not None:
        for img in image:
            self.image_processor.check_image_input(img)
        condition_images = []
        for img in image:
            iw, ih = img.size
            if iw * ih > 1024 * 1024:
                img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                iw, ih = img.size
            multiple_of = self.vae_scale_factor * 2
            iw = (iw // multiple_of) * multiple_of
            ih = (ih // multiple_of) * multiple_of
            img = self.image_processor.preprocess(img, height=ih, width=iw, resize_mode="crop")
            condition_images.append(img)
            height = height or ih
            width = width or iw
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 5. latents
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt, num_latents_channels=num_channels_latents,
        height=height, width=width, dtype=prompt_embeds.dtype, device=device,
        generator=generator, latents=latents,
    )
    image_latents = None
    image_latent_ids = None
    if condition_images is not None:
        image_latents, image_latent_ids = self.prepare_image_latents(
            images=condition_images, batch_size=batch_size * num_images_per_prompt,
            generator=generator, device=device, dtype=self.vae.dtype,
        )

    # 6. timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
        sigmas = None
    mu = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # 7. SpotEdit processors
    text_n = prompt_embeds.shape[1]
    latent_n = latents.shape[1]
    image_n = image_latents.shape[1] if image_latents is not None else 0
    ref_image_latents = image_latents[:, :latent_n] if image_latents is not None else None

    cache_flags = [torch.zeros((n), dtype=torch.bool, device=device) for n in [text_n, latent_n, image_n]]
    cache_flags.append(0)
    _orig_procs = [(name, m.processor) for name, m in self.transformer.named_modules()
                   if isinstance(m, (Flux2Attention, Flux2ParallelSelfAttention))]
    for _, m in self.transformer.named_modules():
        if isinstance(m, Flux2Attention):
            m.set_processor(Flux2SpotAttnProcessor(cache_flags))
        elif isinstance(m, Flux2ParallelSelfAttention):
            m.set_processor(Flux2ParallelSpotAttnProcessor(cache_flags))

    try:
        H_lat = height // self.vae_scale_factor // 2
        W_lat = width // self.vae_scale_factor // 2

        # 8. denoising loop (velocity: reused tokens flow straight to the source)
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        x0_preds = []
        last_noise_pred = None
        cache_final = torch.zeros((latent_n), dtype=torch.bool, device=device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                if len(x0_preds):
                    if i in config.reset_steps or i < config.initial_steps:
                        cache_flags[1] = torch.zeros((latent_n), dtype=torch.bool, device=device)
                        cache_flags[2] = torch.zeros((image_n), dtype=torch.bool, device=device)
                    else:
                        cache_flags[1] = SpotSelect(self, x0_preds[-1], ref_image_latents,
                                                    threshold=config.threshold, method=config.judge_method,
                                                    image_size=(height, width))
                        if config.dilation_radius > 0:
                            cache_flags[1] = dilate_uncached_mask(cache_flags[1], H_lat=H_lat, W_lat=W_lat,
                                                                  dilation_radius=config.dilation_radius)
                        if cache_flags[1].any():
                            cache_final = cache_flags[1]
                            cache_flags[2] = torch.ones((image_n), dtype=torch.bool, device=device)
                        else:
                            cache_flags[2] = torch.zeros((image_n), dtype=torch.bool, device=device)
                        cache_flags[-1] = t.item() / 1000

                self._current_timestep = t
                uncached_latents = latents[:, cache_flags[1].logical_not()]
                latent_image_ids = latent_ids
                if image_latents is not None:
                    uncached_image_latents = image_latents[:, cache_flags[2].logical_not()]
                    latent_model_input = torch.cat([uncached_latents, uncached_image_latents], dim=1)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)
                else:
                    latent_model_input = uncached_latents
                latent_model_input = latent_model_input.to(self.transformer.dtype)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input, timestep=timestep / 1000, guidance=None,
                    encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids,
                    joint_attention_kwargs=self._attention_kwargs, return_dict=False,
                )[0]

                if cache_flags[1].any():
                    uncached_n = cache_flags[1].logical_not().sum().item()
                    sigma = t.item() / 1000  # reused tokens flow straight to the source: v=(x_t-x0_orig)/sigma
                    noisy_copy = (latents - ref_image_latents) / sigma
                    noisy_copy[:, cache_flags[1].logical_not()] = noise_pred[:, :uncached_n]
                    noise_pred = noisy_copy
                else:
                    noise_pred = noise_pred[:, : latents.size(1)]
                last_noise_pred = noise_pred

                x0_preds.append(latents - t.item() / 1000 * noise_pred)
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latents(latents)

        if output_type == "latent":
            image = latents
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)
    finally:
        # restore the original attention processors so the pipe is left unmodified
        for name, proc in _orig_procs:
            self.transformer.get_submodule(name).set_processor(proc)
