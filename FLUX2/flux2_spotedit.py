import torch
import numpy as np

from typing import Any, Callable, Dict, List, Optional, Union

import PIL.Image
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.flux2.pipeline_flux2 import (
    Flux2Pipeline,
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
from .flux2_spot_ultis import (
    SpotEditConfig,
    SpotSelect,
    dilate_uncached_mask,
    boundary_aware_smoothing,
    select_reuse_mask,
)


@torch.no_grad()
def generate(
    self: Flux2Pipeline,
    image: Optional[Union[List[PIL.Image.Image], PIL.Image.Image]] = None,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    sigmas: Optional[List[float]] = None,
    guidance_scale: Optional[float] = 4.0,
    num_images_per_prompt: int = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    text_encoder_out_layers: tuple = (10, 20, 30),
    config: SpotEditConfig = SpotEditConfig(),
    aux: Optional[dict] = None,
):
    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt=prompt,
        height=height,
        width=width,
        prompt_embeds=prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = guidance_scale
    self._attention_kwargs = attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    # 3. Prepare text embeddings
    prompt_embeds, text_ids = self.encode_prompt(
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        text_encoder_out_layers=text_encoder_out_layers,
    )

    # 4. Process reference image(s) -- FLUX.2 appends each as extra sequence tokens
    if image is not None and not isinstance(image, list):
        image = [image]

    condition_images = None
    if image is not None:
        for img in image:
            self.image_processor.check_image_input(img)

        condition_images = []
        for img in image:
            image_width, image_height = img.size
            if image_width * image_height > 1024 * 1024:
                img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                image_width, image_height = img.size

            multiple_of = self.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            condition_images.append(img)
            height = height or image_height
            width = width or image_width

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 5. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_ids = self.prepare_latents(
        batch_size=batch_size * num_images_per_prompt,
        num_latents_channels=num_channels_latents,
        height=height,
        width=width,
        dtype=prompt_embeds.dtype,
        device=device,
        generator=generator,
        latents=latents,
    )

    image_latents = None
    image_latent_ids = None
    if condition_images is not None:
        image_latents, image_latent_ids = self.prepare_image_latents(
            images=condition_images,
            batch_size=batch_size * num_images_per_prompt,
            generator=generator,
            device=device,
            dtype=self.vae.dtype,
        )

    # 6. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
    guidance = guidance.expand(latents.shape[0])

    # 7. Install SpotEdit processors on both block types
    text_n = prompt_embeds.shape[1]
    latent_n = latents.shape[1]
    image_n = image_latents.shape[1] if image_latents is not None else 0

    # reuse decision / restore reference = the first (edited) image's tokens
    ref_image_latents = image_latents[:, :latent_n] if image_latents is not None else None

    cache_flags = [
        torch.zeros((n), dtype=torch.bool, device=device)
        for n in [text_n, latent_n, image_n]
    ]
    cache_flags.append(0)

    # remember original processors so we can restore them on exit (otherwise a later plain
    # pipe() call would run with leftover SpotEdit processors + stale cache state).
    _orig_procs = [(name, module.processor) for name, module in self.transformer.named_modules()
                   if isinstance(module, (Flux2Attention, Flux2ParallelSelfAttention))]

    for _, module in self.transformer.named_modules():
        if isinstance(module, Flux2Attention):
            module.set_processor(Flux2SpotAttnProcessor(cache_flags))
        elif isinstance(module, Flux2ParallelSelfAttention):
            module.set_processor(Flux2ParallelSpotAttnProcessor(cache_flags))

    try:
        H_lat = height // self.vae_scale_factor // 2
        W_lat = width // self.vae_scale_factor // 2

        # 8. Denoising loop
        self.scheduler.set_begin_index(0)

        x0_preds = []
        last_noise_pred = None
        cache_final = torch.zeros((latent_n), dtype=torch.bool, device=device)

        total_cached_tokens, total_latent_tokens = 0, 0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if len(x0_preds):
                    # for the initial and reset steps, we do full computation
                    if i in config.reset_steps or i < config.initial_steps:
                        cache_flags[1] = torch.zeros((latent_n), dtype=torch.bool, device=device)
                        cache_flags[2] = torch.zeros((image_n), dtype=torch.bool, device=device)
                    # for spotedit steps, we do selective computation
                    else:
                        # on full-reuse, lower threshold + re-judge so some tokens stay uncached
                        cache_flags[1] = select_reuse_mask(
                            self, x0_preds[-1], ref_image_latents, H_lat, W_lat,
                            threshold=config.threshold, method=config.judge_method,
                            image_size=(height, width), dilation_radius=config.dilation_radius,
                        )

                        if cache_flags[1].any():
                            cache_final = cache_flags[1]
                            cache_flags[2] = torch.ones((image_n), dtype=torch.bool, device=device)
                        else:
                            cache_flags[2] = torch.zeros((image_n), dtype=torch.bool, device=device)

                        cache_flags[-1] = t.item() / 1000
                        total_cached_tokens += cache_flags[1].sum().item()
                        total_latent_tokens += latent_n

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
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self._attention_kwargs,
                    return_dict=False,
                )[0]

                # update the noise prediction only for edited (uncached) tokens
                if cache_flags[1].any():
                    uncached_n = cache_flags[1].logical_not().sum().item()
                    if config.reuse_mode == "velocity" and ref_image_latents is not None:
                        # reused tokens flow straight to the source: v=(x_t-x0_orig)/sigma
                        # => x0_pred = x_t - sigma*v = x0_orig (smooth, no end-of-run seam).
                        sigma = t.item() / 1000
                        noisy_copy = (latents - ref_image_latents) / sigma
                    else:
                        noisy_copy = last_noise_pred.clone()
                    noisy_copy[:, cache_flags[1].logical_not()] = noise_pred[:, :uncached_n]
                    noise_pred = noisy_copy
                else:
                    noise_pred = noise_pred[:, : latents.size(1)]

                last_noise_pred = noise_pred

                # compute the x_0 prediction
                x0_preds.append(latents - t.item() / 1000 * noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        # expose the final reuse mask (True = non-edited / reused token) for visualisation
        if aux is not None:
            aux["reuse_mask"] = cache_final.detach().to("cpu").clone()
            aux["H_lat"] = H_lat
            aux["W_lat"] = W_lat
        # "overwrite": hard latent paste + boundary smoothing. "velocity": reused tokens already
        # flowed to the source in-loop, so keep the generated latents as-is.
        if cache_final.any() and config.reuse_mode == "overwrite":
            latents[:, cache_final] = ref_image_latents[:, cache_final]
            latents = boundary_aware_smoothing(
                latents, ref_image_latents,
                non_edit_mask=cache_final.unsqueeze(0), lambda0=0.8, hw=(H_lat, W_lat),
            )

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents_with_ids(latents, latent_ids)

            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(latents.device, latents.dtype)
            latents = latents * latents_bn_std + latents_bn_mean
            latents = self._unpatchify_latents(latents)

            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Flux2PipelineOutput(images=image)
    finally:
        # restore the original attention processors so the pipe is left unmodified
        for name, proc in _orig_procs:
            self.transformer.get_submodule(name).set_processor(proc)
