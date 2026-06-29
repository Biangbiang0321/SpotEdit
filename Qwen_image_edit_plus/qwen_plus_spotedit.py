import os
import time
import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    QwenImageEditPlusPipeline,
    calculate_shift,
    retrieve_timesteps,
    calculate_dimensions,
)
from diffusers.models.attention_processor import Attention

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
logger = logging.get_logger(__name__)

from .qwenSpotAttn import QwenSpotEditAttnProcessor
from .qwen_spot_ultis import (
    Spotselect, SpotEditConfig, dilate_uncached_mask, boundary_aware_smoothing, feather_composite,
)

# Qwen-Image-Edit-2509 encodes each reference image twice: a small one for the
# text encoder (semantic conditioning) and a 1MP one for the VAE (pixel conditioning).
CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024


@torch.no_grad()
def generate(
        self: QwenImageEditPlusPipeline,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        config: SpotEditConfig = SpotEditConfig(),
        aux: Optional[dict] = None,
):
    image_size = image[0].size if isinstance(image, list) else image.size
    calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    # 1. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
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

    # 3. Preprocess image(s) -- Qwen-Image-Edit-Plus supports multiple reference images.
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        if not isinstance(image, list):
            image = [image]
        condition_images = []
        vae_image_sizes = []
        vae_images = []
        for img in image:
            image_width, image_height = img.size
            condition_width, condition_height = calculate_dimensions(
                CONDITION_IMAGE_SIZE, image_width / image_height
            )
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            condition_images.append(self.image_processor.resize(img, condition_height, condition_width))
            vae_images.append(self.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    # NOTE: this accelerated SpotEdit path runs a SINGLE transformer forward per step
    # (the speedup methodology assumes no classifier-free guidance -- see CHANGES.md). The
    # denoise loop below never runs the negative/uncond pass, so rather than silently ignore
    # a true-CFG request (which would produce weaker prompt adherence with no error), fail loudly.
    if true_cfg_scale > 1 and has_neg_prompt:
        raise NotImplementedError(
            "SpotEdit generate() runs a single forward per step and does not apply true-CFG. "
            "Call with true_cfg_scale<=1 (and/or no negative_prompt), or use the stock "
            "QwenImageEditPlusPipeline for classifier-free guidance."
        )
    if true_cfg_scale > 1 and not has_neg_prompt:
        logger.warning(
            f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
        )
    elif true_cfg_scale <= 1 and has_neg_prompt:
        logger.warning(
            " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
        )

    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=condition_images,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents = self.prepare_latents(
        vae_images,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [
        [
            (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
            *[
                (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ],
        ]
    ] * batch_size

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
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
    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(
            f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
        )
        guidance = None
    elif not self.transformer.config.guidance_embeds and guidance_scale is None:
        guidance = None

    text_n = prompt_embeds.shape[1]
    latent_n = latents.shape[1]
    image_n = image_latents.shape[1] if image_latents is not None else 0

    # The reuse decision and the final restore are made against the FIRST (edited)
    # reference image, whose tokens occupy the first `latent_n` slots of image_latents.
    ref_image_latents = image_latents[:, :latent_n] if image_latents is not None else None

    cache_flags = [
        torch.zeros((n), dtype=torch.bool, device=device)
        for n in [text_n, latent_n, image_n]
    ]
    cache_flags.append(0)

    # remember the original processors so we can restore them on exit (otherwise a later
    # plain pipe() call would run with leftover SpotEdit processors + stale cache state).
    _orig_procs = [(name, module.processor) for name, module in self.transformer.named_modules()
                   if isinstance(module, Attention)]

    for _, module in self.transformer.named_modules():
        if isinstance(module, Attention):
            module.set_processor(QwenSpotEditAttnProcessor(cache_flags))

    try:
        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        # SDEdit/img2img init: start the denoised latents from a noised version of the source
        # (x = (1-sigma)*x0_orig + sigma*noise) instead of pure noise, and run only the last
        # `source_init_strength` fraction of steps.
        start_idx = 0
        if ref_image_latents is not None and config.source_init_strength < 1.0:
            start_idx = max(0, int(round(len(timesteps) * (1.0 - config.source_init_strength))))
            sigma_start = self.scheduler.sigmas[start_idx].to(latents.device, latents.dtype)
            latents = (1.0 - sigma_start) * ref_image_latents + sigma_start * latents
            timesteps = timesteps[start_idx:]
        self._num_timesteps = len(timesteps)
        self.scheduler.set_begin_index(start_idx)

        x0_preds = []
        last_noise_pred = None

        reuse = torch.zeros((latent_n), dtype=torch.bool, device=device)
        cache_final = torch.zeros((latent_n), dtype=torch.bool, device=device)
        total_cached_tokens, total_latent_tokens = 0, 0
        ac = 0
        step_timing = bool(os.environ.get("SPOTEDIT_STEP_TIMING"))
        step_log = []
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                sel_dt = 0.0

                # for the initial and reset steps, we do full computation
                if i < config.initial_steps or i in config.reset_steps:
                    cache_flags[1] = torch.zeros((latent_n), dtype=torch.bool, device=device)
                    cache_flags[2] = torch.zeros((image_n), dtype=torch.bool, device=device)
                    reuse = torch.zeros((latent_n), dtype=torch.bool, device=device)
                    ac = 0
                    ac += 1
                # for spotedit steps: recompute the reuse mask either every step or once per reset block
                else:
                    if config.select_every_step or ac == 1:
                        if step_timing:
                            torch.cuda.synchronize(); _t_sel = time.perf_counter()
                        if len(x0_preds):
                            reuse = Spotselect(
                                self, x0_preds[-1], ref_image_latents,
                                threshold=config.threshold, method=config.judge_method,
                                image_size=(height, width),
                            )
                        # dilate for stable results
                        H_lat = height // self.vae_scale_factor // 2
                        W_lat = width // self.vae_scale_factor // 2
                        cache_flags[1] = dilate_uncached_mask(reuse, H_lat, W_lat, dilation_radius=config.dilation_radius)
                        if cache_flags[1].any():
                            cache_final = cache_flags[1]
                            cache_flags[2] = torch.ones((image_n), dtype=torch.bool, device=device)
                        else:
                            cache_flags[2] = torch.zeros((image_n), dtype=torch.bool, device=device)
                        if step_timing:
                            torch.cuda.synchronize(); sel_dt = time.perf_counter() - _t_sel
                    ac += 1
                cache_flags[-1] = t.item() / 1000

                cached_token_n = cache_flags[1].sum().item()
                total_cached_tokens += cached_token_n
                total_latent_tokens += latent_n

                self._current_timestep = t

                uncached_latents = latents[:, cache_flags[1].logical_not()]

                if image_latents is not None:
                    uncached_image_latents = image_latents[:, cache_flags[2].logical_not()]
                    latent_model_input = torch.cat([uncached_latents, uncached_image_latents], dim=1)
                else:
                    latent_model_input = uncached_latents

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                if step_timing:
                    torch.cuda.synchronize(); _t_tf = time.perf_counter()
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]
                if step_timing:
                    torch.cuda.synchronize(); tf_dt = time.perf_counter() - _t_tf
                    is_full = (i < config.initial_steps or i in config.reset_steps)
                    step_log.append((i, "FULL" if is_full else "CACHE", latent_model_input.shape[1], sel_dt, tf_dt))
                    print(f"[step] i={i:02d} {'FULL ' if is_full else 'CACHE'} "
                          f"fed_tok={latent_model_input.shape[1]:5d}  select={sel_dt*1000:6.1f}ms  "
                          f"transformer={tf_dt*1000:6.1f}ms", flush=True)

                # update the noise prediction only for edited (uncached) tokens
                if cache_flags[1].any():
                    uncached_n = cache_flags[1].logical_not().sum().item()
                    if config.reuse_mode == "velocity" and ref_image_latents is not None:
                        # reused (non-edited) tokens flow straight to the original image:
                        # v = (x_t - x0_orig)/sigma  =>  x0_pred = x_t - sigma*v = x0_orig.
                        # This converges them to the source smoothly (no end-of-run seam).
                        sigma = t.item() / 1000
                        noisy_copy = (latents - ref_image_latents) / sigma
                    else:
                        noisy_copy = last_noise_pred.clone()
                    noisy_copy[:, cache_flags[1].logical_not()] = noise_pred[:, :uncached_n]
                    noise_pred = noisy_copy
                else:
                    noise_pred = noise_pred[:, : latents.size(1)]

                last_noise_pred = noise_pred

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

        if step_timing and step_log:
            full = [s for s in step_log if s[1] == "FULL"]
            cache = [s for s in step_log if s[1] == "CACHE"]
            tf_full = sum(s[4] for s in full); tf_cache = sum(s[4] for s in cache)
            sel_tot = sum(s[3] for s in step_log)
            print(f"[steptime] FULL steps:  {len(full):2d}  transformer={tf_full:.2f}s", flush=True)
            print(f"[steptime] CACHE steps: {len(cache):2d}  transformer={tf_cache:.2f}s  "
                  f"(avg {tf_cache/max(len(cache),1)*1000:.0f}ms/step)", flush=True)
            print(f"[steptime] selection (judge) total over all steps: {sel_tot:.2f}s", flush=True)
            print(f"[steptime] loop transformer+select = {tf_full+tf_cache+sel_tot:.2f}s", flush=True)

        H_lat = height // self.vae_scale_factor // 2
        W_lat = width // self.vae_scale_factor // 2

        # "overwrite": GitHub default -- hard latent paste + boundary smoothing (leaves a seam).
        # "velocity"/"feather" keep the full generated latents (reused tokens already flowed to the
        # source under "velocity"; "feather" merges in pixel space after decoding).
        if cache_final.any() and config.reuse_mode == "overwrite":
            latents[:, cache_final] = ref_image_latents[:, cache_final]
            latents = boundary_aware_smoothing(
                latents, ref_image_latents,
                non_edit_mask=cache_final.unsqueeze(0), lambda0=0.8, hw=(H_lat, W_lat),
            )

        if output_type == "latent":
            image = latents
        else:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, self.vae.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, self.vae.dtype
            )

            def _decode(packed):
                lat = self._unpack_latents(packed, height, width, self.vae_scale_factor).to(self.vae.dtype)
                lat = lat / latents_std + latents_mean
                return self.vae.decode(lat, return_dict=False)[0][:, :, 0]  # [B,3,H,W], ~[-1,1]

            gen = _decode(latents)
            if config.reuse_mode == "feather" and cache_final.any() and ref_image_latents is not None:
                orig = _decode(ref_image_latents)  # VAE round-trip of the original (same space as gen)
                # cache_final is the REUSE mask (True = non-edited); feather_composite wants the
                # EDIT mask (True = recomputed), so pass its complement.
                gen = feather_composite(
                    gen, orig, cache_final.logical_not(), H_lat, W_lat,
                    tau=config.feather_tau, sigma=config.feather_sigma,
                )
            image = self.image_processor.postprocess(gen, output_type=output_type)

        # expose the final reuse mask (True = non-edited / reused token) on the latent grid,
        # so callers can visualise which region was frozen and how little it moved.
        if aux is not None:
            aux["reuse_mask"] = cache_final.detach().to("cpu").clone()
            aux["H_lat"] = H_lat
            aux["W_lat"] = W_lat

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
    finally:
        # restore the original attention processors so the pipe is left unmodified
        # (a subsequent plain pipe() call must not run with leftover SpotEdit processors)
        for name, proc in _orig_procs:
            self.transformer.get_submodule(name).set_processor(proc)
