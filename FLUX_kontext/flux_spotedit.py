import torch
import numpy as np

from typing import  List, Optional, Union
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    FluxPipelineOutput,
    PREFERRED_KONTEXT_RESOLUTIONS,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
)
from flux_spot_ultis import SpotEditConfig, SpotSelect, boundary_aware_smoothing, dilate_uncached_mask
from fluxSpotAttn import SpotFusionAttnProcessor

@torch.no_grad()
def generate(
    self,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    output_type: Optional[str] = "pil",
    max_sequence_length: int = 512,
    max_area: int = 1024**2,
    _auto_resize: bool = True,
    config: Optional[SpotEditConfig] = None,
):

    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    aspect_ratio = width / height
    width = round((max_area * aspect_ratio) ** 0.5)
    height = round((max_area / aspect_ratio) ** 0.5)

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    self._guidance_scale = guidance_scale
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

    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=1,
    )

    # 3. Preprocess image
    if image is not None and not (
        isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
    ):
        img = image[0] if isinstance(image, list) else image
        image_height, image_width = self.image_processor.get_default_height_width(img)
        aspect_ratio = image_width / image_height
        if _auto_resize:
            # Kontext is trained on specific resolutions, using one of them is recommended
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h)
                for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of
        image = self.image_processor.resize(image, image_height, image_width)
        image = self.image_processor.preprocess(image, image_height, image_width)

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = self.prepare_latents(
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        self.transformer.dtype,
        device,
        None,
        None,
    )
    if image_ids is not None:
        latent_ids = torch.cat(
            [latent_ids, image_ids], dim=0
        )  # dim 0 is sequence dimension

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
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
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * self.scheduler.order, 0
    )
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    # Handle Attention Modules
    text_n = prompt_embeds.shape[1]
    latent_n = latents.shape[1]
    image_n = image_latents.shape[1] if image_latents is not None else 0

    cache_flags = [
        torch.zeros((n), dtype=torch.bool, device=device)
        for n in [text_n, latent_n, image_n]
    ]
    cache_flags.append(0)
    for _, module in self.transformer.named_modules():
        if isinstance(module, FluxAttention):
            module.set_processor(SpotFusionAttnProcessor(cache_flags))

    # 6. Denoising loop
    # We set the index here to remove DtoH sync, helpful especially during compilation.
    # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
    self.scheduler.set_begin_index(0)

    x0_preds = []
    last_noise_pred = None
    reuse = torch.zeros((latent_n), dtype=torch.bool, device=device)
    cache_final = torch.zeros((latent_n), dtype=torch.bool, device=device)

    total_cached_tokens, total_latent_tokens = 0, 0

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if len(x0_preds):
                #for the initial and reset steps, we do full computation
                if i in config.reset_steps or i < config.initial_steps:
                    cache_flags[1] = torch.zeros(
                        (latent_n), dtype=torch.bool, device=device
                    )
                    cache_flags[2] = torch.zeros(
                        (image_n), dtype=torch.bool, device=device
                    )
                #for spotedit steps, we do selective computation
                else:
                    cache_flags[1] = SpotSelect(self, x0_preds[-1], image_latents, threshold=config.threshold, method=config.judge_method)
                    if config.dilation_radius > 0:
                        cache_flags[1] = dilate_uncached_mask(
                            cache_flags[1],
                            H_lat=height // self.vae_scale_factor,
                            W_lat=width // self.vae_scale_factor,
                            dilation_radius=config.dilation_radius,
                        )

                    if cache_flags[1].any():
                        cache_final = cache_flags[1]
                        cache_flags[2] = torch.ones(
                            (image_n), dtype=torch.bool, device=device
                        )
                    else:
                        cache_flags[2] = torch.zeros(
                            (image_n), dtype=torch.bool, device=device
                        )
                    
                    cache_flags[-1] = t.item() / 1000
                    cached_token_n = cache_flags[1].sum().item()
                    total_cached_tokens += cached_token_n
                    total_latent_tokens += latent_n
            # print(
            #     f"Step {i + 1}/{num_inference_steps}, cached latent tokens: {cached_token_n} / {latent_n}"
            # )

            self._current_timestep = t

            uncached_latents = latents[:, cache_flags[1].logical_not()]


            if image_latents is not None:
                uncached_image_latents = image_latents[:, cache_flags[2].logical_not()]
                latent_model_input = torch.cat([uncached_latents, uncached_image_latents], dim=1)
            else:
                latent_model_input = uncached_latents

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # pass the timestep to the scheduler
            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs={},
                return_dict=False,
            )[0]

            if cache_flags[1].any():
                uncached_n = cache_flags[1].logical_not().sum().item()
                noisy_copy = last_noise_pred.clone()
                noisy_copy[:, cache_flags[1].logical_not()] = noise_pred[:, :uncached_n]
                noise_pred = noisy_copy
            else:
                noise_pred = noise_pred[:, : latents.size(1)]

            last_noise_pred = noise_pred
            # compute the x_0 prediction
            x0_preds.append(latents - t.item() / 1000 * noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()

    self._current_timestep = None
    # For non-edited tokens, we explicitly overwrite the generated latents with the original latents
    if cache_final.any():
        latents[:, cache_final] = image_latents[:, cache_final]

    latents = boundary_aware_smoothing(
        latents, image_latents, non_edit_mask=cache_final.unsqueeze(0), lambda0=0.8
    )

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return FluxPipelineOutput(images=image)
