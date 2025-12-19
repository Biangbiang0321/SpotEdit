import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils import deprecate, is_torch_xla_available, logging, replace_example_docstring
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
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
from .qwen_spot_ultis import  Spotselect, SpotEditConfig, dilate_uncached_mask



@torch.no_grad()
def generate(
        self:QwenImageEditPipeline,
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
        config: SpotEditConfig = SpotEditConfig()
):
    image_size = image[0].size if isinstance(image, list) else image.size
    calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
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
    # 3. Preprocess image
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = image
        image = self.image_processor.preprocess(image, calculated_height, calculated_width)
        image = image.unsqueeze(2)

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    if true_cfg_scale > 1 and not has_neg_prompt:
        logger.warning(
            f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
        )
    elif true_cfg_scale <= 1 and has_neg_prompt:
        logger.warning(
            " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
        )

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    prompt_embeds, prompt_embeds_mask = self.encode_prompt(
        image=prompt_image,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents = self.prepare_latents(
        image,
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
            (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
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

    cache_flags = [
        torch.zeros((n), dtype=torch.bool, device=device)
        for n in [text_n, latent_n, image_n]
    ]
    cache_flags.append(0)
    
    for _, module in self.transformer.named_modules():
        if isinstance(module, Attention):
            module.set_processor(QwenSpotEditAttnProcessor(cache_flags))

    if self.attention_kwargs is None:
        self._attention_kwargs = {}

    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
    negative_txt_seq_lens = (
        negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
    )

    # 6. Denoising loop
    self.scheduler.set_begin_index(0)


    x0_preds = []
    last_noise_pred = None


    

    reuse = torch.zeros((latent_n), dtype=torch.bool, device=device)
    cache_final=torch.zeros(
        (latent_n), dtype=torch.bool, device=device
    )
    total_cached_tokens, total_latent_tokens = 0, 0
    ac = 0
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            


            if i < config.initial_steps or i in config.reset_steps:
                cache_flags[1] = torch.zeros(
                    (latent_n), dtype=torch.bool, device = device
                )
                cache_flags[2] = torch.zeros(
                    (image_n),dtype = torch.bool , device = device
                )
                reuse = torch.zeros((latent_n), dtype=torch.bool, device=device)
                ac = 0
                ac += 1
            else:
                if ac % 1 == 0:
                    if len(x0_preds):
                        reuse = Spotselect(self, x0_preds[-1], image_latents, threshold=config.threshold, method=config.judge_method)
                    #dilate for stable results
                    H_lat = height // self.vae_scale_factor // 2
                    W_lat = width // self.vae_scale_factor // 2
                    cache_flags[1] = dilate_uncached_mask(reuse, H_lat, W_lat, dilation_radius=1)
                    if cache_flags[1].any():
                        cache_final = cache_flags[1]
                        cache_flags[2] = torch.ones(
                            (image_n),dtype = torch.bool ,device = device
                        )
                    else:
                        cache_flags[2] = torch.zeros(
                            (image_n),dtype = torch.bool , device = device
                        )
                ac += 1
            cache_flags[-1] = t.item() / 1000

            cached_token_n = cache_flags[1].sum().item()
            total_cached_tokens += cached_token_n
            total_latent_tokens += latent_n

            self._current_timestep = t


            uncached_latents = latents[:, cache_flags[1].logical_not()]

            latent_model_input = latents
            if image_latents is not None:
                uncached_image_latents = image_latents[:,cache_flags[2].logical_not()]
                latent_model_input = torch.cat([uncached_latents, uncached_image_latents], dim=1)
            else:
                latent_model_input = uncached_latents

            
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

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
            #update the noise prediction only for edited tokens
            if cache_flags[1].any():
                uncached_n = cache_flags[1].logical_not().sum().item()
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
    # For non-edited tokens, we explicitly overwrite the generated latents with the original latents
    if cache_final.any():
        latents[:, cache_final] = image_latents[:, cache_final]

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return QwenImagePipelineOutput(images=image)
