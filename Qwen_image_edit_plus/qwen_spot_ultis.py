import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
import numpy as np
import math

from .QwenTokenLPIPS import QwenTokenLPIPS


@dataclass
class SpotEditConfig:
    # ---- cache decision ----
    threshold: float = 0.15
    judge_method: str = "LPIPS"
    initial_steps: int = 4
    reset_steps: list = field(default_factory=lambda: [13, 22, 31])
    dilation_radius: int = 1
    select_every_step: bool = False  # recompute the reuse mask every spotedit step (vs once per reset block)
    # ---- how non-edited (reused) tokens are kept faithful to the source ----
    # "velocity": each step set reused tokens' velocity = (x_t - x0_orig)/sigma so they flow
    #             straight to the original (smooth, no seam) -- recommended.
    # "feather" : keep generated latents, then blend the edit over the original in pixel space.
    # "overwrite": GitHub default -- hard latent paste + boundary smoothing (leaves a seam).
    reuse_mode: str = "velocity"
    feather_tau: float = 0.10    # ("feather") content-diff threshold in [-1,1] image space
    feather_sigma: float = 12.0  # ("feather") feather radius in pixels
    # ---- initialisation ----
    # 1.0 = start the denoised latents from pure noise (standard edit init).
    # <1.0 = SDEdit/img2img: start from a noised source  x = (1-sigma)*x0_orig + sigma*noise
    #        at sigma=strength, running only the last `strength` fraction of steps.
    source_init_strength: float = 1.0


def seed_everything(seed: int = 42):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Spotselect(self, x0_pred, image_latents, threshold=0.1, method='L4', image_size=(1024, 1024)):
    """
    Judge which tokens can be reused based on the selected method.

    `image_latents` must be token-aligned with `x0_pred`, i.e. the latents of the
    first (edited) reference image -- slice it before calling for multi-image inputs.
    """
    if method == 'L4':
        delta = x0_pred - image_latents
        mean_delta = (delta.abs() ** 4).mean(dim=-1).mean(dim=0)
        reuse = mean_delta < threshold
        return reuse
    elif method == 'cosine':
        sim_score = torch.cosine_similarity(
            image_latents.flatten(0, 1),
            x0_pred.flatten(0, 1),
            dim=-1,
        )
        reuse = sim_score > threshold
        return reuse
    elif method == 'LPIPS':
        if not hasattr(self, '_lpips_metric'):
            self.vae.to('cuda')
            self._lpips_metric = QwenTokenLPIPS(self.vae, patch_size=2, t_index=0)
        if self._lpips_metric._z2_cached is None:
            self._lpips_metric.set_reference_z2(
                image_latents,
                image_size=image_size,
                vae_downsample_factor=8,
            )
        token_scores = self._lpips_metric(
            x0_pred,
            image_latents,
            image_size=image_size,
            vae_downsample_factor=8,
            use_cache=True
        )
        reuse = token_scores.mean(dim=0) < threshold
        return reuse


def dilate_uncached_mask(reuse_mask: torch.Tensor, H_lat: int, W_lat: int,
                         dilation_radius: int = 1) -> torch.Tensor:

    # transform reuse mask to uncached mask
    uncached = (~reuse_mask).float().view(1, 1, H_lat, W_lat)

    # define dilation kernel
    kernel_size = 2 * dilation_radius + 1
    dilated = F.max_pool2d(
        uncached,
        kernel_size=kernel_size,
        stride=1,
        padding=dilation_radius
    )

    # turn back to reuse mask
    return (~dilated.squeeze().bool()).view(-1)


def _gaussian_blur(x, sigma):
    """Separable Gaussian blur on [B,1,H,W] (or [B,C,H,W]) with replicate padding."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    kernel = (g[:, None] * g[None, :]).view(1, 1, k, k).expand(x.shape[1], 1, k, k)
    x = F.pad(x, (k // 2,) * 4, mode="replicate")
    return F.conv2d(x, kernel, groups=x.shape[1])


def feather_composite(gen, orig, edit_token_mask, H_lat, W_lat, tau=0.10, sigma=12.0):
    """Pixel-space feathered composite of the generated edit over the original.

    gen, orig: [B,3,H,W] in [-1,1]; edit_token_mask: [N] bool (True = recomputed token).
    Only the pixels that actually changed (|gen-orig| > tau), restricted to the recomputed
    region and Gaussian-feathered, take the generated content; everything else keeps the
    original -- so there is no hard cache<->recompute boundary and unchanged content is preserved.
    """
    B, C, H, W = gen.shape
    if orig.shape[-2:] != (H, W):
        orig = F.interpolate(orig, size=(H, W), mode="bilinear", align_corners=False)
    region = edit_token_mask.float().view(1, 1, H_lat, W_lat)
    region = F.interpolate(region, size=(H, W), mode="nearest")
    diff = (gen - orig).abs().mean(dim=1, keepdim=True)
    m = (diff > tau).float() * region
    m = _gaussian_blur(m, sigma).clamp(0, 1)
    return m * gen + (1 - m) * orig


def boundary_aware_smoothing(
    x_gen: torch.Tensor,          # [B, N, C]
    y_latent: torch.Tensor,       # [B, N, C]
    non_edit_mask: torch.Tensor,  # [B, N], True 表示「非编辑区域」
    lambda0: float = 0.7,
    hw=None,                      # (H, W) of the token grid; defaults to a square grid
) -> torch.Tensor:
    B, N, C = x_gen.shape
    assert y_latent.shape == x_gen.shape, "x_gen and y_latent must have same shape"
    assert non_edit_mask.shape[:2] == (B, N), "non_edit_mask must be [B, N]"

    if hw is not None:
        H, W = hw
    else:
        H = W = int(math.sqrt(N))
    assert H * W == N, "H * W must equal N"

    # [B, N, C] -> [B, C, H, W]
    x_gen_2d = x_gen.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    y_2d = y_latent.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    # non_edit_mask_2d: [B,1,H,W]
    non_edit_mask_2d = non_edit_mask.view(B, H, W).unsqueeze(1).bool()
    edited_mask_2d = ~non_edit_mask_2d

    edited_float = edited_mask_2d.float()
    edited_dilated = F.max_pool2d(
        edited_float, kernel_size=3, stride=1, padding=1
    ) > 0

    boundary_mask_2d = non_edit_mask_2d & edited_dilated

    # interior_non_edited
    interior_non_edited_mask_2d = non_edit_mask_2d & (~boundary_mask_2d)

    x_final_2d = x_gen_2d.clone()

    if interior_non_edited_mask_2d.any():
        m = interior_non_edited_mask_2d.expand_as(x_final_2d)  # [B,C,H,W]
        x_final_2d[m] = y_2d[m]

    if boundary_mask_2d.any():
        m = boundary_mask_2d.expand_as(x_final_2d)
        x_final_2d[m] = lambda0 * y_2d[m] + (1.0 - lambda0) * x_gen_2d[m]

    # reshape back to [B, N, C]
    x_final = x_final_2d.permute(0, 2, 3, 1).contiguous().view(B, N, C)
    return x_final
