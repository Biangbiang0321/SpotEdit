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
    judge_method: str = "LPIPS_kmeans"   # adaptive 1D-kmeans cut (was fixed-threshold "LPIPS")
    initial_steps: int = 4
    reset_steps: list = field(default_factory=lambda: [13, 22, 31])
    dilation_radius: int = 1
    select_every_step: bool = False  # recompute the reuse mask every spotedit step (vs once per reset block)
    compute_mode: str = "sliced"     # "sliced": transformer only sees non-reused tokens (speed);
    #                                  "full": transformer sees every token and the judged mask only drives
    #                                  the write-back -- quality mode for few-step/distilled (Lightning) models.
    full_last_steps: int = 0         # hybrid schedule: run the last K steps in full-compute mode so the
    #                                  whole image settles together (0 = off). With compute_mode="sliced"
    #                                  this recovers most of the "full" quality at a fraction of its cost.
    # ---- interactive / manual region control ----
    manual_reuse_mask: object = None  # optional latent-grid mask (flat or [H_lat, W_lat]; True/1 = keep
    #                                   as-is, False/0 = regenerate). Combined with the judge's decision
    #                                   per manual_mask_policy at every judge point.
    manual_mask_policy: str = "replace"  # "replace" | "intersect" | "union" (how it meets the judge mask)
    preview_after_judge: bool = False    # stop right after the first judge and return the decoded x0
    #                                      draft (aux gets the mask) -- powers interactive mask editing.
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


def _lpips_token_score(self, x0_pred, image_latents, image_size):
    """Per-token LPIPS-like edit score d (mean over channels): high = edited, low = unchanged."""
    if not hasattr(self, '_lpips_metric'):
        self.vae.to('cuda')
        self._lpips_metric = QwenTokenLPIPS(self.vae, patch_size=2, t_index=0)
    if self._lpips_metric._z2_cached is None:
        self._lpips_metric.set_reference_z2(
            image_latents, image_size=image_size, vae_downsample_factor=8,
        )
    token_scores = self._lpips_metric(
        x0_pred, image_latents, image_size=image_size, vae_downsample_factor=8, use_cache=True,
    )
    return token_scores.mean(dim=0)

def _kmeans2_reuse(d):
    """Split per-token scores into reuse/recompute by 1D k-means (k=2, optimal SSE split),
    instead of a fixed threshold. reuse = low-score cluster (tokens close to the source)."""
    x = d.detach().float().cpu().numpy()
    s = np.sort(x.astype(np.float64)); n = len(s)
    if n < 2 or s[-1] <= s[0]:
        return d <= float(s[-1])
    pre = np.cumsum(s); pre2 = np.cumsum(s ** 2); tot, tot2 = pre[-1], pre2[-1]; best, bi = np.inf, 1
    for i in range(1, n):
        nL = i; sL = pre[i - 1]; qL = pre2[i - 1]; nR = n - i; sR = tot - sL; qR = tot2 - qL
        sse = (qL - sL * sL / nL) + (qR - sR * sR / nR)
        if sse < best:
            best, bi = sse, i
    return d <= float((s[bi - 1] + s[bi]) / 2)

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
        reuse = _lpips_token_score(self, x0_pred, image_latents, image_size) < threshold
        return reuse
    elif method == 'LPIPS_kmeans':
        # same LPIPS score as 'LPIPS', but the reuse/recompute cut is chosen adaptively per step
        # by 1D k-means (k=2) instead of the fixed `threshold` (threshold is kept only as the
        # full-reuse guard fallback in select_reuse_mask).
        return _kmeans2_reuse(_lpips_token_score(self, x0_pred, image_latents, image_size))


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


def select_reuse_mask(self, x0_pred, image_latents, H_lat, W_lat, *, threshold, method,
                      image_size, dilation_radius, min_threshold=1e-4, decay=0.5):
    """Reuse mask (True = reuse / non-edited) with a full-reuse guard.

    If the judge would mark EVERY token reusable, the transformer is fed 0 latent tokens to
    recompute -- an empty query that crashes Qwen's RoPE (reshape of a 0-element tensor). In
    that case, repeatedly lower the threshold and re-judge until at least one token is left
    uncached. If the prediction equals the source everywhere (no threshold helps), fall back
    to a full-recompute step (nothing reused)."""
    def _mask(thr):
        reuse = Spotselect(self, x0_pred, image_latents, threshold=thr, method=method, image_size=image_size)
        if dilation_radius > 0:
            reuse = dilate_uncached_mask(reuse, H_lat, W_lat, dilation_radius=dilation_radius)
        return reuse
    mask = _mask(threshold)
    thr = threshold
    while bool(mask.all()) and thr > min_threshold:
        thr = thr * decay
        mask = _mask(thr)
    if bool(mask.all()):
        mask = torch.zeros_like(mask)
    return mask


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
