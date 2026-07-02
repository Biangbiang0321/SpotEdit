import torch
import torch.nn.functional as F
import numpy as np
import math

from dataclasses import dataclass, field

from .FLUX2LPIPS import FLUX2VAETokenLPIPS


@dataclass
class SpotEditConfig:
    # ---- cache decision ----
    threshold: float = 0.4
    judge_method: str = "LPIPS"
    initial_steps: int = 4
    reset_steps: list = field(default_factory=lambda: [13, 22, 31])
    dilation_radius: int = 1
    select_every_step: bool = False  # recompute the reuse mask every spotedit step (vs once per reset block)
    # "velocity" (default): reused tokens' velocity = (x_t - x0_orig)/sigma -> flow to source, no seam.
    # "overwrite": GitHub default -- hard latent paste + boundary smoothing (can leave a seam).
    reuse_mode: str = "velocity"


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


def SpotSelect(self, x0_pred, image_latents, threshold=0.1, method='L4', image_size=(1024, 1024)):
    """Return a boolean reuse mask (True = token can be cached / is non-edited).

    `image_latents` must be token-aligned with `x0_pred` (slice the first/edited
    reference image before calling for multi-image inputs)."""
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
        if not hasattr(self, 'metric') or self.metric is None:
            self.metric = FLUX2VAETokenLPIPS(self.vae)
        if self.metric._z2_cached is None:
            self.metric.set_z2_cache(image_latents, image_size=image_size, vae_downsample_factor=8)
        token_scores = self.metric(
            x0_pred, image_latents,
            image_size=image_size,
            vae_downsample_factor=8,
        )
        reuse = token_scores.mean(dim=0) < threshold
        return reuse
    elif method == 'LPIPS_kmeans':
        # same LPIPS score as 'LPIPS', but the reuse/recompute cut is chosen adaptively per step
        # by 1D k-means (k=2) instead of the fixed `threshold`. Available as an option; not default.
        if not hasattr(self, 'metric') or self.metric is None:
            self.metric = FLUX2VAETokenLPIPS(self.vae)
        if self.metric._z2_cached is None:
            self.metric.set_z2_cache(image_latents, image_size=image_size, vae_downsample_factor=8)
        token_scores = self.metric(
            x0_pred, image_latents,
            image_size=image_size,
            vae_downsample_factor=8,
        )
        return _kmeans2_reuse(token_scores.mean(dim=0))
    else:
        raise NotImplementedError(f"Method {method} not implemented.")


def dilate_uncached_mask(reuse_mask: torch.Tensor, H_lat: int, W_lat: int,
                         dilation_radius: int = 1) -> torch.Tensor:
    """Grow the edited (uncached) region by `dilation_radius` for stable borders."""
    # transform reuse mask to uncached mask
    uncached = (~reuse_mask).float().view(1, 1, H_lat, W_lat)

    kernel_size = 2 * dilation_radius + 1
    dilated = F.max_pool2d(
        uncached,
        kernel_size=kernel_size,
        stride=1,
        padding=dilation_radius,
    )

    # turn back to reuse mask
    return (~dilated.squeeze().bool()).view(-1)


def select_reuse_mask(self, x0_pred, image_latents, H_lat, W_lat, *, threshold, method,
                      image_size, dilation_radius, min_threshold=1e-4, decay=0.5):
    """Reuse mask (True = reuse / non-edited) with a full-reuse guard.

    If the judge would mark EVERY token reusable, the transformer recomputes 0 latent tokens
    (an empty query that can crash RoPE on some backbones). In that case, repeatedly lower the
    threshold and re-judge until at least one token is left uncached. If the prediction equals
    the source everywhere (no threshold helps), fall back to a full-recompute step."""
    def _mask(thr):
        reuse = SpotSelect(self, x0_pred, image_latents, threshold=thr, method=method, image_size=image_size)
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

    non_edit_mask_2d = non_edit_mask.view(B, H, W).unsqueeze(1).bool()
    edited_mask_2d = ~non_edit_mask_2d

    edited_float = edited_mask_2d.float()
    edited_dilated = F.max_pool2d(edited_float, kernel_size=3, stride=1, padding=1) > 0

    boundary_mask_2d = non_edit_mask_2d & edited_dilated
    interior_non_edited_mask_2d = non_edit_mask_2d & (~boundary_mask_2d)

    x_final_2d = x_gen_2d.clone()

    if interior_non_edited_mask_2d.any():
        m = interior_non_edited_mask_2d.expand_as(x_final_2d)
        x_final_2d[m] = y_2d[m]

    if boundary_mask_2d.any():
        m = boundary_mask_2d.expand_as(x_final_2d)
        x_final_2d[m] = lambda0 * y_2d[m] + (1.0 - lambda0) * x_gen_2d[m]

    x_final = x_final_2d.permute(0, 2, 3, 1).contiguous().view(B, N, C)
    return x_final


def seed_everything(seed: int = 42):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
