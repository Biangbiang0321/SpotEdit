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
    reset_steps:  list = field(default_factory=lambda: [13,22,31])
    dilation_radius: int = 1
    
def seed_everything(seed: int = 42):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Spotselect(self, x0_pred, image_latents, threshold=0.1, method='L4'):
    """
    judge which tokens can be reused based on the selected method.
    """
    if method == 'L4':
        delta = x0_pred - image_latents
        mean_delta = (delta.abs()**4).mean(dim=-1).mean(dim=0)
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
                image_size=(1024, 1024),
                vae_downsample_factor=8,
            )
        token_scores = self._lpips_metric(
            x0_pred,  
            image_latents,
            image_size=(1024, 1024),
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

def boundary_aware_smoothing(
    x_gen: torch.Tensor,          # [B, N, C]
    y_latent: torch.Tensor,       # [B, N, C]
    non_edit_mask: torch.Tensor,  # [B, N]，True 表示「非编辑区域」
    lambda0: float = 0.7
) -> torch.Tensor:
    B, N, C = x_gen.shape
    assert y_latent.shape == x_gen.shape, "x_gen and y_latent must have same shape"
    assert non_edit_mask.shape[:2] == (B, N), "non_edit_mask must be [B, N]"

    H = W = int(math.sqrt(N))
    assert H * W == N , "N must be a perfect square"

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