import torch
import torch.nn as nn
import torch.nn.functional as F



class QwenTokenLPIPS(nn.Module):
    def __init__(self, vae, patch_size=2, t_index=0):
        super().__init__()
        self.vae = vae
        self.patch_size = patch_size
        self.t_index = t_index
        
        # cache z2 features
        self._z2_cached = None
        self._z2_feats_cache = None
        self._cached_device = None
        self._cached_dtype = None

    @torch.no_grad()
    def _forward_decoder_first3(self, z5d):
        dec = self.vae.decoder
        feats = {}
        feat_cache = None
        feat_idx = [0]

        x = dec.conv_in(z5d)                 
        feats["conv_in"] = x

        x = dec.mid_block(x, feat_cache, feat_idx)
        feats["mid_block"] = x

        up0 = dec.up_blocks[0]
        x = up0(x, feat_cache, feat_idx)
        feats["up_blocks.0"] = x

        return feats

    def _safe_unpack_tokens_2d(self, z_tokens, image_size, vae_downsample_factor, channels_per_token_div=4):
        B, N, Ctok = z_tokens.shape
        H_img, W_img = image_size

        H_lat = int(2 * (int(H_img) // int(vae_downsample_factor * 2)))
        W_lat = int(2 * (int(W_img) // int(vae_downsample_factor * 2)))

        z = z_tokens.view(B, H_lat // 2, W_lat // 2, Ctok // channels_per_token_div, 2, 2)
        z = z.permute(0, 3, 1, 4, 2, 5).contiguous()
        z = z.view(B, Ctok // channels_per_token_div, H_lat, W_lat)
        return z

    def _apply_qwen_mean_std(self, z5d):
        p = next(self.vae.decoder.parameters())
        device, dtype = p.device, p.dtype

        z_dim = getattr(self.vae.config, "z_dim", z5d.shape[1])

        mean = torch.tensor(getattr(self.vae.config, "latents_mean", 0.0), device=device, dtype=dtype)
        std  = torch.tensor(getattr(self.vae.config, "latents_std", 1.0),  device=device, dtype=dtype)

        mean = mean.view(1, z_dim, 1, 1, 1)
        std  = std.view(1, z_dim, 1, 1, 1)

        z5d = z5d.to(device=device, dtype=dtype)
        z5d = z5d * std + mean
        return z5d

    def _check_z2_cache_valid(self, z2):
        if self._z2_cached is None:
            return False
        return torch.equal(self._z2_cached, z2)
    
    @torch.no_grad()
    def set_reference_z2(self, z2, image_size, vae_downsample_factor):
        """set z2 cache for later use"""
        p = next(self.vae.decoder.parameters())
        device, dtype = p.device, p.dtype
        
        # cache z2
        self._z2_cached = z2.clone()
        self._cached_device = device
        self._cached_dtype = dtype
        
        # cache z2 features
        z2_5d = self._safe_unpack_tokens_2d(z2, image_size, vae_downsample_factor)
        z2_5d = self._apply_qwen_mean_std(z2_5d)
        self._z2_feats_cache = self._forward_decoder_first3(z2_5d)
        

    def clear_cache(self):
        self._z2_cached = None
        self._z2_feats_cache = None
        self._cached_device = None
        self._cached_dtype = None

    @torch.no_grad()
    def forward(self, z1, z2, *, image_size=None, vae_downsample_factor=None, use_cache=True):
        p = next(self.vae.decoder.parameters())
        
        z1_5d = self._safe_unpack_tokens_2d(z1, image_size, vae_downsample_factor)
        z1_5d = self._apply_qwen_mean_std(z1_5d)
        feats1 = self._forward_decoder_first3(z1_5d)
        
        # deal with z2 , use cache if valid
        if use_cache and self._check_z2_cache_valid(z2):
            feats2 = self._z2_feats_cache
        else:
            if use_cache:
                self.set_reference_z2(z2, image_size, vae_downsample_factor)
                feats2 = self._z2_feats_cache
            else:
                z2_5d = self._safe_unpack_tokens_2d(z2, image_size, vae_downsample_factor)
                z2_5d = self._apply_qwen_mean_std(z2_5d)
                feats2 = self._forward_decoder_first3(z2_5d)

        # compute diffs
        B, C_in, T_in, H_in, W_in = z1_5d.shape
        target_size_3d = (T_in, H_in, W_in)
        diffs = []
        for name in ("conv_in", "mid_block", "up_blocks.0"):
            f1, f2 = feats1[name], feats2[name]
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            d = (f1 - f2).pow(2).sum(dim=1, keepdim=True)
            if d.shape[-3:] != target_size_3d:
                d = F.interpolate(d, size=target_size_3d, mode="trilinear", align_corners=False)
            diffs.append(d)

        score_map_3d = torch.stack(diffs, dim=0).mean(dim=0).squeeze(1)
        score_map_2d = score_map_3d.mean(dim=1)

        if (H_in % self.patch_size) != 0 or (W_in % self.patch_size) != 0:
            raise ValueError(f"latent shape {(H_in, W_in)} can not be divided by patch_size={self.patch_size}")
        pooled = F.avg_pool2d(score_map_2d.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)
        token_scores = pooled.flatten(start_dim=1)

        return token_scores