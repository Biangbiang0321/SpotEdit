import torch
import torch.nn as nn
import torch.nn.functional as F

class FLUXVAETokenLPIPS(nn.Module):
    def __init__(self, vae, layers=("conv_in", "mid_block", "up_blocks.0"), patch_size=2):
        super().__init__()
        self.vae = vae
        self.layers = layers
        self.patch_size = patch_size

        self._z2_cached = None
        self._z2_feats_cache = None

    @torch.no_grad()
    def _forward_decoder_first3(self, z_latent):

        dec = self.vae.decoder
        feats = {}
        # get features from first 3 layers
        x = dec.conv_in(z_latent)
        feats["conv_in"] = x

        x = dec.mid_block(x)
        feats["mid_block"] = x

        up0 = dec.up_blocks[0]
        x = up0(x)
        feats["up_blocks.0"] = x

        return feats
    @torch.no_grad()
    def set_z2_cache(self, z2, image_size, vae_downsample_factor):
        z2 = self._safe_unpack_tokens(z2, image_size, vae_downsample_factor)
        z2 = self._apply_scale_shift(z2)
        self._z2_feats_cache = self._forward_decoder_first3(z2)
        return self._z2_feats_cache

    def _safe_unpack_tokens(self, z_tokens, image_size, vae_downsample_factor, channels_per_token_div=4):
        B, N, Ctok = z_tokens.shape
        H_img, W_img = image_size

        # latent size
        h_lat = int(2 * (int(H_img) // int(vae_downsample_factor * 2)))
        w_lat = int(2 * (int(W_img) // int(vae_downsample_factor * 2)))

        expected_N = (h_lat // 2) * (w_lat // 2)
        if N != expected_N:
            raise ValueError(f"num_patches does not match : N={N}, expect {expected_N}, please check image_size/vae_downsample_factor。")
        if (Ctok % channels_per_token_div) != 0:
            raise ValueError(f"C_tok={Ctok} can not be devided by  {channels_per_token_div}")

        z = z_tokens.view(B, h_lat // 2, w_lat // 2, Ctok // channels_per_token_div, 2, 2)
        z = z.permute(0, 3, 1, 4, 2, 5).contiguous()
        z = z.view(B, Ctok // channels_per_token_div, h_lat, w_lat)
        return z  # (B, C_lat, H_lat, W_lat)

    def _apply_scale_shift(self, z_latent):
        scaling = getattr(self.vae.config, "scaling_factor", 1.0)
        shift   = getattr(self.vae.config, "shift_factor", 0.0)
        return (z_latent / scaling) + shift

    def check_z2_cache_valid(self, z2):
        if self._z2_cached is None:
            return False
        if self._z2_feats_cache is None:
            return False
        return torch.equal(self._z2_cached, z2)


    @torch.no_grad()
    def forward(self, z1, z2, *, image_size=None, vae_downsample_factor=None, use_cache =True):
        z1 = self._safe_unpack_tokens(z1, image_size, vae_downsample_factor)
        z2 = self._safe_unpack_tokens(z2, image_size, vae_downsample_factor)

        # 2) scale + shift
        z1 = self._apply_scale_shift(z1)
        z2 = self._apply_scale_shift(z2)

        #set params
        param  = next(self.vae.decoder.parameters())
        device = param.device
        dtype  = param.dtype
        z1 = z1.to(device=device, dtype=dtype)
        z2 = z2.to(device=device, dtype=dtype)

        # collect features
        feats1 = self._forward_decoder_first3(z1)
        if use_cache and self.check_z2_cache_valid(z2):
            feats2 = self._z2_feats_cache
        else:
            feats2 = self._forward_decoder_first3(z2)

        B, _, H_lat, W_lat = z1.shape
        target_hw = (H_lat, W_lat)
        diffs = []
        for name in self.layers:
            f1 = feats1[name]
            f2 = feats2[name]
            f1 = F.normalize(f1, dim=1)
            f2 = F.normalize(f2, dim=1)
            d = (f1 - f2).pow(2).sum(dim=1, keepdim=True)  # (B,1,H_i,W_i)
            if d.shape[-2:] != target_hw:
                d = F.interpolate(d, size=target_hw, mode="bilinear", align_corners=False)
            diffs.append(d)

        score_map = torch.stack(diffs, dim=0).mean(dim=0).squeeze(1)  # (B,H_lat,W_lat)

        if (H_lat % self.patch_size != 0) or (W_lat % self.patch_size != 0):
            raise ValueError(f"latent size {(H_lat, W_lat)} can not be divided by patch_size={self.patch_size} .")
        pooled = F.avg_pool2d(score_map.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size)  # (B,1,H/2,W/2)
        token_scores = pooled.flatten(start_dim=1)  # (B, num_patches)

        return token_scores