import torch
import torch.nn as nn
import torch.nn.functional as F


class FLUX2VAETokenLPIPS(nn.Module):
    """Token-level perceptual distance for FLUX.2 latents.

    FLUX.2 differs from FLUX.1 in two ways that matter here:
      * latents are *patchified* (2x2 -> channel) before packing, so a token has
        ``in_channels`` features and the token grid is ``(H_lat/2, W_lat/2)``;
      * latents are normalised by a running BatchNorm (``vae.bn``) instead of a
        scalar scale/shift.
    We undo both, decode through the first three decoder layers and read off a
    per-token score, exactly as in the FLUX.1 / Qwen variants.
    """

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
        self._z2_cached = z2  # reference image latents are constant across steps
        z2u = self._safe_unpack_tokens(z2, image_size, vae_downsample_factor)
        z2u = self._apply_bn_denorm(z2u)
        z2u = self._unpatchify(z2u)
        self._z2_feats_cache = self._forward_decoder_first3(z2u)
        return self._z2_feats_cache

    def _safe_unpack_tokens(self, z_tokens, image_size, vae_downsample_factor):
        """[B, N, C_tok] tokens -> [B, C_tok, H_lat/2, W_lat/2] patchified latent."""
        B, N, Ctok = z_tokens.shape
        H_img, W_img = image_size

        # latent resolution, then the 2x patchify grid that the tokens live on
        h_lat = int(H_img) // int(vae_downsample_factor)
        w_lat = int(W_img) // int(vae_downsample_factor)
        h_p, w_p = h_lat // 2, w_lat // 2

        expected_N = h_p * w_p
        if N != expected_N:
            raise ValueError(
                f"num_patches does not match: N={N}, expect {expected_N}, please check image_size/vae_downsample_factor."
            )

        z = z_tokens.view(B, h_p, w_p, Ctok).permute(0, 3, 1, 2).contiguous()
        return z  # (B, C_tok, H_lat/2, W_lat/2)

    def _apply_bn_denorm(self, z_patched):
        """Undo the FLUX.2 VAE batch-norm normalisation (token/patch space)."""
        param = next(self.vae.decoder.parameters())
        device, dtype = param.device, param.dtype

        bn = self.vae.bn
        eps = getattr(self.vae.config, "batch_norm_eps", 1e-4)
        mean = bn.running_mean.view(1, -1, 1, 1).to(device=device, dtype=dtype)
        std = torch.sqrt(bn.running_var + eps).view(1, -1, 1, 1).to(device=device, dtype=dtype)

        z_patched = z_patched.to(device=device, dtype=dtype)
        return z_patched * std + mean

    def _unpatchify(self, latents):
        """[B, C*4, H/2, W/2] -> [B, C, H, W] (inverse of Flux2 _patchify_latents)."""
        b, c4, h2, w2 = latents.shape
        c = c4 // 4
        latents = latents.reshape(b, c, 2, 2, h2, w2)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(b, c, h2 * 2, w2 * 2)
        return latents

    def check_z2_cache_valid(self, z2):
        if self._z2_cached is None:
            return False
        if self._z2_feats_cache is None:
            return False
        return torch.equal(self._z2_cached, z2)

    @torch.no_grad()
    def forward(self, z1, z2, *, image_size=None, vae_downsample_factor=None, use_cache=True):
        z1 = self._unpatchify(self._apply_bn_denorm(self._safe_unpack_tokens(z1, image_size, vae_downsample_factor)))

        feats1 = self._forward_decoder_first3(z1)
        if use_cache and self.check_z2_cache_valid(z2):
            feats2 = self._z2_feats_cache
        else:
            z2u = self._unpatchify(self._apply_bn_denorm(self._safe_unpack_tokens(z2, image_size, vae_downsample_factor)))
            feats2 = self._forward_decoder_first3(z2u)
            if use_cache:
                # refresh the cache so a different reference image in a later call
                # is not silently judged against the previous one.
                self._z2_cached = z2
                self._z2_feats_cache = feats2

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
            raise ValueError(
                f"latent size {(H_lat, W_lat)} can not be divided by patch_size={self.patch_size}."
            )
        pooled = F.avg_pool2d(
            score_map.unsqueeze(1), kernel_size=self.patch_size, stride=self.patch_size
        )  # (B,1,H/2,W/2)
        token_scores = pooled.flatten(start_dim=1)  # (B, num_patches)

        return token_scores
