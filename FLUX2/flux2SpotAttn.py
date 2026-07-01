import torch
import torch.nn.functional as F

from typing import Optional

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2ParallelSelfAttention,
    _get_qkv_projections,
)


def _select_query_rope(image_rotary_emb, text_n, latent_n2, cache_flags):
    """Keep only the RoPE positions of the tokens recomputed this step
    (full text + uncached latents + uncached image-condition tokens), matching
    the subset of queries fed through the transformer."""
    return tuple(
        torch.cat(
            [
                each[:text_n],
                each[text_n : text_n + latent_n2][cache_flags[1].logical_not()],
                each[text_n + latent_n2 :][cache_flags[2].logical_not()],
            ],
            dim=0,
        )
        for each in image_rotary_emb
    )


def _reassemble_kv(processor, key, value, text_n, latent_n, latent_n2):
    """Write the freshly computed (uncached) latent keys/values into the running
    cache and return the full-sequence keys/values ``[text | latents | image_cond]``.

    Unlike the FLUX.1 reference, this does not assume ``image_n == latent_n2`` and
    therefore also supports multiple reference images. The FLUX.1 code blended fresh
    and cached KV with ``cos(0.5*pi*t/1000)**2`` which, on the original timestep
    schedule, evaluates to ~1 (i.e. "use the cache"); we reproduce that effect by
    reading straight from the cache after updating the uncached slots."""
    reuse_mask = processor.cache_flags[1]
    uncached = reuse_mask.logical_not()

    # fresh keys/values for the recomputed latent tokens (they sit right after text)
    latent_key = key[:, text_n : text_n + latent_n, :]
    latent_value = value[:, text_n : text_n + latent_n, :]

    # update the running cache at the recomputed latent slots
    processor._cached_keys[:, text_n : text_n + latent_n2][:, uncached] = latent_key
    processor._cached_values[:, text_n : text_n + latent_n2][:, uncached] = latent_value
    processor._cached_t[:, text_n : text_n + latent_n2][:, uncached] = processor.cache_flags[-1]

    # full sequence: fresh text + cached latents (fresh@uncached, cached@reused) + cached image_cond
    key = torch.cat([key[:, :text_n, :], processor._cached_keys[:, text_n:, :]], dim=1)
    value = torch.cat([value[:, :text_n, :], processor._cached_values[:, text_n:, :]], dim=1)
    return key, value


class Flux2SpotAttnProcessor:
    """SpotEdit KV-cached processor for the FLUX.2 double-stream blocks
    (``Flux2Attention``). The joint sequence is ordered ``[text, latents,
    image_cond]``; keys/values for the non-edited latents and the reference
    image(s) are reused from the cache while only the edited (uncached) tokens
    are recomputed."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, cache_flags):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.cache_flags = cache_flags
        self._cached_keys = None
        self._cached_values = None
        self._cached_t = None

    def __call__(
        self,
        attn: "Flux2Attention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            # Order: [text, image]; the image stream itself is [latents, image_cond]
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        image_rotary_emb_query = image_rotary_emb
        if self.cache_flags[1].any():
            text_n = self.cache_flags[0].logical_not().sum().item()
            latent_n = self.cache_flags[1].logical_not().sum().item()
            latent_n2 = self.cache_flags[1].shape[0]

            key, value = _reassemble_kv(self, key, value, text_n, latent_n, latent_n2)
            image_rotary_emb_query = _select_query_rope(
                image_rotary_emb, text_n, latent_n2, self.cache_flags
            )
        else:
            self._cached_keys = key
            self._cached_values = value
            self._cached_t = torch.ones((key.shape[:2]), dtype=key.dtype, device=key.device)
            self._cached_t *= self.cache_flags[-1]

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb_query, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class Flux2ParallelSpotAttnProcessor:
    """SpotEdit KV-cached processor for the FLUX.2 single-stream parallel blocks
    (``Flux2ParallelSelfAttention``). The block fuses QKV with the MLP input
    projection, so the queries and the MLP branch stay on the uncached tokens
    while only the keys/values are expanded back to the full sequence."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, cache_flags):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.cache_flags = cache_flags
        self._cached_keys = None
        self._cached_values = None
        self._cached_t = None

    def __call__(
        self,
        attn: "Flux2ParallelSelfAttention",
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Parallel (QKV + MLP in) projection on the tokens that are fed in
        # (text + uncached latents + uncached image_cond at a cached step).
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        image_rotary_emb_query = image_rotary_emb
        if self.cache_flags[1].any():
            text_n = self.cache_flags[0].logical_not().sum().item()
            latent_n = self.cache_flags[1].logical_not().sum().item()
            latent_n2 = self.cache_flags[1].shape[0]

            key, value = _reassemble_kv(self, key, value, text_n, latent_n, latent_n2)
            image_rotary_emb_query = _select_query_rope(
                image_rotary_emb, text_n, latent_n2, self.cache_flags
            )
        else:
            self._cached_keys = key
            self._cached_values = value
            self._cached_t = torch.ones((key.shape[:2]), dtype=key.dtype, device=key.device)
            self._cached_t *= self.cache_flags[-1]

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb_query, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # Feed-forward branch (kept on the uncached tokens, aligned with the queries)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        hidden_states = attn.to_out(hidden_states)

        return hidden_states
