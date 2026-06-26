import torch
import torch.nn.functional as F


from typing import  Optional
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    _get_qkv_projections,
    dispatch_attention_fn,
)
import torch.nn.functional as F





class SpotFusionAttnProcessor:
    _attention_backend = None

    def __init__(self, cache_flags):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version."
            )
        self.cache_flags = cache_flags

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = (
            _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
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

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
            
        image_rotary_emb_query = image_rotary_emb
        if self.cache_flags[1].any():
            text_n = self.cache_flags[0].logical_not().sum().item()
            latent_n = self.cache_flags[1].logical_not().sum().item()
            latent_n2 = self.cache_flags[1].shape[0]

            # uncached latents
            latent_key = key[:, text_n : text_n + latent_n, :]
            latent_value = value[:, text_n : text_n + latent_n, :]

            # copy the keys and values for the image tokens
            if hasattr(self,'_cached_keys'):
                expanded_key = self._cached_keys[:, text_n + latent_n2 :, :].clone()
                expanded_value = self._cached_values[:, text_n + latent_n2 :, :].clone()
            else:
                expanded_key = key[:, text_n + latent_n :, :].clone()
                expanded_value = value[:, text_n + latent_n :, :].clone()

            # overwrite the keys and values to expanded keys and values
            expanded_key[:, self.cache_flags[1].logical_not()] = latent_key
            expanded_value[:, self.cache_flags[1].logical_not()] = latent_value

            # overwrite the keys and values to cached keys and values
            if self._cached_keys is not None:
                self._cached_keys[:, text_n : text_n + latent_n2][
                    :, self.cache_flags[1].logical_not()
                ] = latent_key
                self._cached_values[:, text_n : text_n + latent_n2][
                    :, self.cache_flags[1].logical_not()
                ] = latent_value
                self._cached_t[:, text_n : text_n + latent_n2][
                    :, self.cache_flags[1].logical_not()
                ] = self.cache_flags[-1]

                t = torch.tensor(self.cache_flags[-1])
                lmd = torch.cos(0.5*torch.pi * (t / 1000))**2
                
                expanded_key = (1 - lmd) * expanded_key + lmd * self._cached_keys[:, text_n : text_n + latent_n2]
                expanded_value = (1 - lmd) * expanded_value + lmd * self._cached_values[:, text_n : text_n + latent_n2]


            # concatenate the keys and values
            key = torch.cat(
                [
                    key[:, :text_n, :],
                    expanded_key,
                    self._cached_keys[:, text_n + latent_n2 :, :],
                ],
                dim=1,
            )
            value = torch.cat(
                [
                    value[:, :text_n, :],
                    expanded_value,
                    self._cached_values[:, text_n + latent_n2 :, :],
                ],
                dim=1,
            )

            # overwrite the query positions for the image tokens
            image_rotary_emb_query = (
                torch.cat(
                    [
                        each[:text_n],
                        each[text_n : text_n + latent_n2][
                            self.cache_flags[1].logical_not()
                        ],
                        each[text_n + latent_n2 :][
                            self.cache_flags[2].logical_not()],
                    ],
                    dim=0,
                )
                for each in image_rotary_emb
            )

        else:
            self._cached_keys = key
            self._cached_values = value
            self._cached_t = torch.ones(
                (key.shape[:2]), dtype=key.dtype, device=key.device
            )
            self._cached_t *= self.cache_flags[-1]

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb_query, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        else:
            self._cached_keys = key
            self._cached_values = value

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

