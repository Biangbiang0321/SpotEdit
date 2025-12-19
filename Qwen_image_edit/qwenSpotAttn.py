import torch
import torch.nn.functional as F
from typing import  Optional
import torch


from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.utils import  is_torch_xla_available, logging

from diffusers.models.attention_processor import Attention
from diffusers.models.transformers.transformer_qwenimage import (
    apply_rotary_emb_qwen,
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
logger = logging.get_logger(__name__)



class QwenSpotEditAttnProcessor:
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
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)


        if self.cache_flags[1].any():
            text_n = self.cache_flags[0].logical_not().sum().item()
            latent_n = self.cache_flags[1].logical_not().sum().item()
            latent_n2 = self.cache_flags[1].shape[0]
            image_n = self.cache_flags[2].shape[0]

            latent_key = img_key[:,:latent_n,:]
            latent_value = img_value[:,:latent_n,:]
            if self._cached_keys is not None:
                expanded_key = self._cached_keys[:,  latent_n2 :, :].clone()
                expanded_value = self._cached_values[:, latent_n2 :, :].clone()
            else:
                expanded_key = img_key[:,  latent_n :, :].clone()
                expanded_value = img_value[:, latent_n :, :].clone()

            expanded_key[:,self.cache_flags[1].logical_not(),:] = latent_key
            expanded_value[:,self.cache_flags[1].logical_not(),:] = latent_value


            if self._cached_keys is not None:
                self._cached_keys[:, :latent_n2][
                    :, self.cache_flags[1].logical_not()
                ] = latent_key
                self._cached_values[:, :latent_n2][
                    :, self.cache_flags[1].logical_not()
                ] = latent_value
                self._cached_t[:, :latent_n2][
                    :, self.cache_flags[1].logical_not()
                ] = self.cache_flags[-1]

                t = torch.tensor(self.cache_flags[-1]).to(device=img_key.device)
                lmd = torch.cos(0.5*torch.pi *(t/1000))**2

                expanded_key = (1 - lmd) * expanded_key + lmd * self._cached_keys[:,  :  latent_n2]
                expanded_value = (1 - lmd) * expanded_value + lmd * self._cached_values[:, : latent_n2]

            img_key = torch.cat([
                expanded_key,
                self._cached_keys[:,latent_n2:,:]
            ],dim=1)

            img_value = torch.cat([
                expanded_value,
                self._cached_values[:,latent_n2:,:]
            ],dim=1)
        else:
            self._cached_keys = img_key
            self._cached_values = img_value
            self._cached_t = torch.ones(
                (img_key.shape[:2]), dtype=img_key.dtype, device=img_key.device
            )
            self._cached_t *= self.cache_flags[-1]
        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            if self.cache_flags[1].any():
                #overwrite the query poisition for sample token
                img_query_rope = img_freqs.clone()
                img_query_rope = torch.cat(
                    [img_query_rope[:latent_n2,][self.cache_flags[1].logical_not()],
                     img_query_rope[latent_n2:][self.cache_flags[2].logical_not()]],dim=0)
                img_query = apply_rotary_emb_qwen(img_query, img_query_rope, use_real=False)
            else:
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
            
        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output

