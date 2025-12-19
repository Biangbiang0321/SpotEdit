"""
Main Usage:
    >>> from FLUX_kontext import generate, SpotEditConfig
    >>> from diffusers import FluxKontextPipeline
    >>> 
    >>> pipe = FluxKontextPipeline.from_pretrained(...)
    >>> config = SpotEditConfig(...)
    >>> result = generate(pipe, image=img, prompt="...",..., config=config)
"""


from .flux_kontext_spotedit import generate
from .flux_spot_ultis import SpotEditConfig

__all__ = [
    'generate',
    'SpotEditConfig',
]