"""
FLUX.2 with SpotEdit - Fast Region-Aware Image Editing

The FLUX.2-dev backbone (Flux2Pipeline) lives on the `backbone/flux2-dev` branch;
this release line ships the GPU-runnable FLUX.2-klein backbone.
"""


from .flux2klein_spotedit import generate
from .flux2_spot_ultis import SpotEditConfig

__all__ = ['generate', 'SpotEditConfig']
