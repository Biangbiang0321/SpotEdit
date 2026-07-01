"""
Qwen-Image-Edit Plus family (2509 / 2511) with SpotEdit - Fast Multi-Reference Image Editing

Both Qwen-Image-Edit-2509 and Qwen-Image-Edit-2511 share the same
``QwenImageEditPlusPipeline`` (same transformer / VAE / attention / multi-image
handling), so this single module accelerates both -- just load the matching repo:
``Qwen/Qwen-Image-Edit-2509`` or ``Qwen/Qwen-Image-Edit-2511``.
"""


from .qwen_plus_spotedit import generate
from .qwen_spot_ultis import SpotEditConfig

__all__ = ['generate', 'SpotEditConfig']
