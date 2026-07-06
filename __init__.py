"""SpotEdit — also loadable as a ComfyUI custom-node package.

When this repository is cloned into ComfyUI's `custom_nodes/`, ComfyUI imports it as
a package and picks up the node mappings below. Outside ComfyUI (notebooks, scripts)
this file is inert: the backbone packages are still imported directly, e.g.
`from Qwen_image_edit import generate`.
"""
try:
    from .comfyui.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    WEB_DIRECTORY = "./comfyui/web"  # served at /extensions/<pkg>/ for custom JS widgets
except Exception:  # missing ComfyUI-side deps must not break library usage
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
