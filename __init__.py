"""ComfyUI Speed MiniMaxH3 custom node."""

from .nodes import MiniMaxH3SpeedCache


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SpeedCache": MiniMaxH3SpeedCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SpeedCache": "MiniMax H3 Speed Cache (Safe)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
