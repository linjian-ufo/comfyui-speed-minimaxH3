"""ComfyUI Speed MiniMaxH3 custom node."""

from .nodes import MiniMaxH3SpeedCache
from .text_encoder import MiniMaxH3TextEncoderCache


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SpeedCache": MiniMaxH3SpeedCache,
    "MiniMaxH3TextEncoderCache": MiniMaxH3TextEncoderCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SpeedCache": "MiniMax H3 Speed Cache (Safe)",
    "MiniMaxH3TextEncoderCache": "MiniMax H3 Text Encoder Cache",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
