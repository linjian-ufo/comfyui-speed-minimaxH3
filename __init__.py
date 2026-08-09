"""ComfyUI Speed MiniMaxH3 custom node."""

from .nodes import MiniMaxH3CacheRuntimeOptions, MiniMaxH3SpeedCache
from .reference_to_video import LinjianMiniMaxH3ReferenceToVideo
from .subgraph_node import LinjianMiniMaxH3ImageToVideo
from .text_encoder import MiniMaxH3TextEncoderCache


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SpeedCache": MiniMaxH3SpeedCache,
    "MiniMaxH3CacheRuntimeOptions": MiniMaxH3CacheRuntimeOptions,
    "MiniMaxH3TextEncoderCache": MiniMaxH3TextEncoderCache,
    "LinjianMiniMaxH3ImageToVideo": LinjianMiniMaxH3ImageToVideo,
    "LinjianMiniMaxH3ReferenceToVideo": LinjianMiniMaxH3ReferenceToVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SpeedCache": "MiniMax H3 Speed Cache (Safe)",
    "MiniMaxH3CacheRuntimeOptions": "MiniMax H3 Cache Runtime Options",
    "MiniMaxH3TextEncoderCache": "MiniMax H3 Text Encoder Cache",
    "LinjianMiniMaxH3ImageToVideo": "linjian Image to Video (MiniMax H3)",
    "LinjianMiniMaxH3ReferenceToVideo": "linjian Reference to Video (MiniMax H3)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
