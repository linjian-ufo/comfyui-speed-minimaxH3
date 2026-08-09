from __future__ import annotations


class LinjianMiniMaxH3ImageToVideo:
    """Frontend entry that is replaced with the bundled native subgraph."""

    DESCRIPTION = (
        "创建 linjian Image to Video (MiniMax H3) 原生子图。其 unet_name 是 MODEL "
        "接口，可直接连接 MiniMax H3 Speed Cache 输出。"
    )
    CATEGORY = "MiniMaxH3"
    RETURN_TYPES = ()
    FUNCTION = "create_subgraph"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def create_subgraph(self):
        raise RuntimeError(
            "linjian MiniMax H3 子图没有被前端展开。请重启 ComfyUI 后按 Ctrl+F5 刷新页面。"
        )


__all__ = ["LinjianMiniMaxH3ImageToVideo"]
