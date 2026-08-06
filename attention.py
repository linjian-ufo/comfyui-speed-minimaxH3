from __future__ import annotations

import logging
from typing import Any, Callable, Optional


LOGGER = logging.getLogger(__name__)


def sage_attention_status() -> tuple[bool, str]:
    """Return whether ComfyUI's SageAttention backend is usable."""
    try:
        import comfy.ldm.modules.attention as attention
    except Exception as exc:
        return False, f"ComfyUI attention module unavailable: {exc}"

    if not bool(getattr(attention, "SAGE_ATTENTION_IS_AVAILABLE", False)):
        return False, "sageattention package is not available to ComfyUI"
    if not hasattr(attention, "attention_sage"):
        return False, "ComfyUI does not expose attention_sage"
    return True, "ComfyUI SageAttention backend"


def make_sage_attention_override(
    *, required: bool = False
) -> tuple[Optional[Callable[..., Any]], str]:
    """Build a ModelPatcher attention override using ComfyUI's backend.

    The override uses ComfyUI's own wrapper so mask support, dtype conversion,
    GQA handling, and runtime fallback stay aligned with the installed ComfyUI
    and sageattention versions.
    """
    available, status = sage_attention_status()
    if not available:
        if required:
            raise RuntimeError(f"无法启用 SageAttention：{status}。")
        LOGGER.warning("MiniMax H3 SageAttention auto fallback: %s", status)
        return None, f"native-fallback ({status})"

    import comfy.ldm.modules.attention as attention

    sage_impl = getattr(attention.attention_sage, "__wrapped__", attention.attention_sage)

    def attention_override_sage(
        _current_attention: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        return sage_impl(*args, **kwargs)

    return attention_override_sage, "sage-enabled"


__all__ = ["make_sage_attention_override", "sage_attention_status"]
