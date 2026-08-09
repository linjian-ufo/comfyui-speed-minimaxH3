from __future__ import annotations

from contextlib import contextmanager
from importlib import metadata
import logging
import threading
from typing import Any, Callable, Optional

import torch


LOGGER = logging.getLogger(__name__)
_QWENVL_CONTEXT = threading.local()
_QWENVL_DISPATCH_LOCK = threading.RLock()
_QWENVL_DISPATCH_INSTALLED = False
_QWENVL_ORIGINAL_DISPATCHERS: dict[str, Callable[..., Any]] = {}


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
    try:
        version = metadata.version("sageattention")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return True, f"ComfyUI SageAttention backend ({version})"


def attention_sage_qwenvl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: Optional[torch.Tensor] = None,
    skip_reshape: bool = False,
    skip_output_reshape: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """SageAttention 2.1.x adapter for QwenVL vision and causal text attention."""
    import comfy.ldm.modules.attention as attention
    import comfy.ops
    from sageattention import sageattn

    if not skip_reshape:
        return attention.attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=skip_reshape,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    batch, _, query_tokens, dim_head = q.shape
    key_tokens = k.shape[-2]
    is_causal = mask is not None and query_tokens == key_tokens
    if mask is not None and not is_causal:
        return attention.attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    if kwargs.get("enable_gqa", False):
        k, v = comfy.ops.repeat_kv_for_gqa(k, v, q.shape[-3], -3)

    try:
        out = sageattn(
            q,
            k,
            v,
            tensor_layout="HND",
            is_causal=is_causal,
            sm_scale=kwargs.get("scale"),
            smooth_k=False,
        )
    except Exception as exc:
        LOGGER.warning("QwenVL SageAttention failed; using PyTorch attention: %s", exc)
        return attention.attention_pytorch(
            q,
            k,
            v,
            heads,
            mask=mask,
            skip_reshape=True,
            skip_output_reshape=skip_output_reshape,
            **kwargs,
        )

    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
    return out


def _make_qwenvl_dispatch_proxy(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    def dispatch(device: Any, mask: bool = False, small_input: bool = False) -> Any:
        if getattr(_QWENVL_CONTEXT, "enabled", False) and getattr(device, "type", None) == "cuda":
            return attention_sage_qwenvl
        return original(device, mask=mask, small_input=small_input)

    return dispatch


def _install_qwenvl_dispatch() -> None:
    global _QWENVL_DISPATCH_INSTALLED

    with _QWENVL_DISPATCH_LOCK:
        if _QWENVL_DISPATCH_INSTALLED:
            return

        import comfy.text_encoders.llama as llama
        import comfy.text_encoders.qwen35 as qwen35
        import comfy.text_encoders.qwen_vl as qwen_vl

        modules = {"llama": llama, "qwen35": qwen35, "qwen_vl": qwen_vl}
        for name, module in modules.items():
            original = module.optimized_attention_for_device
            _QWENVL_ORIGINAL_DISPATCHERS[name] = original
            module.optimized_attention_for_device = _make_qwenvl_dispatch_proxy(original)
        _QWENVL_DISPATCH_INSTALLED = True


@contextmanager
def qwen_vl_sage_context(*, required: bool = False):
    """Select SageAttention only for this thread's MiniMax H3 QwenVL encode."""
    available, status = sage_attention_status()
    if not available:
        if required:
            raise RuntimeError(f"无法启用 QwenVL SageAttention：{status}。")
        yield f"native-fallback ({status})"
        return

    _install_qwenvl_dispatch()
    previous = getattr(_QWENVL_CONTEXT, "enabled", False)
    _QWENVL_CONTEXT.enabled = True
    try:
        yield f"qwen-vl-sage-enabled ({status})"
    finally:
        _QWENVL_CONTEXT.enabled = previous


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


__all__ = [
    "attention_sage_qwenvl",
    "make_sage_attention_override",
    "qwen_vl_sage_context",
    "sage_attention_status",
]
