from __future__ import annotations

from collections import OrderedDict
import hashlib
import logging
import threading
import types
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


def _update_fingerprint(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu").contiguous()
        hasher.update(b"tensor")
        hasher.update(str(tuple(tensor.shape)).encode("ascii"))
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(tensor.view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, dict):
        hasher.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_fingerprint(hasher, key)
            _update_fingerprint(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_fingerprint(hasher, item)
        return
    if isinstance(value, bytes):
        hasher.update(b"bytes")
        hasher.update(value)
        return
    if isinstance(value, (str, int, float, bool, type(None))):
        hasher.update(type(value).__name__.encode("ascii"))
        hasher.update(repr(value).encode("utf-8"))
        return
    hasher.update(type(value).__qualname__.encode("utf-8"))
    hasher.update(repr(value).encode("utf-8", errors="replace"))


def token_fingerprint(tokens: Any, return_pooled: Any, return_dict: bool) -> str:
    hasher = hashlib.blake2b(digest_size=20)
    _update_fingerprint(hasher, tokens)
    _update_fingerprint(hasher, return_pooled)
    _update_fingerprint(hasher, return_dict)
    return hasher.hexdigest()


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    return value


class MiniMaxH3TextEncoderCache:
    """Cache exact MiniMax H3 text/reference-image conditioning results."""

    DESCRIPTION = (
        "缓存完全相同的 MiniMax H3 提示词与参考图编码结果。首次编码仍会完整运行，"
        "后续重复编码可直接复用 CPU 缓存。"
    )
    CATEGORY = "MiniMaxH3/optimization"
    EXPERIMENTAL = True
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    OUTPUT_TOOLTIPS = (
        "带精确结果缓存的 MiniMax H3 CLIP；连接到图生视频或参考生视频条件节点。",
    )
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": (
                            "作用：连接类型为 minimax 的 CLIPLoader 输出，并为它添加精确结果"
                            "缓存。仅支持 MiniMax H3 Qwen3-VL 32B 文本编码器；没有默认连接、"
                            "数值范围或调节步长。"
                        )
                    },
                ),
                "cache_enabled": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "作用：控制是否缓存文本与参考图编码。默认值：开启（True）；可选"
                            "值：开启或关闭，没有数值范围和调节步长。开启后，提示词、参考图"
                            "和编码参数完全相同时复用结果；任意内容变化都会正常重新编码。"
                        ),
                    },
                ),
                "max_cache_entries": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 8,
                        "step": 1,
                        "tooltip": (
                            "作用：限制最多保留多少组文本编码结果。默认值：2；下限：1；"
                            "上限：8；调节步长：1。缓存保存在 CPU 内存；数值越大越方便"
                            "来回切换提示词，但会占用更多内存。"
                        ),
                    },
                ),
            },
            "optional": {
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "作用：在控制台显示文本编码缓存 HIT（命中）和 MISS（未命中）。"
                            "默认值：关闭（False）；可选值：开启或关闭，没有数值范围和调节"
                            "步长。排错时开启，日常使用保持关闭。"
                        ),
                    },
                )
            },
        }

    @staticmethod
    def _is_minimax_clip(clip: Any) -> bool:
        model = getattr(clip, "cond_stage_model", None)
        return type(model).__name__.startswith("MiniMaxH3TEModel")

    def patch(
        self,
        clip: Any,
        cache_enabled: bool,
        max_cache_entries: int,
        verbose: bool = False,
    ) -> tuple[Any]:
        if not self._is_minimax_clip(clip):
            actual = type(getattr(clip, "cond_stage_model", None)).__name__
            raise ValueError(f"该节点仅支持 MiniMax H3 文本编码器，当前为 {actual}。")

        patched_clip = clip.clone()
        if not cache_enabled:
            return (patched_clip,)

        original_encode = patched_clip.encode_from_tokens
        cache: OrderedDict[str, Any] = OrderedDict()
        lock = threading.RLock()
        max_entries = max(1, int(max_cache_entries))

        def cached_encode(
            _self: Any,
            tokens: Any,
            return_pooled: Any = False,
            return_dict: bool = False,
        ) -> Any:
            key = token_fingerprint(tokens, return_pooled, return_dict)
            with lock:
                if key in cache:
                    value = cache.pop(key)
                    cache[key] = value
                    if verbose:
                        LOGGER.info("MiniMax H3 text encoder cache: HIT %s", key[:10])
                    # The stock scheduled encoder pops "cond" from returned
                    # dictionaries, so never expose the stored cache object.
                    return _detach_tree(value)

            if verbose:
                LOGGER.info("MiniMax H3 text encoder cache: MISS %s", key[:10])
            value = original_encode(
                tokens,
                return_pooled=return_pooled,
                return_dict=return_dict,
            )
            cached_value = _detach_tree(value)
            with lock:
                cache[key] = cached_value
                while len(cache) > max_entries:
                    cache.popitem(last=False)
            return value

        patched_clip.encode_from_tokens = types.MethodType(cached_encode, patched_clip)
        patched_clip._minimax_h3_text_cache = cache
        return (patched_clip,)


__all__ = ["MiniMaxH3TextEncoderCache", "token_fingerprint"]
