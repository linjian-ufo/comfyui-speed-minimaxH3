from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Callable, Optional

import torch

import comfy.model_management
import comfy.patcher_extension

from .attention import make_sage_attention_override
from .minimax_patch import ensure_minimax_h3_block_loop_support


LOGGER = logging.getLogger(__name__)
WRAPPER_KEY = "comfyui_speed_minimax_h3"
GIB = 1024 ** 3


@dataclass
class _CacheStats:
    full_steps: int = 0
    skipped_steps: int = 0
    repeated_calls: int = 0
    gpu_cache_steps: int = 0
    cpu_cache_steps: int = 0
    cache_disabled_steps: int = 0


class MiniMaxH3CacheController:
    """Residual cache for the complete MiniMax H3 transformer block loop.

    Only a small, sampled feature signature is retained for change detection.
    The large residual is kept on GPU or CPU according to the selected memory
    policy. The controller owns no model weights and is reset for every sample.
    """

    def __init__(
        self,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int,
        cache_device: str,
        vram_reserve_gb: float,
        ram_reserve_gb: float,
        signature_tokens: int,
        signature_features: int,
        verbose: bool,
    ) -> None:
        self.reuse_threshold = float(reuse_threshold)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.cache_device = cache_device
        self.vram_reserve_bytes = int(float(vram_reserve_gb) * GIB)
        self.ram_reserve_bytes = int(float(ram_reserve_gb) * GIB)
        self.signature_tokens = int(signature_tokens)
        self.signature_features = int(signature_features)
        self.verbose = bool(verbose)
        self.total_steps = 20
        self.reset()

    def reset(self, total_steps: Optional[int] = None) -> None:
        self.cached_residual: Optional[torch.Tensor] = None
        self.previous_signature: Optional[torch.Tensor] = None
        self.cached_metadata: Optional[tuple[Any, ...]] = None
        self.last_step_key: Optional[float] = None
        self.step_counter = 0
        self.consecutive_skips = 0
        self.accumulated_rel_l1 = 0.0
        self.stats = _CacheStats()
        if total_steps is not None:
            self.total_steps = max(1, int(total_steps))

    def finish(self) -> None:
        completed = self.stats.full_steps + self.stats.skipped_steps
        if completed:
            denominator = max(1, completed - self.stats.skipped_steps)
            estimated_speedup = completed / denominator
            LOGGER.info(
                "MiniMax H3 Speed Cache: skipped %d/%d transformer passes "
                "(estimated %.2fx; GPU cache %d, CPU cache %d, cache disabled %d).",
                self.stats.skipped_steps,
                completed,
                estimated_speedup,
                self.stats.gpu_cache_steps,
                self.stats.cpu_cache_steps,
                self.stats.cache_disabled_steps,
            )
        self.reset()

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    @staticmethod
    def _available_memory(device: torch.device) -> int:
        try:
            return int(comfy.model_management.get_free_memory(device))
        except Exception:
            if device.type == "cuda" and torch.cuda.is_available():
                free_bytes, _ = torch.cuda.mem_get_info(device)
                return int(free_bytes)
            return 0

    def _select_storage(self, img: torch.Tensor) -> str:
        if self.cache_device == "gpu":
            return "gpu"
        if self.cache_device == "cpu":
            return "cpu"
        if img.device.type != "cuda":
            return "cpu"

        # A GPU snapshot and the resulting residual coexist briefly.
        tensor_bytes = self._tensor_nbytes(img)
        gpu_required = tensor_bytes * 2 + self.vram_reserve_bytes
        if self._available_memory(img.device) >= gpu_required:
            return "gpu"

        cpu_required = tensor_bytes * 2 + self.ram_reserve_bytes
        if self._available_memory(torch.device("cpu")) >= cpu_required:
            return "cpu"
        return "disabled"

    def _signature(self, img: torch.Tensor, cache_ranges: list[tuple[int, int]]) -> torch.Tensor:
        """Build a small signature without cloning the full hidden-state tensor."""
        ranges = [(max(0, int(a)), min(img.shape[0], int(b))) for a, b in cache_ranges]
        ranges = [(a, b) for a, b in ranges if b > a]
        if not ranges:
            ranges = [(0, int(img.shape[0]))]

        parts = []
        feature_count = max(1, min(self.signature_features, int(img.shape[-1])))
        tokens_per_range = max(1, self.signature_tokens // len(ranges))
        for start, end in ranges:
            stride = max(1, math.ceil((end - start) / tokens_per_range))
            sampled = img[start:end:stride, :feature_count]
            if sampled.numel():
                parts.append(sampled.detach().float().abs().mean(dim=-1))

        if not parts:
            return torch.zeros(1, dtype=torch.float32, device=img.device)
        return torch.cat(parts)[: self.signature_tokens].clone()

    @staticmethod
    def _extract_step_key(args: dict[str, Any]) -> Optional[float]:
        transformer_options = args.get("transformer_options") or {}
        candidates = (
            args.get("step_info"),
            args.get("sigma"),
            args.get("timestep"),
            transformer_options.get("timestep"),
            args.get("t_emb"),
        )
        for candidate in candidates:
            if isinstance(candidate, torch.Tensor) and candidate.numel():
                return float(candidate.detach().flatten()[0].item())
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None

    @staticmethod
    def _metadata(img: torch.Tensor, block_count: Any, cache_ranges: list[tuple[int, int]]) -> tuple[Any, ...]:
        return (
            tuple(img.shape),
            img.dtype,
            img.device.type,
            img.device.index,
            int(block_count) if block_count is not None else None,
            tuple((int(a), int(b)) for a, b in cache_ranges),
        )

    def _reset_cache_only(self) -> None:
        self.cached_residual = None
        self.previous_signature = None
        self.consecutive_skips = 0
        self.accumulated_rel_l1 = 0.0

    def _apply_residual(self, img: torch.Tensor) -> torch.Tensor:
        residual = self.cached_residual
        if residual is None:
            return img
        if residual.device != img.device or residual.dtype != img.dtype:
            residual = residual.to(device=img.device, dtype=img.dtype, non_blocking=False)
        return img + residual

    def _run_and_cache(
        self,
        args: dict[str, Any],
        original_block: Callable[[dict[str, Any]], Any],
        img: torch.Tensor,
        signature: torch.Tensor,
    ) -> Any:
        storage = self._select_storage(img)
        snapshot: Optional[torch.Tensor] = None

        if storage == "disabled":
            self.stats.cache_disabled_steps += 1
            self._reset_cache_only()
            self.previous_signature = signature
            self.stats.full_steps += 1
            return original_block(args)

        try:
            if storage == "gpu":
                snapshot = img.detach().clone()
            else:
                snapshot = img.detach().to(device="cpu", copy=True)
        except torch.cuda.OutOfMemoryError:
            # The fallback encloses the allocation that can actually fail.
            snapshot = img.detach().to(device="cpu", copy=True)
            storage = "cpu"

        result = original_block(args)
        output = result["img"] if isinstance(result, dict) else result

        try:
            if storage == "gpu":
                residual = (output - snapshot).detach()
            else:
                output_cpu = output.detach().to(device="cpu", copy=True)
                residual = output_cpu.sub_(snapshot)
        except torch.cuda.OutOfMemoryError:
            # A residual allocation may fail even if the initial snapshot fit.
            snapshot_cpu = snapshot.detach().to(device="cpu", copy=True)
            snapshot = None
            comfy.model_management.soft_empty_cache()
            output_cpu = output.detach().to(device="cpu", copy=True)
            residual = output_cpu.sub_(snapshot_cpu)
            storage = "cpu"

        self.cached_residual = residual
        self.previous_signature = signature
        self.accumulated_rel_l1 = 0.0
        self.consecutive_skips = 0
        self.stats.full_steps += 1
        if storage == "gpu":
            self.stats.gpu_cache_steps += 1
        else:
            self.stats.cpu_cache_steps += 1
        return result

    def __call__(self, args: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        original_block = kwargs.get("original_block")
        if original_block is None or not isinstance(args, dict):
            raise RuntimeError("MiniMax H3 Speed Cache received an incompatible block-loop callback.")

        img = args["img"]
        cache_ranges = args.get("cache_ranges") or []
        block_count = args.get("block_count")
        metadata = self._metadata(img, block_count, cache_ranges)
        if metadata != self.cached_metadata:
            self._reset_cache_only()
            self.cached_metadata = metadata

        step_key = self._extract_step_key(args)
        if step_key is None:
            return original_block(args)

        # Repeated calls at the same sigma can represent another CFG condition.
        # Passing them through prevents residuals from crossing condition lanes.
        if self.last_step_key is not None and step_key == self.last_step_key:
            self.stats.repeated_calls += 1
            return original_block(args)

        self.last_step_key = step_key
        self.step_counter += 1
        progress = self.step_counter / max(1, self.total_steps)
        signature = self._signature(img, cache_ranges)

        should_skip = False
        rel_l1 = math.inf
        if self.cached_residual is not None and self.previous_signature is not None:
            if signature.shape == self.previous_signature.shape:
                difference = (signature - self.previous_signature).abs().mean().item()
                denominator = self.previous_signature.abs().mean().item() + 1e-6
                rel_l1 = difference / denominator
                self.accumulated_rel_l1 += rel_l1
                should_skip = (
                    self.start_percent <= progress <= self.end_percent
                    and self.accumulated_rel_l1 < self.reuse_threshold
                    and self.consecutive_skips < self.max_consecutive_skips
                )

        if should_skip:
            self.consecutive_skips += 1
            self.stats.skipped_steps += 1
            output = self._apply_residual(img)
            if self.verbose:
                LOGGER.info(
                    "MiniMax H3 Speed Cache step %d: SKIP (accumulated %.5f < %.5f).",
                    self.step_counter,
                    self.accumulated_rel_l1,
                    self.reuse_threshold,
                )
            return {"img": output}

        if self.verbose:
            LOGGER.info(
                "MiniMax H3 Speed Cache step %d: RUN (relative %.5f, progress %.1f%%).",
                self.step_counter,
                rel_l1,
                progress * 100.0,
            )
        return self._run_and_cache(args, original_block, img, signature)


class _SamplingScope:
    def __init__(self, cache: MiniMaxH3CacheController) -> None:
        self.cache = cache

    @staticmethod
    def _extract_sigmas(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[torch.Tensor]:
        candidate = kwargs.get("sigmas")
        if isinstance(candidate, torch.Tensor) and candidate.ndim == 1 and candidate.numel() > 1:
            return candidate
        if len(args) > 3:
            candidate = args[3]
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 1 and candidate.numel() > 1:
                return candidate
        for candidate in args:
            if isinstance(candidate, torch.Tensor) and candidate.ndim == 1 and candidate.numel() > 1:
                return candidate
        return None

    def __call__(self, executor: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        sigmas = self._extract_sigmas(args, kwargs)
        total_steps = int(sigmas.numel() - 1) if sigmas is not None else 20
        self.cache.reset(total_steps=total_steps)
        try:
            return executor(*args, **kwargs)
        finally:
            self.cache.finish()


class MiniMaxH3SpeedCache:
    """MiniMax H3 transformer cache with safe memory fallback."""

    DESCRIPTION = (
        "MiniMax H3 专用 SageAttention 与缓存加速 / Dedicated SageAttention "
        "and cache acceleration for MiniMax H3. "
        "默认采用实测速度与画质兼顾参数；不会修改显卡频率、功耗或风扇设置。 "
        "Uses a tested speed/quality profile and never changes GPU clocks, "
        "power limits, voltage, or fan settings."
    )
    CATEGORY = "MiniMaxH3/optimization"
    EXPERIMENTAL = True
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "已应用安全缓存策略的 MiniMax H3 模型 / MiniMax H3 model with the safe cache policy applied.",
    )
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "作用：接入需要加速的 MiniMax H3 扩散模型。仅支持 MiniMaxH3Model，"
                            "不支持其他视频或图片模型。请把模型加载节点的 MODEL 输出连接到这里。"
                            "该参数没有数值上下限，也没有默认模型。"
                        )
                    },
                ),
                "reuse_threshold": (
                    "FLOAT",
                    {
                        "default": 0.12,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.005,
                        "tooltip": (
                            "缓存复用阈值。作用：决定当前特征变化足够小时是否跳过一次完整的 "
                            "Transformer 计算。默认值：0.12；下限：0.00；上限：1.00；"
                            "调节步长：0.005。数值越高，命中缓存越容易、速度越快，但细节、"
                            "运动连续性和音画一致性的风险越高；数值越低越保画质。设为 0.00 "
                            "时不会跳步，相当于关闭加速判断。推荐画质优先使用 0.06～0.08，"
                            "均衡使用 0.10，速度优先可尝试 0.12。"
                        ),
                    },
                ),
                "start_percent": (
                    "FLOAT",
                    {
                        "default": 0.10,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "加速开始位置。作用：指定从采样总进度的哪个位置开始允许复用缓存。"
                            "默认值：0.10（10%）；下限：0.00（0%）；上限：1.00（100%）；"
                            "调节步长：0.01（1%）。开始得越早速度潜力越高，但采样前期结构"
                            "变化较大，过早跳步更容易影响构图和运动。必须小于 end_percent。"
                        ),
                    },
                ),
                "end_percent": (
                    "FLOAT",
                    {
                        "default": 0.90,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "加速结束位置。作用：指定采样进行到哪个位置后停止跳步，后续步骤恢复"
                            "完整计算以收敛细节。默认值：0.90（90%）；下限：0.00（0%）；"
                            "上限：1.00（100%）；调节步长：0.01（1%）。结束得越晚可能更快，"
                            "但最后阶段的纹理、边缘、动作及音频细节风险更高。必须大于 "
                            "start_percent。"
                        ),
                    },
                ),
                "max_consecutive_skips": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 5,
                        "step": 1,
                        "tooltip": (
                            "最大连续跳步数。作用：限制最多连续复用多少次缓存，达到上限后强制"
                            "执行一次完整计算。默认值：2；下限：1；上限：5；调节步长：1。"
                            "数值越高可能越快，但快速运动、主体稳定性和音画同步风险会明显增加。"
                            "画质优先保持 1；均衡或速度优先可尝试 2，不建议轻易超过 2。"
                        ),
                    },
                ),
                "cache_device": (
                    ["auto", "gpu", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "缓存保存位置。默认值：auto；可选值只有 auto、gpu、cpu，没有数值"
                            "上下限。auto：根据真实可用显存、系统内存和预留空间自动选择，推荐；"
                            "gpu：缓存常驻显存，通常最快但会增加显存占用；cpu：缓存保存在系统"
                            "内存，更省显存但跳步时需要传回显卡，速度可能下降。"
                        ),
                    },
                ),
            },
            "optional": {
                "vram_reserve_gb": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 0.5,
                        "max": 16.0,
                        "step": 0.5,
                        "tooltip": (
                            "显存预留量，单位 GB，仅在 cache_device=auto 时用于选择缓存位置。"
                            "作用：至少给模型权重、激活张量、CUDA 工作区和 VAE 留出这些显存。"
                            "默认值：2.0 GB；下限：0.5 GB；上限：16.0 GB；调节步长：0.5 GB。"
                            "经常显存不足时应调高；显存非常充足并追求速度时可以适当调低。"
                            "该设置不会修改显卡的物理显存或功耗。"
                        ),
                    },
                ),
                "ram_reserve_gb": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.0,
                        "max": 32.0,
                        "step": 1.0,
                        "tooltip": (
                            "系统内存预留量，单位 GB，仅在 cache_device=auto 时用于判断能否安全"
                            "使用 CPU 缓存。默认值：4.0 GB；下限：1.0 GB；上限：32.0 GB；"
                            "调节步长：1.0 GB。内存紧张、同时运行其他程序或不希望 Windows "
                            "使用页面文件时应调高；内存充足时保持默认即可。"
                        ),
                    },
                ),
                "signature_tokens": (
                    "INT",
                    {
                        "default": 128,
                        "min": 32,
                        "max": 512,
                        "step": 32,
                        "tooltip": (
                            "特征签名采样的 Token 数量。作用：从音频和视频隐藏状态中抽取多少个"
                            "位置来判断相邻步骤变化，不会改变生成分辨率或帧数。默认值：128；"
                            "下限：32；上限：512；调节步长：32。数值越大判断更全面，但会略微"
                            "增加检测开销；数值太小可能误判。一般保持 128。"
                        ),
                    },
                ),
                "signature_features": (
                    "INT",
                    {
                        "default": 64,
                        "min": 16,
                        "max": 256,
                        "step": 16,
                        "tooltip": (
                            "每个采样 Token 用于特征签名的通道数量。作用：决定变化检测观察多少"
                            "个隐藏特征通道，不会改变模型实际计算精度。默认值：64；下限：16；"
                            "上限：256；调节步长：16。数值越大检测更细致但略增开销；数值太小"
                            "可能降低判断可靠性。一般保持 64。"
                        ),
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "详细日志开关。作用：在 ComfyUI 控制台打印每一步是完整计算 RUN "
                            "还是缓存跳过 SKIP，并在结束时显示跳步数、缓存位置和估算加速比。"
                            "默认值：关闭（False）；可选值：关闭或开启，没有数值上下限。"
                            "首次测试和排查画质/速度问题时建议开启，长期使用可关闭以减少日志。"
                        ),
                    },
                ),
                # Keep new widgets after every legacy widget. ComfyUI stores
                # widget values by position, so inserting this in the middle
                # would shift old workflow values into the wrong controls.
                "sage_attention": (
                    ["auto", "enabled", "disabled"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "MiniMax H3 扩散模型注意力后端。默认 auto：检测到 SageAttention "
                            "便自动启用；不可用时安全回退到 ComfyUI 默认注意力后端；enabled：要求"
                            "必须启用，失败时明确报错；disabled：不由本节点设置注意力后端。"
                        ),
                    },
                ),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, start_percent: float, end_percent: float, **_: Any) -> Any:
        if start_percent >= end_percent:
            return "start_percent 必须小于 end_percent。"
        return True

    @staticmethod
    def _get_diffusion_model(model: Any) -> Any:
        try:
            return model.get_model_object("diffusion_model")
        except Exception:
            return getattr(getattr(model, "model", None), "diffusion_model", None)

    def patch(
        self,
        model: Any,
        reuse_threshold: float,
        start_percent: float,
        end_percent: float,
        max_consecutive_skips: int,
        cache_device: str,
        vram_reserve_gb: float = 2.0,
        ram_reserve_gb: float = 4.0,
        signature_tokens: int = 128,
        signature_features: int = 64,
        verbose: bool = False,
        sage_attention: str = "auto",
    ) -> tuple[Any]:
        if sage_attention not in {"auto", "enabled", "disabled"}:
            LOGGER.warning(
                "MiniMax H3 repaired an invalid legacy SageAttention widget value %r to auto.",
                sage_attention,
            )
            sage_attention = "auto"

        diffusion_model = self._get_diffusion_model(model)
        if diffusion_model is None or diffusion_model.__class__.__name__ != "MiniMaxH3Model":
            actual = type(diffusion_model).__name__ if diffusion_model is not None else "None"
            raise ValueError(f"该节点仅支持 MiniMaxH3Model，当前模型为 {actual}。")

        patch_status = ensure_minimax_h3_block_loop_support()
        if verbose:
            LOGGER.info("MiniMax H3 block-loop support: %s", patch_status)

        patched_model = model.clone()
        transformer_options = patched_model.model_options.setdefault("transformer_options", {})
        existing = (
            transformer_options.get("patches_replace", {})
            .get("dit", {})
            .get(("block_loop", 0))
        )
        if existing is not None:
            raise RuntimeError(
                "检测到已有 MiniMax H3 / EasyCache block_loop 缓存。请不要串联多个缓存节点。"
            )

        attention_status = "disabled"
        existing_attention = transformer_options.get("optimized_attention_override")
        if sage_attention == "auto" and existing_attention is not None:
            attention_status = "existing-external-override"
        elif sage_attention != "disabled":
            attention_override, attention_status = make_sage_attention_override(
                required=sage_attention == "enabled"
            )
            if attention_override is not None:
                transformer_options["optimized_attention_override"] = attention_override
        LOGGER.info("MiniMax H3 attention backend: %s", attention_status)

        cache = MiniMaxH3CacheController(
            reuse_threshold=reuse_threshold,
            start_percent=start_percent,
            end_percent=end_percent,
            max_consecutive_skips=max_consecutive_skips,
            cache_device=cache_device,
            vram_reserve_gb=vram_reserve_gb,
            ram_reserve_gb=ram_reserve_gb,
            signature_tokens=signature_tokens,
            signature_features=signature_features,
            verbose=verbose,
        )
        patched_model.set_model_patch_replace(cache, "dit", "block_loop", 0)
        patched_model.remove_wrappers_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            WRAPPER_KEY,
        )
        patched_model.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            WRAPPER_KEY,
            _SamplingScope(cache),
        )
        return (patched_model,)


class MiniMaxH3CacheRuntimeOptions:
    """Override safe runtime-only settings on an already patched cache model."""

    DESCRIPTION = (
        "供 linjian Image to Video 子图内部使用：覆盖外接 MiniMax H3 Speed Cache "
        "模型的缓存设备和详细日志开关，不会叠加第二套 block_loop 缓存。"
    )
    CATEGORY = "MiniMaxH3/internal"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = (
        "保持原加速策略、但已应用子图 cache_device 与 verbose 设置的 MiniMax H3 MODEL。",
    )
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "作用：接收已经经过 MiniMax H3 Speed Cache (Safe) 的 MODEL。"
                            "该连接没有默认值、数值范围或步长；若直接连接原始 UNET，节点会"
                            "明确报错，避免界面参数看似生效但实际没有缓存。"
                        )
                    },
                ),
                "cache_device": (
                    ["auto", "gpu", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "作用：覆盖外接加速节点本次运行的缓存位置。默认 auto；可选 auto、"
                            "gpu、cpu，没有数值范围或步长。auto 根据实际可用显存、内存和预留"
                            "空间选择，推荐；gpu 通常最快但更占显存；cpu 更省显存，但每次复用"
                            "可能需要同步传回显卡，长视频时可能变慢。"
                        ),
                    },
                ),
                "verbose": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "作用：覆盖外接加速节点的详细日志开关。默认关闭（False）；可选"
                            "开启或关闭，没有数值范围或步长。开启后控制台会打印 RUN/SKIP、"
                            "缓存设备、跳步数量和估算加速比；首次测速或排错建议开启，日常可关闭。"
                        ),
                    },
                ),
            }
        }

    @staticmethod
    def _controller(model: Any) -> MiniMaxH3CacheController | None:
        transformer_options = getattr(model, "model_options", {}).get(
            "transformer_options", {}
        )
        controller = (
            transformer_options.get("patches_replace", {})
            .get("dit", {})
            .get(("block_loop", 0))
        )
        if isinstance(controller, MiniMaxH3CacheController):
            return controller
        return None

    def apply(
        self,
        model: Any,
        cache_device: str = "auto",
        verbose: bool = False,
    ) -> tuple[Any]:
        if cache_device not in {"auto", "gpu", "cpu"}:
            LOGGER.warning(
                "MiniMax H3 repaired an invalid subgraph cache_device %r to auto.",
                cache_device,
            )
            cache_device = "auto"

        controller = self._controller(model)
        if controller is None:
            raise RuntimeError(
                "linjian Image to Video 的 unet_name 必须连接 MiniMax H3 Speed Cache "
                "(Safe) 输出，不能直接连接原始 UNETLoader。"
            )

        controller.cache_device = cache_device
        controller.verbose = bool(verbose)
        if verbose:
            LOGGER.info(
                "MiniMax H3 subgraph runtime options: cache_device=%s verbose=true",
                cache_device,
            )
        return (model,)


__all__ = [
    "MiniMaxH3SpeedCache",
    "MiniMaxH3CacheController",
    "MiniMaxH3CacheRuntimeOptions",
]
