from __future__ import annotations

from typing import Any

import nodes
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo

from .text_encoder import MiniMaxH3TextEncoderCache


class LinjianMiniMaxH3ReferenceToVideo(MiniMaxH3ReferenceToVideo):
    """MiniMax H3 reference conditioning with transparent QwenVL Sage support."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LinjianMiniMaxH3ReferenceToVideo",
            display_name="linjian Reference to Video (MiniMax H3)",
            description=(
                "MiniMax H3 参考图、参考视频和参考音频条件节点。内部会自动适配 "
                "SageAttention 2.1.1 的 QwenVL 内核接口，不兼容时安全回退到 PyTorch。"
            ),
            category="MiniMaxH3",
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip=(
                        "作用：连接类型设为 minimax 的 CLIPLoader 输出，负责理解提示词和"
                        "参考素材；内部会自动尝试 QwenVL SageAttention 2.1.1。默认无连接，"
                        "没有数值范围或步长。模型不匹配会导致编码失败；首次编码可能较慢。"
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "作用：连接 MiniMax H3 视频 VAE，用于编码参考图和参考视频并创建视频"
                        "潜空间。默认无连接，没有数值范围或步长。必须使用匹配的视频 VAE；"
                        "错误模型可能导致尺寸、颜色或编码异常。"
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    tooltip=(
                        "作用：连接 MiniMax H3 音频 VAE，用于编码参考视频音轨和独立参考音频。"
                        "默认无连接，没有数值范围或步长。必须使用匹配的音频 VAE；错误模型"
                        "可能导致无声、编码失败或音频异常。"
                    ),
                ),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "作用：描述目标视频内容、动作、镜头、风格和声音，并使用 <Picture 1>、"
                        "<Video 1>、<Audio 1> 引用已连接素材。默认空白，没有数值范围或步长。"
                        "编号按同类素材的连接顺序计算；漏写或写错标签会削弱对应参考素材作用。"
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                    tooltip=(
                        "作用：设置输出视频宽度。默认 1344；下限 32，上限为 ComfyUI 分辨率"
                        "限制，步长 32。宽度越大细节潜力越高，但显存占用和生成时间明显增加；"
                        "建议保持 32 的倍数并与目标画幅匹配。"
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                    tooltip=(
                        "作用：设置输出视频高度。默认 768；下限 32，上限为 ComfyUI 分辨率"
                        "限制，步长 32。高度越大细节潜力越高，但显存占用和生成时间明显增加；"
                        "建议保持 32 的倍数并与目标画幅匹配。"
                    ),
                ),
                io.Int.Input(
                    "length",
                    default=124,
                    min=5,
                    max=3600,
                    step=17,
                    tooltip=(
                        "作用：设置 24 fps 下的生成帧数。默认 124 帧（约 5 秒）；范围 5–3600，"
                        "步长 17，模型常用训练范围约 124–362 帧。帧数越多，生成时间、显存和"
                        "保持主体一致性的难度越高；首次建议用 124 帧测试。"
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["match", "max"],
                    default="match",
                    tooltip=(
                        "作用：选择参考图处理尺寸。默认 match；可选 match 或 max，没有数值"
                        "步长。match 按输出像素面积缩小，速度快、显存低，推荐日常使用；max "
                        "保留短边约 2048 像素的参考流程，身份和细节可能更强，但编码可能慢数倍。"
                    ),
                ),
                io.Image.Input(
                    "ref_image_0",
                    optional=True,
                    tooltip=(
                        "作用：连接第 1 张参考图，并在提示词中用 <Picture 1> 引用。默认不连接，"
                        "没有数值范围或步长。大图会按 ref_image_size 缩小，小图不会放大；"
                        "清晰、主体完整的图片通常更利于保持身份。"
                    ),
                ),
                io.Image.Input(
                    "ref_image_1",
                    optional=True,
                    tooltip=(
                        "作用：连接第 2 张参考图，并在提示词中用 <Picture 2> 引用。默认不连接，"
                        "没有数值范围或步长。大图会按 ref_image_size 缩小，小图不会放大；"
                        "与第 1 张冲突的构图或身份可能降低一致性。"
                    ),
                ),
                io.Image.Input(
                    "ref_image_2",
                    optional=True,
                    tooltip=(
                        "作用：连接第 3 张参考图，并在提示词中用 <Picture 3> 引用。默认不连接，"
                        "没有数值范围或步长。大图会按 ref_image_size 缩小，小图不会放大；"
                        "参考素材越多，提示词越需要明确每张图的用途。"
                    ),
                ),
                io.Image.Input(
                    "ref_video_0",
                    optional=True,
                    tooltip=(
                        "作用：连接第 1 段参考视频帧，并在提示词中用 <Video 1> 引用。默认不连接，"
                        "没有数值范围或步长。按 24 fps 处理，建议 2–15 秒；视频越长编码越慢、"
                        "占用越高，画面稳定且主体清晰的片段更适合作为动作或风格参考。"
                    ),
                ),
                io.Audio.Input(
                    "ref_video_audio_0",
                    optional=True,
                    tooltip=(
                        "作用：连接与 ref_video_0 同一段视频对应的音轨。默认不连接，没有数值"
                        "范围或步长。应与参考视频时间内容匹配；错配音轨可能削弱音画对应关系。"
                    ),
                ),
                io.Audio.Input(
                    "ref_audio_0",
                    optional=True,
                    tooltip=(
                        "作用：连接不依附参考视频的第 1 段独立参考音频，并在提示词中用 "
                        "<Audio 1> 引用。默认不连接，没有数值范围或步长。较长或噪声过多的"
                        "音频会增加编码时间并可能降低声音参考的明确性。"
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip=(
                        "输出：包含提示词及全部参考图片、视频和音频信息的 MiniMax H3 正向"
                        "条件，连接到采样器或引导器的 positive/conditioning 接口。"
                    ),
                ),
                io.Latent.Output(
                    display_name="Latent",
                    tooltip=(
                        "输出：按所选宽度、高度和帧数创建的 MiniMax H3 音视频潜空间，连接到"
                        "采样器的 latent_image 接口；尺寸和帧数越大，占用与耗时越高。"
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip: Any,
        vae: Any,
        audio_vae: Any,
        prompt: str,
        width: int,
        height: int,
        length: int,
        ref_image_size: str = "match",
        ref_image_0: Any | None = None,
        ref_image_1: Any | None = None,
        ref_image_2: Any | None = None,
        ref_video_0: Any | None = None,
        ref_video_audio_0: Any | None = None,
        ref_audio_0: Any | None = None,
    ) -> io.NodeOutput:
        # Keep the public UI identical to the built-in conditioning node. Sage is
        # intentionally internal: adding a selector would change the requested UI.
        sage_clip = MiniMaxH3TextEncoderCache().patch(
            clip,
            cache_enabled=False,
            max_cache_entries=2,
            verbose=False,
            sage_attention="auto",
        )[0]
        return super().execute(
            sage_clip,
            vae,
            audio_vae,
            prompt,
            width,
            height,
            length,
            ref_image_size,
            {
                "ref_image_0": ref_image_0,
                "ref_image_1": ref_image_1,
                "ref_image_2": ref_image_2,
            },
            {"ref_video_0": ref_video_0},
            {"ref_video_audio_0": ref_video_audio_0},
            {"ref_audio_0": ref_audio_0},
        )


__all__ = ["LinjianMiniMaxH3ReferenceToVideo"]
