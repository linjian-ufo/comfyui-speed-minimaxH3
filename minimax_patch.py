from __future__ import annotations

import inspect
import threading
from typing import Any

import torch


PATCH_VERSION = 1
_PATCH_LOCK = threading.Lock()


def _has_block_loop_support(model_class: type) -> bool:
    if getattr(model_class, "_speed_minimax_h3_block_loop_version", 0) >= PATCH_VERSION:
        return True
    if not hasattr(model_class, "_run_blocks"):
        return False
    try:
        source = inspect.getsource(model_class._forward)
    except (OSError, TypeError):
        source = ""
    return '"block_loop"' in source or "'block_loop'" in source or bool(
        getattr(model_class, "_is_minimax_patched", False)
    )


def ensure_minimax_h3_block_loop_support() -> str:
    """Install an in-memory compatibility hook only when ComfyUI lacks one.

    No file in ComfyUI is overwritten. Existing compatible hooks from a newer
    ComfyUI or an older MiniMax H3 cache project are reused.
    """
    import comfy.ldm.minimax.model as minimax_model

    model_class = getattr(minimax_model, "MiniMaxH3Model", None)
    if model_class is None:
        raise RuntimeError("当前 ComfyUI 中没有 MiniMaxH3Model，请先升级 ComfyUI。")

    with _PATCH_LOCK:
        if _has_block_loop_support(model_class):
            return "existing-compatible-hook"

        def _run_blocks(
            self: Any,
            h: torch.Tensor,
            t_emb: torch.Tensor,
            mod_segments: Any,
            rope_freqs: torch.Tensor,
            transformer_options: dict[str, Any],
            start: int = 0,
            end: int | None = None,
        ) -> torch.Tensor:
            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})
            end = len(self.blocks) if end is None else end
            prefetch_queue = minimax_model.comfy.model_prefetch.make_prefetch_queue(
                list(self.blocks[start:end]), h.device, transformer_options
            )
            for index in range(start, end):
                block = self.blocks[index]
                minimax_model.comfy.model_prefetch.prefetch_queue_pop(
                    prefetch_queue, h.device, block
                )
                if ("double_block", index) in blocks_replace:
                    def block_wrap(args: dict[str, Any], current_block: Any = block) -> dict[str, torch.Tensor]:
                        return {
                            "img": current_block(
                                args["img"],
                                args["t_emb"],
                                args["mod_segments"],
                                args["rope_freqs"],
                                transformer_options=args["transformer_options"],
                            )
                        }

                    h = blocks_replace[("double_block", index)](
                        {
                            "img": h,
                            "t_emb": t_emb,
                            "mod_segments": mod_segments,
                            "rope_freqs": rope_freqs,
                            "transformer_options": transformer_options,
                        },
                        {"original_block": block_wrap},
                    )["img"]
                else:
                    h = block(
                        h,
                        t_emb,
                        mod_segments,
                        rope_freqs,
                        transformer_options=transformer_options,
                    )
            if prefetch_queue is not None:
                minimax_model.comfy.model_prefetch.prefetch_queue_pop(
                    prefetch_queue, h.device, None
                )
            return h

        def patched_forward(
            self: Any,
            x: Any,
            timestep: torch.Tensor,
            context: torch.Tensor,
            transformer_options: dict[str, Any] | None = None,
            minimax_payload: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> Any:
            transformer_options = transformer_options or {}
            video_x, audio_x = x[0], x[1]
            orig_t, orig_h, orig_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
            video_x = minimax_model.comfy.ldm.common_dit.pad_to_patch_size(
                video_x, self.patch_size
            )
            if video_x.shape[0] != 1:
                raise ValueError("MiniMax H3 supports batch size 1")
            payload = minimax_payload or {}
            device = video_x.device
            dtype = context.dtype

            latent_t, lat_h, lat_w = video_x.shape[2], video_x.shape[3], video_x.shape[4]
            audio_t = audio_x.shape[-1]
            text_len = context.shape[1]

            layout = payload.get("layout")
            if layout is None or layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t):
                layout = minimax_model.PackedLayout(
                    text_len,
                    latent_t,
                    lat_h,
                    lat_w,
                    audio_t,
                    keyframes=payload.get("keyframes"),
                    refs=payload.get("refs"),
                    frame_count=payload.get("frame_count"),
                )

            shift_v = float(
                transformer_options.get(
                    "minimax_h3_sigma_shift_video", self.sigma_shift_video
                )
            )
            shift_a = float(
                transformer_options.get(
                    "minimax_h3_sigma_shift_audio", self.sigma_shift_audio
                )
            )
            sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
            t_v = float(1.0 - sigma_v)
            t_a = float(
                1.0 - minimax_model.time_shift_sigma(sigma_v, shift_v, shift_a)
            )

            vis_aug = float(
                payload.get("visual_cond_noise_aug", minimax_model.VISUAL_COND_TIMESTEP)
            )
            aud_aug = float(
                payload.get("audio_cond_noise_aug", minimax_model.AUDIO_COND_TIMESTEP)
            )
            has_vis_cond = any(
                kind in ("cond", "ref_img") for _, _, kind in layout.segments
            )
            has_aud_cond = any(kind == "ref_audio" for _, _, kind in layout.segments)
            seg_t = {
                "text": t_v,
                "video": t_v,
                "audio": t_a,
                "cond": max(t_v, vis_aug),
                "ref_img": max(t_v, vis_aug),
                "ref_audio": max(t_a, aud_aug),
            }
            unique_t = sorted(
                {t_v, t_a}
                | ({seg_t["cond"]} if has_vis_cond else set())
                | ({seg_t["ref_audio"]} if has_aud_cond else set())
            )
            t_row = {value: index for index, value in enumerate(unique_t)}
            seg_tag = {
                "text": 1,
                "video": 0,
                "audio": 2,
                "cond": 0,
                "ref_img": 0,
                "ref_audio": 2,
            }

            text_tags = payload.get("text_token_tags")
            mod_segments = []
            for start, end, kind in layout.segments:
                row_base = t_row[seg_t[kind]] * 3
                if kind == "text" and text_tags is not None:
                    tags = text_tags.view(-1).tolist()
                    run_start = 0
                    for index in range(1, end - start + 1):
                        if index == end - start or tags[index] != tags[run_start]:
                            mod_segments.append(
                                (
                                    start + run_start,
                                    start + index,
                                    row_base + int(tags[run_start]),
                                )
                            )
                            run_start = index
                else:
                    mod_segments.append((start, end, row_base + seg_tag[kind]))

            img_update = layout.img_update.to(device)
            audio_update = layout.audio_update.to(device)
            video_rows = minimax_model.patchify_video(
                video_x.to(torch.float32), self.patch_size
            )
            audio_rows = minimax_model.pack_audio(audio_x.to(torch.float32))
            cond_video_rows = self._cond_video_rows(payload, device)
            cond_audio_rows = self._cond_audio_rows(payload, device)

            all_video_rows = video_rows
            if cond_video_rows is not None:
                all_video_rows = torch.empty(
                    img_update.shape[0],
                    video_rows.shape[1],
                    dtype=torch.float32,
                    device=device,
                )
                all_video_rows[~img_update] = cond_video_rows
                all_video_rows[img_update] = video_rows
            all_audio_rows = audio_rows
            if cond_audio_rows is not None:
                all_audio_rows = torch.empty(
                    audio_update.shape[0],
                    audio_rows.shape[1],
                    dtype=torch.float32,
                    device=device,
                )
                all_audio_rows[~audio_update] = cond_audio_rows
                all_audio_rows[audio_update] = audio_rows

            video_embed = self.video_patch_proj(all_video_rows).to(dtype)
            audio_embed = self.audio_patch_proj(all_audio_rows).to(dtype)
            text_states = context[0]
            if text_states.shape[-1] != self.hidden_size:
                text_states = self.token_refiner(
                    self.condition_proj(text_states),
                    transformer_options=transformer_options,
                )

            h = torch.empty(layout.seq_len, self.hidden_size, dtype=dtype, device=device)
            video_offset = 0
            audio_offset = 0
            for start, end, kind in layout.segments:
                length = end - start
                if kind == "text":
                    h[start:end] = text_states
                elif kind in ("cond", "ref_img", "video"):
                    h[start:end] = video_embed[video_offset : video_offset + length]
                    video_offset += length
                else:
                    h[start:end] = audio_embed[audio_offset : audio_offset + length]
                    audio_offset += length

            t_vals = torch.tensor(unique_t, dtype=torch.float32, device=device)
            if self.use_adaln_curves:
                table = minimax_model.comfy.model_management.cast_to(
                    self.adaln_t_table, device=device
                )
                position = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
                index0 = position.floor().long().clamp(max=table.shape[0] - 2)
                t_emb = torch.lerp(
                    table[index0], table[index0 + 1], (position - index0).unsqueeze(1)
                )
            else:
                t_emb = self.time_embedder(t_vals).to(dtype)

            rope_freqs = minimax_model.rope_rotation_table(
                self.rope_freqs(layout.position_ids, device), dtype
            )
            patches_replace = transformer_options.get("patches_replace", {})
            blocks_replace = patches_replace.get("dit", {})
            cache_ranges = [
                (start, end)
                for start, end, kind in layout.segments
                if kind in ("audio", "video")
            ]
            if ("block_loop", 0) in blocks_replace:
                def block_loop_wrap(args: dict[str, Any]) -> dict[str, torch.Tensor]:
                    return {
                        "img": self._run_blocks(
                            args["img"],
                            args["t_emb"],
                            args["mod_segments"],
                            args["rope_freqs"],
                            args["transformer_options"],
                            args.get("start", 0),
                            args.get("end"),
                        )
                    }

                h = blocks_replace[("block_loop", 0)](
                    {
                        "img": h,
                        "t_emb": t_emb,
                        "timestep": timestep,
                        "step_info": sigma_v,
                        "mod_segments": mod_segments,
                        "rope_freqs": rope_freqs,
                        "transformer_options": transformer_options,
                        "cache_ranges": cache_ranges,
                        "block_count": len(self.blocks),
                    },
                    {"original_block": block_loop_wrap},
                )["img"]
            else:
                h = self._run_blocks(
                    h, t_emb, mod_segments, rope_freqs, transformer_options
                )

            video_seg = next(
                (start, end, t_row[seg_t["video"]])
                for start, end, kind in layout.segments
                if kind == "video"
            )
            audio_seg = next(
                (start, end, t_row[seg_t["audio"]])
                for start, end, kind in layout.segments
                if kind == "audio"
            )
            video_output, audio_output = self.final_layer(
                h, t_emb, video_seg, audio_seg
            )
            video_output = minimax_model.unpatchify_video(
                video_output,
                latent_t,
                lat_h // 2,
                lat_w // 2,
                self.latents_dim,
                self.patch_size,
            )
            video_output = video_output[:, :, :orig_t, :orig_h, :orig_w]
            audio_output = minimax_model.unpack_audio(audio_output)
            slope_a = minimax_model.time_shift_slope(
                sigma_v, shift_v, shift_a
            ).to(audio_output.dtype)
            return [
                -video_output.to(video_x.dtype),
                (-slope_a) * audio_output.to(audio_x.dtype),
            ]

        model_class._run_blocks = _run_blocks
        model_class._forward = patched_forward
        model_class._speed_minimax_h3_block_loop_version = PATCH_VERSION
        return "installed-runtime-hook"

__all__ = ["ensure_minimax_h3_block_loop_support"]
