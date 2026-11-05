"""CPU parity checks against a caller-supplied, unmodified ComfyUI model.py.

Run with COMFY_MINIMAX_SOURCE=/path/to/ComfyUI/comfy/ldm/minimax/model.py.
The native packing, timestep/modulation, final head (including PDD), and output
code run with small random weights. Attention blocks and ComfyUI memory/wrapper
services are lightweight doubles. This is not an end-to-end GPU generation test.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from test_compatibility import installed_modules, minimax_patch, module_tree

SOURCE = os.environ.get("COMFY_MINIMAX_SOURCE")


@unittest.skipUnless(SOURCE, "Set COMFY_MINIMAX_SOURCE for upstream source parity tests")
class UpstreamCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.modules = module_tree([
            "comfy", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model",
            "comfy.ldm.common_dit", "comfy.model_prefetch", "comfy.model_management",
            "comfy.ops", "comfy.patcher_extension",
        ])
        self.enterContext(installed_modules(self.modules))
        comfy = self.modules["comfy"]
        comfy.ldm.common_dit.pad_to_patch_size = lambda x, size: torch.nn.functional.pad(
            x, (0, -x.shape[-1] % size[-1], 0, -x.shape[-2] % size[-2], 0, -x.shape[-3] % size[-3])
        )
        comfy.model_management.cast_to = lambda x, device=None: x.to(device)
        comfy.model_prefetch.make_prefetch_queue = lambda *args: None
        comfy.model_prefetch.prefetch_queue_pop = lambda *args, **kwargs: None
        comfy.model_prefetch.malloc_graph_enabled = lambda device: False
        comfy.patcher_extension.WrappersMP = SimpleNamespace(DIFFUSION_MODEL="diffusion")
        comfy.patcher_extension.get_all_wrappers = lambda *args: []
        comfy.patcher_extension.WrapperExecutor = SimpleNamespace(
            new_class_executor=lambda forward, *args: SimpleNamespace(execute=forward)
        )
        @contextmanager
        def weights(head, h, **kwargs):
            yield head.weight, head.bias
        comfy.ops.CastBiasWeightContext = weights
        self.backend = self.modules["comfy.ldm.minimax.model"]
        self.backend.__dict__.update(
            torch=torch, nn=torch.nn, math=math, comfy=comfy,
            ComfyAttention=lambda: None,
        )
        # Keep all source definitions unchanged; only imports are substituted to
        # avoid loading ComfyUI's GPU, server, and optional attention dependencies.
        tree = ast.parse(Path(SOURCE).read_text(encoding="utf-8"), filename=SOURCE)
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(tree, SOURCE, "exec"), self.backend.__dict__)
        self.native_forward = self.backend.MiniMaxH3Model._forward
        self.has_masks = hasattr(self.backend, "mask_row_values")
        self.has_pdd = "sample_sigmas" in self.backend.FinalLayer.forward.__code__.co_varnames
        torch.manual_seed(1234)
        self.model = self.backend.MiniMaxH3Model(
            hidden_size=8, num_layers=0, token_refiner_num_layers=0,
            num_attention_heads=2, attention_head_dim=4, ffn_hidden_size=16,
            latents_dim=1, audio_latents_dim=4, text_dim=8,
            timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4,
            rope_inv_freq_len=1, dtype=torch.float32, operations=torch.nn,
        )
        self.model.rope.inv_freq.fill_(0.1)
        class Block(torch.nn.Module):
            def forward(self, h, t_emb, mod_segments, rope_freqs, transformer_options, attention=None):
                h = h.clone()
                for a, b, row in mod_segments:
                    values = t_emb[row // 3].mean(dim=-1)
                    if values.ndim:
                        values = values.unsqueeze(-1)
                    h[a:b] += values * 0.1
                return h
        self.model.blocks = torch.nn.ModuleList([Block(), Block()])
        self.video = torch.randn(1, 1, 1, 2, 4)
        self.audio = torch.randn(1, 4, 2, 2)
        self.context = torch.randn(1, 2, 8)

    def assert_parity(self, *, payload=None, extra=None, pdd=False, cached=False, outer=False, curves=False):
        if pdd:
            if not self.has_pdd:
                self.skipTest("This ComfyUI version predates PDD final heads")
            for head in (self.model.final_layer.video_out, self.model.final_layer.audio_out):
                head.weight = torch.nn.Parameter(head.weight.repeat(3, 1))
                head.bias = torch.nn.Parameter(head.bias.repeat(3))
        if curves:
            self.model.use_adaln_curves = True
            self.model.adaln_t_table = torch.randn(5, 4)
        payload = payload or {}
        extra = extra or {}
        options = {
            "minimax_h3_sigma_shift_video": 8.0,
            "minimax_h3_sigma_shift_audio": 2.0,
            "sample_sigmas": torch.tensor([1.0, 0.5, 0.25, 0.0]),
        }
        x = [self.video, self.audio]
        timestep = torch.tensor([500.0])
        with torch.no_grad():
            if outer:
                expected = self.model.forward(x, timestep, self.context, dict(options), payload, **extra)
            else:
                expected = self.native_forward(self.model, x, timestep, self.context, dict(options), payload, **extra)
            self.assertEqual(minimax_patch.ensure_minimax_h3_block_loop_support(), "installed-runtime-hook")
            calls = []
            if cached:
                def block_loop(args, extras):
                    calls.append(args)
                    # Exercise partial-block entry points used by cache plugins.
                    mid = extras["original_block"]({**args, "end": 1})["img"]
                    return extras["original_block"]({**args, "img": mid, "start": 1})
                options["patches_replace"] = {"dit": {("block_loop", 0): block_loop}}
            forward = self.model.forward if outer else self.model._forward
            actual = forward(x, timestep, self.context, options, payload, **extra)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
            self.assertEqual(got.dtype, want.dtype)
        if cached:
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["block_count"], 2)

    def test_plain_forward(self):
        self.assert_parity()

    def test_block_loop_and_partial_block_execution(self):
        self.assert_parity(cached=True)

    def test_adaln_curve_interpolation(self):
        self.assert_parity(curves=True, cached=True)

    def test_outer_audio_schedule_conversion(self):
        self.assert_parity(payload={"audio_scale": 0.5}, outer=True, cached=True)

    def test_non_patch_aligned_video_is_cropped(self):
        self.video = self.video[..., :3]
        self.assert_parity(cached=True)

    def test_text_modality_tags(self):
        self.assert_parity(payload={"text_token_tags": torch.tensor([0, 1])}, cached=True)

    def test_pdd_final_heads(self):
        self.assert_parity(pdd=True, cached=True)

    def test_keyframe_audio_and_video_conditioning(self):
        if not self.has_masks:
            self.skipTest("This ComfyUI version predates keyframe audio")
        video = torch.randn_like(self.video)
        audio = torch.randn_like(self.audio)
        self.assert_parity(payload={
            "keyframes": [{"resolved_frame_index": 0, "latent": video, "audio_latent": audio}],
            "cond_video_latents": [video], "cond_audio_latents": [audio],
        }, cached=True)

    def test_all_one_masks(self):
        if not self.has_masks:
            self.skipTest("This ComfyUI version predates denoise masks")
        self.assert_parity(extra={
            "denoise_mask": torch.ones(1, 1, 1, 2, 4),
            "audio_denoise_mask": torch.ones(1, 1, 2, 2),
        }, cached=True, outer=True)

    def test_uniform_masks(self):
        if not self.has_masks:
            self.skipTest("This ComfyUI version predates denoise masks")
        self.assert_parity(extra={
            "denoise_mask": torch.full((1, 1, 1, 2, 4), 0.5),
            "audio_denoise_mask": torch.zeros(1, 1, 2, 2),
        }, cached=True, outer=True)

    def test_mixed_per_row_masks(self):
        if not self.has_masks:
            self.skipTest("This ComfyUI version predates denoise masks")
        self.assert_parity(extra={
            "denoise_mask": torch.tensor([[[[[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]]]]]),
            "audio_denoise_mask": torch.tensor([[[[0.0, 0.25], [0.5, 1.0]]]]),
        }, cached=True, outer=True)


if __name__ == "__main__":
    unittest.main()
