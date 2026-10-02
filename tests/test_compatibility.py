"""CPU regression tests; no ComfyUI installation or model weights required."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"minimax_test_{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attention_adapter = load_module("attention")
minimax_patch = load_module("minimax_patch")


def module_tree(names):
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        if "." in name:
            parent, child = name.rsplit(".", 1)
            setattr(modules[parent], child, module)
    return modules


@contextmanager
def installed_modules(modules):
    # Restore only our stubs: torch may lazily import modules during a test.
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class AttentionCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.modules = module_tree([
            "comfy", "comfy.ldm", "comfy.ldm.modules",
            "comfy.ldm.modules.attention", "comfy.ops", "sageattention",
        ])
        self.native = Mock(return_value=object())
        self.sage = Mock(side_effect=lambda q, k, v, **kwargs: q.clone())
        self.modules["comfy.ldm.modules.attention"].attention_pytorch = self.native
        self.modules["sageattention"].sageattn = self.sage
        self.repeat = Mock(side_effect=lambda k, v, *args: (k, v))
        self.modules["comfy.ops"].repeat_kv_for_gqa = self.repeat
        self.enterContext(installed_modules(self.modules))

    def test_unsupported_or_mixed_dtypes_use_native_without_warnings(self):
        for dtypes in [
            (torch.float32,) * 3,
            (torch.float32, torch.float32, torch.float16),
            (torch.float16, torch.float32, torch.float16),
            (torch.float16, torch.float16, torch.float32),
            (torch.bfloat16, torch.float16, torch.bfloat16),
        ]:
            with self.subTest(dtypes=dtypes):
                q, k, v = [torch.ones(1, 2, 3, 4, dtype=d) for d in dtypes]
                mask = torch.zeros(3, 3)
                with self.assertNoLogs(attention_adapter.LOGGER, level="WARNING"):
                    result = attention_adapter.attention_sage_qwenvl(
                        q, k, v, 2, mask=mask, skip_reshape=True,
                        skip_output_reshape=True, enable_gqa=True, scale=0.5,
                    )
                self.assertIs(result, self.native.return_value)
                args, kwargs = self.native.call_args
                for actual, expected in zip(args[:3], (q, k, v)):
                    dtype = torch.promote_types(torch.promote_types(q.dtype, k.dtype), v.dtype)
                    torch.testing.assert_close(actual, expected.to(dtype))
                self.assertIs(kwargs["mask"], mask)
                self.assertTrue(kwargs["skip_reshape"])
                self.assertTrue(kwargs["skip_output_reshape"])
                self.assertTrue(kwargs["enable_gqa"])
                self.assertEqual(kwargs["scale"], 0.5)
        self.sage.assert_not_called()
        self.repeat.assert_not_called()

    def test_mixed_vision_inputs_run_real_pytorch_sdpa(self):
        def native(q, k, v, heads, **kwargs):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        self.native.side_effect = native
        for value_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(value_dtype=value_dtype):
                q = torch.randn(1, 2, 3, 4, dtype=torch.float32)
                k = torch.randn_like(q)
                v = torch.randn(q.shape, dtype=value_dtype)
                result = attention_adapter.attention_sage_qwenvl(
                    q, k, v, 2, skip_reshape=True, skip_output_reshape=True,
                )
                expected = torch.nn.functional.scaled_dot_product_attention(q, k, v.float())
                torch.testing.assert_close(result, expected)
        self.sage.assert_not_called()

    def test_matching_half_dtypes_still_use_sage_and_reshape(self):
        for dtype in (torch.float16, torch.bfloat16):
            for keep_heads in (False, True):
                with self.subTest(dtype=dtype, keep_heads=keep_heads):
                    q = torch.ones(1, 2, 3, 4, dtype=dtype)
                    result = attention_adapter.attention_sage_qwenvl(
                        q, q, q, 2, skip_reshape=True,
                        skip_output_reshape=keep_heads,
                    )
                    self.assertEqual(result.shape, (1, 2, 3, 4) if keep_heads else (1, 3, 8))
                    self.assertEqual(result.dtype, dtype)
        self.assertEqual(self.sage.call_count, 4)
        self.native.assert_not_called()

    def test_kernel_failure_still_falls_back(self):
        self.sage.side_effect = RuntimeError("unsupported kernel")
        q = torch.ones(1, 2, 3, 4, dtype=torch.float16)
        with self.assertLogs(attention_adapter.LOGGER, level="WARNING"):
            result = attention_adapter.attention_sage_qwenvl(q, q, q, 2, skip_reshape=True)
        self.assertIs(result, self.native.return_value)

    def test_unreshaped_inputs_still_use_native(self):
        q = torch.ones(1, 3, 8, dtype=torch.float16)
        attention_adapter.attention_sage_qwenvl(q, q, q, 2)
        self.sage.assert_not_called()
        self.assertFalse(self.native.call_args.kwargs["skip_reshape"])


class MiniMaxCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.modules = module_tree([
            "comfy", "comfy.ldm", "comfy.ldm.minimax", "comfy.ldm.minimax.model",
            "comfy.ldm.common_dit", "comfy.model_prefetch",
        ])
        self.backend = self.modules["comfy.ldm.minimax.model"]
        self.backend.comfy = self.modules["comfy"]
        self.backend.VISUAL_COND_TIMESTEP = 0.99
        self.backend.AUDIO_COND_TIMESTEP = 0.99
        self.backend.time_shift_sigma = lambda sigma, v, a: a * sigma / (v + (a - v) * sigma)
        self.backend.patchify_video = lambda x, size: x.reshape(1, 4)
        self.backend.pack_audio = lambda x: x.reshape(1, 4)
        self.backend.unpatchify_video = lambda x, *args: x.reshape(1, 1, 1, 2, 2)
        self.backend.unpack_audio = lambda x: x.reshape(1, 4, 1, 1)
        self.backend.rope_rotation_table = lambda x, dtype: x.to(dtype)
        self.modules["comfy.ldm.common_dit"].pad_to_patch_size = lambda x, size: x
        self.modules["comfy.model_prefetch"].make_prefetch_queue = lambda *args: None
        self.modules["comfy.model_prefetch"].prefetch_queue_pop = lambda *args: None
        self.layout = SimpleNamespace(
            signature=(1, 1, 2, 2, 1), seq_len=3,
            segments=[(0, 1, "text"), (1, 2, "audio"), (2, 3, "video")],
            img_update=torch.tensor([True]), audio_update=torch.tensor([True]),
            position_ids=torch.zeros(3, 3),
        )
        self.backend.MiniMaxH3Model = type("MiniMaxH3Model", (), {
            "patch_size": (1, 2, 2), "hidden_size": 4, "latents_dim": 1,
            "sigma_shift_video": 12.0, "sigma_shift_audio": 3.0,
            "use_adaln_curves": False, "blocks": [],
            "_cond_video_rows": lambda self, *args: None,
            "_cond_audio_rows": lambda self, *args: None,
            "video_patch_proj": lambda self, x: x,
            "audio_patch_proj": lambda self, x: x,
            "time_embedder": lambda self, x: x.unsqueeze(1),
            "rope_freqs": lambda self, x, device: x,
            "final_layer": lambda self, h, *args: (h[2:3], h[1:2]),
        })
        self.enterContext(installed_modules(self.modules))

    def run_forward(self, use_cache=False):
        self.assertEqual(minimax_patch.ensure_minimax_h3_block_loop_support(), "installed-runtime-hook")
        self.assertEqual(minimax_patch.ensure_minimax_h3_block_loop_support(), "existing-compatible-hook")
        video = torch.full((1, 1, 1, 2, 2), 2.0, dtype=torch.float16)
        audio = torch.full((1, 4, 1, 1), 3.0, dtype=torch.bfloat16)
        options = {"minimax_h3_sigma_shift_video": 8.0, "minimax_h3_sigma_shift_audio": 2.0}
        cache = Mock(side_effect=lambda args, extra: extra["original_block"](args))
        if use_cache:
            options["patches_replace"] = {"dit": {("block_loop", 0): cache}}
        result = self.backend.MiniMaxH3Model()._forward(
            [video, audio], torch.tensor([500.0]), torch.ones(1, 1, 4),
            transformer_options=options, minimax_payload={"layout": self.layout},
        )
        self.assertTrue(torch.equal(result[0], -video))
        self.assertEqual(result[0].dtype, video.dtype)
        if use_cache:
            cache.assert_called_once()
        return result[1], audio

    def test_legacy_audio_keeps_derivative_scaling(self):
        slope = Mock(return_value=torch.tensor(0.64))
        self.backend.time_shift_slope = slope
        actual, audio = self.run_forward()
        # Match the original operation order and dtype promotion exactly.
        expected = -torch.tensor(0.64) * audio
        torch.testing.assert_close(actual, expected)
        sigma, video_shift, audio_shift = slope.call_args.args
        self.assertAlmostEqual(float(sigma), 0.5)
        self.assertEqual((video_shift, audio_shift), (8.0, 2.0))

    def test_new_audio_is_unscaled_without_cache(self):
        actual, audio = self.run_forward()
        torch.testing.assert_close(actual, -audio)
        self.assertEqual(actual.dtype, audio.dtype)

    def test_new_audio_is_unscaled_with_cache(self):
        actual, audio = self.run_forward(use_cache=True)
        torch.testing.assert_close(actual, -audio)
        self.assertEqual(actual.dtype, audio.dtype)

    def test_existing_native_hook_is_not_replaced(self):
        class NativeModel:
            _is_minimax_patched = True
            def _run_blocks(self):
                pass
            def _forward(self):
                pass
        original = NativeModel._forward
        self.backend.MiniMaxH3Model = NativeModel
        self.assertEqual(minimax_patch.ensure_minimax_h3_block_loop_support(), "existing-compatible-hook")
        self.assertIs(NativeModel._forward, original)


if __name__ == "__main__":
    unittest.main()
