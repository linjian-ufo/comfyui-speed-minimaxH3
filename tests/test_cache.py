from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "comfyui_speed_minimaxh3_test_target"
if PACKAGE_NAME not in sys.modules:
    package_spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    package_module = importlib.util.module_from_spec(package_spec)
    sys.modules[PACKAGE_NAME] = package_module
    package_spec.loader.exec_module(package_module)

from comfyui_speed_minimaxh3_test_target import nodes as nodes_module  # noqa: E402
from comfyui_speed_minimaxh3_test_target.nodes import (  # noqa: E402
    MiniMaxH3CacheController,
    MiniMaxH3SpeedCache,
)
from comfyui_speed_minimaxh3_test_target.text_encoder import (  # noqa: E402
    MiniMaxH3TextEncoderCache,
    token_fingerprint,
)


class CacheControllerTests(unittest.TestCase):
    def make_cache(self, **overrides):
        values = {
            "reuse_threshold": 10.0,
            "start_percent": 0.0,
            "end_percent": 1.0,
            "max_consecutive_skips": 1,
            "cache_device": "cpu",
            "vram_reserve_gb": 0.5,
            "ram_reserve_gb": 1.0,
            "signature_tokens": 32,
            "signature_features": 16,
            "verbose": False,
        }
        values.update(overrides)
        cache = MiniMaxH3CacheController(**values)
        cache.reset(total_steps=4)
        return cache

    @staticmethod
    def args(img, step):
        return {
            "img": img,
            "step_info": torch.tensor([step]),
            "cache_ranges": [(2, img.shape[0])],
            "block_count": 50,
            "transformer_options": {},
        }

    @staticmethod
    def original(args):
        return {"img": args["img"] + 2.0}

    def test_first_step_runs_and_second_step_can_skip(self):
        cache = self.make_cache()
        first = torch.ones(16, 32)
        result1 = cache(self.args(first, 0.9), {"original_block": self.original})
        self.assertTrue(torch.allclose(result1["img"], first + 2.0))
        second = first + 0.001
        result2 = cache(self.args(second, 0.8), {"original_block": self.original})
        self.assertTrue(torch.allclose(result2["img"], second + 2.0))
        self.assertEqual(cache.stats.full_steps, 1)
        self.assertEqual(cache.stats.skipped_steps, 1)

    def test_repeated_sigma_is_not_cross_cached(self):
        cache = self.make_cache()
        first = torch.ones(16, 32)
        cache(self.args(first, 0.9), {"original_block": self.original})

        calls = {"count": 0}

        def repeated_original(args):
            calls["count"] += 1
            return {"img": args["img"] + 3.0}

        result = cache(
            self.args(torch.ones(16, 32) * 7.0, 0.9),
            {"original_block": repeated_original},
        )
        self.assertEqual(calls["count"], 1)
        self.assertTrue(torch.allclose(result["img"], torch.ones(16, 32) * 10.0))
        self.assertEqual(cache.stats.repeated_calls, 1)

    def test_shape_change_invalidates_cached_residual(self):
        cache = self.make_cache()
        cache(self.args(torch.ones(16, 32), 0.9), {"original_block": self.original})
        changed = torch.ones(20, 32)
        cache(self.args(changed, 0.8), {"original_block": self.original})
        self.assertEqual(cache.stats.full_steps, 2)
        self.assertEqual(cache.stats.skipped_steps, 0)

    def test_threshold_zero_never_skips(self):
        cache = self.make_cache(reuse_threshold=0.0)
        cache(self.args(torch.ones(16, 32), 0.9), {"original_block": self.original})
        cache(self.args(torch.ones(16, 32), 0.8), {"original_block": self.original})
        self.assertEqual(cache.stats.full_steps, 2)
        self.assertEqual(cache.stats.skipped_steps, 0)


class NodePatchTests(unittest.TestCase):
    class FakeModel:
        def __init__(self):
            self.diffusion_model = type("MiniMaxH3Model", (), {})()
            self.model_options = {"transformer_options": {}}
            self.wrappers = {}

        def get_model_object(self, name):
            if name == "diffusion_model":
                return self.diffusion_model
            raise KeyError(name)

        def clone(self):
            clone = self.__class__()
            clone.diffusion_model = self.diffusion_model
            clone.model_options = {
                "transformer_options": dict(
                    self.model_options.get("transformer_options", {})
                )
            }
            return clone

        def set_model_patch_replace(self, patch, name, block_name, number):
            replacements = (
                self.model_options.setdefault("transformer_options", {})
                .setdefault("patches_replace", {})
                .setdefault(name, {})
            )
            replacements[(block_name, number)] = patch

        def remove_wrappers_with_key(self, wrapper_type, key):
            self.wrappers.pop((wrapper_type, key), None)

        def add_wrapper_with_key(self, wrapper_type, key, wrapper):
            self.wrappers[(wrapper_type, key)] = wrapper

    def test_node_defaults_match_recommended_profile(self):
        inputs = MiniMaxH3SpeedCache.INPUT_TYPES()
        required = inputs["required"]
        optional = inputs["optional"]

        self.assertEqual(required["reuse_threshold"][1]["default"], 0.12)
        self.assertEqual(required["start_percent"][1]["default"], 0.10)
        self.assertEqual(required["end_percent"][1]["default"], 0.90)
        self.assertEqual(required["max_consecutive_skips"][1]["default"], 2)
        self.assertEqual(required["cache_device"][1]["default"], "auto")
        self.assertEqual(optional["vram_reserve_gb"][1]["default"], 2.0)
        self.assertEqual(optional["ram_reserve_gb"][1]["default"], 4.0)
        self.assertEqual(optional["signature_tokens"][1]["default"], 128)
        self.assertEqual(optional["signature_features"][1]["default"], 64)
        self.assertIs(optional["verbose"][1]["default"], False)
        self.assertEqual(optional["sage_attention"][1]["default"], "auto")

    def test_invalid_shifted_sage_value_is_repaired_to_auto(self):
        original_ensure = nodes_module.ensure_minimax_h3_block_loop_support
        original_make_sage = nodes_module.make_sage_attention_override
        calls = []
        nodes_module.ensure_minimax_h3_block_loop_support = lambda: "test-hook"
        nodes_module.make_sage_attention_override = lambda required=False: (
            calls.append(required) or object(),
            "sage-enabled",
        )
        try:
            MiniMaxH3SpeedCache().patch(
                self.FakeModel(),
                reuse_threshold=0.12,
                start_percent=0.1,
                end_percent=0.9,
                max_consecutive_skips=2,
                cache_device="auto",
                sage_attention=2.0,
            )
        finally:
            nodes_module.ensure_minimax_h3_block_loop_support = original_ensure
            nodes_module.make_sage_attention_override = original_make_sage

        self.assertEqual(calls, [False])

    def test_new_sage_widget_is_appended_after_all_legacy_widgets(self):
        inputs = MiniMaxH3SpeedCache.INPUT_TYPES()
        self.assertEqual(
            list(inputs["required"]),
            [
                "model",
                "reuse_threshold",
                "start_percent",
                "end_percent",
                "max_consecutive_skips",
                "cache_device",
            ],
        )
        self.assertEqual(
            list(inputs["optional"]),
            [
                "vram_reserve_gb",
                "ram_reserve_gb",
                "signature_tokens",
                "signature_features",
                "verbose",
                "sage_attention",
            ],
        )

    def test_every_input_has_native_bilingual_hover_help(self):
        inputs = MiniMaxH3SpeedCache.INPUT_TYPES()
        input_names = set(inputs["required"]) | set(inputs["optional"])

        for group in ("required", "optional"):
            for name, definition in inputs[group].items():
                self.assertIn("tooltip", definition[1], name)
                self.assertTrue(definition[1]["tooltip"].strip(), name)

        for language in ("en", "zh"):
            locale_path = PACKAGE_ROOT / "locales" / language / "nodeDefs.json"
            with locale_path.open("r", encoding="utf-8") as handle:
                node_definition = json.load(handle)["MiniMaxH3SpeedCache"]

            self.assertEqual(set(node_definition["inputs"]), input_names)
            self.assertTrue(node_definition["display_name"].strip())
            self.assertTrue(node_definition["description"].strip())
            for name in input_names:
                localized = node_definition["inputs"][name]
                self.assertTrue(localized["name"].strip(), f"{language}:{name}:name")
                self.assertTrue(
                    localized["tooltip"].strip(),
                    f"{language}:{name}:tooltip",
                )

            for critical in (
                "reuse_threshold",
                "start_percent",
                "end_percent",
                "max_consecutive_skips",
                "sage_attention",
            ):
                self.assertTrue(
                    node_definition["inputs"][critical]["name"].startswith("★"),
                    f"{language}:{critical}:highlight",
                )

    def test_node_clones_model_and_registers_one_cache(self):
        original_ensure = nodes_module.ensure_minimax_h3_block_loop_support
        original_make_sage = nodes_module.make_sage_attention_override
        sage_override = object()
        nodes_module.ensure_minimax_h3_block_loop_support = lambda: "test-hook"
        nodes_module.make_sage_attention_override = lambda required=False: (
            sage_override,
            "sage-enabled",
        )
        try:
            original = self.FakeModel()
            patched = MiniMaxH3SpeedCache().patch(
                original,
                reuse_threshold=0.08,
                start_percent=0.2,
                end_percent=0.8,
                max_consecutive_skips=1,
                cache_device="auto",
            )[0]
        finally:
            nodes_module.ensure_minimax_h3_block_loop_support = original_ensure
            nodes_module.make_sage_attention_override = original_make_sage

        self.assertIsNot(patched, original)
        self.assertIs(
            patched.model_options["transformer_options"]["optimized_attention_override"],
            sage_override,
        )
        replacements = patched.model_options["transformer_options"]["patches_replace"]["dit"]
        self.assertIn(("block_loop", 0), replacements)
        self.assertEqual(len(patched.wrappers), 1)

    def test_auto_sage_preserves_an_external_attention_override(self):
        original_ensure = nodes_module.ensure_minimax_h3_block_loop_support
        original_make_sage = nodes_module.make_sage_attention_override
        existing_override = object()
        nodes_module.ensure_minimax_h3_block_loop_support = lambda: "test-hook"

        def unexpected_sage_call(required=False):
            raise AssertionError("auto must preserve an existing override")

        nodes_module.make_sage_attention_override = unexpected_sage_call
        try:
            original = self.FakeModel()
            original.model_options["transformer_options"] = {
                "optimized_attention_override": existing_override
            }
            patched = MiniMaxH3SpeedCache().patch(
                original,
                reuse_threshold=0.08,
                start_percent=0.2,
                end_percent=0.8,
                max_consecutive_skips=1,
                cache_device="auto",
            )[0]
        finally:
            nodes_module.ensure_minimax_h3_block_loop_support = original_ensure
            nodes_module.make_sage_attention_override = original_make_sage

        self.assertIs(
            patched.model_options["transformer_options"]["optimized_attention_override"],
            existing_override,
        )

    def test_node_rejects_an_existing_block_loop_cache(self):
        original_ensure = nodes_module.ensure_minimax_h3_block_loop_support
        nodes_module.ensure_minimax_h3_block_loop_support = lambda: "test-hook"
        try:
            original = self.FakeModel()
            original.model_options["transformer_options"] = {
                "patches_replace": {"dit": {("block_loop", 0): object()}}
            }
            with self.assertRaisesRegex(RuntimeError, "不要串联"):
                MiniMaxH3SpeedCache().patch(
                    original,
                    reuse_threshold=0.08,
                    start_percent=0.2,
                    end_percent=0.8,
                    max_consecutive_skips=1,
                    cache_device="auto",
                    sage_attention="disabled",
                )
        finally:
            nodes_module.ensure_minimax_h3_block_loop_support = original_ensure


class TextEncoderCacheTests(unittest.TestCase):
    class FakeClip:
        def __init__(self):
            self.cond_stage_model = type("MiniMaxH3TEModel_", (), {})()
            self.calls = 0

        def clone(self):
            clone = self.__class__()
            clone.cond_stage_model = self.cond_stage_model
            clone.calls = self.calls
            return clone

        def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
            self.calls += 1
            value = torch.tensor([float(self.calls)])
            if return_dict:
                return {"cond": value, "pooled_output": value + 1}
            if return_pooled:
                return value, value + 1
            return value

    def test_token_fingerprint_changes_with_tensor_content(self):
        first = token_fingerprint({"x": torch.tensor([1, 2])}, False, False)
        second = token_fingerprint({"x": torch.tensor([1, 3])}, False, False)
        self.assertNotEqual(first, second)

    def test_text_cache_defaults_and_locales_match(self):
        inputs = MiniMaxH3TextEncoderCache.INPUT_TYPES()
        input_names = set(inputs["required"]) | set(inputs["optional"])
        self.assertIs(inputs["required"]["cache_enabled"][1]["default"], True)
        self.assertEqual(inputs["required"]["max_cache_entries"][1]["default"], 2)

        for group in ("required", "optional"):
            for name, definition in inputs[group].items():
                self.assertTrue(definition[1]["tooltip"].strip(), name)

        for language in ("en", "zh"):
            locale_path = PACKAGE_ROOT / "locales" / language / "nodeDefs.json"
            with locale_path.open("r", encoding="utf-8") as handle:
                node_definition = json.load(handle)["MiniMaxH3TextEncoderCache"]
            self.assertEqual(set(node_definition["inputs"]), input_names)
            self.assertTrue(node_definition["display_name"].strip())
            self.assertTrue(node_definition["description"].strip())

    def test_repeated_tokens_reuse_text_encoder_result(self):
        patched = MiniMaxH3TextEncoderCache().patch(
            self.FakeClip(), cache_enabled=True, max_cache_entries=2
        )[0]
        tokens = {"qwen3vl_32b": [(1, 1.0), (2, 1.0)]}
        first = patched.encode_from_tokens(tokens)
        second = patched.encode_from_tokens(tokens)
        self.assertEqual(patched.calls, 1)
        self.assertTrue(torch.equal(first, second))

    def test_cached_dictionary_survives_scheduled_encoder_mutation(self):
        patched = MiniMaxH3TextEncoderCache().patch(
            self.FakeClip(), cache_enabled=True, max_cache_entries=2
        )[0]
        tokens = {"qwen3vl_32b": [(1, 1.0)]}
        first = patched.encode_from_tokens(tokens, return_dict=True)
        first.pop("cond")
        second = patched.encode_from_tokens(tokens, return_dict=True)
        second.pop("cond")
        third = patched.encode_from_tokens(tokens, return_dict=True)
        self.assertIn("cond", third)
        self.assertEqual(patched.calls, 1)

    def test_text_encoder_cache_rejects_other_clip_types(self):
        other = self.FakeClip()
        other.cond_stage_model = object()
        with self.assertRaisesRegex(ValueError, "仅支持 MiniMax H3"):
            MiniMaxH3TextEncoderCache().patch(
                other, cache_enabled=True, max_cache_entries=2
            )

if __name__ == "__main__":
    unittest.main()
