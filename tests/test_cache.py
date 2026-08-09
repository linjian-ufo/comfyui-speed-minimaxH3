from __future__ import annotations

from contextlib import contextmanager
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
from comfyui_speed_minimaxh3_test_target import reference_to_video as reference_module  # noqa: E402
from comfyui_speed_minimaxh3_test_target import text_encoder as text_encoder_module  # noqa: E402
from comfyui_speed_minimaxh3_test_target.nodes import (  # noqa: E402
    MiniMaxH3CacheController,
    MiniMaxH3CacheRuntimeOptions,
    MiniMaxH3SpeedCache,
)
from comfyui_speed_minimaxh3_test_target.text_encoder import (  # noqa: E402
    MiniMaxH3TextEncoderCache,
    token_fingerprint,
)
from comfyui_speed_minimaxh3_test_target.reference_to_video import (  # noqa: E402
    LinjianMiniMaxH3ReferenceToVideo,
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

    def test_subgraph_runtime_options_update_existing_controller_only(self):
        original_ensure = nodes_module.ensure_minimax_h3_block_loop_support
        nodes_module.ensure_minimax_h3_block_loop_support = lambda: "test-hook"
        try:
            patched = MiniMaxH3SpeedCache().patch(
                self.FakeModel(),
                reuse_threshold=0.12,
                start_percent=0.1,
                end_percent=0.9,
                max_consecutive_skips=2,
                cache_device="gpu",
                sage_attention="disabled",
            )[0]
        finally:
            nodes_module.ensure_minimax_h3_block_loop_support = original_ensure

        controller = patched.model_options["transformer_options"]["patches_replace"][
            "dit"
        ][("block_loop", 0)]
        returned = MiniMaxH3CacheRuntimeOptions().apply(
            patched,
            cache_device="cpu",
            verbose=True,
        )[0]

        self.assertIs(returned, patched)
        self.assertIs(
            returned.model_options["transformer_options"]["patches_replace"]["dit"][
                ("block_loop", 0)
            ],
            controller,
        )
        self.assertEqual(controller.cache_device, "cpu")
        self.assertIs(controller.verbose, True)

    def test_subgraph_runtime_options_reject_raw_unet(self):
        with self.assertRaisesRegex(RuntimeError, "必须连接 MiniMax H3 Speed Cache"):
            MiniMaxH3CacheRuntimeOptions().apply(self.FakeModel())

    def test_subgraph_runtime_options_have_chinese_hover_help(self):
        inputs = MiniMaxH3CacheRuntimeOptions.INPUT_TYPES()["required"]
        self.assertEqual(set(inputs), {"model", "cache_device", "verbose"})
        for name, definition in inputs.items():
            tooltip = definition[1]["tooltip"]
            self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in tooltip), name)
            self.assertIn("默认", tooltip, name)


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
        self.assertEqual(inputs["optional"]["sage_attention"][1]["default"], "auto")

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

    def test_text_encoder_uses_qwenvl_sage_context(self):
        original_context = text_encoder_module.qwen_vl_sage_context
        calls = []

        @contextmanager
        def fake_context(*, required=False):
            calls.append(required)
            yield "test-qwenvl-sage"

        text_encoder_module.qwen_vl_sage_context = fake_context
        try:
            patched = MiniMaxH3TextEncoderCache().patch(
                self.FakeClip(),
                cache_enabled=False,
                max_cache_entries=2,
                sage_attention="enabled",
            )[0]
            patched.encode_from_tokens({"qwen3vl_32b": [(1, 1.0)]})
        finally:
            text_encoder_module.qwen_vl_sage_context = original_context

        self.assertEqual(calls, [True])

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


class SubgraphBundleTests(unittest.TestCase):
    def test_linjian_subgraph_replaces_unet_combo_with_model_socket(self):
        source = (PACKAGE_ROOT / "web" / "linjian_minimax_h3_subgraph.js").read_text(
            encoding="utf-8"
        )
        self.assertIn('const SUBGRAPH_NAME = "linjian Image to Video (MiniMax H3)"', source)
        self.assertIn('slot("unet_name", "MODEL", [229]', source)
        self.assertIn('target_id: 9, target_slot: 0, type: "MODEL"', source)
        self.assertIn('target_id: 16, target_slot: 0, type: "MODEL"', source)
        self.assertNotIn('"UNETLoader",', source)
        self.assertIn("subgraph.configure(definition)", source)
        self.assertIn('"MiniMaxH3TextEncoderCache"', source)
        self.assertIn('"sage_attention", "COMBO"', source)
        self.assertIn('origin_id: 112, origin_slot: 0, target_id: 104', source)
        self.assertIn('minimax-H3\\\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors', source)
        self.assertIn('minimax-H3\\\\minimax_h3_video_vae_fp16.safetensors', source)
        self.assertIn('minimax-H3\\\\minimax_h3_audio_vae_fp32.safetensors', source)
        self.assertIn("function createOuterNodeConfig", source)
        self.assertIn('{ name: "prompt", type: "STRING", widget: { name: "prompt" }', source)
        self.assertIn('{ name: "steps", type: "INT", widget: { name: "steps" }', source)
        self.assertIn('{ name: "unet_name", type: "MODEL", link: null, tooltip:', source)
        self.assertIn('{ name: "cache_device", type: "COMBO", widget: { name: "cache_device" }', source)
        self.assertIn('{ name: "verbose", type: "BOOLEAN", widget: { name: "verbose" }', source)
        self.assertIn('"MiniMaxH3CacheRuntimeOptions"', source)
        self.assertIn('input("steps", "INT", 232, true)', source)
        self.assertIn("function migrateLegacyImageNode", source)
        self.assertIn("function registerImageTooltipNodeDef", source)
        self.assertIn("app.updateVueAppNodeDefs(nodeDefs)", source)
        self.assertIn("replacement.configure(outerConfig)", source)
        self.assertIn("replacement.setSize?.([480, 690])", source)

    def test_linjian_image_locales_cover_every_outer_input(self):
        expected_inputs = {
            "first_frame",
            "last_frame",
            "prompt",
            "width",
            "height",
            "value_1",
            "steps",
            "noise_seed",
            "unet_name",
            "cache_device",
            "verbose",
            "clip_name",
            "vae_name",
            "vae_name_1",
        }
        for language in ("en", "zh"):
            locale_path = PACKAGE_ROOT / "locales" / language / "nodeDefs.json"
            with locale_path.open("r", encoding="utf-8") as handle:
                definition = json.load(handle)["LinjianMiniMaxH3ImageToVideo"]
            self.assertEqual(set(definition["inputs"]), expected_inputs)
            for input_definition in definition["inputs"].values():
                self.assertTrue(input_definition["name"].strip())
                self.assertTrue(input_definition["tooltip"].strip())


class ReferenceToVideoTests(unittest.TestCase):
    def test_schema_matches_requested_reference_layout(self):
        info = LinjianMiniMaxH3ReferenceToVideo.GET_NODE_INFO_V1()
        self.assertEqual(
            info["display_name"], "linjian Reference to Video (MiniMax H3)"
        )
        self.assertEqual(info["category"], "MiniMaxH3")
        self.assertEqual(
            info["input_order"]["required"],
            [
                "clip",
                "vae",
                "audio_vae",
                "prompt",
                "width",
                "height",
                "length",
                "ref_image_size",
            ],
        )
        self.assertEqual(
            info["input_order"]["optional"],
            [
                "ref_image_0",
                "ref_image_1",
                "ref_image_2",
                "ref_video_0",
                "ref_video_audio_0",
                "ref_audio_0",
            ],
        )
        optional = info["input"]["optional"]
        self.assertEqual(optional["ref_image_0"][0], "IMAGE")
        self.assertEqual(optional["ref_image_1"][0], "IMAGE")
        self.assertEqual(optional["ref_image_2"][0], "IMAGE")
        self.assertEqual(optional["ref_video_0"][0], "IMAGE")
        self.assertEqual(optional["ref_video_audio_0"][0], "AUDIO")
        self.assertEqual(optional["ref_audio_0"][0], "AUDIO")
        self.assertEqual(info["output"], ["CONDITIONING", "LATENT"])
        self.assertEqual(info["output_name"], ["positive", "Latent"])

    def test_execute_adds_internal_auto_qwenvl_sage_adapter(self):
        original_patch = reference_module.MiniMaxH3TextEncoderCache.patch
        original_execute = reference_module.MiniMaxH3ReferenceToVideo.__dict__["execute"]
        patched_clip = object()
        calls = []

        def fake_patch(
            _self,
            clip,
            cache_enabled,
            max_cache_entries,
            verbose=False,
            sage_attention="auto",
        ):
            calls.append(
                (clip, cache_enabled, max_cache_entries, verbose, sage_attention)
            )
            return (patched_clip,)

        @classmethod
        def fake_execute(cls, clip, *args):
            calls.append((cls, clip, args))
            return "reference-result"

        reference_module.MiniMaxH3TextEncoderCache.patch = fake_patch
        reference_module.MiniMaxH3ReferenceToVideo.execute = fake_execute
        try:
            clip = object()
            result = LinjianMiniMaxH3ReferenceToVideo.execute(
                clip,
                object(),
                object(),
                "prompt",
                1344,
                768,
                124,
            )
        finally:
            reference_module.MiniMaxH3TextEncoderCache.patch = original_patch
            reference_module.MiniMaxH3ReferenceToVideo.execute = original_execute

        self.assertEqual(result, "reference-result")
        self.assertEqual(calls[0], (clip, False, 2, False, "auto"))
        self.assertIs(calls[1][1], patched_clip)

    def test_reference_locales_cover_every_logical_input(self):
        expected_inputs = {
            "clip",
            "vae",
            "audio_vae",
            "prompt",
            "width",
            "height",
            "length",
            "ref_image_size",
            "ref_image_0",
            "ref_image_1",
            "ref_image_2",
            "ref_video_0",
            "ref_video_audio_0",
            "ref_audio_0",
        }
        for language in ("en", "zh"):
            locale_path = PACKAGE_ROOT / "locales" / language / "nodeDefs.json"
            with locale_path.open("r", encoding="utf-8") as handle:
                definition = json.load(handle)["LinjianMiniMaxH3ReferenceToVideo"]
            self.assertEqual(set(definition["inputs"]), expected_inputs)
            self.assertEqual(
                definition["display_name"],
                "linjian Reference to Video (MiniMax H3)",
            )
            for input_definition in definition["inputs"].values():
                self.assertTrue(input_definition["name"].strip())
                self.assertTrue(input_definition["tooltip"].strip())

    def test_reference_backend_tooltips_are_detailed_chinese(self):
        info = LinjianMiniMaxH3ReferenceToVideo.GET_NODE_INFO_V1()
        for group in ("required", "optional"):
            for name, definition in info["input"][group].items():
                tooltip = definition[1]["tooltip"]
                self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in tooltip), name)
                self.assertIn("默认", tooltip, name)

    def test_reference_frontend_uses_requested_tall_layout(self):
        source = (
            PACKAGE_ROOT / "web" / "linjian_minimax_h3_reference.js"
        ).read_text(encoding="utf-8")
        self.assertIn('const NODE_NAME = "LinjianMiniMaxH3ReferenceToVideo"', source)
        self.assertIn("this.setSize?.([365, 485])", source)

if __name__ == "__main__":
    unittest.main()
