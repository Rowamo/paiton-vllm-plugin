import os
import unittest
from unittest import mock

from paiton_vllm_plugin.benchmarks.offline_benchmark import (
    apply_preset_defaults,
    build_parser,
    build_prompts,
    enable_vllm_plugin,
)


class OfflineBenchmarkPresetTests(unittest.TestCase):
    def test_deepseek_v4_flash_preset_sets_runtime_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--preset",
            "deepseek-v4-flash",
            "--num-prompts",
            "1",
        ])

        apply_preset_defaults(args)

        self.assertEqual(args.model, "deepseek-ai/DeepSeek-V4-Flash")
        self.assertEqual(
            args.compiled_model_dir,
            "/app/paiton-compiler/tmp/DeepSeek-V4-Flash",
        )
        self.assertEqual(args.kv_cache_dtype, "fp8")
        self.assertEqual(args.max_model_len, 4096)
        self.assertEqual(args.max_num_batched_tokens, 4096)
        self.assertEqual(
            build_prompts(args),
            ["Write one sentence about why compilers are useful."],
        )

    def test_prompt_overrides_preset_prompts(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--preset",
            "deepseek-v4-flash",
            "--prompt",
            "Say hello.",
            "--num-prompts",
            "1",
        ])

        apply_preset_defaults(args)

        self.assertEqual(build_prompts(args), ["Say hello."])

    def test_enable_vllm_plugin_preserves_existing_plugins(self) -> None:
        with mock.patch.dict("os.environ", {"VLLM_PLUGINS": "foo"}, clear=False):
            enable_vllm_plugin("register_paiton_models")
            enable_vllm_plugin("paiton_platform")

            self.assertEqual(
                os.environ["VLLM_PLUGINS"],
                "foo,register_paiton_models,paiton_platform",
            )


if __name__ == "__main__":
    unittest.main()
