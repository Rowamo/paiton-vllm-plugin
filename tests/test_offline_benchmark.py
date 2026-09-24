import os
import unittest
from unittest import mock

from paiton_vllm_plugin.benchmarks.offline_benchmark import (
    apply_preset_defaults,
    build_parser,
    build_prompts,
    enable_vllm_plugin,
    summarize_measurements,
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
        self.assertEqual(args.max_model_len, 8192)
        self.assertEqual(args.max_num_batched_tokens, 512)
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

    def test_benchmark_defaults_are_deterministic(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.top_p, 1.0)
        self.assertTrue(args.ignore_eos)

    def test_summarize_measurements_uses_all_iterations(self) -> None:
        def make_output(token_count: int):
            output = mock.Mock()
            output.outputs = [mock.Mock(token_ids=list(range(token_count)))]
            return output

        summary = summarize_measurements(
            [
                [make_output(3), make_output(2)],
                [make_output(4)],
            ],
            [2.0, 1.0],
        )

        self.assertEqual(summary["per_iter_generated_tokens"], [5, 4])
        self.assertEqual(summary["generated_tokens"], 9)
        self.assertEqual(summary["avg_generated_tokens"], 4.5)
        self.assertEqual(summary["avg_latency_s"], 1.5)
        self.assertEqual(summary["generated_toks_per_s"], 3.0)


if __name__ == "__main__":
    unittest.main()
