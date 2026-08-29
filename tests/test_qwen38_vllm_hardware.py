"""Opt-in end-to-end vLLM scheduler gates for Qwen3.8 artifacts."""

import json
import os
from pathlib import Path
import unittest

try:
    from tests.qwen38_capture import (
        assert_finite_completion,
        assert_reference_completion,
        capture_completion,
    )
except ModuleNotFoundError:
    from qwen38_capture import (
        assert_finite_completion,
        assert_reference_completion,
        capture_completion,
    )


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_VLLM") == "1",
    "set PAITON_RUN_QWEN38_VLLM=1 for the gfx1201 vLLM gate",
)
class Qwen38VllmHardwareTest(unittest.TestCase):
    def test_scheduler_prefill_and_decode(self) -> None:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        model_path = Path(os.environ["PAITON_QWEN38_VLLM_MODEL"])
        prefix_caching = os.environ.get("PAITON_QWEN38_VLLM_PREFIX") == "1"
        batch_mode = os.environ.get("PAITON_QWEN38_VLLM_BATCH") == "1"
        full_mode = os.environ.get("PAITON_QWEN38_VLLM_FULL") == "1"
        self.assertLessEqual(sum((prefix_caching, batch_mode, full_mode)), 1)
        max_model_len = 8192 if full_mode else (1024 if prefix_caching else 16)
        max_num_batched_tokens = (
            8192
            if full_mode
            else (800 if prefix_caching else (12 if batch_mode else 6))
        )
        if full_mode:
            # Two greedy decode steps bring this request to the exact qualified
            # 8,192-token context without ever probing beyond the contract.
            prompts = [[151644, *([198] * 8189)]]
        elif prefix_caching:
            prompts = [[151644, *([198] * 799)]]
        else:
            prompts = [
                [151644, 8948, 198, 151645],
                *([[151644, 9707, 198, 151645]] if batch_mode else []),
            ]
        self.assertTrue(model_path.is_dir())
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
            "gfx1201",
        )

        llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=2 if batch_mode else 1,
            block_size=16,
            gpu_memory_utilization=0.80,
            kv_cache_memory_bytes=(
                2 * 1024**3
                if full_mode or os.environ.get("PAITON_QWEN38_RESULT_PATH")
                else None
            ),
            enforce_eager=True,
            enable_prefix_caching=prefix_caching,
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts],
            SamplingParams(
                temperature=0.0,
                max_tokens=2,
                logprobs=(20 if os.environ.get("PAITON_QWEN38_RESULT_PATH") else None),
            ),
            use_tqdm=False,
        )

        self.assertEqual(len(outputs), len(prompts))
        if full_mode:
            self.assertEqual(len(prompts[0]) + 2, 8192)
        for output in outputs:
            self.assertEqual(len(output.outputs), 1)
            self.assertEqual(len(output.outputs[0].token_ids), 2)
            self.assertEqual(output.outputs[0].finish_reason, "length")
        result_path = os.environ.get("PAITON_QWEN38_RESULT_PATH")
        if result_path:
            self.assertEqual(len(outputs), 1)
            result = capture_completion(outputs[0], prompts[0])
            Path(result_path).write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            printable_result = result
            if len(prompts[0]) > 64:
                printable_result = {
                    **result,
                    "prompt_token_ids": {
                        "length": len(prompts[0]),
                        "head": prompts[0][:4],
                        "tail": prompts[0][-4:],
                    },
                }
            print(
                "QWEN38_PAITON_RESULT="
                + json.dumps(printable_result, sort_keys=True)
            )
            assert_finite_completion(result)
            reference_path = os.environ.get("PAITON_QWEN38_REFERENCE_PATH")
            if reference_path:
                reference = json.loads(
                    Path(reference_path).read_text(encoding="utf-8")
                )
                assert_reference_completion(result, reference)
        if prefix_caching:
            repeated = llm.generate(
                [TokensPrompt(prompt_token_ids=prompts[0])],
                SamplingParams(temperature=0.0, max_tokens=2),
                use_tqdm=False,
            )
            self.assertEqual(
                repeated[0].outputs[0].token_ids,
                outputs[0].outputs[0].token_ids,
            )
            self.assertEqual(outputs[0].num_cached_tokens, 0)
            self.assertEqual(repeated[0].num_cached_tokens, 784)
        elif batch_mode:
            for prompt, expected in zip(prompts, outputs, strict=True):
                repeated = llm.generate(
                    [TokensPrompt(prompt_token_ids=prompt)],
                    SamplingParams(temperature=0.0, max_tokens=2),
                    use_tqdm=False,
                )
                self.assertEqual(
                    repeated[0].outputs[0].token_ids,
                    expected.outputs[0].token_ids,
                )


if __name__ == "__main__":
    unittest.main()
