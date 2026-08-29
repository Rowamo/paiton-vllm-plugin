"""Opt-in end-to-end vLLM scheduler gate for a reduced Qwen3.8 artifact."""

import os
from pathlib import Path
import unittest


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
        max_model_len = 1024 if prefix_caching else 16
        max_num_batched_tokens = 800 if prefix_caching else 6
        prompt_token_ids = (
            [151644, *([198] * 799)]
            if prefix_caching
            else [151644, 8948, 198, 151645]
        )
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
            max_num_seqs=1,
            block_size=16,
            gpu_memory_utilization=0.80,
            enforce_eager=True,
            enable_prefix_caching=prefix_caching,
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            SamplingParams(temperature=0.0, max_tokens=2),
            use_tqdm=False,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 2)
        self.assertEqual(outputs[0].outputs[0].finish_reason, "length")
        if prefix_caching:
            repeated = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt_token_ids)],
                SamplingParams(temperature=0.0, max_tokens=2),
                use_tqdm=False,
            )
            self.assertEqual(
                repeated[0].outputs[0].token_ids,
                outputs[0].outputs[0].token_ids,
            )
            self.assertEqual(outputs[0].num_cached_tokens, 0)
            self.assertEqual(repeated[0].num_cached_tokens, 784)


if __name__ == "__main__":
    unittest.main()
