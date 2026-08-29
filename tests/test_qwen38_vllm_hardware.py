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
        self.assertTrue(model_path.is_dir())
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
            "gfx1201",
        )

        llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=16,
            max_num_batched_tokens=6,
            max_num_seqs=1,
            block_size=16,
            gpu_memory_utilization=0.80,
            enforce_eager=True,
            # Qualify contract v2's page-first physical attention cache first;
            # aligned prefix reuse remains a separate state-lifecycle gate.
            enable_prefix_caching=False,
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=[151644, 8948, 198, 151645])],
            SamplingParams(temperature=0.0, max_tokens=2),
            use_tqdm=False,
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs), 1)
        self.assertEqual(len(outputs[0].outputs[0].token_ids), 2)
        self.assertEqual(outputs[0].outputs[0].finish_reason, "length")


if __name__ == "__main__":
    unittest.main()
