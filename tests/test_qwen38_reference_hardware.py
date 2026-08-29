"""Opt-in upstream-vLLM reference gate for the exact Qwen3.8 checkpoint."""

import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    from tests.qwen38_capture import capture_completion
except ModuleNotFoundError:
    from qwen38_capture import capture_completion


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_REFERENCE") == "1",
    "set PAITON_RUN_QWEN38_REFERENCE=1 for the upstream gfx1201 reference",
)
class Qwen38ReferenceHardwareTest(unittest.TestCase):
    def test_upstream_quark_prefill_and_decode(self) -> None:
        import torch
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        self.assertEqual(
            os.environ.get("VLLM_PAITON_VANILLA_ROCM_PLATFORM"), "1"
        )
        metadata = Path(os.environ["PAITON_QWEN38_REFERENCE_METADATA"])
        checkpoint = Path(os.environ["PAITON_QWEN38_CHECKPOINT"])
        result_path = Path(os.environ["PAITON_QWEN38_RESULT_PATH"])
        self.assertTrue((metadata / "config.json").is_file())
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
            "gfx1201",
        )

        runtime_owner = tempfile.TemporaryDirectory(prefix="qwen38_reference_")
        runtime = Path(runtime_owner.name)
        for source in metadata.iterdir():
            if source.is_file():
                (runtime / source.name).symlink_to(source.resolve())
        (runtime / checkpoint.name).symlink_to(checkpoint.resolve())

        prompt = [151644, 8948, 198, 151645]
        llm = LLM(
            model=str(runtime),
            dtype="bfloat16",
            max_model_len=16,
            max_num_batched_tokens=6,
            max_num_seqs=1,
            block_size=16,
            kv_cache_memory_bytes=2 * 1024**3,
            enforce_eager=True,
            enable_prefix_caching=False,
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt)],
            SamplingParams(temperature=0.0, max_tokens=2, logprobs=20),
            use_tqdm=False,
        )
        self.assertEqual(len(outputs), 1)
        result = capture_completion(outputs[0], prompt)
        self.assertEqual(len(result["generated_token_ids"]), 2)
        self.assertEqual(len(result["top_logprobs"]), 2)
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("QWEN38_REFERENCE_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
