"""Full Qwen3.8 image/video vLLM parity gate on gfx1201."""

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


def deterministic_image(size: int, frame: int = 0):
    import numpy as np
    from PIL import Image

    y, x = np.mgrid[:size, :size]
    pixels = np.stack(
        (
            (x + 17 * frame) % 256,
            (y * 3 + 29 * frame) % 256,
            ((x + y) * 5 + 11 * frame) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_MULTIMODAL_VLLM") == "1",
    "set PAITON_RUN_QWEN38_MULTIMODAL_VLLM=1 for the full gfx1201 gate",
)
class Qwen38MultimodalVllmHardwareTest(unittest.TestCase):
    def test_image_or_video_prefill_and_decode(self):
        import torch
        from vllm import LLM, SamplingParams

        model_path = Path(os.environ["PAITON_QWEN38_MULTIMODAL_MODEL"])
        modality = os.environ.get("PAITON_QWEN38_MULTIMODAL_MODALITY", "image")
        result_path = Path(os.environ["PAITON_QWEN38_RESULT_PATH"])
        self.assertIn(modality, ("image", "video"))
        self.assertTrue(model_path.is_dir())
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1201"
        )

        if modality == "image":
            media = deterministic_image(256)
            placeholder = "<|image_pad|>"
            question = "Describe the dominant colors in this image briefly."
        else:
            import numpy as np

            frames = np.stack(
                [np.asarray(deterministic_image(64, frame)) for frame in range(4)]
            )
            media = (
                frames,
                {
                    "fps": 2.0,
                    "duration": 2.0,
                    "total_num_frames": 4,
                    "frames_indices": [0, 1, 2, 3],
                    "video_backend": "opencv",
                    "do_sample_frames": False,
                },
            )
            placeholder = "<|video_pad|>"
            question = "Describe how the colors change in this video briefly."
        prompt = (
            "<|im_start|>user\n<|vision_start|>"
            + placeholder
            + "<|vision_end|>"
            + question
            + "<|im_end|>\n<|im_start|>assistant\n"
        )

        llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=1,
            block_size=16,
            kv_cache_memory_bytes=2 * 1024**3,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {modality: media},
                }
            ],
            SamplingParams(temperature=0.0, max_tokens=2, logprobs=20),
            use_tqdm=False,
        )
        self.assertEqual(len(outputs), 1)
        prompt_ids = list(outputs[0].prompt_token_ids)
        result = capture_completion(outputs[0], prompt_ids)
        result["modality"] = modality
        result["prompt_text"] = prompt
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("QWEN38_MULTIMODAL_RESULT=" + json.dumps(result, sort_keys=True))
        self.assertEqual(len(result["generated_token_ids"]), 2)
        assert_finite_completion(result)
        reference_path = os.environ.get("PAITON_QWEN38_REFERENCE_PATH")
        if reference_path:
            reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
            self.assertEqual(reference["modality"], modality)
            self.assertEqual(reference["prompt_text"], prompt)
            assert_reference_completion(result, reference)


if __name__ == "__main__":
    unittest.main()
