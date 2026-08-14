import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from paiton_vllm_plugin.models.paiton_deepseek_v4 import (
    PaitonDeepseekV4ForCausalLM,
)


class _FakeRuntimeModel:
    def get_output_name_to_index_map(self):
        return {
            "logits": 0,
            "topk_ids_layer_3": 1,
            "topk_ids_layer_7": 2,
        }

    def get_output_maximum_shape(self, name):
        if name.startswith("topk_ids_layer_"):
            return [8, 2]
        return [1, 16]


def _make_capture_model(
    flush_path: str,
    *,
    ring_max: int = 2,
    max_tokens: int = 4,
    capture_steps: int = 3,
    current_rank: int = 0,
    capture_rank: int = 0,
):
    model = PaitonDeepseekV4ForCausalLM.__new__(PaitonDeepseekV4ForCausalLM)
    model.model = _FakeRuntimeModel()
    model.model_name = "test-artifact"
    model._moe_topk_capture = None
    model._moe_topk_ring = None
    model._moe_topk_metadata = []
    model._moe_topk_ring_idx = 0
    model._moe_topk_step = 0
    model._moe_topk_recorded = 0
    model._moe_topk_chunk_idx = 0
    model._moe_topk_done = False
    model._moe_topk_flush_path = flush_path
    model._moe_topk_ring_max = ring_max
    model._moe_topk_max_tokens = max_tokens
    model._moe_topk_capture_steps = capture_steps
    model._moe_topk_current_rank = current_rank
    model._moe_topk_capture_rank = capture_rank
    return model


class MoeTopkCaptureTests(unittest.TestCase):
    def test_persistent_output_and_ring_pointers_are_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _make_capture_model(str(Path(tmp) / "routes.pt"))
            self.assertTrue(model._init_moe_topk_capture(torch.device("cpu")))
            output_ptr = model._moe_topk_buffer.data_ptr()
            ring_ptr = model._moe_topk_ring.data_ptr()

            first = {"logits": object()}
            second = {"logits": object()}
            with mock.patch.object(
                torch, "empty", side_effect=AssertionError("unexpected allocation")
            ):
                model._bind_moe_topk_outputs(first, 2)
                model._bind_moe_topk_outputs(second, 1)

                model._moe_topk_buffer.fill_(11)
                model._record_moe_topk_capture(
                    2, 0, {"graph_mode": True, "use_bound": True}
                )

            self.assertEqual(model._moe_topk_buffer.data_ptr(), output_ptr)
            self.assertEqual(model._moe_topk_ring.data_ptr(), ring_ptr)
            for name in ("topk_ids_layer_3", "topk_ids_layer_7"):
                self.assertEqual(first[name].data_ptr, second[name].data_ptr)

            self.assertEqual(model._moe_topk_ring.data_ptr(), ring_ptr)
            self.assertEqual(model._moe_topk_ring_idx, 1)
            self.assertEqual(model._moe_topk_metadata[0]["metadata"]["graph_mode"], True)
            self.assertFalse(any(Path(tmp).iterdir()))

    def test_full_and_final_partial_chunks_are_unique_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "routes.pt"
            model = _make_capture_model(str(base))
            model._init_moe_topk_capture(torch.device("cpu"))

            model._moe_topk_buffer.fill_(10)
            model._record_moe_topk_capture(2, 0)
            model._moe_topk_buffer.fill_(20)
            model._record_moe_topk_capture(1, 1)

            chunk0 = Path(tmp) / "routes.chunk00000.pt"
            self.assertTrue(chunk0.is_file())
            payload0 = torch.load(chunk0, weights_only=False)
            self.assertEqual(payload0["topk_ids"].shape, (2, 2, 4, 2))
            self.assertEqual([e["num_tokens"] for e in payload0["entries"]], [2, 1])
            self.assertTrue(bool((payload0["topk_ids"][0, :, :2] == 10).all()))
            self.assertTrue(bool((payload0["topk_ids"][1, :, :1] == 20).all()))

            model._moe_topk_buffer.fill_(30)
            model._record_moe_topk_capture(1, 2)
            chunk1 = Path(tmp) / "routes.chunk00001.pt"
            self.assertTrue(chunk1.is_file())
            payload1 = torch.load(chunk1, weights_only=False)
            self.assertEqual(len(payload1["entries"]), 1)
            self.assertTrue(bool((payload1["topk_ids"][0, :, :1] == 30).all()))
            self.assertTrue(model._moe_topk_done)

            model._record_moe_topk_capture(1, 3)
            self.assertEqual(sorted(Path(tmp).glob("*.pt")), [chunk0, chunk1])

    def test_explicit_flush_writes_partial_ring(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _make_capture_model(
                str(Path(tmp) / "partial.pt"), ring_max=4, capture_steps=10
            )
            model._init_moe_topk_capture(torch.device("cpu"))
            model._moe_topk_buffer.fill_(5)
            model._record_moe_topk_capture(1, 0)
            path = model._flush_moe_topk_capture()
            self.assertEqual(path, str(Path(tmp) / "partial.chunk00000.pt"))
            self.assertEqual(model._moe_topk_ring_idx, 0)
            self.assertEqual(model._moe_topk_metadata, [])

    def test_capture_rank_is_independent_of_current_rank(self):
        with tempfile.TemporaryDirectory() as tmp:
            selected = _make_capture_model(
                str(Path(tmp) / "rank1.pt"), current_rank=1, capture_rank=1
            )
            selected._init_moe_topk_capture(torch.device("cpu"))
            self.assertIsNotNone(selected._moe_topk_ring)
            selected._record_moe_topk_capture(1, 0)
            self.assertEqual(selected._moe_topk_ring_idx, 1)

            other = _make_capture_model(
                str(Path(tmp) / "rank0.pt"), current_rank=0, capture_rank=1
            )
            other._init_moe_topk_capture(torch.device("cpu"))
            self.assertIsNone(other._moe_topk_ring)
            other._record_moe_topk_capture(1, 0)
            self.assertEqual(other._moe_topk_ring_idx, 0)

    def test_prefill_larger_than_capture_bound_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = _make_capture_model(str(Path(tmp) / "routes.pt"), max_tokens=4)
            model._init_moe_topk_capture(torch.device("cpu"))
            model._record_moe_topk_capture(5, 0)
            self.assertEqual(model._moe_topk_ring_idx, 0)
            self.assertEqual(model._moe_topk_recorded, 0)

    def test_positive_capture_environment_validation(self):
        for raw in ("0", "-1", "bad"):
            with mock.patch.dict(
                "os.environ", {"PAITON_MOE_TOPK_RING_MAX": raw}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    PaitonDeepseekV4ForCausalLM._positive_capture_env(
                        "PAITON_MOE_TOPK_RING_MAX", 256
                    )


if __name__ == "__main__":
    unittest.main()
