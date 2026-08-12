import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from paiton_vllm_plugin.models.runtime_compat import (
    current_device_stream_ptr,
    normalize_paiton_kv_cache,
    run_with_current_stream,
)


class KVCacheCompatibilityTests(unittest.TestCase):
    BLOCK_SIZE = 16
    NUM_KV_HEADS = 8
    HEAD_SIZE = 128

    def make_cache(
        self,
        *,
        num_blocks: int = 40,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        return torch.empty(
            2,
            num_blocks,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            dtype=dtype,
        )

    def normalize(self, binding, **overrides) -> torch.Tensor:
        arguments = {
            "expected_dtype": torch.bfloat16,
            "expected_device": torch.device("cpu"),
            "block_size": self.BLOCK_SIZE,
            "num_kv_heads": self.NUM_KV_HEADS,
            "head_size": self.HEAD_SIZE,
            "max_seq_len": 513,
            "name": "kv_cache_0",
        }
        arguments.update(overrides)
        return normalize_paiton_kv_cache(binding, **arguments)

    def assert_same_view(self, expected: torch.Tensor, actual: torch.Tensor) -> None:
        self.assertIs(actual, expected)
        self.assertEqual(actual.data_ptr(), expected.data_ptr())
        self.assertEqual(actual.storage_offset(), expected.storage_offset())
        self.assertEqual(actual.stride(), expected.stride())
        self.assertEqual(
            actual.untyped_storage().data_ptr(),
            expected.untyped_storage().data_ptr(),
        )

    def test_installed_direct_5d_layout_is_preserved_zero_copy(self) -> None:
        cache = self.make_cache()
        self.assert_same_view(cache, self.normalize(cache))

    def test_legacy_single_virtual_engine_layout_is_preserved_zero_copy(self) -> None:
        cache = self.make_cache()
        self.assert_same_view(cache, self.normalize([cache]))

    def test_nonzero_storage_offset_is_preserved(self) -> None:
        shape = (2, 40, self.BLOCK_SIZE, self.NUM_KV_HEADS, self.HEAD_SIZE)
        elements = 1
        for dimension in shape:
            elements *= dimension
        backing = torch.empty(elements + 7, dtype=torch.bfloat16)
        strides = (
            40 * self.BLOCK_SIZE * self.NUM_KV_HEADS * self.HEAD_SIZE,
            self.BLOCK_SIZE * self.NUM_KV_HEADS * self.HEAD_SIZE,
            self.NUM_KV_HEADS * self.HEAD_SIZE,
            self.HEAD_SIZE,
            1,
        )
        cache = backing.as_strided(shape, strides, storage_offset=7)
        self.assert_same_view(cache, self.normalize(cache))

    def test_invalid_rank_fails_closed(self) -> None:
        cache = torch.empty(
            40,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "rank 5"):
            self.normalize(cache)

    def test_invalid_kv_plane_dimension_fails_closed(self) -> None:
        cache = torch.empty(
            1,
            40,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS,
            self.HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "plane of size 2"):
            self.normalize(cache)

    def test_invalid_tail_dimensions_fail_closed(self) -> None:
        cache = torch.empty(
            2,
            40,
            self.BLOCK_SIZE,
            self.NUM_KV_HEADS + 1,
            self.HEAD_SIZE,
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(ValueError, "unsupported KV-cache dimensions"):
            self.normalize(cache)

    def test_invalid_block_size_configuration_fails_closed(self) -> None:
        cache = self.make_cache()
        with self.assertRaisesRegex(ValueError, "positive multiple of 16"):
            self.normalize(cache, block_size=15)

    def test_invalid_stride_pattern_fails_closed(self) -> None:
        cache = torch.empty(
            2,
            40,
            self.BLOCK_SIZE,
            self.HEAD_SIZE,
            self.NUM_KV_HEADS,
            dtype=torch.bfloat16,
        ).transpose(-1, -2)
        self.assertEqual(
            cache.shape,
            (
                2,
                40,
                self.BLOCK_SIZE,
                self.NUM_KV_HEADS,
                self.HEAD_SIZE,
            ),
        )
        with self.assertRaisesRegex(ValueError, "unsupported KV-cache strides"):
            self.normalize(cache)

    def test_wrong_dtype_fails_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "implicit reinterpretation"):
            self.normalize(self.make_cache(dtype=torch.float16))

    def test_wrong_device_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected meta"):
            self.normalize(self.make_cache(), expected_device=torch.device("meta"))

    def test_unknown_binding_type_fails_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "unsupported KV-cache binding type"):
            self.normalize({"cache": self.make_cache()})

    def test_ambiguous_legacy_binding_fails_closed(self) -> None:
        cache = self.make_cache()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.normalize([cache, cache])

    def test_block_boundaries_15_16_and_17(self) -> None:
        cache = self.make_cache(num_blocks=2)
        for context_length in (15, 16, 17):
            with self.subTest(context_length=context_length):
                self.assert_same_view(
                    cache,
                    self.normalize(cache, max_seq_len=context_length),
                )

    def test_partition_boundary_511_512_and_context_513(self) -> None:
        cache = self.make_cache(num_blocks=33)
        for context_length in (511, 512, 513):
            with self.subTest(context_length=context_length):
                self.assert_same_view(
                    cache,
                    self.normalize(cache, max_seq_len=context_length),
                )

    def test_insufficient_blocks_fail_closed(self) -> None:
        cache = self.make_cache(num_blocks=32)
        with self.assertRaisesRegex(ValueError, "requires 33 blocks"):
            self.normalize(cache, max_seq_len=513)


class StreamCompatibilityTests(unittest.TestCase):
    class RecordingModel:
        def __init__(self) -> None:
            self.calls = []

        def run_with_tensors(self, inputs, outputs, **kwargs):
            self.calls.append((inputs, outputs, kwargs))
            return dict(outputs)

    @staticmethod
    def stream(handle: int, device: str = "cuda:0") -> SimpleNamespace:
        return SimpleNamespace(cuda_stream=handle, device=torch.device(device))

    def run_with_stream(self, stream, *, sync: bool):
        model = self.RecordingModel()
        inputs = {"input_ids": object()}
        outputs = {"logits": object()}
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "synchronize") as synchronize,
        ):
            result = run_with_current_stream(
                model,
                inputs,
                outputs,
                device=torch.device("cuda:0"),
                sync=sync,
            )
        synchronize.assert_not_called()
        return model, result

    def test_synchronous_execution_preserves_sync_flag(self) -> None:
        model, _ = self.run_with_stream(self.stream(91), sync=True)
        self.assertEqual(model.calls[0][2], {"stream_ptr": 91, "sync": True})

    def test_async_default_stream_is_explicitly_passed(self) -> None:
        model, _ = self.run_with_stream(self.stream(0), sync=False)
        call = model.calls[0][2]
        self.assertIsNotNone(call["stream_ptr"])
        self.assertEqual(call, {"stream_ptr": 0, "sync": False})

    def test_async_nondefault_stream_is_passed(self) -> None:
        model, _ = self.run_with_stream(self.stream(12345), sync=False)
        self.assertEqual(
            model.calls[0][2], {"stream_ptr": 12345, "sync": False}
        )

    def test_stream_is_queried_on_each_successive_request(self) -> None:
        model = self.RecordingModel()
        streams = [self.stream(11), self.stream(22), self.stream(11)]
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", side_effect=streams) as current,
        ):
            for _ in streams:
                run_with_current_stream(
                    model,
                    {},
                    {},
                    device="cuda:0",
                    sync=False,
                )
        self.assertEqual(current.call_count, 3)
        self.assertEqual(
            [call[2]["stream_ptr"] for call in model.calls], [11, 22, 11]
        )

    def test_wrong_active_device_is_rejected(self) -> None:
        with patch.object(torch.cuda, "current_device", return_value=1):
            with self.assertRaisesRegex(RuntimeError, "is not the active device"):
                current_device_stream_ptr("cuda:0")

    def test_wrong_device_stream_is_rejected(self) -> None:
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=self.stream(99, device="cuda:1"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "stream belongs to cuda:1"):
                current_device_stream_ptr("cuda:0")

    def test_missing_or_stale_stream_handle_is_rejected(self) -> None:
        stale = SimpleNamespace(device=torch.device("cuda:0"))
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "current_stream", return_value=stale),
        ):
            with self.assertRaisesRegex(RuntimeError, "no HIP stream handle"):
                current_device_stream_ptr("cuda:0")

    def test_stream_query_failure_is_clear(self) -> None:
        with (
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(
                torch.cuda,
                "current_stream",
                side_effect=RuntimeError("driver failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "unable to obtain"):
                current_device_stream_ptr("cuda:0")


if __name__ == "__main__":
    unittest.main()
