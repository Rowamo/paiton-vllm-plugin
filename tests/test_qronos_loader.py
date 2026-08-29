import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from paiton_vllm_plugin.runtime.core.utils.qronos_loader import (
    QronosLinearSpec,
    QronosParallelism,
    QronosStreamingTransformer,
)


def tensors(k, n, offset=0):
    weight = torch.full((k, n // 8), offset, dtype=torch.int32)
    scale = torch.arange(k // 128 * n, dtype=torch.float32).reshape(
        k // 128, n
    ) + offset
    zero = torch.zeros((k // 128, n // 8), dtype=torch.int32)
    return weight, scale, zero


def consume_triple(transformer, spec, values, order=("weight", "scale", "zero")):
    suffix = {
        "weight": ".weight",
        "scale": ".weight_scale",
        "zero": ".weight_zero_point",
    }
    mapping = dict(zip(("weight", "scale", "zero"), values))
    result = None
    for component in order:
        value = transformer.consume(
            spec.source_prefix + suffix[component], mapping[component]
        )
        if value is not None:
            result = value
    return result


class TestQronosStreamingTransformer(unittest.TestCase):
    def test_rejects_ambiguous_target_constant_names(self):
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            QronosStreamingTransformer((
                QronosLinearSpec("layer.q", "same", "same", 128, 8),
            ))
        with self.assertRaisesRegex(ValueError, "duplicate Qronos target"):
            QronosStreamingTransformer((
                QronosLinearSpec("layer.q", "shared", "q_scale", 128, 8),
                QronosLinearSpec("layer.k", "k_weight", "shared", 128, 8),
            ))

    def test_finalizes_each_linear_without_model_dictionary(self):
        specs = (
            QronosLinearSpec("layer.0.q", "l0_q_weight", "l0_q_scale", 256, 64),
            QronosLinearSpec("layer.0.k", "l0_k_weight", "l0_k_scale", 256, 32),
        )
        loader = QronosStreamingTransformer(specs, max_pending_linears=1)
        first = consume_triple(loader, specs[0], tensors(256, 64))
        self.assertIsNotNone(first)
        self.assertEqual(loader.pending_bytes, 0)
        self.assertEqual([name for name, _ in first.constants()], [
            "l0_q_weight", "l0_q_scale"
        ])
        second = consume_triple(
            loader, specs[1], tensors(256, 32), order=("zero", "weight", "scale")
        )
        self.assertIsNotNone(second)
        loader.finish()
        self.assertEqual(loader.peak_pending_linears, 1)

    def test_rejects_non_streaming_tensor_order(self):
        specs = tuple(
            QronosLinearSpec(f"layer.{i}.q", f"w{i}", f"s{i}", 128, 8)
            for i in range(3)
        )
        loader = QronosStreamingTransformer(specs, max_pending_linears=2)
        for spec in specs[:2]:
            self.assertIsNone(
                loader.consume(spec.source_prefix + ".weight", tensors(128, 8)[0])
            )
        with self.assertRaisesRegex(MemoryError, "streaming window"):
            loader.consume(specs[2].source_prefix + ".weight", tensors(128, 8)[0])

    def test_random_access_ignores_physical_order_and_fetches_one_triple(self):
        specs = (
            QronosLinearSpec("layer.0.q", "w0", "s0", 128, 16),
            QronosLinearSpec("layer.1.q", "w1", "s1", 128, 16),
        )
        values = {}
        for index, spec in enumerate(specs):
            weight, scale, zero = tensors(128, 16, offset=index)
            values[spec.source_prefix + ".weight_scale"] = scale
            values[spec.source_prefix + ".weight"] = weight
            values[spec.source_prefix + ".weight_zero_point"] = zero

        class Source:
            def __init__(self):
                self.accesses = []

            def keys(self):
                # Deliberately grouped like the pinned checkpoint data order.
                return [
                    *(spec.source_prefix + ".weight_scale" for spec in specs),
                    *(spec.source_prefix + ".weight" for spec in specs),
                    *(spec.source_prefix + ".weight_zero_point" for spec in specs),
                ]

            def get_tensor(self, name):
                self.accesses.append(name)
                return values[name]

        source = Source()
        loader = QronosStreamingTransformer(specs, max_pending_linears=1)
        results = list(loader.iter_from_random_access_source(source))
        loader.finish()
        self.assertEqual(len(results), 2)
        self.assertEqual(
            source.accesses,
            [
                "layer.0.q.weight",
                "layer.0.q.weight_scale",
                "layer.0.q.weight_zero_point",
                "layer.1.q.weight",
                "layer.1.q.weight_scale",
                "layer.1.q.weight_zero_point",
            ],
        )
        self.assertEqual(loader.peak_pending_linears, 1)

    def test_random_access_rejects_undeclared_quantized_linear(self):
        spec = QronosLinearSpec("layer.q", "w", "s", 128, 8)
        weight, scale, zero = tensors(128, 8)
        source = {
            "layer.q.weight": weight,
            "layer.q.weight_scale": scale,
            "layer.q.weight_zero_point": zero,
            "layer.extra.weight_scale": scale,
        }
        loader = QronosStreamingTransformer((spec,))
        with self.assertRaisesRegex(ValueError, "undeclared quantized"):
            list(loader.iter_from_random_access_source(source))

    def test_safetensors_random_access_path(self):
        spec = QronosLinearSpec("layer.q", "w", "s", 128, 16)
        weight, scale, zero = tensors(128, 16)
        with tempfile.TemporaryDirectory(prefix="paiton_qronos_source_") as root:
            path = Path(root) / "model.safetensors"
            save_file(
                {
                    "layer.q.weight_scale": scale,
                    "layer.q.weight": weight,
                    "layer.q.weight_zero_point": zero,
                },
                path,
            )
            loader = QronosStreamingTransformer((spec,), max_pending_linears=1)
            results = list(loader.iter_safetensors(path))
            loader.finish()
        self.assertEqual(len(results), 1)
        self.assertEqual(tuple(results[0].weights.packed_weight.shape), (16, 16))
        self.assertEqual(loader.peak_pending_linears, 1)

    def test_column_parallel_shards_packed_n_before_transform(self):
        spec = QronosLinearSpec(
            "layer.q", "q_weight", "q_scale", 256, 64, QronosParallelism.COLUMN
        )
        source = tensors(256, 64)
        rank1 = QronosStreamingTransformer((spec,), tp_rank=1, tp_size=2)
        result = consume_triple(rank1, spec, source)
        rank1.finish()
        self.assertEqual(tuple(result.weights.packed_weight.shape), (32, 32))
        self.assertEqual(tuple(result.weights.scales.shape), (32, 2))
        torch.testing.assert_close(result.weights.scales, source[1][:, 32:].t())

    def test_row_parallel_shards_group_aligned_k(self):
        spec = QronosLinearSpec(
            "layer.o", "o_weight", "o_scale", 256, 16, QronosParallelism.ROW
        )
        source = tensors(256, 16)
        rank1 = QronosStreamingTransformer((spec,), tp_rank=1, tp_size=2)
        result = consume_triple(rank1, spec, source)
        rank1.finish()
        self.assertEqual(tuple(result.weights.packed_weight.shape), (16, 16))
        self.assertEqual(tuple(result.weights.scales.shape), (16, 1))
        torch.testing.assert_close(result.weights.scales[:, 0], source[1][1])

    def test_exact_shape_dtype_duplicate_and_missing_checks(self):
        spec = QronosLinearSpec("layer.q", "q_weight", "q_scale", 256, 64)
        loader = QronosStreamingTransformer((spec,))
        with self.assertRaisesRegex(ValueError, "shape must be exactly"):
            loader.consume(
                "layer.q.weight",
                torch.zeros((32, 256), dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "dtype must be"):
            loader.consume(
                "layer.q.weight",
                torch.zeros((256, 8), dtype=torch.int64),
            )
        loader.consume("layer.q.weight", tensors(256, 64)[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            loader.consume("layer.q.weight", tensors(256, 64)[0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            loader.finish()

    def test_pending_byte_limit_is_enforced(self):
        spec = QronosLinearSpec("layer.q", "q_weight", "q_scale", 256, 64)
        loader = QronosStreamingTransformer((spec,), max_pending_bytes=1024)
        with self.assertRaisesRegex(MemoryError, "memory contract"):
            loader.consume("layer.q.weight", tensors(256, 64)[0])


if __name__ == "__main__":
    unittest.main()
