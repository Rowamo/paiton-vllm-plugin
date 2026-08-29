import unittest

import torch

from paiton_vllm_plugin.runtime.core.utils.qronos_w4a16 import (
    EXLLAMA_K_SHIFTS,
    QUARK_REORDER,
    transform_qronos_w4a16,
)


def pack_quark_reorder(signed_weight_kn: torch.Tensor) -> torch.Tensor:
    """Independent checkpoint fixture packer for signed logical [K, N]."""
    k, n = signed_weight_kn.shape
    assert n % 8 == 0
    values = (signed_weight_kn.to(torch.int32) & 0xF).reshape(k, n // 8, 8)
    packed = torch.zeros((k, n // 8), dtype=torch.int32)
    for logical_index, packed_index in enumerate(QUARK_REORDER):
        packed.bitwise_or_(values[:, :, logical_index] << (packed_index * 4))
    return packed


def unpack_exllama_signed(packed_weight_nk8: torch.Tensor) -> torch.Tensor:
    """Independent kernel-layout decoder returning signed logical [N, K]."""
    values = torch.stack(
        [(packed_weight_nk8 >> shift) & 0xF for shift in EXLLAMA_K_SHIFTS], dim=-1
    )
    return values.reshape(packed_weight_nk8.shape[0], -1).to(torch.int32) - 8


def pinned_vllm_oracle(signed_weight_kn: torch.Tensor) -> torch.Tensor:
    """Reference layout from vLLM commit 39bd959b, expressed independently."""
    # canonicalize_quark_packed_int4 applies XOR 8, AWQ conversion restores
    # logical [K,N], and RDNAHybridW4A16 packs transposed [N,K] in this order.
    biased_nk = (signed_weight_kn.t().contiguous().to(torch.int32) & 0xF) ^ 0x8
    n, k = biased_nk.shape
    groups = biased_nk.reshape(n, k // 8, 8)
    packed = torch.zeros((n, k // 8), dtype=torch.int32)
    for logical_index, shift in enumerate(EXLLAMA_K_SHIFTS):
        packed.bitwise_or_(groups[:, :, logical_index] << shift)
    return packed


class TestQronosW4A16Transform(unittest.TestCase):
    def _transform(self, signed_weight, scales, *, padded_output_size=None, chunk=256):
        k, n = signed_weight.shape
        return transform_qronos_w4a16(
            pack_quark_reorder(signed_weight),
            scales,
            torch.zeros((k // 128, n // 8), dtype=torch.int32),
            padded_output_size=padded_output_size,
            output_chunk_size=chunk,
        )

    def test_all_sixteen_nibbles_and_pinned_oracle(self):
        signed = torch.arange(-8, 8, dtype=torch.int32).repeat(128, 1)
        transformed = self._transform(signed, torch.ones((1, 16), dtype=torch.float32))

        torch.testing.assert_close(
            transformed.packed_weight, pinned_vllm_oracle(signed), rtol=0, atol=0
        )
        torch.testing.assert_close(
            unpack_exllama_signed(transformed.packed_weight),
            signed.t().contiguous(),
            rtol=0,
            atol=0,
        )

    def test_random_group128_dequantization_and_chunking(self):
        generator = torch.Generator().manual_seed(1201)
        k, n = 384, 264
        signed = torch.randint(-8, 8, (k, n), generator=generator, dtype=torch.int32)
        scales = torch.rand((k // 128, n), generator=generator, dtype=torch.float32)

        transformed_8 = self._transform(signed, scales, chunk=8)
        transformed_256 = self._transform(signed, scales, chunk=256)
        torch.testing.assert_close(
            transformed_8.packed_weight,
            transformed_256.packed_weight,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            transformed_8.packed_weight, pinned_vllm_oracle(signed), rtol=0, atol=0
        )

        kernel_signed = unpack_exllama_signed(transformed_8.packed_weight)
        kernel_dequant = kernel_signed * transformed_8.scales.repeat_interleave(128, dim=1)
        checkpoint_dequant = signed.t() * scales.t().repeat_interleave(128, dim=1)
        torch.testing.assert_close(kernel_dequant, checkpoint_dequant, rtol=0, atol=0)

    def test_output_padding_is_zero_scaled(self):
        signed = torch.arange(-8, 8, dtype=torch.int32).repeat(128, 1)
        transformed = self._transform(
            signed,
            torch.ones((1, 16), dtype=torch.float32),
            padded_output_size=24,
        )
        self.assertEqual(tuple(transformed.packed_weight.shape), (24, 16))
        self.assertEqual(tuple(transformed.scales.shape), (24, 1))
        self.assertEqual(torch.count_nonzero(transformed.scales[16:]).item(), 0)

    def test_nonzero_symmetric_zero_points_rejected(self):
        signed = torch.zeros((128, 8), dtype=torch.int32)
        zero_points = torch.zeros((1, 1), dtype=torch.int32)
        zero_points[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "must contain only zero"):
            transform_qronos_w4a16(
                pack_quark_reorder(signed),
                torch.ones((1, 8), dtype=torch.float32),
                zero_points,
            )

    def test_all_distinct_qwen38_projection_shapes(self):
        # (K, N) covers GDN, full-attention, MLP, and the unusually small
        # 48-output a/b projections. Tensors stay packed throughout this test.
        shapes = (
            (5120, 48),
            (5120, 1024),
            (5120, 5120),
            (5120, 6144),
            (6144, 5120),
            (5120, 10240),
            (5120, 12288),
            (5120, 17408),
            (17408, 5120),
        )
        for k, n in shapes:
            with self.subTest(k=k, n=n):
                checkpoint_weight = torch.zeros((k, n // 8), dtype=torch.int32)
                scales = torch.ones((k // 128, n), dtype=torch.float32)
                zero_points = torch.zeros((k // 128, n // 8), dtype=torch.int32)
                transformed = transform_qronos_w4a16(
                    checkpoint_weight, scales, zero_points
                )
                self.assertEqual(tuple(transformed.packed_weight.shape), (n, k // 8))
                self.assertEqual(tuple(transformed.scales.shape), (n, k // 128))
                self.assertEqual(
                    transformed.packed_weight.numel()
                    * transformed.packed_weight.element_size(),
                    checkpoint_weight.numel() * checkpoint_weight.element_size(),
                )

    def test_checkpoint_shape_dtype_and_scale_validation(self):
        signed = torch.zeros((128, 8), dtype=torch.int32)
        packed = pack_quark_reorder(signed)
        zeros = torch.zeros((1, 1), dtype=torch.int32)
        invalid = (
            (packed.to(torch.int64), torch.ones((1, 8)), zeros, "dtype int32"),
            (packed, torch.ones((1, 8), dtype=torch.bfloat16), zeros, "dtype float32"),
            (packed, torch.full((1, 8), float("nan")), zeros, "must be finite"),
            (packed, -torch.ones((1, 8)), zeros, "must be non-negative"),
            (packed, torch.ones((2, 8)), zeros, "scales shape"),
        )
        for weight, scales, zero_points, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                transform_qronos_w4a16(weight, scales, zero_points)


if __name__ == "__main__":
    unittest.main()
