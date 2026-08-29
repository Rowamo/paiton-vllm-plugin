"""Strict Quark Qronos checkpoint-to-kernel W4A16 transformation.

The supported checkpoint stores packed signed INT4 weights as I32 [K, N/8]
using Quark's reorder convention, F32 scales as [K/128, N], and packed I32
zero-point metadata as [K/128, N/8]. Paiton's gfx12 kernel layout is
ExLlama-shuffled I32 [N_padded, K/8] with F32 scales [N_padded, K/128].

No BF16 weight expansion is created. Repacking uses bounded output chunks so
large projections do not also allocate a full unpacked I32 matrix.
"""

from dataclasses import dataclass

import torch


QUARK_REORDER = (0, 4, 1, 5, 2, 6, 3, 7)
EXLLAMA_K_SHIFTS = (0, 16, 4, 20, 8, 24, 12, 28)
GROUP_SIZE = 128
LAYOUT_VERSION = "paiton_w4a16_g128_v1"


@dataclass(frozen=True)
class QronosW4A16Weights:
    packed_weight: torch.Tensor
    scales: torch.Tensor
    input_size: int
    output_size: int
    padded_output_size: int
    group_size: int = GROUP_SIZE
    layout_version: str = LAYOUT_VERSION


def _require_cpu_contiguous(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be transformed on CPU, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _validate_checkpoint_tensors(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    group_size: int,
) -> tuple[int, int]:
    if group_size != GROUP_SIZE:
        raise ValueError(f"Qronos W4A16 requires group_size=128, got {group_size}")
    for tensor, name in (
        (packed_weight, "packed_weight"),
        (scales, "scales"),
        (packed_zero_points, "packed_zero_points"),
    ):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank 2, got shape {tuple(tensor.shape)}")
        _require_cpu_contiguous(tensor, name)
    if packed_weight.dtype is not torch.int32:
        raise ValueError(f"packed_weight must have dtype int32, got {packed_weight.dtype}")
    if scales.dtype is not torch.float32:
        raise ValueError(f"scales must have dtype float32, got {scales.dtype}")
    if packed_zero_points.dtype is not torch.int32:
        raise ValueError(
            f"packed_zero_points must have dtype int32, got {packed_zero_points.dtype}"
        )

    input_size, packed_output_size = packed_weight.shape
    output_size = scales.shape[1]
    if input_size % group_size != 0:
        raise ValueError(
            f"input size K={input_size} must be divisible by group_size={group_size}"
        )
    if input_size % 8 != 0:
        raise ValueError(f"input size K={input_size} must be divisible by pack factor 8")
    if output_size <= 0 or output_size > packed_output_size * 8:
        raise ValueError(
            f"scale output size N={output_size} is incompatible with packed capacity "
            f"{packed_output_size * 8}"
        )
    if packed_output_size != (output_size + 7) // 8:
        raise ValueError(
            f"packed output dimension must be ceil(N/8)={(output_size + 7) // 8}, "
            f"got {packed_output_size}"
        )
    expected_scale_shape = (input_size // group_size, output_size)
    if tuple(scales.shape) != expected_scale_shape:
        raise ValueError(
            f"scales shape must be {expected_scale_shape}, got {tuple(scales.shape)}"
        )
    expected_zero_shape = (input_size // group_size, packed_output_size)
    if tuple(packed_zero_points.shape) != expected_zero_shape:
        raise ValueError(
            "packed_zero_points shape must be "
            f"{expected_zero_shape}, got {tuple(packed_zero_points.shape)}"
        )
    if torch.count_nonzero(packed_zero_points).item() != 0:
        raise ValueError(
            "symmetric Qronos checkpoint zero-point metadata must contain only zero"
        )
    if not torch.isfinite(scales).all().item():
        raise ValueError("Qronos scales must be finite")
    if (scales < 0).any().item():
        raise ValueError("Qronos scales must be non-negative")
    return input_size, output_size


def repack_qronos_reorder_to_exllama(
    packed_weight: torch.Tensor,
    output_size: int,
    padded_output_size: int,
    *,
    output_chunk_size: int = 256,
) -> torch.Tensor:
    """Repack signed Quark output-packed I4 into biased ExLlama K-packing."""
    if output_chunk_size <= 0 or output_chunk_size % 8 != 0:
        raise ValueError("output_chunk_size must be a positive multiple of 8")
    if padded_output_size < output_size or padded_output_size % 8 != 0:
        raise ValueError(
            "padded_output_size must be at least output_size and divisible by 8"
        )

    input_size = packed_weight.shape[0]
    packed_kernel_weight = torch.zeros(
        (padded_output_size, input_size // 8), dtype=torch.int32, device="cpu"
    )
    source_shifts = tuple(index * 4 for index in QUARK_REORDER)

    for output_start in range(0, output_size, output_chunk_size):
        output_end = min(output_start + output_chunk_size, output_size)
        source_word_start = output_start // 8
        source_word_end = (output_end + 7) // 8
        source = packed_weight[:, source_word_start:source_word_end]

        # Quark symmetric I4 is two's-complement in the checkpoint. XOR 8
        # maps it to the uint4b8 representation expected by the kernel.
        logical = torch.stack(
            [((source >> shift) & 0xF) ^ 0x8 for shift in source_shifts], dim=-1
        ).reshape(input_size, -1)
        logical = logical[:, : output_end - output_start].t().contiguous()
        groups = logical.reshape(output_end - output_start, input_size // 8, 8)

        destination = torch.zeros(
            (output_end - output_start, input_size // 8), dtype=torch.int32
        )
        for value_index, shift in enumerate(EXLLAMA_K_SHIFTS):
            destination.bitwise_or_(groups[:, :, value_index] << shift)
        packed_kernel_weight[output_start:output_end].copy_(destination)

    return packed_kernel_weight


def transform_qronos_w4a16(
    packed_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    *,
    padded_output_size: int | None = None,
    group_size: int = GROUP_SIZE,
    output_chunk_size: int = 256,
) -> QronosW4A16Weights:
    """Validate and transform one checkpoint-native Qronos linear tensor set."""
    input_size, output_size = _validate_checkpoint_tensors(
        packed_weight, scales, packed_zero_points, group_size
    )
    if padded_output_size is None:
        padded_output_size = (output_size + 7) // 8 * 8
    packed_kernel_weight = repack_qronos_reorder_to_exllama(
        packed_weight,
        output_size,
        padded_output_size,
        output_chunk_size=output_chunk_size,
    )
    kernel_scales = torch.zeros(
        (padded_output_size, input_size // group_size), dtype=torch.float32
    )
    kernel_scales[:output_size].copy_(scales.t())
    return QronosW4A16Weights(
        packed_weight=packed_kernel_weight,
        scales=kernel_scales,
        input_size=input_size,
        output_size=output_size,
        padded_output_size=padded_output_size,
        group_size=group_size,
    )
