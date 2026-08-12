"""Runtime-boundary validation for Paiton model wrappers.

The helpers in this module intentionally know only about contracts shared by
vLLM and the generated Paiton runtime.  They do not key behavior on a vLLM
version string: supported layouts are recognized by their type, rank,
dimensions, dtype, device, storage bounds, and strides.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for dimension in reversed(shape):
        result.append(stride)
        stride *= dimension
    return tuple(reversed(result))


def normalize_paiton_kv_cache(
    binding: Any,
    *,
    expected_dtype: torch.dtype,
    expected_device: torch.device | str,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    max_seq_len: int,
    name: str,
) -> torch.Tensor:
    """Return an evidenced, zero-copy Paiton K/V cache tensor.

    Supported bindings are:

    * current vLLM: the complete five-dimensional tensor is bound directly;
    * legacy vLLM: a single-virtual-engine sequence contains that tensor.

    The Paiton attention backend defines the semantic layout as
    ``(2, num_blocks, block_size, num_kv_heads, head_size)``.  No shape is
    reconstructed from an element count and no copy or dtype conversion is
    performed.
    """

    source: str
    if isinstance(binding, torch.Tensor):
        tensor = binding
        source = "direct"
    elif isinstance(binding, Sequence) and not isinstance(
        binding, (str, bytes, bytearray)
    ):
        if len(binding) != 1:
            raise ValueError(
                f"{name}: legacy KV-cache binding must contain exactly one "
                f"virtual-engine tensor, got {len(binding)}"
            )
        tensor = binding[0]
        source = "legacy"
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name}: legacy KV-cache entry must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
    else:
        raise TypeError(
            f"{name}: unsupported KV-cache binding type "
            f"{type(binding).__name__}; expected a tensor or a "
            "single-virtual-engine sequence"
        )

    expected_device = torch.device(expected_device)
    if tensor.device != expected_device:
        raise ValueError(
            f"{name}: KV cache is on {tensor.device}, expected {expected_device}"
        )
    if tensor.dtype != expected_dtype:
        raise TypeError(
            f"{name}: KV cache has dtype {tensor.dtype}, expected "
            f"{expected_dtype}; implicit reinterpretation is not supported"
        )
    if tensor.ndim != 5:
        raise ValueError(
            f"{name}: {source} KV cache must have rank 5, got rank "
            f"{tensor.ndim} with shape {tuple(tensor.shape)}"
        )

    shape = tuple(tensor.shape)
    if shape[0] != 2:
        raise ValueError(
            f"{name}: KV cache dimension 0 must be the K/V plane of size 2, "
            f"got shape {shape}"
        )
    if shape[1] <= 0:
        raise ValueError(f"{name}: KV cache must contain at least one block")
    if not isinstance(block_size, int) or block_size <= 0 or block_size % 16:
        raise ValueError(
            f"{name}: configured block size must be a positive multiple of 16, "
            f"got {block_size!r}"
        )
    expected_tail = (block_size, num_kv_heads, head_size)
    if shape[2:] != expected_tail:
        raise ValueError(
            f"{name}: unsupported KV-cache dimensions {shape}; expected "
            f"(2, num_blocks, {block_size}, {num_kv_heads}, {head_size})"
        )
    if not isinstance(max_seq_len, int) or max_seq_len <= 0:
        raise ValueError(
            f"{name}: max_seq_len must be a positive integer, got "
            f"{max_seq_len!r}"
        )
    required_blocks = (max_seq_len + block_size - 1) // block_size
    if required_blocks > shape[1]:
        raise ValueError(
            f"{name}: context length {max_seq_len} requires "
            f"{required_blocks} blocks, but the cache has {shape[1]}"
        )

    expected_strides = _contiguous_strides(shape)
    actual_strides = tuple(tensor.stride())
    if actual_strides != expected_strides:
        raise ValueError(
            f"{name}: unsupported KV-cache strides {actual_strides}; "
            f"expected plane-major contiguous strides {expected_strides}"
        )

    element_size = tensor.element_size()
    offset_bytes = tensor.storage_offset() * element_size
    required_bytes = tensor.numel() * element_size
    storage_bytes = tensor.untyped_storage().nbytes()
    if offset_bytes < 0 or offset_bytes + required_bytes > storage_bytes:
        raise ValueError(
            f"{name}: KV-cache view exceeds its storage: offset={offset_bytes}, "
            f"required={required_bytes}, storage={storage_bytes}"
        )

    return tensor


def current_device_stream_ptr(device: torch.device | str) -> int:
    """Query and validate the active PyTorch/HIP stream for ``device``.

    The query is deliberately performed for every invocation.  Handle value
    zero is valid for PyTorch's current legacy/default HIP stream; it remains
    distinct from omitting the stream argument at the Python boundary.
    """

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(
            f"Paiton execution requires a CUDA/HIP device, got {device}"
        )

    try:
        active_index = int(torch.cuda.current_device())
    except Exception as error:
        raise RuntimeError("unable to determine the active HIP device") from error

    expected_index = active_index if device.index is None else device.index
    if expected_index != active_index:
        raise RuntimeError(
            f"Paiton input device cuda:{expected_index} is not the active "
            f"device cuda:{active_index}"
        )

    try:
        stream = torch.cuda.current_stream(device=torch.device("cuda", active_index))
    except Exception as error:
        raise RuntimeError(
            f"unable to obtain the current HIP stream for cuda:{active_index}"
        ) from error

    stream_device = torch.device(stream.device)
    stream_index = active_index if stream_device.index is None else stream_device.index
    if stream_device.type != "cuda" or stream_index != active_index:
        raise RuntimeError(
            f"current stream belongs to {stream_device}, expected cuda:{active_index}"
        )

    handle = getattr(stream, "cuda_stream", None)
    if handle is None:
        raise RuntimeError(
            f"current stream for cuda:{active_index} has no HIP stream handle"
        )
    try:
        handle = int(handle)
    except (TypeError, ValueError, OverflowError) as error:
        raise RuntimeError(
            f"current stream for cuda:{active_index} has an invalid HIP handle"
        ) from error
    if handle < 0:
        raise RuntimeError(
            f"current stream for cuda:{active_index} has a negative HIP handle"
        )
    return handle


def run_with_current_stream(
    model: Any,
    inputs: Mapping[str, torch.Tensor],
    outputs: Mapping[str, torch.Tensor],
    *,
    device: torch.device | str,
    sync: bool,
) -> dict[str, torch.Tensor]:
    """Invoke a Paiton model on the caller's freshly queried HIP stream."""

    stream_ptr = current_device_stream_ptr(device)
    return model.run_with_tensors(
        inputs,
        outputs,
        stream_ptr=stream_ptr,
        sync=sync,
    )


__all__ = [
    "current_device_stream_ptr",
    "normalize_paiton_kv_cache",
    "run_with_current_stream",
]
