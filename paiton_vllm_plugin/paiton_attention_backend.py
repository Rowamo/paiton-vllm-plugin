# SPDX-License-Identifier: Apache-2.0
"""
Paiton attention backend shim.

We don't actually *run* vLLM attention ops for Paiton-compiled models, but vLLM
still uses the selected attention backend to decide:
- KV cache tensor layout / shape
- metadata builder type

Paiton Qwen3.8 contract v3 consumes vLLM's page-first KV layout directly:
  (num_blocks, 2, block_size, num_kv_heads, head_size)

Keeping the physical block axis first is required by vLLM's hybrid cache
manager: GDN state pages can enlarge a scheduler block, and vLLM splits those
pages into contiguous attention-kernel blocks along this axis.
"""

from __future__ import annotations

from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


class PaitonTritonAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # The compiled artifact has a fixed 16-token physical block ABI.
        # Returning an exact size (rather than inheriting Triton's
        # MultipleOf(16)) tells vLLM to split a larger hybrid-manager page into
        # virtual 16-token blocks and expand its kernel block tables.
        return [16]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Contract v3 page-first physical layout:
        # (num_blocks, 2, block_size, num_kv_heads, head_size)
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # The overridden Paiton shape is five-dimensional, unlike Triton's
        # packed four-dimensional shape. It is already in the physical layout
        # consumed by the compiled kernels, so its stride permutation is the
        # identity. Preserve an outer layer dimension when vLLM asks for its
        # cross-layer form.
        rank = 6 if include_num_layers_dimension else 5
        return tuple(range(rank))
