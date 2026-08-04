# SPDX-License-Identifier: Apache-2.0
"""
Paiton attention backend shim.

We don't actually *run* vLLM attention ops for Paiton-compiled models, but vLLM
still uses the selected attention backend to decide:
- KV cache tensor layout / shape
- metadata builder type

Most Paiton compiled kernels in this repo expect KV cache layout:
  (2, num_blocks, block_size, num_kv_heads, head_size)

Kimi K3 is a hybrid attention/Mamba model and must use blocks-first pages:
  (num_blocks, 2, block_size, num_kv_heads, head_size)

These backends subclass Triton's backend and override both shape and stride
order.  Triton's native cache became rank four, so inheriting its stride order
for Paiton's rank-five tensors trips vLLM's cache initialization assertion.
"""

from __future__ import annotations

from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend


class PaitonTritonAttentionBackend(TritonAttentionBackend):
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Match the compiled Paiton model's expected layout:
        # (2, num_blocks, block_size, num_kv_heads, head_size)
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # The logical K/V-first shape is also physically contiguous.  Do not
        # inherit Triton's rank-four stride permutation.
        rank = 6 if include_num_layers_dimension else 5
        return tuple(range(rank))


class PaitonKimiK3AttentionBackend(PaitonTritonAttentionBackend):
    """Blocks-first latent MLA cache backend for hybrid Kimi K3.

    vLLM requires a block ID to identify the same contiguous page in every
    cache group of a hybrid attention/Mamba model.  Keeping the block dimension
    outermost also makes prefix-cache block copies visible to Paiton's kernels.
    """

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # Logical [layers, blocks, 2, block_size, heads, head_size] is
            # physically [blocks, layers, 2, block_size, heads, head_size].
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def indexes_kv_by_block_stride() -> bool:
        """Keep scheduler block IDs aligned with blocks-first MLA pages."""
        return True
