# SPDX-License-Identifier: Apache-2.0
"""
Paiton attention backend shim.

We don't actually *run* vLLM attention ops for Paiton-compiled models, but vLLM
still uses the selected attention backend to decide:
- KV cache tensor layout / shape
- metadata builder type

Paiton compiled kernels in this repo expect KV cache layout:
  (2, num_blocks, block_size, num_kv_heads, head_size)

Whereas vLLM's Triton backend uses:
  (num_blocks, 2, block_size, num_kv_heads, head_size)

This backend subclasses Triton's backend and only overrides the KV cache shape
to match the compiled model expectation.
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


