"""Compatibility helpers for multiple vLLM package layouts."""

from __future__ import annotations

try:
    from vllm.attention import Attention, AttentionType
except ModuleNotFoundError:
    try:
        from vllm.attention.layer import Attention, AttentionType
    except ModuleNotFoundError:
        try:
            from vllm.model_executor.layers.attention import Attention
        except ModuleNotFoundError:
            from vllm.model_executor.layers.attention.attention import Attention
        from vllm.v1.attention.backend import AttentionType

__all__ = ["Attention", "AttentionType"]

