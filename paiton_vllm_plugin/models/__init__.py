# SPDX-License-Identifier: Apache-2.0
"""
Paiton Model Implementations for vLLM.

This module provides vLLM-compatible wrappers for Paiton-compiled models.
"""

from importlib import import_module

__all__ = [
    "PaitonLlamaForCausalLM",
    "PaitonQwen2ForCausalLM",
    "PaitonQwen3ForCausalLM",
    "PaitonQwen3MoeForCausalLM",
    "PaitonQwen38ForCausalLM",
    "PaitonQwen38ForConditionalGeneration",
]

_MODEL_MODULES = {
    "PaitonLlamaForCausalLM": ".paiton_llama",
    "PaitonQwen2ForCausalLM": ".paiton_qwen",
    "PaitonQwen3ForCausalLM": ".paiton_qwen3",
    "PaitonQwen3MoeForCausalLM": ".paiton_qwen3_moe",
    "PaitonQwen38ForCausalLM": ".paiton_qwen38",
    "PaitonQwen38ForConditionalGeneration": ".paiton_qwen38_multimodal",
}


def __getattr__(name: str):
    """Keep CPU-safe helpers importable without importing vLLM or Torch."""
    module_name = _MODEL_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
