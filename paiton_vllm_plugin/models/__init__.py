# SPDX-License-Identifier: Apache-2.0
"""
Paiton Model Implementations for vLLM.

This module provides vLLM-compatible wrappers for Paiton-compiled models.
"""

from paiton_vllm_plugin.models.paiton_llama import PaitonLlamaForCausalLM
from paiton_vllm_plugin.models.paiton_qwen import PaitonQwen2ForCausalLM
from paiton_vllm_plugin.models.paiton_qwen3 import PaitonQwen3ForCausalLM
from paiton_vllm_plugin.models.paiton_qwen3_moe import PaitonQwen3MoeForCausalLM

__all__ = [
    "PaitonLlamaForCausalLM",
    "PaitonQwen2ForCausalLM",
    "PaitonQwen3ForCausalLM",
    "PaitonQwen3MoeForCausalLM",
]
