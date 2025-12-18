# SPDX-License-Identifier: Apache-2.0
"""
Paiton Qwen Model Implementation for vLLM.

This module provides a vLLM-compatible wrapper for Paiton-compiled Qwen models.
"""

import torch
from typing import Dict, List, Optional

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.models.paiton_base import PaitonModelBase


class PaitonQwen2ForCausalLM(PaitonModelBase):
    """
    vLLM-compatible wrapper for Paiton-compiled Qwen2 models.
    
    This class handles:
    - Loading compiled Qwen2 .so files
    - Weight mapping with proper tensor parallel distribution
    - Integration with vLLM's inference pipeline
    
    Usage:
        To use this model, ensure your model's config.json contains:
        {
            "architectures": ["PaitonQwen2ForCausalLM"],
            ...
        }
    """
    
    # Qwen2-specific weight fusions (same as Llama)
    packed_modules_mapping: Dict[str, List[str]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config, prefix)
        
        # Qwen2-specific configuration validation
        assert hasattr(self.config, "num_hidden_layers"), \
            "Qwen2 config must have num_hidden_layers"
        assert hasattr(self.config, "num_attention_heads"), \
            "Qwen2 config must have num_attention_heads"
        
        # Qwen2 may have different num_key_value_heads per layer
        # Handle sliding window attention if present
        self.sliding_window = getattr(self.config, "sliding_window", None)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Qwen2 model.
        
        Delegates to the base class implementation which handles
        the Paiton runtime execution.
        """
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

