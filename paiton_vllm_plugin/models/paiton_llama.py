# SPDX-License-Identifier: Apache-2.0
"""
Paiton Llama Model Implementation for vLLM.

This module provides a vLLM-compatible wrapper for Paiton-compiled Llama models.
"""

import torch
from typing import Dict, List, Optional

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.models.paiton_base import PaitonModelBase


class PaitonLlamaForCausalLM(PaitonModelBase):
    """
    vLLM-compatible wrapper for Paiton-compiled Llama models.
    
    This class handles:
    - Loading compiled Llama .so files
    - Weight mapping with proper tensor parallel distribution
    - Integration with vLLM's inference pipeline
    
    Usage:
        To use this model, ensure your model's config.json contains:
        {
            "architectures": ["PaitonLlamaForCausalLM"],
            ...
        }
        
        Or specify it via command line:
        --model /path/to/model --trust-remote-code
    """
    
    # Llama-specific weight fusions
    packed_modules_mapping: Dict[str, List[str]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config, prefix)
        
        # Llama-specific configuration validation
        assert hasattr(self.config, "num_hidden_layers"), \
            "Llama config must have num_hidden_layers"
        assert hasattr(self.config, "num_attention_heads"), \
            "Llama config must have num_attention_heads"
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Llama model.
        
        Delegates to the base class implementation which handles
        the Paiton runtime execution.
        """
        return super().forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

