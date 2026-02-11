# SPDX-License-Identifier: Apache-2.0
"""
Base class for Paiton-compiled models in vLLM.

Provides common functionality for loading and running Paiton models.
"""

import torch
from torch import nn
from typing import Any, Dict, Iterable, Set, Tuple, List, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from vllm.config import VllmConfig
try:
    # vLLM <=0.14
    from vllm.attention import Attention, AttentionType
except ImportError:
    # vLLM >=0.15 moved these symbols.
    from vllm.attention.layer import Attention, AttentionType
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext, get_forward_context


class PaitonModelBase(nn.Module, ABC):
    """
    Base class for Paiton-compiled models.
    
    This class provides:
    - Model loading from compiled .so files
    - Weight mapping and distribution for tensor parallelism
    - Forward pass integration with vLLM's attention metadata
    - FP8 quantization support
    """
    
    # Subclasses should override these for model-specific weight fusions
    packed_modules_mapping: Dict[str, List[str]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        
        self.config = vllm_config.model_config.hf_config
        self.model_path = Path(vllm_config.model_config.model)
        
        # Load the compiled Paiton model
        model_so_name = f"{self.model_path.name}_tp{self.tp_size}.so"
        model_so_path = self.model_path / model_so_name
        
        from paiton.core.model import Model
        self.model = Model(model_so_path)
        
        self.dtype = self.config.torch_dtype
        self.cache_dtype = (
            self.dtype 
            if vllm_config.cache_config.cache_dtype == "auto" 
            else torch.float8_e4m3fnuz
        )
        self.quantized = vllm_config.quant_config is not None
        self.unpadded_vocab_size = self.config.vocab_size
        
        # Set up logits processor
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            self.config.vocab_size,
            logit_scale,
            logits_as_input=True,  # Paiton model output is logits
        )
        
        # Set up static forward context for KV caches
        self.num_layers = self.config.num_hidden_layers
        self.compilation_config = vllm_config.compilation_config
        num_q_heads = self.config.num_attention_heads // self.tp_size
        num_kv_heads = max(1, self.config.num_key_value_heads // self.tp_size)
        head_size = self.config.hidden_size // self.config.num_attention_heads
        scale = head_size ** -0.5
        
        self.compilation_config.static_forward_context = {
            str(i): Attention(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=str(i),
                attn_type=AttentionType.DECODER,
            )
            for i in range(self.num_layers)
        }
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input IDs.
        
        For Paiton models, embeddings are handled internally by the compiled
        model, so this returns a dummy tensor to satisfy the VllmModel protocol.
        """
        hidden_size = getattr(self.config, "hidden_size", 4096)
        return torch.zeros(
            (*input_ids.shape, hidden_size),
            dtype=self.dtype,
            device=input_ids.device,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Paiton compiled model.
        
        Args:
            input_ids: Token IDs [batch_size]
            positions: Position IDs [batch_size]
            intermediate_tensors: For pipeline parallelism (not used)
            inputs_embeds: Pre-computed embeddings (not used for Paiton)
            
        Returns:
            Logits tensor [batch_size, vocab_size]
        """
        output = torch.empty(
            [input_ids.shape[0], self.config.vocab_size],
            dtype=torch.float32,
            device="cuda"
        )
        
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        
        if not attn_metadata:
            return output
        
        # Get attention metadata (same for all layers)
        attn_metadata = attn_metadata['0']
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len
        
        # Prepare inputs for Paiton model
        inputs = {
            "input_ids": input_ids,
            "position_ids": positions,
            "slot_mapping": attn_metadata.slot_mapping,
            "query_start_locations": attn_metadata.query_start_loc,
            "context_lengths": attn_metadata.seq_lens,
            "block_tables": attn_metadata.block_table,
            "max_query_len": torch.empty(
                [max_query_len, 0], dtype=torch.int32, device="cuda"
            ),
            "max_seq_len": torch.empty(
                [max_seq_len, 0], dtype=torch.int32, device="cuda"
            ),
        }
        
        # Add KV caches
        for i in range(self.num_layers):
            idx = f"kv_cache_{i}"
            kv_cache = self.compilation_config.static_forward_context[str(i)].kv_cache[0]
            inputs[idx] = kv_cache.view(self.cache_dtype)
        
        outputs = {"logits": output}
        model_output = self.model.run_with_tensors(inputs, outputs, sync=False)
        
        return model_output["logits"]
    
    def compute_logits(self, hidden_states: torch.Tensor) -> Optional[torch.Tensor]:
        """Process logits through the logits processor."""
        if self.tp_rank == 0:
            return self.logits_processor(None, hidden_states)
        return None
    
    def get_rank_weight(self, weight: torch.Tensor, dim: int) -> torch.Tensor:
        """Get the weight slice for the current tensor parallel rank."""
        if weight.dim() == 0:
            weight = weight.reshape([1])
        local_weight = torch.split(
            weight, weight.shape[dim] // self.tp_size, dim
        )[self.tp_rank]
        return local_weight
    
    def get_rank_bias(self, bias: torch.Tensor) -> torch.Tensor:
        """Get bias for the current rank (only rank 0 uses bias)."""
        if self.tp_rank == 0:
            return bias
        return torch.zeros_like(bias)
    
    def _convert_fp8_weights(self, paiton_param: torch.Tensor) -> torch.Tensor:
        """Convert FP8 weights from fn to fnuz format for AMD GPUs."""
        if paiton_param.dtype == torch.float8_e4m3fn:
            weight_as_int8 = paiton_param.view(torch.int8).cuda()
            # e4m3fn `-0` is `NaN` in e4m3fnuz, set to `0`
            weight_as_int8[weight_as_int8 == -128] = 0
            paiton_param = weight_as_int8.view(torch.float8_e4m3fnuz)
        return paiton_param
    
    def _handle_fused_weights(
        self,
        unfused_name: str,
        fused_module: str,
        unfused_module: str,
        full_fused_name: str,
        pt_params: Dict[str, torch.Tensor],
        convert_name: callable,
    ) -> Optional[torch.Tensor]:
        """Handle fused weight parameters (QKV, gate_up, etc.)."""
        modules_to_fuse = self.packed_modules_mapping[fused_module]
        # IMPORTANT: `unfused_name` still contains the original sub-module
        # (e.g. "gate_proj"). We must build the list of parameter names from
        # that original name; using `full_fused_name` would incorrectly produce
        # "gate_up_proj.*" for all fused components.
        params_names = [
            unfused_name.replace(unfused_module, module) for module in modules_to_fuse
        ]
        params = [pt_params[param_name] for param_name in params_names]
        
        if full_fused_name.endswith(".weight"):
            # Column parallel weights, split across dim=0
            params = [self.get_rank_weight(param, dim=0).cuda() for param in params]
            
            if self.quantized:
                # Handle fused quantized weights with different scales
                scales_names = [n.replace(".weight", ".weight_scale") for n in params_names]
                scales = [pt_params[n] for n in scales_names]
                
                # De-quantize, apply uniform scale, re-quantize
                dequant_weights = [
                    w.to(torch.float32) * s for (w, s) in zip(params, scales)
                ]
                max_weight_scale = max(scale.item() for scale in scales)
                params = [
                    (w / max_weight_scale).to(params[0].dtype) 
                    for w in dequant_weights
                ]
            
            return torch.cat(params, dim=0)
            
        elif full_fused_name.endswith(".bias"):
            bias = torch.cat(params, dim=0)
            return self.get_rank_bias(bias)
            
        elif full_fused_name.endswith(".input_scale"):
            for param in params[1:]:
                assert param == params[0], f"input_scale must be equal for all of {params_names}"
            return params[0] * 2
            
        elif full_fused_name.endswith(".weight_scale"):
            max_weight_scale = max(scale.item() for scale in params)
            return torch.FloatTensor([max_weight_scale * 2])
        
        return None
    
    def map_pt_params(self, pt_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Map PyTorch parameters to Paiton parameter format.
        
        Handles:
        - Weight fusion (QKV, gate_up projections)
        - Tensor parallel distribution
        - Quantization scale handling
        - FP8 format conversion
        """
        def find_fusion(name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            """Check if module is fused, return fused info."""
            for fused_module, modules in self.packed_modules_mapping.items():
                for module in modules:
                    if module in name:
                        return fused_module, module, name.replace(module, fused_module)
            return None, None, None
        
        def convert_name(name: str) -> str:
            """Convert PyTorch param name to Paiton format (. -> _)."""
            return name.replace("model.", "").replace(".", "_")
        
        params_paiton: Dict[str, torch.Tensor] = {}
        
        for name, param in pt_params.items():
            unfused_name = name
            fused_module, unfused_module, full_fused_name = find_fusion(name)
            
            if fused_module is not None:
                # Skip if already processed
                if convert_name(full_fused_name) in params_paiton:
                    continue
                
                paiton_param = self._handle_fused_weights(
                    unfused_name,
                    fused_module,
                    unfused_module,
                    full_fused_name,
                    pt_params,
                    convert_name,
                )
                name = full_fused_name
                
            elif name.endswith("down_proj.weight") or name.endswith("o_proj.weight"):
                # Row parallel, split across dim=1
                paiton_param = self.get_rank_weight(param, dim=1)
                
            elif name.endswith(".bias"):
                paiton_param = self.get_rank_bias(param)
                
            elif name.endswith("norm.weight"):
                # Normalization weights are not split
                paiton_param = param
                
            elif name.endswith(".input_scale"):
                paiton_param = param * 2
                
            elif name.endswith(".weight_scale"):
                paiton_param = param * 2
                
            elif name.endswith(".kv_scale"):
                # Split kv_scale into k_scale and v_scale
                k_scale_name = name.replace("kv_scale", "k_scale")
                v_scale_name = name.replace("kv_scale", "v_scale")
                params_paiton[convert_name(k_scale_name)] = (param * 2).cuda()
                params_paiton[convert_name(v_scale_name)] = (param * 2).cuda()
                continue
                
            else:
                paiton_param = self.get_rank_weight(param, dim=0)
            
            if paiton_param is not None:
                paiton_param = self._convert_fp8_weights(paiton_param)
                params_paiton[convert_name(name)] = paiton_param.cuda()
        
        return params_paiton
    
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        """Load weights into the Paiton model."""
        self.model.set_many_constants_with_tensors(self.map_pt_params(dict(weights)))
        return set()  # Return empty set as Paiton handles all weights internally

