# SPDX-License-Identifier: Apache-2.0
"""
Paiton vLLM Plugin - Entry points for vLLM plugin system.

This plugin registers:
1. Paiton platform (based on ROCm) for optimized execution
2. Paiton model architectures for compiled models
"""

import os


def paiton_platform_plugin() -> str | None:
    """
    Platform plugin entry point.
    
    Returns the fully qualified name of the PaitonPlatform class if
    running on a supported AMD GPU, otherwise returns None.
    """
    # Allow explicit opt-out so users can run vanilla vLLM on MI300 systems.
    # (Paiton's platform changes KV-cache layout and is only compatible with
    # Paiton-compiled model runtimes.)
    if os.environ.get("VLLM_DISABLE_PAITON_PLATFORM", "0") == "1":
        return None

    # Check if we're on ROCm/AMD GPU
    try:
        import torch
        if torch.cuda.is_available():
            # Check for AMD GPU (ROCm uses cuda interface)
            device_name = torch.cuda.get_device_properties(0).name.lower()
            gcn_arch = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '')
            
            # Enable Paiton platform on MI300 series or when explicitly requested
            is_mi3xx = any(arch in gcn_arch for arch in ["gfx942", "gfx950"])
            force_paiton = os.environ.get("VLLM_USE_PAITON_PLATFORM", "0") == "1"
            
            if is_mi3xx or force_paiton:
                # vLLM platform plugins must return a dot-qualified class name
                # (module.Class), not the entry-point style (module:Class).
                return "paiton_vllm_plugin.paiton_platform.PaitonPlatform"
    except Exception:
        pass
    
    return None


def register_paiton_models() -> None:
    """
    General plugin entry point to register Paiton model architectures.
    
    This registers the PaitonLlamaForCausalLM and other Paiton-compiled
    model classes with the vLLM ModelRegistry.
    """
    from vllm import ModelRegistry
    
    # Register Paiton model architectures
    # Users can specify these in their model config's architectures field
    model_registrations = {
        "PaitonLlamaForCausalLM": "paiton_vllm_plugin.models.paiton_llama:PaitonLlamaForCausalLM",
        "PaitonQwen2ForCausalLM": "paiton_vllm_plugin.models.paiton_qwen:PaitonQwen2ForCausalLM",
        "PaitonQwen3ForCausalLM": "paiton.qwen.qwen3_vllm:PaitonQwen3ForCausalLM",
        "PaitonQwen3MoeForCausalLM": "paiton.qwen.qwen3_moe_vllm:PaitonQwen3MoeForCausalLM",
    }
    
    for arch, model_path in model_registrations.items():
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, model_path)

