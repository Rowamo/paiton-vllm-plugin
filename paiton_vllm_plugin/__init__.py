# SPDX-License-Identifier: Apache-2.0
"""
Paiton vLLM Plugin - Entry points for vLLM plugin system.

This plugin registers:
1. Paiton platform (based on ROCm) for optimized execution
2. Paiton model architectures for compiled models
"""

import os

from paiton_vllm_plugin.artifact_manifest import (
    ArtifactCompatibilityError,
    detect_runtime_gpu_arch,
)


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
    # The pinned vLLM reference stack relies on AMD SMI for built-in ROCm
    # discovery, but that probe can fail after Torch initializes ROCm. Allow
    # correctness harnesses to select vLLM's unmodified ROCmPlatform through
    # this already-discovered entry point without enabling any Paiton layout
    # or attention overrides.
    if os.environ.get("VLLM_PAITON_VANILLA_ROCM_PLATFORM", "0") == "1":
        return "vllm.platforms.rocm.RocmPlatform"

    force_paiton = os.environ.get("VLLM_USE_PAITON_PLATFORM", "0") == "1"
    try:
        selected_arch = detect_runtime_gpu_arch()
    except ArtifactCompatibilityError:
        selected_arch = None
    supported_arches = {"gfx942", "gfx950", "gfx1200", "gfx1201"}
    if selected_arch in supported_arches or force_paiton:
        return "paiton_vllm_plugin.paiton_platform.PaitonPlatform"
    
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
        "PaitonQwen3ForCausalLM": "paiton_vllm_plugin.models.paiton_qwen3:PaitonQwen3ForCausalLM",
        "PaitonQwen3MoeForCausalLM": "paiton_vllm_plugin.models.paiton_qwen3_moe:PaitonQwen3MoeForCausalLM",
        "PaitonQwen38ForCausalLM": "paiton_vllm_plugin.models.paiton_qwen38:PaitonQwen38ForCausalLM",
    }
    
    for arch, model_path in model_registrations.items():
        if arch not in ModelRegistry.get_supported_archs():
            ModelRegistry.register_model(arch, model_path)
