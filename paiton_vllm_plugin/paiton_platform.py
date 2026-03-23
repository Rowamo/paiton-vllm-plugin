# SPDX-License-Identifier: Apache-2.0
"""
Paiton Platform - vLLM platform implementation for Paiton compiled models.

Extends the ROCm platform with Paiton-specific optimizations and configurations.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms.rocm import RocmPlatform

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class PaitonPlatform(RocmPlatform):
    """
    Platform implementation for running Paiton-compiled models on AMD GPUs.
    
    This platform extends RocmPlatform with:
    - Custom attention backend support for Paiton models
    - Optimized configuration defaults for compiled models
    - Integration with Paiton runtime
    """
    
    device_name: str = "paiton"
    
    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """
        Update vLLM configuration for optimal Paiton model execution.
        
        This sets up:
        - Worker class configuration
        - Block size optimization
        - Custom ops configuration
        """
        # First apply ROCm base configuration
        super().check_and_update_config(vllm_config)
        
        cache_config = vllm_config.cache_config
        compilation_config = vllm_config.compilation_config
        parallel_config = vllm_config.parallel_config
        
        # Paiton-specific optimizations
        if cache_config and cache_config.block_size is None:
            # Paiton models work well with block size 16
            cache_config.block_size = 16

        # If the model is configured for FP8 KV-cache (common for Paiton
        # "FP8-KV" builds), vLLM's default kv_cache_dtype=auto would pick the
        # model dtype (e.g. bf16), which will mismatch the compiled runtime.
        # Only override when the user hasn't explicitly set a cache dtype.
        if cache_config and cache_config.cache_dtype == "auto":
            qc = getattr(vllm_config.model_config.hf_config, "quantization_config",
                         None)

            def _qget(obj, key: str):
                if obj is None:
                    return None
                if isinstance(obj, dict):
                    return obj.get(key)
                return getattr(obj, key, None)

            quant_method = _qget(qc, "quant_method")
            kv_cache_scheme = _qget(qc, "kv_cache_scheme")
            if (quant_method == "fp8") and (kv_cache_scheme in ("static", "fp8")):
                cache_config.cache_dtype = "fp8"

        # Paiton compiled models already include fused kernels (attention, etc.)
        # and generally are *not* compatible with vLLM's cudagraph capture /
        # torch.compile pipelines (which assume PyTorch graph capture).
        #
        # If cudagraph is enabled, we have observed decode-step corruption
        # (e.g. repetitive special tokens / gibberish) even when the first token
        # looks correct. Default to eager execution and disable cudagraph unless
        # the user explicitly opted in.
        # NOTE: Import lazily to avoid circular imports during platform
        # initialization (vllm.config.compilation imports current_platform).
        from vllm.config.compilation import CUDAGraphMode, CompilationMode

        if compilation_config.mode != CompilationMode.NONE:
            compilation_config.mode = CompilationMode.NONE
        if compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        if compilation_config.cudagraph_capture_sizes:
            compilation_config.cudagraph_capture_sizes = []
        
        # Use standard GPU worker - Paiton models run through the model forward
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"
        
        # Enable custom ops for Paiton
        if "all" not in compilation_config.custom_ops:
            compilation_config.custom_ops = ["all"]
        
        logger.info("Paiton platform configured with block_size=%s",
                   cache_config.block_size if cache_config else "N/A")
    
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads: int | None = None,
    ) -> str:
        """
        Get the attention backend class for Paiton models.
        
        Paiton compiled models handle attention internally, but we must ensure
        vLLM allocates the KV cache with the layout expected by the compiled
        runtime. We therefore return a Triton-backend subclass that only
        overrides KV cache shape/layout.
        """
        use_sparse = getattr(attn_selector_config, "use_sparse", False)
        use_mla = getattr(attn_selector_config, "use_mla", False)

        if use_sparse:
            raise NotImplementedError("Sparse Attention is not supported for Paiton.")
        if use_mla:
            # Not currently supported by the Paiton compiled path in this plugin.
            raise NotImplementedError("MLA is not supported for Paiton.")

        return "paiton_vllm_plugin.paiton_attention_backend.PaitonTritonAttentionBackend"
    
    @classmethod
    def supports_fp8(cls) -> bool:
        """Paiton supports FP8 on MI300 series."""
        return True
    
    @classmethod
    def is_fp8_fnuz(cls) -> bool:
        """MI300 series uses fnuz FP8 format."""
        gcn_arch = torch.cuda.get_device_properties(0).gcnArchName
        return "gfx94" in gcn_arch
    
    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        """Return the appropriate FP8 dtype for the current device."""
        if cls.is_fp8_fnuz():
            return torch.float8_e4m3fnuz
        return torch.float8_e4m3fn
