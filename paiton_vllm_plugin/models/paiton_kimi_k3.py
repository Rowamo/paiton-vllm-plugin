# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Paiton-compiled Kimi K3 models.

Kimi K3 is a hybrid model with 93 transformer layers:
- 69 Kimi Delta Attention (KDA) layers with stateful conv + recurrent caches.
- 24 gated full-MLA layers with NoPE latent paged KV cache.
- Attention Residuals (AttnRes) with block size 12.
- Stable LatentMoE with 896 routed experts (MXFP4), top-16, SiTU-GLU.
- Two shared SiTU experts per MoE layer.

The checkpoint stores weights under ``language_model.model.layers.N.*``.
The compiled artifact expects Paiton-style constant names (dots replaced by
underscores, ``model.`` prefix stripped). This class handles:

- KDA conv state allocation (Q/K/V, per-layer)
- KDA recurrent state allocation (per-layer)
- MLA latent paged KV cache (for the 24 full-MLA layers)
- AttnRes block-residual bank
- MXFP4 expert weight packing (shared with the GLM MoE DSA path)
- K3-specific weight name mapping (g_proj, o_norm, conv1d, A_log, dt_bias)
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import Tensor

from vllm.distributed.parallel_state import get_ep_group
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid

from paiton_vllm_plugin.models.paiton_glm_moe_dsa import (
    PaitonGlmMoeDsaForCausalLM,
    _packed_to_uint8,
    _scale_to_uint8,
    _fix_fp8,
    _preshuffle_flatmm_mxfp4_weight,
    _preshuffle_flatmm_mxfp4_scale,
    _preshuffle_mxfp4_weight,
    _preshuffle_mxfp4_scale,
    _artifact_has_flatmm_constant_names,
    _artifact_has_fused_shared_flatmm_constants,
    _resolve_compiled_moe_kernel,
)
from paiton_vllm_plugin.runtime.core import (
    PData,
    runtime_uses_fnuz_fp8,
    torch_dtype_to_string,
    torch_to_paiton_data,
)


# Per-expert checkpoint tensor suffixes for K3's compressed-tensors MXFP4.
_EXPERT_PROJ_NAMES = ("w1", "w2", "w3")  # w1=gate, w2=down, w3=up


class PaitonKimiK3ForCausalLM(
    PaitonGlmMoeDsaForCausalLM, HasInnerState, IsHybrid
):
    """Runtime wrapper for Paiton-compiled Kimi K3 artifacts.

    Inherits MXFP4 expert packing and MLA latent cache management from
    :class:`PaitonGlmMoeDsaForCausalLM`, and adds:
    - KDA conv/recurrent state allocation and binding.
    - AttnRes block-residual bank binding.
    - K3 checkpoint weight name mapping.
    """

    # K3 does not fuse q/k/v or gate/up at the module level (the compiled
    # graph handles fusion internally), so disable vLLM's module fusion.
    packed_modules_mapping: Dict[str, list] = {}

    def __init__(self, vllm_config, prefix: str = ""):
        # K3 is a multimodal model: the HF config is KimiK3Config with a
        # nested text_config (KimiLinearConfig). The base class reads
        # vocab_size, num_hidden_layers, num_attention_heads, etc. from
        # self.config, which must be the text config for Paiton. Override
        # self.config to the text config before calling super().__init__.
        hf_config = vllm_config.model_config.hf_config
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            # Copy top-level fields the base class may need (ep_size,
            # paiton_logits_all_gather, decode_partition_size, etc.) onto
            # the text config so getattr() picks them up.
            for key in (
                "ep_size",
                "paiton_logits_all_gather",
                "decode_partition_size",
                "kv_cache_block_size",
                "fp8_kv_cache",
                "quantization_config",
                "paiton_kimi_k3_contract",
            ):
                if hasattr(hf_config, key) and not hasattr(text_config, key):
                    setattr(text_config, key, getattr(hf_config, key))
            # Also copy torch_dtype if the text config doesn't have it.
            if not hasattr(text_config, "torch_dtype"):
                text_config.torch_dtype = getattr(hf_config, "torch_dtype", "bfloat16")
            vllm_config.model_config.hf_config = text_config

        super().__init__(vllm_config, prefix=prefix)

        cfg = self.config
        if bool(getattr(cfg, "fp8_kv_cache", False)):
            raise ValueError(
                "The Paiton Kimi K3 MLA kernels do not support FP8 KV "
                "caches. Recompile and run with fp8_kv_cache=false."
            )
        # Validate the K3 artifact/runtime contract. Legacy artifacts that
        # predate the contract are rejected with a recompile message; the
        # runtime TP/block-size/dtype/Mamba-mode must match the compiled
        # artifact.
        self._k3_contract = self._validate_k3_contract(vllm_config)
        # K3-specific config values.
        self._kda_num_heads = self._get_kda_num_heads()
        self._kda_head_dim = self._get_kda_head_dim()
        self._kda_conv_kernel_size = self._get_kda_conv_kernel_size()
        self._attn_res_block_size = int(getattr(cfg, "attn_res_block_size", 12) or 12)

        # Compute KDA per-layer local dims (after TP sharding).
        tp = int(self.tp_size)
        self._kda_local_proj_dim = (self._kda_num_heads * self._kda_head_dim) // tp
        self._kda_local_num_heads = self._kda_num_heads // tp

        # KDA layer classification: layer is KDA if (layer+1) in kda_layers.
        lac = getattr(cfg, "linear_attn_config", None) or {}
        kda_layers = set(lac.get("kda_layers", []) or [])
        self._is_kda_layer = [
            (layer_idx + 1) in kda_layers
            for layer_idx in range(self.num_layers)
        ]

        # Allocate KDA conv/recurrent state banks (lazily, on first forward).
        self._kda_conv_caches: Dict[str, Dict[int, torch.Tensor]] = {
            "q": {}, "k": {}, "v": {}
        }
        self._kda_recurrent_caches: Dict[int, torch.Tensor] = {}

        # AttnRes block count. The block-residual bank itself is per-forward
        # transient storage allocated in forward() sized by the scheduled
        # token count; no instance bank is retained across forwards.
        self._num_attn_res_blocks = (
            self.num_layers + self._attn_res_block_size - 1
        ) // self._attn_res_block_size

    # ------------------------------------------------------------------ #
    # KDA config helpers.
    # ------------------------------------------------------------------ #
    def _get_kda_num_heads(self) -> int:
        lac = getattr(self.config, "linear_attn_config", None) or {}
        return int(lac.get("num_heads", getattr(self.config, "num_attention_heads", 96)))

    def _get_kda_head_dim(self) -> int:
        lac = getattr(self.config, "linear_attn_config", None) or {}
        return int(lac.get("head_dim", 128))

    def _get_kda_conv_kernel_size(self) -> int:
        lac = getattr(self.config, "linear_attn_config", None) or {}
        return int(lac.get("short_conv_kernel_size", 4))

    def _mla_head_dim(self) -> int:
        return int(getattr(self.config, "kv_lora_rank", 512)) + int(
            getattr(self.config, "qk_rope_head_dim", 64)
        )

    # ------------------------------------------------------------------ #
    # Artifact/runtime contract validation.
    # ------------------------------------------------------------------ #
    def _validate_k3_contract(self, vllm_config) -> dict:
        """Validate the ``paiton_kimi_k3_contract`` against the runtime config.

        Rejects legacy artifacts that predate the contract (no
        ``paiton_kimi_k3_contract`` in the config) and checks TP, block size,
        model dtype, and Mamba cache mode against the compiled artifact. The
        per-token ``all`` Mamba mode is not implemented by the compiled K3
        graph and is rejected here.
        """
        contract = getattr(self.config, "paiton_kimi_k3_contract", None)
        if not isinstance(contract, dict):
            raise RuntimeError(
                "This Kimi K3 artifact predates the paiton_kimi_k3_contract "
                "metadata and is not supported by the current runtime. "
                "Recompile the model with the Paiton K3 compiler so the "
                "artifact records its contract (TP/EP, block size, dtype, "
                "Mamba mode, FlatMM requirement)."
            )

        compiled_tp = int(contract.get("tp_size", 1))
        if compiled_tp != int(self.tp_size):
            raise RuntimeError(
                "Kimi K3 TP size mismatch: artifact compiled for "
                f"tp_size={compiled_tp} but runtime tp_size={self.tp_size}. "
                "Recompile with the matching --tp_size."
            )

        cache_config = getattr(vllm_config, "cache_config", None)
        compiled_block = int(contract.get("kv_cache_block_size", 0))
        if cache_config is not None and compiled_block > 0:
            runtime_block = int(getattr(cache_config, "block_size", 0))
            if runtime_block > 0 and runtime_block != compiled_block:
                raise RuntimeError(
                    "Kimi K3 kv_cache_block_size mismatch: artifact compiled "
                    f"with block_size={compiled_block} but runtime "
                    f"block_size={runtime_block}. Recompile or reconfigure "
                    "vLLM to use the matching block size."
                )

        compiled_dtype = str(contract.get("model_dtype", "")).replace("torch.", "")
        runtime_dtype = str(self.dtype).replace("torch.", "")
        if compiled_dtype and runtime_dtype and compiled_dtype != runtime_dtype:
            raise RuntimeError(
                "Kimi K3 model dtype mismatch: artifact compiled for "
                f"{compiled_dtype} but runtime dtype={runtime_dtype}."
            )

        supported_modes = tuple(contract.get("mamba_cache_modes", ()))
        mamba_mode = str(getattr(cache_config, "mamba_cache_mode", "none"))
        if mamba_mode == "all" or (
            supported_modes and mamba_mode not in supported_modes
        ):
            raise RuntimeError(
                f"Kimi K3 does not support mamba_cache_mode={mamba_mode!r}. "
                f"Supported modes: {list(supported_modes) or ['none', 'align']}. "
                "The per-token 'all' mode is not implemented by the compiled "
                "K3 KDA graph; use 'none' or 'align'."
            )

        return contract

    # ------------------------------------------------------------------ #
    # KDA state allocation.
    # ------------------------------------------------------------------ #
    def _ensure_kda_states(self, num_state_slots: int, device: torch.device) -> None:
        """Allocate or grow KDA state banks without discarding live rows."""
        state_len = self._kda_conv_kernel_size - 1
        local_proj = self._kda_local_proj_dim
        local_heads = self._kda_local_num_heads

        for layer_idx in range(self.num_layers):
            if not self._is_kda_layer[layer_idx]:
                continue
            for proj in ("q", "k", "v"):
                current = self._kda_conv_caches[proj].get(layer_idx)
                if (
                    current is None
                    or current.shape[0] < num_state_slots
                    or current.device != device
                ):
                    replacement = torch.zeros(
                        num_state_slots, local_proj, state_len,
                        dtype=self.dtype, device=device,
                    )
                    if current is not None and current.device == device:
                        replacement[: current.shape[0]].copy_(current)
                    self._kda_conv_caches[proj][layer_idx] = replacement
            current = self._kda_recurrent_caches.get(layer_idx)
            if (
                current is None
                or current.shape[0] < num_state_slots
                or current.device != device
            ):
                replacement = torch.zeros(
                    num_state_slots, local_heads, self._kda_head_dim,
                    self._kda_head_dim,
                    dtype=torch.float32, device=device,
                )
                if current is not None and current.device == device:
                    replacement[: current.shape[0]].copy_(current)
                self._kda_recurrent_caches[layer_idx] = replacement

    @staticmethod
    def _find_kda_metadata(attn_metadata):
        """Return vLLM's linear/KDA metadata from a hybrid metadata map."""
        values = attn_metadata.values() if isinstance(attn_metadata, dict) else ()
        for metadata in values:
            if hasattr(metadata, "non_spec_state_indices_tensor") or hasattr(
                metadata, "state_indices_tensor"
            ):
                return metadata
        return None

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
        )

        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateShapeCalculator,
        )

        config = vllm_config.model_config.hf_config
        config = getattr(config, "text_config", config)
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_attn_config["num_heads"],
            config.linear_attn_config["head_dim"],
            conv_kernel_size=config.linear_attn_config["short_conv_kernel_size"],
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator,
        )

        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def _attn_res_block_residual_for_forward(
        self, num_tokens: int, device: torch.device
    ) -> torch.Tensor:
        """Allocate the AttnRes block-residual bank for a single forward.

        The block-residual bank is per-forward transient storage sized by the
        scheduled token count (one residual stream per token), not by the
        request count. It is zero-initialized here and written/read only
        within this forward: block-write layers (every ``attn_res_block_size``
        layers) copy ``prefix[token]`` into their slot, and subsequent layers
        mix the valid slots back into the prefix via AttnRes. No state crosses
        a forward boundary, so a fresh allocation each step matches the
        compiled ``[num_tokens, num_blocks, hidden]`` input shape.
        """
        return torch.zeros(
            num_tokens, self._num_attn_res_blocks,
            int(self.config.hidden_size),
            dtype=self.dtype, device=device,
        )

    # ------------------------------------------------------------------ #
    # KV cache spec: K3 uses MLA latent cache for all layers (vLLM allocates
    # the same shape; KDA layers don't read it but vLLM still needs it).
    # ------------------------------------------------------------------ #
    def _configure_glm_kv_cache_spec(self, vllm_config) -> None:
        num_q_heads = int(self.config.num_attention_heads) // int(self.tp_size)
        head_size = self._mla_head_dim()
        scale = float(getattr(self.config, "qk_head_dim", head_size)) ** -0.5

        static_context = getattr(
            self.compilation_config,
            "static_forward_context",
            {},
        )
        for i in range(self.num_layers):
            ctx = static_context[str(i)]
            ctx.num_heads = num_q_heads
            ctx.head_size = head_size
            ctx.head_size_v = head_size
            ctx.num_kv_heads = 1
            if hasattr(ctx, "scale"):
                ctx.scale = scale

            impl = getattr(ctx, "impl", None)
            if impl is not None:
                impl.num_heads = num_q_heads
                impl.head_size = head_size
                if hasattr(impl, "head_size_v"):
                    impl.head_size_v = head_size
                impl.num_kv_heads = 1
                if hasattr(impl, "scale"):
                    impl.scale = scale
                if hasattr(impl, "num_queries_per_kv"):
                    impl.num_queries_per_kv = num_q_heads

    # ------------------------------------------------------------------ #
    # Forward pass.
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors=None,
        inputs_embeds=None,
    ) -> torch.Tensor:
        from vllm.forward_context import ForwardContext, get_forward_context

        forward_context: ForwardContext = get_forward_context()
        all_attn_metadata = forward_context.attn_metadata
        if not all_attn_metadata:
            return torch.empty(
                [input_ids.shape[0], self.config.vocab_size],
                dtype=torch.float32, device="cuda",
            )

        attn_metadata = all_attn_metadata["0"]
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len

        input_ids_i32 = input_ids.to(dtype=torch.int32, copy=False).contiguous()
        position_ids_i64 = positions.to(dtype=torch.int64, copy=False).contiguous()
        slot_mapping_i64 = attn_metadata.slot_mapping.to(
            dtype=torch.int64, copy=False
        ).contiguous()
        query_start_loc_i32 = attn_metadata.query_start_loc.to(
            dtype=torch.int32, copy=False
        ).contiguous()
        seq_lens_i32 = attn_metadata.seq_lens.to(
            dtype=torch.int32, copy=False
        ).contiguous()
        block_table_i32 = attn_metadata.block_table.to(
            dtype=torch.int32, copy=False
        ).contiguous()

        max_query_len_backing = torch.empty([1], dtype=torch.int32, device="cuda")
        max_seq_len_backing = torch.empty([1], dtype=torch.int32, device="cuda")
        max_query_len_backing.fill_(int(max_query_len))
        max_seq_len_backing.fill_(int(max_seq_len))

        # Determine batch size from query_start_loc.
        batch_size = int(query_start_loc_i32.shape[0]) - 1
        device = input_ids.device

        # KDA state rows are scheduler-owned. In particular, their indices
        # remain stable when continuous batching reorders requests, unlike a
        # batch-order arange(). Prefer KDA metadata, with the linear-attention
        # metadata spelling supported for older vLLM releases.
        kda_metadata = self._find_kda_metadata(all_attn_metadata)
        state_indices = (
            getattr(kda_metadata, "non_spec_state_indices_tensor", None)
            if kda_metadata is not None
            else None
        )
        if state_indices is None and kda_metadata is not None:
            state_indices = getattr(kda_metadata, "state_indices_tensor", None)
        if state_indices is None:
            # Compatibility fallback for old metadata: the first block-table
            # column is the scheduler's stable state slot, not batch order.
            state_indices = block_table_i32[:, 0]
        state_indices = state_indices.to(dtype=torch.int32, copy=False).contiguous()
        if state_indices.shape[0] != batch_size:
            raise RuntimeError(
                "Kimi K3 requires non-speculative KDA state metadata; "
                f"got {state_indices.shape[0]} state rows for {batch_size} requests."
            )

        has_initial_state = (
            getattr(kda_metadata, "has_initial_state", None)
            if kda_metadata is not None
            else None
        )
        if has_initial_state is None:
            query_lens = query_start_loc_i32[1:] - query_start_loc_i32[:-1]
            has_initial_state = seq_lens_i32 > query_lens
        has_initial_state = has_initial_state.to(
            dtype=torch.int32, copy=False
        ).contiguous()

        # State slots are stable scheduler IDs and may exceed the current
        # batch size. Grow instead of recreating the banks so surviving
        # requests retain both short-convolution and recurrent state.
        num_state_slots = max(int(state_indices.max().item()) + 1, 1)
        self._ensure_kda_states(num_state_slots, device)

        # Build inputs dict.
        inputs = {
            "input_ids": torch_to_paiton_data(input_ids_i32),
            "position_ids": torch_to_paiton_data(position_ids_i64),
            "slot_mapping": torch_to_paiton_data(slot_mapping_i64),
            "query_start_locations": torch_to_paiton_data(query_start_loc_i32),
            "context_lengths": torch_to_paiton_data(seq_lens_i32),
            "block_tables": torch_to_paiton_data(block_table_i32),
            "max_query_len": PData(
                max_query_len_backing.data_ptr(),
                [max_query_len, 0],
                torch_dtype_to_string(torch.int32),
            ),
            "max_seq_len": PData(
                max_seq_len_backing.data_ptr(),
                [max_seq_len, 0],
                torch_dtype_to_string(torch.int32),
            ),
            "state_indices": torch_to_paiton_data(state_indices),
            "has_initial_state": torch_to_paiton_data(has_initial_state),
        }

        # Bind KV caches for all layers (MLA layers read it; KDA layers get
        # a dummy that the compiled graph ignores).
        for i in range(self.num_layers):
            idx = f"kv_cache_{i}"
            kv_cache = self.compilation_config.static_forward_context[
                str(i)
            ].kv_cache[0]
            # For MLA layers, use the latent KV cache shape.
            if not self._is_kda_layer[i]:
                latent_kv = self._get_glm_latent_kv_cache(i, kv_cache)
                inputs[idx] = torch_to_paiton_data(
                    latent_kv.view(self.cache_dtype)
                )
            else:
                # KDA layers: bind a dummy (the compiled graph doesn't read
                # kv_cache for KDA layers; the conv/recurrent states carry
                # the cache).
                dummy = torch.zeros(1, dtype=self.cache_dtype, device=device)
                inputs[idx] = torch_to_paiton_data(dummy)

        # Bind KDA conv states (Q/K/V per KDA layer).
        for layer_idx in range(self.num_layers):
            if not self._is_kda_layer[layer_idx]:
                # Dummy for MLA layers (compiled graph ignores).
                for proj in ("q", "k", "v"):
                    key = f"kda_conv_state_{proj}_{layer_idx}"
                    dummy = torch.zeros(1, dtype=self.dtype, device=device)
                    inputs[key] = torch_to_paiton_data(dummy)
                continue
            for proj in ("q", "k", "v"):
                key = f"kda_conv_state_{proj}_{layer_idx}"
                inputs[key] = torch_to_paiton_data(
                    self._kda_conv_caches[proj][layer_idx]
                )

        # Bind KDA recurrent states.
        for layer_idx in range(self.num_layers):
            key = f"kda_recurrent_state_{layer_idx}"
            if self._is_kda_layer[layer_idx]:
                inputs[key] = torch_to_paiton_data(
                    self._kda_recurrent_caches[layer_idx]
                )
            else:
                dummy = torch.zeros(1, dtype=torch.float32, device=device)
                inputs[key] = torch_to_paiton_data(dummy)

        # Bind AttnRes block-residual bank (per-forward transient, sized by
        # the scheduled token count).
        num_tokens = int(input_ids.shape[0])
        block_residual = self._attn_res_block_residual_for_forward(
            num_tokens, device
        )
        inputs["block_residual"] = torch_to_paiton_data(block_residual)

        # Output.
        output = torch.empty(
            [input_ids.shape[0], self.config.vocab_size],
            dtype=torch.float32, device="cuda",
        )
        outputs = {"logits": torch_to_paiton_data(output)}

        stream_ptr = torch.cuda.current_stream().cuda_stream
        self.model.run(inputs, outputs, stream_ptr=stream_ptr, sync=False)
        return output

    # ------------------------------------------------------------------ #
    # Weight mapping.
    # ------------------------------------------------------------------ #
    def map_pt_params(
        self,
        pt_params: Dict[str, Tensor],
        expected_constant_names: Optional[Set[str]] = None,
    ) -> Dict[str, Tensor]:
        """Map K3 checkpoint weights to Paiton constant names.

        K3 checkpoint names: ``language_model.model.layers.N.*``
        Paiton constant names: ``layers_N_*`` (dots -> underscores, prefix
        stripped, ``language_model.`` and ``model.`` removed).
        """
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        def get_rank_weight(weight: Tensor, dim: int) -> Tensor:
            if weight.dim() == 0:
                weight = weight.reshape([1])
            return torch.split(weight, weight.shape[dim] // tp_size, dim)[tp_rank]

        def convert_name(name: str) -> str:
            # Map checkpoint names to compiled constant names.
            # Checkpoint: language_model.model.layers.N.block_sparse_moe.*
            # Compiled:   layers_N_mlp_*  (block_sparse_moe -> mlp)
            # Checkpoint: language_model.model.layers.N.self_attn.*
            # Compiled:   layers_N_self_attn_*
            # Checkpoint: language_model.lm_head.weight
            # Compiled:   lm_head_weight
            # Checkpoint: language_model.model.norm.weight
            # Compiled:   norm_weight
            name = name.replace("language_model.model.", "")
            name = name.replace("language_model.", "")
            name = name.replace("model.", "")
            # Map block_sparse_moe -> mlp (compiled module is self.mlp).
            name = name.replace("block_sparse_moe", "mlp")
            # Map mlp.gate.weight -> mlp.gate_proj.weight (router projection).
            name = name.replace("mlp.gate.weight", "mlp.gate_proj.weight")
            # Conv1d: compiled stores as [dim, k] Parameter (bare, no .weight),
            # checkpoint has [dim, 1, k] .weight. Strip .weight for conv1d.
            name = name.replace("conv1d.weight", "conv1d")
            # o_norm: compiled stores as a bare Parameter (not .weight).
            name = name.replace("o_norm.weight", "o_norm")
            # A_log and dt_bias: bare Parameters (no .weight suffix).
            # Already correct since checkpoint uses bare names.
            return name.replace(".", "_")

        def maybe_emit(out_name: str, value: Tensor) -> None:
            if (
                expected_constant_names is None
                or out_name in expected_constant_names
            ) and out_name not in params_paiton:
                params_paiton[out_name] = value

        params_paiton: Dict[str, Tensor] = {}

        # EP setup.
        enable_ep = bool(self.parallel_config.enable_expert_parallel)
        ep_group = get_ep_group()
        ep_rank = ep_group.rank_in_group if enable_ep else 0
        ep_size = ep_group.world_size if enable_ep else 1
        self._validate_compiled_ep_size(ep_size)

        num_experts = int(getattr(self.config, "num_experts", 896))
        num_local_experts = num_experts // ep_size
        placement = getattr(
            self.parallel_config, "expert_placement_strategy", "linear"
        )
        if placement == "round_robin":
            local_expert_ids = list(range(ep_rank, num_experts, ep_size))
        else:
            expert_start = ep_rank * num_local_experts
            local_expert_ids = list(
                range(expert_start, expert_start + num_local_experts)
            )

        # Buffer per-layer expert weights for MXFP4 packing.
        layers_routed_experts = [
            [{} for _ in range(num_experts)]
            for _ in range(self.config.num_hidden_layers)
        ]
        layers_shared_experts = [{} for _ in range(self.config.num_hidden_layers)]

        # Regex patterns for expert and shared expert weights.
        routed_re = re.compile(
            r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(.+)"
        )
        shared_re = re.compile(
            r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.shared_experts\.(.+)"
        )

        num_hidden_layers = self.config.num_hidden_layers

        for name, param in pt_params.items():
            # Skip MTP/predictor layers.
            m_layer = re.match(
                r"language_model\.model\.layers\.(\d+)\.", name
            )
            if m_layer is not None and int(m_layer[1]) >= num_hidden_layers:
                continue

            # Skip rotary_emb constants.
            if "rotary_emb" in name:
                continue

            # ---- Routed MoE experts: buffer for MXFP4 packing. ---------- #
            m = routed_re.match(name)
            if m is not None:
                layer_id, expert_id, weight_name = int(m[1]), int(m[2]), m[3]
                # Map w1/w2/w3 prefixes to gate_proj/down_proj/up_proj.
                # The checkpoint stores experts as w1.weight_packed, w2.weight_packed,
                # w3.weight_packed (and corresponding _weight_scale). Map the
                # prefix so the packing code below finds them by the expected names.
                proj_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
                for src_prefix, dst_prefix in proj_map.items():
                    if weight_name.startswith(src_prefix + "."):
                        weight_name = dst_prefix + weight_name[len(src_prefix):]
                        break
                layers_routed_experts[layer_id][expert_id][weight_name] = param
                continue

            # ---- Shared experts: buffer for MXFP4 packing. --------------- #
            m = shared_re.match(name)
            if m is not None:
                layer_id, weight_name = int(m[1]), m[2]
                layers_shared_experts[layer_id][weight_name] = param
                continue

            # ---- Router gate weight and correction bias. ----------------- #
            if name.endswith("block_sparse_moe.gate.weight"):
                out_name = convert_name(name)
                maybe_emit(out_name, param.cuda())
                continue

            if name.endswith("block_sparse_moe.gate.e_score_correction_bias"):
                out_name = convert_name(name)
                maybe_emit(out_name, param.cuda())
                continue

            # ---- Conv1d weights: TP-shard the channel axis. ------------- #
            # Checkpoint stores [dim, 1, kernel_size] with dim = num_heads *
            # head_dim (depthwise over the projection channels). The compiled
            # model expects [local_proj_dim, kernel_size] where local_proj_dim
            # = dim // tp_size, so shard dim 0 per rank after squeezing the
            # singleton channel axis.
            if name.endswith("conv1d.weight"):
                if param.dim() == 3 and param.shape[1] == 1:
                    param = param.squeeze(1)
                if param.dim() != 2 or param.shape[0] % tp_size != 0:
                    raise RuntimeError(
                        f"K3 conv1d weight {name} has shape {tuple(param.shape)}; "
                        f"expected [dim, kernel_size] with dim divisible by "
                        f"tp_size={tp_size}."
                    )
                value = get_rank_weight(param, dim=0)
                maybe_emit(convert_name(name), value.cuda())
                continue

            # ---- A_log: per-head decay log, TP-sliced to [num_local_heads]. -- #
            # The checkpoint flattens A_log to 1-D (current [128]) or a legacy
            # 4-D form; only the first num_heads values are meaningful (one
            # per KDA head). Slice rank r to [r*local_heads:(r+1)*local_heads],
            # ignoring unused trailing values, so each rank loads its own
            # heads' decay values. The compiled constant is [num_local_heads]
            # and the KDA kernel indexes A_log[head] directly.
            if name.endswith("self_attn.A_log"):
                local_heads = self._kda_local_num_heads
                total_heads = self._kda_num_heads
                flat = param.detach().float().reshape(-1)
                if flat.shape[0] < total_heads:
                    raise RuntimeError(
                        f"K3 A_log has {flat.shape[0]} values but K3 needs "
                        f"{total_heads} per-head decay values."
                    )
                start = tp_rank * local_heads
                sliced = flat[start:start + local_heads].contiguous()
                maybe_emit(convert_name(name), sliced.cuda())
                continue

            # ---- o_norm: tile [head_dim] to [local_proj_dim]. ------------ #
            # The checkpoint stores o_norm.weight as [head_dim] (shared
            # across all heads). The compiled model expects
            # [local_proj_dim] = [num_local_heads * head_dim], so we
            # replicate the weight across all heads.
            if name.endswith("self_attn.o_norm.weight"):
                head_dim = self._kda_head_dim
                local_proj = self._kda_local_proj_dim
                num_local_heads = local_proj // head_dim
                if param.shape[0] == head_dim:
                    tiled = param.repeat(num_local_heads)
                elif param.shape[0] == local_proj:
                    tiled = param
                else:
                    raise RuntimeError(
                        f"K3 o_norm weight {name} has shape {tuple(param.shape)}; "
                        f"expected [{head_dim}] or [{local_proj}]."
                    )
                maybe_emit(convert_name(name), tiled.cuda())
                continue

            # ---- dt_bias: [projection_size] -> [local_proj_dim]. -------- #
            if name.endswith("self_attn.dt_bias"):
                value = get_rank_weight(param, dim=0)
                maybe_emit(convert_name(name), value.cuda())
                continue

            # ---- b_proj: [num_heads, hidden] -> column-parallel. -------- #
            if name.endswith("self_attn.b_proj.weight"):
                value = get_rank_weight(param, dim=0)
                maybe_emit(convert_name(name), value.cuda())
                continue

            # ---- f_a_proj: replicated (not column-parallel). ------------ #
            # g1 = f_b(f_a(hidden)); f_a_proj maps hidden_size -> head_dim
            # and is shared across all heads/ranks, so it must not be
            # TP-sharded (the default fallthrough below would wrongly split
            # it on dim 0).
            if name.endswith("self_attn.f_a_proj.weight"):
                maybe_emit(convert_name(name), param.cuda())
                continue

            # ---- g_proj (KDA output gate): column-parallel. ------------- #
            if name.endswith("self_attn.g_proj.weight"):
                value = get_rank_weight(param, dim=0)
                maybe_emit(convert_name(name), value.cuda())
                continue

            # ---- Row-parallel weights (down_proj, o_proj). -------------- #
            if name.endswith("down_proj.weight") or name.endswith("o_proj.weight"):
                value = get_rank_weight(param, dim=1)
                maybe_emit(
                    convert_name(name),
                    _fix_fp8(value) if value.dtype == torch.float8_e4m3fn
                    else value.cuda(),
                )
                continue

            # ---- MLA low-rank projections (replicated). ----------------- #
            if name.endswith("q_a_proj.weight") or name.endswith(
                "kv_a_proj_with_mqa.weight"
            ):
                maybe_emit(convert_name(name), param.cuda())
                continue

            # ---- q_b_proj, kv_b_proj: column-parallel. ------------------ #
            if name.endswith("q_b_proj.weight") or name.endswith(
                "kv_b_proj.weight"
            ):
                value = get_rank_weight(param, dim=0)
                maybe_emit(convert_name(name), value.cuda())
                continue

            # ---- Norm weights (replicated). ----------------------------- #
            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                maybe_emit(convert_name(name), param.cuda())
                continue

            # ---- Res proj weights [1, hidden] -> [hidden]. -------------- #
            if name.endswith("res_proj.weight"):
                if param.dim() == 2 and param.shape[0] == 1:
                    param = param.squeeze(0)
                maybe_emit(convert_name(name), param.cuda())
                continue

            # ---- Dense MLP: fuse gate_proj + up_proj -> gate_up_proj. --- #
            # (handled in a second pass below)

            # ---- embed_tokens, lm_head: column-parallel. --------------- #
            # ---- All other column-parallel weights. ---------------------- #
            value = get_rank_weight(param, dim=0)
            maybe_emit(
                convert_name(name),
                _fix_fp8(value) if value.dtype == torch.float8_e4m3fn
                else value.cuda(),
            )

        # ---- Dense-layer MLP fusion (gate_proj + up_proj). ------------- #
        for layer_id in range(num_hidden_layers):
            if self._is_sparse_layer(layer_id):
                continue
            gp_name = (
                f"language_model.model.layers.{layer_id}.mlp.gate_proj.weight"
            )
            up_name = (
                f"language_model.model.layers.{layer_id}.mlp.up_proj.weight"
            )
            if gp_name in pt_params and up_name in pt_params:
                fused = torch.cat(
                    [
                        get_rank_weight(pt_params[gp_name], dim=0),
                        get_rank_weight(pt_params[up_name], dim=0),
                    ],
                    dim=0,
                )
                maybe_emit(
                    convert_name(
                        f"language_model.model.layers.{layer_id}"
                        f".mlp.gate_up_proj.weight"
                    ),
                    fused.cuda(),
                )

        # ---- Pack routed MXFP4 experts into fused w13/w2 (+UE8M0). ----- #
        import os as _os
        _moe_kernel = _resolve_compiled_moe_kernel(
            _os.environ.get("PAITON_MOE_KERNEL"),
            expected_constant_names,
        )
        _use_ck_moe = _moe_kernel == "ck_moe_fp4"
        _use_flatmm_moe = _moe_kernel == "ck_flatmm_fp4"
        _fused_shared_flatmm = _artifact_has_fused_shared_flatmm_constants(
            expected_constant_names
        )
        # FlatMM artifacts use layout-specific constant names with a
        # "_flatmm" suffix; the GLM plugin uses the same convention.
        _routed_layout_suffix = (
            "_flatmm_fused_shared"
            if _fused_shared_flatmm
            else ("_flatmm" if _use_flatmm_moe else "")
        )

        # K3 LatentMoE: experts operate on the latent dim (routed_expert_hidden_size),
        # not the hidden_size. The w13/w2 shapes use latent_dim and moe_intermediate.
        _latent_dim = int(getattr(self.config, "routed_expert_hidden_size", 3584))
        _inter = int(getattr(self.config, "moe_intermediate_size", 3072))

        for layer_id, experts in enumerate(layers_routed_experts):
            if not any(experts):
                continue
            local_experts_list = [experts[i] for i in local_expert_ids]
            shared = layers_shared_experts[layer_id]

            # Pack w13 = [gate_proj.weight_packed; up_proj.weight_packed].
            w13_parts = [
                torch.cat(
                    [
                        _packed_to_uint8(e["gate_proj.weight_packed"]),
                        _packed_to_uint8(e["up_proj.weight_packed"]),
                    ],
                    dim=0,
                )
                for e in local_experts_list
            ]
            w13 = torch.stack(w13_parts, dim=0)
            if _use_flatmm_moe:
                w13 = _preshuffle_flatmm_mxfp4_weight(
                    w13.cuda(), _latent_dim, gate_up=True
                )
            elif _use_ck_moe:
                w13 = _preshuffle_mxfp4_weight(
                    w13.cuda().reshape(-1, _latent_dim // 2), _latent_dim
                ).reshape(w13.shape)
            else:
                w13 = w13.cuda()
            maybe_emit(
                convert_name(
                    f"language_model.model.layers.{layer_id}"
                    f".block_sparse_moe.experts.w13_weight{_routed_layout_suffix}"
                ),
                w13,
            )

            # Pack w13_scale.
            w13_scale_parts = [
                torch.cat(
                    [
                        _scale_to_uint8(e["gate_proj.weight_scale"]),
                        _scale_to_uint8(e["up_proj.weight_scale"]),
                    ],
                    dim=0,
                )
                for e in local_experts_list
            ]
            w13_scale = torch.stack(w13_scale_parts, dim=0)
            if _use_flatmm_moe:
                w13_scale = _preshuffle_flatmm_mxfp4_scale(
                    w13_scale.cuda(), gate_up=True
                )
            elif _use_ck_moe:
                w13_scale = _preshuffle_mxfp4_scale(
                    w13_scale.cuda().reshape(-1, _latent_dim // 32),
                    _latent_dim // 32,
                ).reshape(w13_scale.shape)
            else:
                w13_scale = w13_scale.cuda()
            maybe_emit(
                convert_name(
                    f"language_model.model.layers.{layer_id}"
                    f".block_sparse_moe.experts.w13_weight_scale{_routed_layout_suffix}"
                ),
                w13_scale,
            )

            # Pack w2 = down_proj.weight_packed.
            w2_parts = [
                _packed_to_uint8(e["down_proj.weight_packed"])
                for e in local_experts_list
            ]
            w2 = torch.stack(w2_parts, dim=0)
            if _use_flatmm_moe:
                w2 = _preshuffle_flatmm_mxfp4_weight(
                    w2.cuda(), _inter, gate_up=False
                )
            elif _use_ck_moe:
                w2 = _preshuffle_mxfp4_weight(
                    w2.cuda().reshape(-1, _inter // 2), _inter
                ).reshape(w2.shape)
            else:
                w2 = w2.cuda()
            maybe_emit(
                convert_name(
                    f"language_model.model.layers.{layer_id}"
                    f".block_sparse_moe.experts.w2_weight{_routed_layout_suffix}"
                ),
                w2,
            )

            # Pack w2_scale.
            w2_scale_parts = [
                _scale_to_uint8(e["down_proj.weight_scale"])
                for e in local_experts_list
            ]
            w2_scale = torch.stack(w2_scale_parts, dim=0)
            if _use_flatmm_moe:
                w2_scale = _preshuffle_flatmm_mxfp4_scale(
                    w2_scale.cuda(), gate_up=False
                )
            elif _use_ck_moe:
                w2_scale = _preshuffle_mxfp4_scale(
                    w2_scale.cuda().reshape(-1, _inter // 32),
                    _inter // 32,
                ).reshape(w2_scale.shape)
            else:
                w2_scale = w2_scale.cuda()
            maybe_emit(
                convert_name(
                    f"language_model.model.layers.{layer_id}"
                    f".block_sparse_moe.experts.w2_weight_scale{_routed_layout_suffix}"
                ),
                w2_scale,
            )

            # Local expert mask (EP).
            mask_name = convert_name(
                f"language_model.model.layers.{layer_id}"
                f".block_sparse_moe.experts.local_expert_mask"
            )
            if expected_constant_names is None or mask_name in expected_constant_names:
                local_mask = torch.zeros(
                    (num_experts,), dtype=torch.int32
                )
                local_mask[local_expert_ids] = 1
                params_paiton[mask_name] = local_mask.cuda()

        # ---- Shared experts (dense SiTU MLP, not MXFP4). --------------- #
        # K3's shared experts are unquantized bf16 gate_proj/up_proj/down_proj.
        # The compiled model fuses them into gate_up_proj.
        for layer_id, shared in enumerate(layers_shared_experts):
            if not shared:
                continue
            # Fuse gate_proj + up_proj -> gate_up_proj (column-parallel).
            if "gate_proj.weight" in shared and "up_proj.weight" in shared:
                fused = torch.cat(
                    [
                        get_rank_weight(shared["gate_proj.weight"], dim=0),
                        get_rank_weight(shared["up_proj.weight"], dim=0),
                    ],
                    dim=0,
                )
                maybe_emit(
                    convert_name(
                        f"language_model.model.layers.{layer_id}"
                        f".block_sparse_moe.shared_experts.gate_up_proj.weight"
                    ),
                    fused.cuda(),
                )
            # down_proj (row-parallel).
            if "down_proj.weight" in shared:
                value = get_rank_weight(shared["down_proj.weight"], dim=1)
                maybe_emit(
                    convert_name(
                        f"language_model.model.layers.{layer_id}"
                        f".block_sparse_moe.shared_experts.down_proj.weight"
                    ),
                    value.cuda(),
                )

        return params_paiton

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        import os

        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            torch.cuda.set_device(int(local_rank_env))

        pt_params = {name: tensor.detach().cpu() for name, tensor in weights}

        expected_all = set(
            self.model.get_constant_names(unbound_constants_only=False)
        )
        mapped = self.map_pt_params(
            pt_params,
            expected_constant_names=expected_all,
        )

        missing = sorted(expected_all - set(mapped.keys()))
        extra = sorted(set(mapped.keys()) - expected_all)
        if missing:
            raise RuntimeError(
                "Paiton K3 constants mismatch: did not provide values for "
                f"some expected constants. (mapped={len(mapped)}, "
                f"expected={len(expected_all)}). "
                "First 50 missing:\n- " + "\n- ".join(missing[:50])
            )
        if extra:
            raise RuntimeError(
                "Paiton K3 constants mismatch: produced constant names that "
                f"the compiled artifact does not expect. "
                f"(mapped={len(mapped)}, expected={len(expected_all)}). "
                "First 50 extra:\n- " + "\n- ".join(extra[:50])
            )

        self.model.set_many_constants_with_tensors(mapped)
        return set()
