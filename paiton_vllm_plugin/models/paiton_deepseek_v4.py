"""vLLM wrapper for Paiton-compiled DeepSeek V4 models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Set

import torch

from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.models.deepseek_v4_weights import DeepseekV4WeightsMixin
from paiton_vllm_plugin.models.paiton_qwen3_moe import PaitonQwen3MoeForCausalLM
from paiton_vllm_plugin.models.paiton_deepseek_v4_sparse import (
    DeepseekV4SparseRuntimeMixin,
)
from paiton_vllm_plugin.models.moe_topk_capture import MoeTopKCaptureMixin
from paiton_vllm_plugin.runtime.core import (
    PData,
    torch_dtype_to_string,
    torch_to_paiton_data,
)


@dataclass
class _DeepseekLayerInputBinding:
    layer_idx: int
    ctx: object
    kv_cache: torch.Tensor
    kv_cache_view: torch.Tensor
    kv_cache_pdata: PData
    kv_cache_data_ptr: int
    kv_cache_name: str
    compress_ratio: int
    needs_sparse: bool
    has_sparse_kv: bool
    has_compressor_state_cache: bool
    has_sparse_mla_compressed_offset: bool
    has_indexer_state_cache: bool
    has_sparse_mla_indexer_kv: bool
    has_indexer_q_fp8: bool
    has_indexer_weights: bool
    has_indexer_k_fp8: bool
    has_indexer_k_scale: bool
    has_cu_seqlen_ks: bool
    has_cu_seqlen_ke: bool
    has_indexer_num_existing_rows: bool
    has_sparse_mla_indices: bool
    has_sparse_mla_topk_length: bool


@dataclass
class _DeepseekInputPlan:
    signature: tuple
    expected_inputs: Set[str]
    ordered_input_names: tuple[str, ...]
    input_name_to_position: Dict[str, int]
    layer_bindings: tuple[_DeepseekLayerInputBinding, ...]
    needs_sparse_any: bool
    first_kv_cache: Optional[torch.Tensor]


class PaitonDeepseekV4ForCausalLM(
    MoeTopKCaptureMixin,
    DeepseekV4SparseRuntimeMixin,
    DeepseekV4WeightsMixin,
    PaitonQwen3MoeForCausalLM,
):
    """Runtime wrapper for DeepSeek V4 Flash Paiton artifacts.

    DeepSeek V4 uses native checkpoint names (`layers.*`, `embed.weight`,
    `head.weight`) and MXFP4 routed expert tensors.  The base Qwen3-MoE wrapper
    provides vLLM scheduler/runtime integration; this class supplies the
    DeepSeek-specific input filtering and constant mapping.
    """

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__(vllm_config, prefix=prefix)
        self._deepseek_sliding_window = getattr(self.config, "sliding_window", None)
        self._disable_vllm_sliding_window_check()
        self._paiton_graph_mode = os.getenv("PAITON_ENABLE_GRAPHS", "1") == "1"
        self._paiton_graph_max_seq_len = int(
            vllm_config.model_config.max_model_len
        )
        self._paiton_physical_block_high_water: int = 0
        self._configure_moe_topk_capture_state()

    def _disable_vllm_sliding_window_check(self) -> None:
        """Force sliding_window=None to avoid the vLLM MLA+SW assertion.

        vLLM's standard attention layer asserts
        ``not vllm_config.model_config.use_mla`` when ``sliding_window`` is set.
        DeepSeek V4 sets both, which would raise
        "MLA is not supported for slidingwindow" during KV-cache spec
        construction. PAITON handles attention through its own compiled
        artifact, so we clear sliding_window on both the config and the dummy
        vLLM attention layers to skip the vLLM check.
        """
        if getattr(self.config, "sliding_window", None) is not None:
            self.config.sliding_window = None
        static_context = getattr(self.compilation_config, "static_forward_context", {})
        for attn_layer in static_context.values():
            if hasattr(attn_layer, "sliding_window"):
                attn_layer.sliding_window = None
            impl = getattr(attn_layer, "impl", None)
            if hasattr(impl, "sliding_window"):
                impl.sliding_window = None

    @staticmethod
    def _get_kv_cache_tensor(ctx) -> Optional[torch.Tensor]:
        kv_cache = getattr(ctx, "kv_cache", None)
        if kv_cache is None:
            return None
        if torch.is_tensor(kv_cache):
            return kv_cache
        if isinstance(kv_cache, (list, tuple)):
            if len(kv_cache) == 0:
                return None
            return kv_cache[0]
        return kv_cache

    @staticmethod
    def _first_tensor_attr(obj, names: tuple[str, ...]) -> Optional[torch.Tensor]:
        for name in names:
            value = getattr(obj, name, None)
            if torch.is_tensor(value):
                return value
        return None

    def _sparse_mla_index_width(self) -> int:
        base_topk = int(getattr(self.config, "index_topk", 512))
        ratios = getattr(self.config, "compress_ratios", None) or []
        has_compressed_layers = any(max(1, int(ratio)) > 1 for ratio in ratios)
        window_size = int(getattr(self, "_deepseek_sliding_window", 0) or 0)
        if not has_compressed_layers or window_size <= 0:
            return base_topk
        combined = base_topk + window_size
        alignment = 128
        return ((combined + alignment - 1) // alignment) * alignment

    def _layer_compress_ratio(self, layer_idx: int) -> int:
        ratios = getattr(self.config, "compress_ratios", None) or []
        if layer_idx < len(ratios):
            return max(1, int(ratios[layer_idx]))
        return 1

    def _index_head_dim(self) -> int:
        return int(getattr(self.config, "index_head_dim",
                           getattr(self.config, "head_dim", 512)))

    def _index_n_heads(self) -> int:
        return int(getattr(self.config, "index_n_heads", 64))

    @staticmethod
    def _rounded_runtime_capacity(
        required: int,
        *,
        current: Optional[int] = None,
        quantum: int = 256,
        maximum: Optional[int] = None,
    ) -> int:
        required = max(1, int(required))
        quantum = max(1, int(quantum))
        rounded = ((required + quantum - 1) // quantum) * quantum
        if current is not None and current > 0:
            # Grow geometrically to keep cache pointers stable.
            current = int(current)
            growth_target = current * 2 if required > current else current
            rounded = max(rounded, growth_target)
        if maximum is not None and maximum > 0:
            rounded = min(rounded, int(maximum))
            if rounded < required:
                rounded = required
        return rounded

    def _compiled_indexer_overwrites_sparse_inputs(self) -> bool:
        """Whether sparse index inputs are write-only scratch for this model."""
        return False

    def _replicate_logits_if_needed(self, logits: torch.Tensor) -> None:
        """Broadcast logits only for legacy/single-rank compiled artifacts."""
        if (
            self.tp_size > 1
            and not bool(getattr(self, "_paiton_logits_replicated", False))
        ):
            get_tp_group().broadcast(logits, src=0)

    @staticmethod
    def _ordered_input_names(input_name_to_index) -> tuple[str, ...]:
        if isinstance(input_name_to_index, dict):
            return tuple(
                name
                for name, _ in sorted(
                    input_name_to_index.items(), key=lambda item: item[1]
                )
            )
        return tuple(input_name_to_index)

    def _deepseek_input_plan_signature(
        self,
        ordered_input_names: tuple[str, ...],
    ) -> tuple:
        static_context = getattr(
            self.compilation_config,
            "static_forward_context",
            {},
        )
        compress_ratios = tuple(getattr(self.config, "compress_ratios", None) or ())
        return (
            ordered_input_names,
            id(static_context),
            int(self.num_layers),
            self.cache_dtype,
            compress_ratios,
        )

    def _refresh_deepseek_kv_binding(
        self,
        binding: _DeepseekLayerInputBinding,
        *,
        validate_context: bool = False,
    ) -> tuple[torch.Tensor, PData]:
        kv_cache = binding.kv_cache
        if validate_context:
            current = self._get_kv_cache_tensor(binding.ctx)
            if current is not None:
                kv_cache = current
        current_ptr = kv_cache.data_ptr()
        if current_ptr == binding.kv_cache_data_ptr and kv_cache is binding.kv_cache:
            return binding.kv_cache, binding.kv_cache_pdata

        kv_cache_view = kv_cache.view(self.cache_dtype)
        binding.kv_cache = kv_cache
        binding.kv_cache_view = kv_cache_view
        binding.kv_cache_pdata = torch_to_paiton_data(kv_cache_view)
        binding.kv_cache_data_ptr = kv_cache.data_ptr()
        return binding.kv_cache, binding.kv_cache_pdata

    def _indexer_num_existing_rows(
        self,
        layer_idx: int,
        sparse_mla_indexer_kv: Optional[torch.Tensor],
    ) -> int:
        if sparse_mla_indexer_kv is not None:
            return int(sparse_mla_indexer_kv.shape[0])
        return 0

    def _get_indexer_k_quant_cache(
        self,
        layer_idx: int,
        num_rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return persistent FP8 K rows and scales for one sparse indexer.

        The compiled GLM K writer quantizes only slots appended by the current
        step.  These buffers must therefore outlive token-count-specific eager
        and graph scratch caches: prefill and decode normally have different
        token counts, but decode still reads every K row written by prefill.
        Preserve initialized rows when the physical sparse-cache capacity
        grows as the context grows.
        """
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        caches = getattr(self, "_indexer_k_quant_caches", None)
        if caches is None:
            caches = {}
            self._indexer_k_quant_caches = caches

        head_dim = self._index_head_dim()
        current = caches.get(layer_idx)
        current_fp8 = current[0] if current is not None else None
        current_scale = current[1] if current is not None else None
        current_capacity = (
            int(current_fp8.shape[0])
            if current_fp8 is not None
            and current_fp8.device == device
            and current_fp8.dtype == torch.float8_e4m3fnuz
            and current_fp8.ndim == 2
            and int(current_fp8.shape[1]) == head_dim
            and current_scale is not None
            and current_scale.device == device
            and current_scale.dtype == torch.float32
            else 0
        )

        if current_capacity < num_rows:
            capacity = self._rounded_runtime_capacity(
                num_rows,
                current=current_capacity or None,
                quantum=256,
            )
            next_fp8 = torch.empty(
                (capacity, head_dim),
                dtype=torch.float8_e4m3fnuz,
                device=device,
            )
            next_scale = torch.empty(
                (capacity,), dtype=torch.float32, device=device
            )
            if current_capacity:
                next_fp8[:current_capacity].copy_(
                    current_fp8[:current_capacity]
                )
                next_scale[:current_capacity].copy_(
                    current_scale[:current_capacity]
                )
            current_fp8, current_scale = next_fp8, next_scale
            caches[layer_idx] = (current_fp8, current_scale)

        return current_fp8[:num_rows], current_scale[:num_rows]

    def _sparse_mla_compressed_offset_input_value(
        self,
        step_sparse_slot_offset: int,
    ) -> int:
        return step_sparse_slot_offset

    def _sparse_mla_runtime_alias_key(self, layer_idx: int) -> int:
        """Return the sparse-index buffer alias group for a layer.

        DeepSeek mutates sparse-index inputs independently per layer, so the
        default is one backing per layer. GLM DSA overrides this for "shared"
        indexer layers that must consume a previous full-indexer layer's output.
        """
        return layer_idx

    @staticmethod
    def _validate_sparse_runtime_capacity(
        name: str,
        tensor: torch.Tensor,
        required_slots: int,
    ) -> None:
        capacity = int(tensor.shape[0])
        if capacity < required_slots:
            raise RuntimeError(
                f"{name} has {capacity} rows, but this step can reference "
                f"{required_slots} physical sparse MLA slots."
            )

    def _get_deepseek_input_plan(self) -> _DeepseekInputPlan:
        input_name_to_index = self.model.get_input_name_to_index_map()
        ordered_input_names = self._ordered_input_names(input_name_to_index)
        signature = self._deepseek_input_plan_signature(ordered_input_names)
        cached = getattr(self, "_deepseek_input_plan", None)
        if cached is not None and cached.signature == signature:
            return cached

        expected_inputs = set(ordered_input_names)
        static_context = getattr(
            self.compilation_config,
            "static_forward_context",
            {},
        )
        layer_bindings: list[_DeepseekLayerInputBinding] = []
        first_kv_cache = None
        needs_sparse_any = False
        for layer_idx in range(self.num_layers):
            ctx = static_context.get(str(layer_idx))
            if ctx is None:
                continue
            kv_cache = self._get_kv_cache_tensor(ctx)
            if kv_cache is None:
                continue
            if first_kv_cache is None:
                first_kv_cache = kv_cache

            prefix = {
                "kv_cache": f"kv_cache_{layer_idx}",
                "sparse_kv": f"sparse_mla_kv_{layer_idx}",
                "compressor_state": f"compressor_state_cache_{layer_idx}",
                "compressed_offset": f"sparse_mla_compressed_offset_{layer_idx}",
                "indexer_state": f"indexer_state_cache_{layer_idx}",
                "indexer_kv": f"sparse_mla_indexer_kv_{layer_idx}",
                "indexer_q_fp8": f"indexer_q_fp8_{layer_idx}",
                "indexer_weights": f"indexer_weights_{layer_idx}",
                "indexer_k_fp8": f"indexer_k_fp8_{layer_idx}",
                "indexer_k_scale": f"indexer_k_scale_{layer_idx}",
                "cu_seqlen_ks": f"cu_seqlen_ks_{layer_idx}",
                "cu_seqlen_ke": f"cu_seqlen_ke_{layer_idx}",
                "indexer_num_existing_rows": f"indexer_num_existing_rows_{layer_idx}",
                "sparse_indices": f"sparse_mla_indices_{layer_idx}",
                "sparse_topk": f"sparse_mla_topk_length_{layer_idx}",
            }
            has_sparse_kv = prefix["sparse_kv"] in expected_inputs
            has_indexer_kv = prefix["indexer_kv"] in expected_inputs
            has_sparse_indices = prefix["sparse_indices"] in expected_inputs
            has_sparse_topk = prefix["sparse_topk"] in expected_inputs
            has_compressed_offset = prefix["compressed_offset"] in expected_inputs
            needs_sparse = (
                has_sparse_kv
                or has_indexer_kv
                or has_sparse_indices
                or has_sparse_topk
                or has_compressed_offset
            )
            needs_sparse_any = needs_sparse_any or needs_sparse

            kv_cache_view = kv_cache.view(self.cache_dtype)
            layer_bindings.append(
                _DeepseekLayerInputBinding(
                    layer_idx=layer_idx,
                    ctx=ctx,
                    kv_cache=kv_cache,
                    kv_cache_view=kv_cache_view,
                    kv_cache_pdata=torch_to_paiton_data(kv_cache_view),
                    kv_cache_data_ptr=kv_cache.data_ptr(),
                    kv_cache_name=prefix["kv_cache"],
                    compress_ratio=self._layer_compress_ratio(layer_idx),
                    needs_sparse=needs_sparse,
                    has_sparse_kv=has_sparse_kv,
                    has_compressor_state_cache=(
                        prefix["compressor_state"] in expected_inputs
                    ),
                    has_sparse_mla_compressed_offset=has_compressed_offset,
                    has_indexer_state_cache=(
                        prefix["indexer_state"] in expected_inputs
                    ),
                    has_sparse_mla_indexer_kv=has_indexer_kv,
                    has_indexer_q_fp8=prefix["indexer_q_fp8"] in expected_inputs,
                    has_indexer_weights=(
                        prefix["indexer_weights"] in expected_inputs
                    ),
                    has_indexer_k_fp8=prefix["indexer_k_fp8"] in expected_inputs,
                    has_indexer_k_scale=(
                        prefix["indexer_k_scale"] in expected_inputs
                    ),
                    has_cu_seqlen_ks=prefix["cu_seqlen_ks"] in expected_inputs,
                    has_cu_seqlen_ke=prefix["cu_seqlen_ke"] in expected_inputs,
                    has_indexer_num_existing_rows=(
                        prefix["indexer_num_existing_rows"] in expected_inputs
                    ),
                    has_sparse_mla_indices=has_sparse_indices,
                    has_sparse_mla_topk_length=has_sparse_topk,
                )
            )

        plan = _DeepseekInputPlan(
            signature=signature,
            expected_inputs=expected_inputs,
            ordered_input_names=ordered_input_names,
            input_name_to_position={
                name: idx for idx, name in enumerate(ordered_input_names)
            },
            layer_bindings=tuple(layer_bindings),
            needs_sparse_any=needs_sparse_any,
            first_kv_cache=first_kv_cache,
        )
        self._deepseek_input_plan = plan
        return plan

    def _ensure_sparse_mla_inputs(
        self,
        expected_inputs: Set[str],
        inputs: Dict[str, PData],
        input_ids: torch.Tensor,
        keepalive: list[torch.Tensor],
    ) -> None:
        """Validate sparse MLA inputs and optionally fill debug placeholders."""
        missing_sparse = {
            name
            for name in expected_inputs
            if name.startswith("sparse_mla_") and name not in inputs
        }
        if not missing_sparse:
            return

        if os.getenv("PAITON_ALLOW_SPARSE_MLA_PLACEHOLDERS", "0") != "1":
            raise RuntimeError(
            "DeepSeek V4 sparse MLA runtime inputs are missing. The "
                "compiled artifact expects sparse MLA KV, index, top-k, or "
                "indexer KV tensors, but the runtime did not provide them. "
                "Placeholder sparse MLA inputs are numerically invalid and "
                "are disabled by default. Missing inputs: "
                + ", ".join(sorted(missing_sparse))
            )

        device = input_ids.device
        num_tokens = int(input_ids.shape[0])
        head_dim = int(getattr(self.config, "head_dim", 512))
        index_topk = self._sparse_mla_index_width()

        empty_sparse_kv = torch.empty(
            (0, 1, head_dim),
            dtype=self.dtype,
            device=device,
        )
        empty_sparse_indexer_kv = torch.empty(
            (0, 1, self._index_head_dim()),
            dtype=self.dtype,
            device=device,
        )
        zero_sparse_indices = torch.zeros(
            (num_tokens, index_topk),
            dtype=torch.int32,
            device=device,
        )
        zero_topk_length = torch.zeros(
            (num_tokens,),
            dtype=torch.int32,
            device=device,
        )
        keepalive.extend([empty_sparse_kv, zero_sparse_indices, zero_topk_length])
        if any(name.startswith("sparse_mla_indexer_kv_") for name in missing_sparse):
            keepalive.append(empty_sparse_indexer_kv)

        sparse_kv_pdata = torch_to_paiton_data(empty_sparse_kv)
        sparse_indexer_kv_pdata = torch_to_paiton_data(empty_sparse_indexer_kv)
        sparse_indices_pdata = torch_to_paiton_data(zero_sparse_indices)
        sparse_topk_length_pdata = torch_to_paiton_data(zero_topk_length)

        for name in missing_sparse:
            if name in inputs:
                continue
            if name.startswith("sparse_mla_kv_"):
                inputs[name] = sparse_kv_pdata
            elif name.startswith("sparse_mla_indexer_kv_"):
                inputs[name] = sparse_indexer_kv_pdata
            elif name.startswith("sparse_mla_indices_"):
                inputs[name] = sparse_indices_pdata
            elif name.startswith("sparse_mla_topk_length_"):
                inputs[name] = sparse_topk_length_pdata

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        del intermediate_tensors, inputs_embeds
        forward_context: ForwardContext = get_forward_context()
        all_attn_metadata = forward_context.attn_metadata
        if not all_attn_metadata:
            output = torch.empty(
                [input_ids.shape[0], self.config.vocab_size],
                dtype=torch.float32,
                device=input_ids.device,
            )
            return output

        attn_metadata = all_attn_metadata["0"]
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len
        # Prefill remains eager because its shapes are transient.
        graph_mode = self._paiton_graph_mode and int(max_query_len) == 1
        num_runtime_rows = int(input_ids.shape[0])
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is not None and seq_lens.ndim == 1 and seq_lens.numel() > 0:
            num_runtime_rows = int(seq_lens.shape[0])
        else:
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            if query_start_loc is not None and query_start_loc.numel() >= 2:
                num_runtime_rows = int(query_start_loc.numel() - 1)

        def allocate_logits(rows: int) -> torch.Tensor:
            if not graph_mode:
                return torch.empty(
                    [rows, self.config.vocab_size],
                    dtype=torch.float32,
                    device=input_ids.device,
                )
            cache = getattr(self, "_paiton_graph_output_cache", None)
            if cache is None:
                cache = {}
                self._paiton_graph_output_cache = cache
            key = (input_ids.device, rows)
            cached = cache.get(key)
            if cached is None:
                cached = torch.empty(
                    [rows, self.config.vocab_size],
                    dtype=torch.float32,
                    device=input_ids.device,
                )
                cache[key] = cached
            return cached

        output = allocate_logits(int(input_ids.shape[0]))
        if num_runtime_rows == output.shape[0]:
            runtime_output = output
        else:
            output.zero_()
            runtime_output = allocate_logits(num_runtime_rows)

        device = input_ids.device

        def prepare_runtime_input(
            name: str,
            tensor: torch.Tensor,
            dtype: torch.dtype,
        ) -> torch.Tensor:
            if not graph_mode:
                return tensor.to(
                    device=device, dtype=dtype, copy=False
                ).contiguous()

            # Captured kernels require stable input addresses.
            cache = getattr(self, "_paiton_graph_input_cache", None)
            if cache is None:
                cache = {}
                self._paiton_graph_input_cache = cache
            key = (device, name, tuple(tensor.shape), dtype)
            persistent = cache.get(key)
            if persistent is None:
                persistent = torch.empty(
                    tensor.shape, dtype=dtype, device=device
                )
                cache[key] = persistent
            persistent.copy_(tensor)
            return persistent

        input_ids_i32 = prepare_runtime_input(
            "input_ids", input_ids, torch.int32
        )
        run_input_backings: list[torch.Tensor] = [input_ids_i32]
        input_plan = self._get_deepseek_input_plan()
        expected_inputs = input_plan.expected_inputs
        ordered_input_names = input_plan.ordered_input_names
        ordered_inputs: list[Optional[PData]] = [None] * len(ordered_input_names)
        ordered_ptrs: list[int] = [0] * len(ordered_input_names)
        ordered_shape_sigs: list[Optional[tuple]] = [None] * len(ordered_input_names)
        bound_input_count = [0]

        class _TrackedInputs(dict):
            def __setitem__(tracked_self, key, value):  # type: ignore[no-untyped-def]
                super().__setitem__(key, value)
                idx = input_plan.input_name_to_position.get(key)
                if idx is None:
                    return
                if ordered_inputs[idx] is None:
                    bound_input_count[0] += 1
                ordered_inputs[idx] = value
                ordered_ptrs[idx] = int(value.data_ptr)
                ordered_shape_sigs[idx] = (tuple(value.shape), value.dtype)

        inputs: Dict[str, PData] = _TrackedInputs()
        inputs["input_ids"] = torch_to_paiton_data(input_ids_i32)

        slot_mapping_i64 = None
        query_start_loc_i32 = None
        seq_lens_i32 = None
        block_table_i32 = None

        # Capture context for the indexer FP8 MQA runtime hooks.
        def num_tokens_for_input_alloc() -> int:
            return int(input_ids.shape[0])

        if positions is not None:
            position_ids_i64 = prepare_runtime_input(
                "position_ids", positions, torch.int64
            )
            run_input_backings.append(position_ids_i64)
            inputs["position_ids"] = torch_to_paiton_data(position_ids_i64)

        if hasattr(attn_metadata, "slot_mapping"):
            slot_mapping_i64 = prepare_runtime_input(
                "slot_mapping", attn_metadata.slot_mapping, torch.int64
            )
            run_input_backings.append(slot_mapping_i64)
            inputs["slot_mapping"] = torch_to_paiton_data(slot_mapping_i64)

        if hasattr(attn_metadata, "query_start_loc"):
            query_start_loc_i32 = prepare_runtime_input(
                "query_start_locations",
                attn_metadata.query_start_loc,
                torch.int32,
            )
            run_input_backings.append(query_start_loc_i32)
            inputs["query_start_locations"] = torch_to_paiton_data(
                query_start_loc_i32)

        if hasattr(attn_metadata, "seq_lens"):
            seq_lens_i32 = prepare_runtime_input(
                "context_lengths", attn_metadata.seq_lens, torch.int32
            )
            run_input_backings.append(seq_lens_i32)
            inputs["context_lengths"] = torch_to_paiton_data(seq_lens_i32)

        if hasattr(attn_metadata, "block_table"):
            block_table_i32 = prepare_runtime_input(
                "block_tables", attn_metadata.block_table, torch.int32
            )
            run_input_backings.append(block_table_i32)
            inputs["block_tables"] = torch_to_paiton_data(block_table_i32)


        generated_sparse_indices = None
        generated_sparse_topk_length = None

        if graph_mode:
            scalar_cache = getattr(self, "_paiton_graph_scalar_cache", None)
            if scalar_cache is None:
                scalar_cache = {}
                self._paiton_graph_scalar_cache = scalar_cache
            scalar_key = (device, "max_lengths")
            scalar_pair = scalar_cache.get(scalar_key)
            if scalar_pair is None:
                scalar_pair = (
                    torch.empty([1], dtype=torch.int32, device=device),
                    torch.empty([1], dtype=torch.int32, device=device),
                )
                scalar_cache[scalar_key] = scalar_pair
            max_query_len_backing, max_seq_len_backing = scalar_pair
        else:
            max_query_len_backing = torch.empty(
                [1], dtype=torch.int32, device=device
            )
            max_seq_len_backing = torch.empty(
                [1], dtype=torch.int32, device=device
            )
        max_query_len_backing.fill_(int(max_query_len))
        max_seq_len_backing.fill_(int(max_seq_len))
        run_input_backings.extend([max_query_len_backing, max_seq_len_backing])
        # Match graph scalar shapes to the fixed indexer workspace.
        bound_max_seq_len = int(max_seq_len)
        if graph_mode:
            bound_max_seq_len = max(
                bound_max_seq_len,
                int(
                    getattr(
                        self,
                        "_paiton_graph_max_seq_len",
                        bound_max_seq_len,
                    )
                ),
            )
            bound_max_seq_len = (
                (bound_max_seq_len + 255) // 256
            ) * 256
        inputs["max_query_len"] = PData(
            max_query_len_backing.data_ptr(),
            [max_query_len, 0],
            torch_dtype_to_string(torch.int32),
        )
        inputs["max_seq_len"] = PData(
            max_seq_len_backing.data_ptr(),
            [bound_max_seq_len, 0],
            torch_dtype_to_string(torch.int32),
        )

        needs_sparse_any = input_plan.needs_sparse_any
        first_kv_cache = input_plan.first_kv_cache
        compiled_indexer_outputs = (
            self._compiled_indexer_overwrites_sparse_inputs()
        )
        if needs_sparse_any and not compiled_indexer_outputs:
            if (
                generated_sparse_indices is None
                and query_start_loc_i32 is not None
                and seq_lens_i32 is not None
                and block_table_i32 is not None
                and first_kv_cache is not None
            ):
                generated_sparse_indices, generated_sparse_topk_length = (
                    self._build_recent_sparse_mla_indices(
                        query_start_loc_i32,
                        seq_lens_i32,
                        block_table_i32,
                        int(input_ids.shape[0]),
                        self._sparse_mla_index_width(),
                        int(first_kv_cache.shape[2]),
                        input_ids.device,
                        int(self._deepseek_sliding_window or 0),
                    )
                )
                run_input_backings.extend(
                    [generated_sparse_indices, generated_sparse_topk_length]
                )
            elif generated_sparse_indices is None and needs_sparse_any:
                raise RuntimeError(
                    "DeepSeek V4 sparse MLA needs query_start_loc, seq_lens, "
                    "and block_table metadata to build sparse indices."
                )


        # Compute shared slot/block extents once per step.
        step_required_slots, step_max_slot, step_max_block, step_required_blocks_bt = (
            self._compute_step_slot_extents(
                slot_mapping_i64,
                generated_sparse_indices,
                block_table_i32,
            )
        )


        step_sparse_slot_offset = self._rounded_runtime_capacity(
            step_required_slots,
            quantum=256,
            maximum=(
                int(first_kv_cache.shape[1]) * int(first_kv_cache.shape[2])
                if first_kv_cache is not None
                else None
            ),
        )
        compressed_slot_offset_input_value = (
            self._sparse_mla_compressed_offset_input_value(step_sparse_slot_offset)
        )
        compressed_slot_offset_tensor = None
        if needs_sparse_any:
            if graph_mode:
                scalar_key = (device, "compressed_slot_offset")
                compressed_slot_offset_tensor = scalar_cache.get(scalar_key)
                if compressed_slot_offset_tensor is None:
                    compressed_slot_offset_tensor = torch.empty(
                        [1], dtype=torch.int64, device=device
                    )
                    scalar_cache[scalar_key] = compressed_slot_offset_tensor
                compressed_slot_offset_tensor.fill_(
                    compressed_slot_offset_input_value
                )
            else:
                compressed_slot_offset_tensor = torch.tensor(
                    [compressed_slot_offset_input_value],
                    dtype=torch.int64,
                    device=device,
                )

        # Reuse scratch buffers across decode steps.
        nt = num_tokens_for_input_alloc()
        if graph_mode:
            # Preserve graph storage independently for each token count.
            graph_scratch_caches = getattr(
                self, "_paiton_graph_scratch_caches", None
            )
            if graph_scratch_caches is None:
                graph_scratch_caches = {}
                self._paiton_graph_scratch_caches = graph_scratch_caches
            sc = graph_scratch_caches.setdefault(nt, {"nt": nt})
        else:
            if (
                not hasattr(self, "_scratch_cache")
                or self._scratch_cache.get("nt") != nt
            ):
                self._scratch_cache = {"nt": nt}
            sc = self._scratch_cache

        # Full-indexer layers share one MFMA logits workspace.
        if "indexer_logits_workspace" in expected_inputs:
            logits_stride = int(max_seq_len)
            if graph_mode:
                # A fixed stride avoids graph recapture as context grows.
                logits_stride = max(
                    logits_stride,
                    int(
                        getattr(
                            self,
                            "_paiton_graph_max_seq_len",
                            logits_stride,
                        )
                    ),
                )
                logits_stride = (
                    (logits_stride + 255) // 256
                ) * 256
            index_topk = int(getattr(self.config, "index_topk", 2048))
            required_logits = nt * logits_stride if logits_stride > index_topk else 1
            logits_capacity = 1 << max(0, required_logits - 1).bit_length()
            logits_scratch = sc.get("indexer_logits_workspace")
            if logits_scratch is None or logits_scratch.numel() < logits_capacity:
                logits_scratch = torch.empty(
                    logits_capacity, dtype=torch.float32, device=device
                )
                sc["indexer_logits_workspace"] = logits_scratch
            run_input_backings.append(logits_scratch)
            inputs["indexer_logits_workspace"] = PData(
                logits_scratch.data_ptr(),
                [nt, logits_stride],
                torch_dtype_to_string(torch.float32),
            )
        validate_cached_kv = os.getenv("PAITON_VALIDATE_KV_BINDINGS", "0") == "1"
        c128_sparse_input_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        sparse_input_alias_cache: Dict[
            int,
            tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        ] = {}
        for layer_binding in input_plan.layer_bindings:
            i = layer_binding.layer_idx
            ctx = layer_binding.ctx
            layer_attn_metadata = (
                all_attn_metadata.get(str(i), attn_metadata)
                if isinstance(all_attn_metadata, dict)
                else attn_metadata
            )
            kv_cache, kv_cache_pdata = self._refresh_deepseek_kv_binding(
                layer_binding,
                validate_context=validate_cached_kv,
            )
            run_input_backings.append(kv_cache)
            inputs[layer_binding.kv_cache_name] = kv_cache_pdata

            sparse_kv = None
            if layer_binding.has_sparse_kv:
                sparse_kv = self._first_tensor_attr(
                    ctx,
                    (
                        "sparse_mla_kv",
                        "sparse_mla_kv_cache",
                        "mla_kv_cache",
                        "kv_c_and_k_pe_cache",
                        "compressed_kv_cache",
                    ),
                )
                if sparse_kv is None:
                    if slot_mapping_i64 is not None and (
                        generated_sparse_indices is not None
                        or compiled_indexer_outputs
                    ):
                        compressed_slot_offset = (
                            step_sparse_slot_offset
                            if layer_binding.compress_ratio > 1
                            else None
                        )
                        sparse_kv = self._get_sparse_mla_kv_cache(
                            i,
                            kv_cache,
                            slot_mapping_i64,
                            generated_sparse_indices,
                            compressed_slot_offset=compressed_slot_offset,
                            required_slots=step_required_slots,
                        )
                    else:
                        raise RuntimeError(
                            f"sparse_mla_kv_{i} is expected but runtime sparse "
                            "KV cache metadata is unavailable."
                        )
            if sparse_kv is not None:
                self._validate_sparse_runtime_capacity(
                    f"sparse_mla_kv_{i}",
                    sparse_kv,
                    step_required_slots,
                )
                run_input_backings.append(sparse_kv)
                inputs[f"sparse_mla_kv_{i}"] = torch_to_paiton_data(sparse_kv)

            if layer_binding.has_compressor_state_cache:
                compressor_state_cache = self._get_deepseek_v4_state_cache(
                    i,
                    kv_cache,
                    slot_mapping_i64,
                    block_table_i32,
                    indexer=False,
                    max_slot=step_max_slot,
                    max_block=step_max_block,
                )
                run_input_backings.append(compressor_state_cache)
                inputs[f"compressor_state_cache_{i}"] = torch_to_paiton_data(
                    compressor_state_cache)

            if layer_binding.has_sparse_mla_compressed_offset:
                if compressed_slot_offset_tensor is None:
                    raise RuntimeError(
                        f"sparse_mla_compressed_offset_{i} is expected but "
                        "the sparse MLA runtime offset tensor is unavailable."
                    )
                run_input_backings.append(compressed_slot_offset_tensor)
                inputs[f"sparse_mla_compressed_offset_{i}"] = torch_to_paiton_data(
                    compressed_slot_offset_tensor)

            if layer_binding.has_indexer_state_cache:
                indexer_state_cache = self._get_deepseek_v4_state_cache(
                    i,
                    kv_cache,
                    slot_mapping_i64,
                    block_table_i32,
                    indexer=True,
                    max_slot=step_max_slot,
                    max_block=step_max_block,
                )
                run_input_backings.append(indexer_state_cache)
                inputs[f"indexer_state_cache_{i}"] = torch_to_paiton_data(
                    indexer_state_cache)

            sparse_mla_indexer_kv = None
            if layer_binding.has_sparse_mla_indexer_kv:
                if slot_mapping_i64 is None:
                    raise RuntimeError(
                        f"sparse_mla_indexer_kv_{i} is expected but slot_mapping "
                        "metadata is unavailable.")
                sparse_mla_indexer_kv = self._get_sparse_mla_indexer_kv_cache(
                    i,
                    kv_cache,
                    slot_mapping_i64,
                    generated_sparse_indices,
                    compressed_slot_offset=(
                        step_sparse_slot_offset
                        if layer_binding.compress_ratio > 1
                        else None
                    ),
                    required_slots=step_required_slots,
                )
                self._validate_sparse_runtime_capacity(
                    f"sparse_mla_indexer_kv_{i}",
                    sparse_mla_indexer_kv,
                    step_required_slots,
                )
                run_input_backings.append(sparse_mla_indexer_kv)
                inputs[f"sparse_mla_indexer_kv_{i}"] = torch_to_paiton_data(
                    sparse_mla_indexer_kv)

            # The compiled artifact produces the C4 sparse-indexer aux tensors
            # in-graph. The runtime only allocates storage and binds it here.
            if layer_binding.has_indexer_q_fp8:
                key = f"iq8_{i}"
                if key not in sc:
                    sc[key] = torch.empty(
                        (nt, self._index_n_heads(), self._index_head_dim()),
                        dtype=torch.float8_e4m3fnuz, device=device,
                    )
                indexer_q_fp8 = sc[key]
                run_input_backings.append(indexer_q_fp8)
                inputs[f"indexer_q_fp8_{i}"] = torch_to_paiton_data(indexer_q_fp8)

            if layer_binding.has_indexer_weights:
                key = f"iw_{i}"
                if key not in sc:
                    sc[key] = torch.empty(
                        (nt, self._index_n_heads()),
                        dtype=torch.float32, device=device,
                    )
                indexer_weights = sc[key]
                run_input_backings.append(indexer_weights)
                inputs[f"indexer_weights_{i}"] = torch_to_paiton_data(indexer_weights)

            if (
                layer_binding.has_indexer_k_fp8
                or layer_binding.has_indexer_k_scale
            ):
                if sparse_mla_indexer_kv is None:
                    raise RuntimeError(
                        f"indexer_k_fp8_{i} / indexer_k_scale_{i} require "
                        f"sparse_mla_indexer_kv_{i}, but it was not bound."
                    )
                num_sparse_rows = int(sparse_mla_indexer_kv.shape[0])
                indexer_k_fp8, indexer_k_scale = (
                    self._get_indexer_k_quant_cache(
                        i, num_sparse_rows, device
                    )
                )
                if layer_binding.has_indexer_k_fp8:
                    run_input_backings.append(indexer_k_fp8)
                    inputs[f"indexer_k_fp8_{i}"] = torch_to_paiton_data(indexer_k_fp8)
                if layer_binding.has_indexer_k_scale:
                    run_input_backings.append(indexer_k_scale)
                    inputs[f"indexer_k_scale_{i}"] = torch_to_paiton_data(indexer_k_scale)

            if (
                layer_binding.has_cu_seqlen_ks
                or layer_binding.has_cu_seqlen_ke
            ):
                cks_key = f"cs_{i}"
                cke_key = f"ce_{i}"
                if cks_key not in sc:
                    sc[cks_key] = torch.empty(
                        (nt,), dtype=torch.int32, device=device,
                    )
                    sc[cke_key] = torch.empty(
                        (nt,), dtype=torch.int32, device=device,
                    )
                cu_seqlen_ks = sc[cks_key]
                cu_seqlen_ke = sc[cke_key]
                if layer_binding.has_cu_seqlen_ks:
                    run_input_backings.append(cu_seqlen_ks)
                    inputs[f"cu_seqlen_ks_{i}"] = torch_to_paiton_data(cu_seqlen_ks)
                if layer_binding.has_cu_seqlen_ke:
                    run_input_backings.append(cu_seqlen_ke)
                    inputs[f"cu_seqlen_ke_{i}"] = torch_to_paiton_data(                    cu_seqlen_ke)

            if layer_binding.has_indexer_num_existing_rows:
                inr_key = f"inr_{i}"
                if inr_key not in sc:
                    sc[inr_key] = torch.empty((1,), dtype=torch.int32, device=device)
                # Quantize only newly appended K rows.
                num_existing = self._indexer_num_existing_rows(
                    i,
                    sparse_mla_indexer_kv,
                )
                sc[inr_key][0] = num_existing
                run_input_backings.append(sc[inr_key])
                inputs[f"indexer_num_existing_rows_{i}"] = torch_to_paiton_data(
                    sc[inr_key])

            if (
                layer_binding.has_sparse_mla_indices
                or layer_binding.has_sparse_mla_topk_length
            ):
                sparse_alias_key = self._sparse_mla_runtime_alias_key(i)
                cached_sparse_inputs = sparse_input_alias_cache.get(sparse_alias_key)
                if cached_sparse_inputs is not None:
                    sparse_indices, sparse_topk_length = cached_sparse_inputs
                    if sparse_indices is not None:
                        run_input_backings.append(sparse_indices)
                        inputs[f"sparse_mla_indices_{i}"] = torch_to_paiton_data(
                            sparse_indices)
                    elif layer_binding.has_sparse_mla_indices:
                        raise RuntimeError(
                            f"sparse_mla_indices_{i} aliases sparse MLA group "
                            f"{sparse_alias_key}, but that group has no indices."
                        )
                    if sparse_topk_length is not None:
                        run_input_backings.append(sparse_topk_length)
                        inputs[f"sparse_mla_topk_length_{i}"] = torch_to_paiton_data(
                            sparse_topk_length)
                    elif layer_binding.has_sparse_mla_topk_length:
                        raise RuntimeError(
                            f"sparse_mla_topk_length_{i} aliases sparse MLA group "
                            f"{sparse_alias_key}, but that group has no top-k lengths."
                        )
                    continue

                compressed_slot_offset_value = (
                    step_sparse_slot_offset
                    if (
                        slot_mapping_i64 is not None
                        and generated_sparse_indices is not None
                        and layer_binding.compress_ratio > 1
                    )
                    else None
                )
                sparse_indices = None
                sparse_topk_length = None
                sparse_inputs_need_copy = not compiled_indexer_outputs
                if not compiled_indexer_outputs:
                    sparse_indices = self._first_tensor_attr(
                        ctx,
                        (
                            "sparse_mla_indices",
                            "sparse_mla_topk_indices",
                            "topk_indices",
                            "topk_indices_buffer",
                        ),
                    )
                    if sparse_indices is None:
                        sparse_indices = self._first_tensor_attr(
                            layer_attn_metadata,
                            (
                                "sparse_mla_indices",
                                "sparse_mla_topk_indices",
                                "topk_indices",
                                "topk_indices_buffer",
                                "c128a_prefill_topk_indices",
                                "c128a_global_decode_topk_indices",
                            ),
                        )
                    sparse_topk_length = self._first_tensor_attr(
                        ctx,
                        (
                            "sparse_mla_topk_length",
                            "sparse_mla_topk_lengths",
                            "topk_length",
                            "topk_lengths",
                        ),
                    )
                    if sparse_topk_length is None:
                        sparse_topk_length = self._first_tensor_attr(
                            layer_attn_metadata,
                            (
                                "sparse_mla_topk_length",
                                "sparse_mla_topk_lengths",
                                "topk_length",
                                "topk_lengths",
                                "c128a_decode_topk_lens",
                            ),
                        )

                # The full indexer overwrites both scratch buffers.
                if compiled_indexer_outputs:
                    indices_key = f"compiled_sparse_indices_{sparse_alias_key}"
                    sparse_indices = sc.get(indices_key)
                    expected_indices_shape = (
                        nt,
                        self._sparse_mla_index_width(),
                    )
                    if (
                        sparse_indices is None
                        or tuple(sparse_indices.shape) != expected_indices_shape
                    ):
                        sparse_indices = torch.empty(
                            expected_indices_shape,
                            dtype=torch.int32,
                            device=device,
                        )
                        sc[indices_key] = sparse_indices
                    length_key = f"compiled_sparse_length_{sparse_alias_key}"
                    sparse_topk_length = sc.get(length_key)
                    if (
                        sparse_topk_length is None
                        or tuple(sparse_topk_length.shape) != (nt,)
                    ):
                        sparse_topk_length = torch.empty(
                            (nt,), dtype=torch.int32, device=device
                        )
                        sc[length_key] = sparse_topk_length

                if (
                    layer_binding.compress_ratio == 128
                    and compressed_slot_offset_value is not None
                    and query_start_loc_i32 is not None
                    and generated_sparse_indices is not None
                ):
                    c128_decode_indices = self._first_tensor_attr(
                        layer_attn_metadata,
                        ("c128a_global_decode_topk_indices",),
                    )
                    c128_decode_topk_length = self._first_tensor_attr(
                        layer_attn_metadata,
                        ("c128a_decode_topk_lens",),
                    )
                    c128_prefill_local = self._first_tensor_attr(
                        layer_attn_metadata,
                        ("c128a_prefill_topk_indices",),
                    )
                    c128_block_table = getattr(layer_attn_metadata, "block_table", None)
                    c128_cache_key = (
                        layer_binding.compress_ratio,
                        int(compressed_slot_offset_value),
                        int(getattr(layer_attn_metadata, "block_size", 64)),
                        id(c128_decode_indices),
                        id(c128_decode_topk_length),
                        id(c128_prefill_local),
                        id(c128_block_table),
                        id(query_start_loc_i32),
                        id(generated_sparse_indices),
                        id(generated_sparse_topk_length),
                        id(position_ids_i64) if positions is not None else None,
                        id(slot_mapping_i64),
                    )
                    cached_sparse_inputs = c128_sparse_input_cache.get(c128_cache_key)
                    if cached_sparse_inputs is None:
                        cached_sparse_inputs = self._build_c128a_sparse_inputs(
                            layer_attn_metadata,
                            query_start_loc_i32,
                            compressed_slot_offset_value,
                            generated_sparse_indices,
                            generated_sparse_topk_length,
                            layer_idx=i,
                            positions=(
                                position_ids_i64 if positions is not None else None
                            ),
                            slot_mapping=slot_mapping_i64,
                        )
                        c128_sparse_input_cache[c128_cache_key] = cached_sparse_inputs
                    sparse_indices, sparse_topk_length = cached_sparse_inputs
                elif sparse_indices is None:
                    sparse_indices = generated_sparse_indices
                if sparse_indices is not None:
                    sparse_indices_backing = None
                    if graph_mode and sparse_inputs_need_copy:
                        backing_key = f"graph_sparse_indices_{sparse_alias_key}"
                        sparse_indices_backing = sc.get(backing_key)
                        if (
                            sparse_indices_backing is None
                            or sparse_indices_backing.shape != sparse_indices.shape
                        ):
                            sparse_indices_backing = torch.empty_like(
                                sparse_indices,
                                dtype=torch.int32,
                                memory_format=torch.contiguous_format,
                            )
                            sc[backing_key] = sparse_indices_backing
                    if sparse_inputs_need_copy:
                        sparse_indices = self._layer_sparse_input_copy(
                            sparse_indices,
                            dtype=torch.int32,
                            out=sparse_indices_backing,
                        )
                    run_input_backings.append(sparse_indices)
                    inputs[f"sparse_mla_indices_{i}"] = torch_to_paiton_data(
                        sparse_indices)
                elif layer_binding.has_sparse_mla_indices:
                    raise RuntimeError(
                        f"sparse_mla_indices_{i} is expected but could not be generated. "
                        "This indicates a bug in the sparse MLA input generation logic."
                    )

                if sparse_topk_length is None:
                    sparse_topk_length = generated_sparse_topk_length
                if sparse_topk_length is None and sparse_indices is not None:
                    sparse_topk_length = (sparse_indices >= 0).sum(dim=-1).to(
                        dtype=torch.int32
                    )
                if sparse_topk_length is not None:
                    sparse_length_backing = None
                    if graph_mode and sparse_inputs_need_copy:
                        backing_key = f"graph_sparse_length_{sparse_alias_key}"
                        sparse_length_backing = sc.get(backing_key)
                        if (
                            sparse_length_backing is None
                            or sparse_length_backing.shape
                            != sparse_topk_length.shape
                        ):
                            sparse_length_backing = torch.empty_like(
                                sparse_topk_length,
                                dtype=torch.int32,
                                memory_format=torch.contiguous_format,
                            )
                            sc[backing_key] = sparse_length_backing
                    if sparse_inputs_need_copy:
                        sparse_topk_length = self._layer_sparse_input_copy(
                            sparse_topk_length,
                            dtype=torch.int32,
                            out=sparse_length_backing,
                        )
                    run_input_backings.append(sparse_topk_length)
                    inputs[f"sparse_mla_topk_length_{i}"] = torch_to_paiton_data(
                        sparse_topk_length)
                elif layer_binding.has_sparse_mla_topk_length:
                    raise RuntimeError(
                        f"sparse_mla_topk_length_{i} is expected but could not be generated. "
                        "This indicates a bug in the sparse MLA input generation logic."
                    )
                sparse_input_alias_cache[sparse_alias_key] = (
                    sparse_indices,
                    sparse_topk_length,
                )

        self._ensure_sparse_mla_inputs(
            expected_inputs,
            inputs,
            input_ids,
            run_input_backings,
        )
        if bound_input_count[0] != len(ordered_input_names):
            for idx, name in enumerate(ordered_input_names):
                if ordered_inputs[idx] is None and name in inputs:
                    pd = inputs[name]
                    ordered_inputs[idx] = pd
                    ordered_ptrs[idx] = int(pd.data_ptr)
                    ordered_shape_sigs[idx] = (tuple(pd.shape), pd.dtype)

        missing_inputs = {
            name
            for idx, name in enumerate(ordered_input_names)
            if ordered_inputs[idx] is None
        }
        if missing_inputs:
            raise RuntimeError(
                "Paiton DeepSeek V4 artifact input mismatch. Missing inputs: "
                + ", ".join(sorted(missing_inputs)))

        outputs = {"logits": torch_to_paiton_data(runtime_output)}

        self._ensure_moe_topk_capture_state()
        if self._moe_topk_capture is None:
            self._init_moe_topk_capture(runtime_output.device)
        self._bind_moe_topk_outputs(outputs, nt)
        stream_ptr = torch.cuda.current_stream().cuda_stream
        self._run_input_backings = run_input_backings

        # Stable decode updates pointers without rebuilding input metadata.
        runtime_shape_sig = tuple(ordered_shape_sigs)
        can_bind = (
            os.getenv("PAITON_DISABLE_BOUND_RUN", "0") != "1"
            and len(ordered_inputs) == len(expected_inputs)
            and getattr(self.model, "_bound_run_available", True)
        )
        use_bound = False
        if can_bind:
            try:
                previous_shape_sig = getattr(self, "_bound_runtime_shape_sig", None)
                if (
                    previous_shape_sig is None
                    or previous_shape_sig != runtime_shape_sig
                ):
                    self.model.bind_inputs(ordered_inputs)
                    self._bound_runtime_shape_sig = runtime_shape_sig
                    self._bound_input_names = list(ordered_input_names)
                    self._bound_runtime_ptrs = list(ordered_ptrs)
                    use_bound = True
                else:
                    if (
                        graph_mode
                        and os.getenv("PAITON_GRAPH_DEBUG", "0") == "1"
                    ):
                        previous_ptrs = getattr(
                            self, "_paiton_graph_debug_ptrs", None
                        )
                        debug_step = getattr(
                            self, "_paiton_graph_debug_step", 0
                        ) + 1
                        self._paiton_graph_debug_step = debug_step
                        debug_names = ordered_input_names + ("logits",)
                        current_ptrs = tuple(ordered_ptrs) + (
                            int(runtime_output.data_ptr()),
                        )
                        if previous_ptrs is not None and debug_step <= 32:
                            changed = [
                                debug_names[idx]
                                for idx, (old_ptr, new_ptr) in enumerate(
                                    zip(previous_ptrs, current_ptrs)
                                )
                                if old_ptr != new_ptr
                            ]
                            print(
                                "[paiton-graph] "
                                f"step={debug_step} changed_inputs="
                                f"{len(changed)} names={changed}",
                                flush=True,
                            )
                        self._paiton_graph_debug_ptrs = current_ptrs
                    previous_ptrs = getattr(self, "_bound_runtime_ptrs", None)
                    if previous_ptrs != ordered_ptrs:
                        self.model.update_input_pointers(ordered_ptrs)
                        self._bound_runtime_ptrs = list(ordered_ptrs)
                    use_bound = True
            except AttributeError:
                # Older artifacts do not export persistent binding.
                self.model._bound_run_available = False
                use_bound = False

        # Decode is asynchronous; compact-logit prefill completes in order.
        _need_sync = runtime_output is not output
        if use_bound:
            self.model.run_bound(
                outputs, stream_ptr=stream_ptr, sync=_need_sync,
                graph_mode=graph_mode,
            )
        else:
            filtered_inputs = {
                name: ordered_inputs[idx]
                for idx, name in enumerate(ordered_input_names)
            }
            self.model.run(
                filtered_inputs, outputs,
                stream_ptr=stream_ptr, sync=_need_sync,
                graph_mode=graph_mode,
            )
        if self._moe_topk_capture:
            self._record_moe_topk_capture(
                nt, self._moe_topk_step,
                metadata={
                    "graph_mode": graph_mode,
                    "use_bound": use_bound,
                },
            )

        self._replicate_logits_if_needed(runtime_output)
        if runtime_output is not output:
            if query_start_loc_i32 is None or query_start_loc_i32.numel() < num_runtime_rows + 1:
                raise RuntimeError(
                    "DeepSeek V4 compact logits output requires query_start_loc metadata "
                    "to expand per-request logits back into vLLM's per-token shape."
            )
            sample_rows = (query_start_loc_i32[1:] - 1).to(dtype=torch.int64)
            output.index_copy_(0, sample_rows, runtime_output)
        return output
