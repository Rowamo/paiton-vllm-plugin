"""vLLM wrapper for Paiton-compiled DeepSeek V4 models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from torch import Tensor

from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.models.paiton_qwen3_moe import PaitonQwen3MoeForCausalLM
from paiton_vllm_plugin.models.paiton_deepseek_v4_sparse import (
    DeepseekV4SparseRuntimeMixin,
)
from paiton_vllm_plugin.runtime.core import (
    PData,
    runtime_uses_fnuz_fp8,
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
    DeepseekV4SparseRuntimeMixin,
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
        # D1: PAITON internal graph capture (stream capture in the compiled
        # .so) was tested on MI355X and found to be ~10% SLOWER than eager
        # launches — hipGraphLaunch has higher per-call overhead than
        # individual hipModuleLaunchKernel calls on this platform. Disabled
        # by default. Set PAITON_ENABLE_GRAPHS=1 to enable for testing on
        # other platforms where graph replay may be faster.
        import os as _os
        self._paiton_graph_mode = _os.getenv("PAITON_ENABLE_GRAPHS", "0") == "1"

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
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    def _maybe_profile_compiled_runtime(
        self,
        ordered_inputs: list[PData],
        ordered_input_names: tuple[str, ...],
        outputs: Dict[str, PData],
        stream_ptr: int,
    ) -> None:
        if os.getenv("PAITON_DEEPSEEK_PROFILE", "0") != "1":
            return

        step = getattr(self, "_paiton_profile_step", 0) + 1
        self._paiton_profile_step = step
        target_step = self._env_int("PAITON_DEEPSEEK_PROFILE_STEP", 0)
        if getattr(self, "_paiton_profile_done", False):
            return
        if target_step > 0 and step != target_step:
            return

        profile_dir = os.getenv(
            "PAITON_DEEPSEEK_PROFILE_DIR",
            "/tmp/paiton_deepseek_profiles",
        )
        os.makedirs(profile_dir, exist_ok=True)
        rank = os.getenv("RANK")
        if rank is None and torch.distributed.is_available():
            if torch.distributed.is_initialized():
                rank = str(torch.distributed.get_rank())
        if rank is None:
            rank = "0"
        local_rank = os.getenv("LOCAL_RANK")
        if local_rank is None and torch.cuda.is_available():
            local_rank = str(torch.cuda.current_device())
        if local_rank is None:
            local_rank = rank
        filename = os.path.join(
            profile_dir,
            f"deepseek_rank{rank}_local{local_rank}_step{step}.json",
        )
        iters = max(1, self._env_int("PAITON_DEEPSEEK_PROFILE_ITERS", 1))
        filtered_inputs = {
            name: ordered_inputs[idx]
            for idx, name in enumerate(ordered_input_names)
        }
        print(
            f"[paiton-deepseek] profiling compiled runtime to {filename} "
            f"({iters} iter(s), forward step {step})",
            flush=True,
        )
        self.model.profile(
            filtered_inputs,
            outputs,
            num_iters=iters,
            filename=filename,
            stream_ptr=stream_ptr,
        )
        self._paiton_profile_done = True

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
            rounded = max(rounded, min(int(current) * 2, rounded))
        if maximum is not None and maximum > 0:
            rounded = min(rounded, int(maximum))
            if rounded < required:
                rounded = required
        return rounded

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
        output = torch.empty(
            [input_ids.shape[0], self.config.vocab_size],
            dtype=torch.float32,
            device=input_ids.device,
        )
        if not all_attn_metadata:
            return output

        attn_metadata = all_attn_metadata["0"]
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len
        num_runtime_rows = int(input_ids.shape[0])
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        if seq_lens is not None and seq_lens.ndim == 1 and seq_lens.numel() > 0:
            num_runtime_rows = int(seq_lens.shape[0])
        else:
            query_start_loc = getattr(attn_metadata, "query_start_loc", None)
            if query_start_loc is not None and query_start_loc.numel() >= 2:
                num_runtime_rows = int(query_start_loc.numel() - 1)
        if num_runtime_rows == output.shape[0]:
            runtime_output = output
        else:
            output.zero_()
            runtime_output = torch.empty(
                [num_runtime_rows, self.config.vocab_size],
                dtype=torch.float32,
                device=input_ids.device,
            )

        input_ids_i32 = input_ids.to(dtype=torch.int32, copy=False).contiguous()
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

        device = input_ids.device

        if positions is not None:
            position_ids_i64 = positions.to(
                device=device, dtype=torch.int64, copy=False).contiguous()
            run_input_backings.append(position_ids_i64)
            inputs["position_ids"] = torch_to_paiton_data(position_ids_i64)

        if hasattr(attn_metadata, "slot_mapping"):
            slot_mapping_i64 = attn_metadata.slot_mapping.to(
                device=device, dtype=torch.int64, copy=False).contiguous()
            run_input_backings.append(slot_mapping_i64)
            inputs["slot_mapping"] = torch_to_paiton_data(slot_mapping_i64)

        if hasattr(attn_metadata, "query_start_loc"):
            query_start_loc_i32 = attn_metadata.query_start_loc.to(
                device=device, dtype=torch.int32, copy=False).contiguous()
            run_input_backings.append(query_start_loc_i32)
            inputs["query_start_locations"] = torch_to_paiton_data(
                query_start_loc_i32)

        if hasattr(attn_metadata, "seq_lens"):
            seq_lens_i32 = attn_metadata.seq_lens.to(
                device=device, dtype=torch.int32, copy=False).contiguous()
            run_input_backings.append(seq_lens_i32)
            inputs["context_lengths"] = torch_to_paiton_data(seq_lens_i32)

        if hasattr(attn_metadata, "block_table"):
            block_table_i32 = attn_metadata.block_table.to(
                device=device, dtype=torch.int32, copy=False).contiguous()
            run_input_backings.append(block_table_i32)
            inputs["block_tables"] = torch_to_paiton_data(block_table_i32)


        generated_sparse_indices = None
        generated_sparse_topk_length = None

        max_query_len_backing = torch.empty([1], dtype=torch.int32, device=device)
        max_seq_len_backing = torch.empty([1], dtype=torch.int32, device=device)
        max_query_len_backing.fill_(int(max_query_len))
        max_seq_len_backing.fill_(int(max_seq_len))
        run_input_backings.extend([max_query_len_backing, max_seq_len_backing])
        inputs["max_query_len"] = PData(
            max_query_len_backing.data_ptr(),
            [max_query_len, 0],
            torch_dtype_to_string(torch.int32),
        )
        inputs["max_seq_len"] = PData(
            max_seq_len_backing.data_ptr(),
            [max_seq_len, 0],
            torch_dtype_to_string(torch.int32),
        )

        # Hoist sparse-index construction out of the per-layer loop. The indices
        # are identical across all layers for a given step, so building them once
        # here (rather than lazily on the first layer that needs them) lets us
        # also compute the per-step slot/block extents once below.
        needs_sparse_any = input_plan.needs_sparse_any
        first_kv_cache = input_plan.first_kv_cache
        if needs_sparse_any:
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


        # Compute per-step slot/block extents ONCE. This replaces ~150-170
        # per-layer .item() GPU->CPU syncs (43 layers x ~3-4 calls) with at
        # most two .item() calls per step (max slot, max block-table entry).
        step_required_slots, step_max_slot, step_max_block, step_required_blocks_bt = (
            self._compute_step_slot_extents(
                slot_mapping_i64,
                generated_sparse_indices,
                block_table_i32,
            )
        )


        # Pre-allocate compressed_slot_offset tensor once (same value for all
        # layers). Previously allocated per-layer with torch.tensor().
        step_sparse_slot_offset = self._rounded_runtime_capacity(
            step_required_slots,
            quantum=256,
            maximum=(
                int(first_kv_cache.shape[1]) * int(first_kv_cache.shape[2])
                if first_kv_cache is not None
                else None
            ),
        )
        compressed_slot_offset_tensor = (
            torch.tensor([step_sparse_slot_offset], dtype=torch.int64, device=device)
            if needs_sparse_any else None
        )

        # Pre-allocate per-layer scratch buffers ONCE and reuse them across
        # decode steps. Previously, torch.empty() was called per-layer per-step
        # (258+ cudaMalloc calls per forward), which dominated the 550ms
        # per-layer loop overhead.
        nt = num_tokens_for_input_alloc()
        if not hasattr(self, "_scratch_cache") or self._scratch_cache.get("nt") != nt:
            self._scratch_cache = {"nt": nt}
        sc = self._scratch_cache

        validate_cached_kv = os.getenv("PAITON_VALIDATE_KV_BINDINGS", "0") == "1"
        c128_sparse_input_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
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
                    if (
                        slot_mapping_i64 is not None
                        and generated_sparse_indices is not None
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

            if (
                layer_binding.has_sparse_mla_compressed_offset
                and slot_mapping_i64 is not None
            ):
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
                kfp8_key = f"ik8_{i}"
                ksc_key = f"iks_{i}"
                if kfp8_key not in sc or sc[kfp8_key].shape[0] < num_sparse_rows:
                    sc[kfp8_key] = torch.empty(
                        (num_sparse_rows, self._index_head_dim()),
                        dtype=torch.float8_e4m3fnuz, device=device,
                    )
                    sc[ksc_key] = torch.empty(
                        (num_sparse_rows,), dtype=torch.float32, device=device,
                    )
                indexer_k_fp8 = sc[kfp8_key][:num_sparse_rows]
                indexer_k_scale = sc[ksc_key][:num_sparse_rows]
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
                # Write the current K-row high-water mark so the compiler's
                # k_quant kernel only processes newly appended rows.
                if sparse_mla_indexer_kv is not None:
                    num_existing = int(sparse_mla_indexer_kv.shape[0])
                else:
                    num_existing = 0
                sc[inr_key][0] = num_existing
                run_input_backings.append(sc[inr_key])
                inputs[f"indexer_num_existing_rows_{i}"] = torch_to_paiton_data(
                    sc[inr_key])

            if (
                layer_binding.has_sparse_mla_indices
                or layer_binding.has_sparse_mla_topk_length
            ):
                compressed_slot_offset_value = (
                    step_sparse_slot_offset
                    if (
                        slot_mapping_i64 is not None
                        and generated_sparse_indices is not None
                        and layer_binding.compress_ratio > 1
                    )
                    else None
                )
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
                    sparse_indices = self._layer_sparse_input_copy(
                        sparse_indices,
                        dtype=torch.int32,
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
                    sparse_topk_length = self._layer_sparse_input_copy(
                        sparse_topk_length,
                        dtype=torch.int32,
                    )
                    run_input_backings.append(sparse_topk_length)
                    inputs[f"sparse_mla_topk_length_{i}"] = torch_to_paiton_data(
                        sparse_topk_length)
                elif layer_binding.has_sparse_mla_topk_length:
                    raise RuntimeError(
                        f"sparse_mla_topk_length_{i} is expected but could not be generated. "
                        "This indicates a bug in the sparse MLA input generation logic."
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
        stream_ptr = torch.cuda.current_stream().cuda_stream
        self._run_input_backings = run_input_backings

        # Persistent input binding fast path. During stable decode, the input
        # name set and shapes are stable across steps, so we update only the
        # ordered data pointers. Avoid rebuilding filtered input dicts,
        # ordered PData lists, and full per-input shape signatures per token.
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
                    use_bound = True
                else:
                    # Shapes unchanged: fast path, only update pointers.
                    self.model.update_input_pointers(ordered_ptrs)
                    use_bound = True
            except (AttributeError, RuntimeError, OSError):
                # The loaded .so doesn't export the new binding API (built
                # before the persistent-binding C changes). Fall back to the
                # regular run path and don't retry the bound path.
                self.model._bound_run_available = False
                use_bound = False

        self._maybe_profile_compiled_runtime(
            ordered_inputs,
            ordered_input_names,
            outputs,
            stream_ptr,
        )

        if use_bound:
            self.model.run_bound(
                outputs, stream_ptr=stream_ptr, sync=True,
                graph_mode=self._paiton_graph_mode,
            )
        else:
            filtered_inputs = {
                name: ordered_inputs[idx]
                for idx, name in enumerate(ordered_input_names)
            }
            self.model.run(
                filtered_inputs, outputs,
                stream_ptr=stream_ptr, sync=True,
                graph_mode=self._paiton_graph_mode,
            )

        if runtime_output is not output:
            if query_start_loc_i32 is None or query_start_loc_i32.numel() < num_runtime_rows + 1:
                raise RuntimeError(
                    "DeepSeek V4 compact logits output requires query_start_loc metadata "
                    "to expand per-request logits back into vLLM's per-token shape."
            )
            sample_rows = (query_start_loc_i32[1:] - 1).to(dtype=torch.int64)
            output.index_copy_(0, sample_rows, runtime_output)
        return output

    def map_pt_params(
        self,
        pt_params: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        def convert_name(name: str) -> str:
            return name.replace("model.", "").replace(".", "_")

        def fix_fp8(w: torch.Tensor) -> torch.Tensor:
            if runtime_uses_fnuz_fp8() and w.dtype == torch.float8_e4m3fn:
                w_int8 = w.view(torch.int8).cuda()
                w_int8[w_int8 == -128] = 0
                return w_int8.view(torch.float8_e4m3fnuz)
            return w.cuda()

        def shuffle_weight(weight: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
            """Pre-shuffle dynamic-FP8 weights for CK blockscale GEMMs."""
            in_rows, in_cols = layout
            block_cols = in_cols * 2
            elems_per_16b = 16 // weight.element_size()
            assert weight.shape[-2] % in_rows == 0
            assert weight.shape[-1] % block_cols == 0

            weight_view = weight.view(
                -1,
                weight.shape[-2] // in_rows,
                in_rows,
                weight.shape[-1] // block_cols,
                block_cols // elems_per_16b,
                elems_per_16b,
            )
            weight_view = weight_view.permute(0, 1, 3, 4, 2, 5).contiguous()
            return weight_view.view(*weight.shape)

        def scale_to_float(scale: torch.Tensor) -> torch.Tensor:
            return scale.float()

        def scale_to_uint8(scale: torch.Tensor) -> torch.Tensor:
            if scale.dtype == torch.uint8:
                return scale
            if scale.element_size() == 1:
                return scale.view(torch.uint8)
            return scale.to(torch.uint8)

        def packed_to_uint8(weight: torch.Tensor) -> torch.Tensor:
            if weight.dtype == torch.uint8:
                return weight
            if weight.element_size() == 1:
                return weight.view(torch.uint8)
            return weight.to(torch.uint8)

        def fuse_wkv_wgate(name: str, param: torch.Tensor) -> tuple[str, torch.Tensor]:
            wgate_name = name.replace(".wkv.", ".wgate.")
            out_name = convert_name(name.replace(".wkv.", ".fused_wkv_wgate."))
            value = torch.cat([param, pt_params[wgate_name]], dim=0)
            return out_name, value

        def fuse_wkv_wgate_scale(name: str,
                                 param: torch.Tensor) -> tuple[str, torch.Tensor]:
            wgate_name = name.replace(".wkv.", ".wgate.")
            out_name = convert_name(
                name.replace(".wkv.scale", ".fused_wkv_wgate.weight_scale_inv"))
            value = torch.cat(
                [scale_to_float(param),
                 scale_to_float(pt_params[wgate_name])],
                dim=0,
            ) * fp8_scale_factor
            return out_name, value

        def maybe_emit(name: str, value: torch.Tensor) -> None:
            if expected_constant_names is None or name in expected_constant_names:
                params_paiton[name] = value.cuda()

        fp8_scale_factor = 2.0 if current_platform.is_fp8_fnuz() else 1.0
        params_paiton: Dict[str, torch.Tensor] = {}

        try:
            ep_group = get_ep_group()
            ep_rank = ep_group.rank_in_group
            ep_size = ep_group.world_size
        except Exception:
            ep_rank = 0
            ep_size = 1

        num_experts = getattr(self.config, "n_routed_experts",
                              getattr(self.config, "num_experts", 0))
        if num_experts and num_experts % ep_size != 0:
            raise ValueError(
                f"EP world_size must divide n_routed_experts "
                f"(ep_size={ep_size}, n_routed_experts={num_experts})")
        num_local_experts = num_experts // ep_size if num_experts else 0
        placement = getattr(self.parallel_config, "expert_placement_strategy",
                            "linear")
        if placement == "round_robin":
            local_expert_ids = list(range(ep_rank, num_experts, ep_size))
        else:
            start = ep_rank * num_local_experts
            local_expert_ids = list(range(start, start + num_local_experts))

        expert_regex = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)")
        layers_experts = [
            [{} for _ in range(num_experts)]
            for _ in range(getattr(self.config, "num_hidden_layers", 0))
        ]

        for name, param in pt_params.items():
            expert_match = expert_regex.match(name)
            if expert_match:
                layer_id = int(expert_match[1])
                expert_id = int(expert_match[2])
                layers_experts[layer_id][expert_id][expert_match[3]] = param
                continue

            if name == "embed.weight":
                out_name = "embed_tokens_weight"
                value = self.get_rank_weight(param, dim=0)
            elif name == "head.weight":
                out_name = "lm_head_weight"
                value = self.get_rank_weight(param, dim=0)
            elif name == "norm.weight" or name.startswith("hc_head"):
                out_name = convert_name(name)
                value = param
            elif ".attn.wq_a.weight" in name:
                wq_a = param
                wkv = pt_params[name.replace(".wq_a.weight", ".wkv.weight")]
                out_name = convert_name(name.replace(
                    ".wq_a.weight", ".fused_wqa_wkv.weight"))
                value = self.get_rank_weight(torch.cat([wq_a, wkv], dim=0), dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif ".attn.wq_a.scale" in name:
                wq_a = scale_to_float(param)
                wkv = scale_to_float(pt_params[name.replace(".wq_a.scale",
                                                            ".wkv.scale")])
                out_name = convert_name(name.replace(
                    ".wq_a.scale", ".fused_wqa_wkv.weight_scale_inv"))
                value = self.get_rank_weight(
                    torch.cat([wq_a, wkv], dim=0), dim=0) * fp8_scale_factor
            elif ".attn.wkv." in name:
                continue
            elif ".attn.wq_b.weight" in name:
                out_name = convert_name(name)
                value = self.get_rank_weight(param, dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif ".attn.wo_a.weight" in name:
                out_name = convert_name(name)
                # wo_a is consumed by the fused grouped blockscale kernel, not
                # CK, so keep it in normal row-major [N, K] layout.
                value = self.get_rank_weight(param, dim=0)
            elif ".attn.wq_b.scale" in name or ".attn.wo_a.scale" in name:
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = self.get_rank_weight(
                    scale_to_float(param), dim=0) * fp8_scale_factor
            elif ".attn.wo_b.weight" in name:
                out_name = convert_name(name)
                value = self.get_rank_weight(param, dim=1)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif ".attn.wo_b.scale" in name:
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = self.get_rank_weight(
                    scale_to_float(param), dim=1) * fp8_scale_factor
            elif name.endswith(".attn.q_norm.weight") or name.endswith(
                    ".attn.kv_norm.weight") or name.endswith(".attn_norm.weight"):
                out_name = convert_name(name)
                value = param
            elif name.endswith(".attn.attn_sink"):
                out_name = convert_name(name)
                local_sink = self.get_rank_weight(param.float(), dim=0)
                padded_heads = max(int(local_sink.numel()), 64)
                value = torch.full(
                    (padded_heads,),
                    -float("inf"),
                    dtype=torch.float32,
                    device=local_sink.device,
                )
                value[: local_sink.numel()].copy_(local_sink)
            elif ".attn.compressor.wkv.weight" in name:
                out_name, value = fuse_wkv_wgate(name, param)
            elif ".attn.compressor.wkv.scale" in name:
                out_name, value = fuse_wkv_wgate_scale(name, param)
            elif ".attn.compressor.wgate." in name:
                continue
            elif (
                name.endswith(".attn.compressor.ape")
                or name.endswith(".attn.compressor.norm.weight")
                or name.endswith(".attn.compressor.fused_wkv_wgate.weight")
                or name.endswith(".attn.compressor.fused_wkv_wgate.scale")
            ):
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = scale_to_float(param) * fp8_scale_factor if name.endswith(
                    ".scale") else param
            elif ".attn.indexer.compressor.wkv.weight" in name:
                out_name, value = fuse_wkv_wgate(name, param)
            elif ".attn.indexer.compressor.wkv.scale" in name:
                out_name, value = fuse_wkv_wgate_scale(name, param)
            elif ".attn.indexer.compressor.wgate." in name:
                continue
            elif (
                name.endswith(".attn.indexer.compressor.ape")
                or name.endswith(".attn.indexer.compressor.norm.weight")
                or name.endswith(".attn.indexer.compressor.fused_wkv_wgate.weight")
                or name.endswith(".attn.indexer.compressor.fused_wkv_wgate.scale")
            ):
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = scale_to_float(param) * fp8_scale_factor if name.endswith(
                    ".scale") else param
            elif ".attn.indexer.wq_b.weight" in name or (
                    ".attn.indexer.weights_proj.weight" in name):
                out_name = convert_name(name)
                value = param
                if getattr(self, "dynamic_quant", False) and ".attn.indexer.wq_b.weight" in name:
                    value = shuffle_weight(value)
            elif ".attn.indexer.wq_b.scale" in name or (
                    ".attn.indexer.weights_proj.scale" in name):
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = scale_to_float(param) * fp8_scale_factor
            elif name.endswith(".ffn.shared_experts.w1.weight"):
                w1 = param
                w3 = pt_params[name.replace(".w1.weight", ".w3.weight")]
                out_name = convert_name(name.replace(
                    ".w1.weight", ".gate_up_proj.weight"))
                value = self.get_rank_weight(torch.cat([w1, w3], dim=0), dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif name.endswith(".ffn.shared_experts.w1.scale"):
                w1 = scale_to_float(param)
                w3 = scale_to_float(pt_params[name.replace(".w1.scale",
                                                           ".w3.scale")])
                out_name = convert_name(name.replace(
                    ".w1.scale", ".gate_up_proj.weight_scale_inv"))
                value = self.get_rank_weight(
                    torch.cat([w1, w3], dim=0), dim=0) * fp8_scale_factor
            elif name.endswith(".ffn.shared_experts.w3.weight") or name.endswith(
                    ".ffn.shared_experts.w3.scale"):
                continue
            elif name.endswith(".ffn.shared_experts.w2.weight"):
                out_name = convert_name(name.replace(".w2.weight",
                                                     ".down_proj.weight"))
                value = self.get_rank_weight(param, dim=1)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif name.endswith(".ffn.shared_experts.w2.scale"):
                out_name = convert_name(name.replace(
                    ".w2.scale", ".down_proj.weight_scale_inv"))
                value = self.get_rank_weight(
                    scale_to_float(param), dim=1) * fp8_scale_factor
            elif name.endswith(".ffn.gate.tid2eid"):
                out_name = convert_name(
                    name.replace(".gate.tid2eid",
                                 ".experts.hash_indices_table"))
                value = param.to(dtype=torch.int32)
            elif name.endswith(".ffn.gate.weight"):
                out_name = convert_name(name)
                value = param
            elif name.endswith(".ffn.gate.bias"):
                out_name = convert_name(name.replace(
                    ".gate.bias", ".experts.e_score_correction_bias"))
                value = param.float()
            elif name.endswith(".ffn_norm.weight"):
                out_name = convert_name(name)
                value = param
            elif name.endswith("_fn") or name.endswith("_base") or name.endswith(
                    "_scale"):
                out_name = convert_name(name)
                value = param
            else:
                continue

            if out_name.endswith("_weight_scale_inv"):
                value = value.float()
            maybe_emit(out_name, fix_fp8(value) if value.dtype == torch.float8_e4m3fn
                       else value)

        for layer_id, experts in enumerate(layers_experts):
            if not any(experts):
                continue
            local_experts = [experts[i] for i in local_expert_ids]

            w13 = torch.stack(
                [
                    torch.cat(
                        [
                            packed_to_uint8(expert["w1.weight"]),
                            packed_to_uint8(expert["w3.weight"]),
                        ],
                        dim=0,
                    )
                    for expert in local_experts
                ],
                dim=0,
            )
            maybe_emit(
                convert_name(f"layers.{layer_id}.ffn.experts.w13_weight"),
                w13,
            )

            w13_scale = torch.stack(
                [
                    torch.cat(
                        [
                            scale_to_uint8(expert["w1.scale"]),
                            scale_to_uint8(expert["w3.scale"]),
                        ],
                        dim=0,
                    )
                    for expert in local_experts
                ],
                dim=0,
            )
            maybe_emit(
                convert_name(f"layers.{layer_id}.ffn.experts.w13_weight_scale"),
                w13_scale,
            )

            w2 = torch.stack(
                [packed_to_uint8(expert["w2.weight"]) for expert in local_experts],
                dim=0,
            )
            maybe_emit(
                convert_name(f"layers.{layer_id}.ffn.experts.w2_weight"),
                w2,
            )

            w2_scale = torch.stack(
                [scale_to_uint8(expert["w2.scale"]) for expert in local_experts],
                dim=0,
            )
            maybe_emit(
                convert_name(f"layers.{layer_id}.ffn.experts.w2_weight_scale"),
                w2_scale,
            )

            mask_name = convert_name(
                f"layers.{layer_id}.ffn.experts.local_expert_mask")
            if expected_constant_names is None or mask_name in expected_constant_names:
                local_mask = torch.zeros((num_experts,), dtype=torch.int32)
                local_mask[local_expert_ids] = 1
                params_paiton[mask_name] = local_mask.cuda()

        self._add_deepseek_router_defaults(params_paiton, expected_constant_names)
        self._add_deepseek_attention_defaults(params_paiton, expected_constant_names,
                                             fp8_scale_factor)
        return params_paiton

    def _add_deepseek_attention_defaults(
        self,
        params_paiton: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]],
        fp8_scale_factor: float,
    ) -> None:
        if expected_constant_names is None:
            return

        for name in expected_constant_names:
            if name in params_paiton:
                continue
            if name.endswith("_attn_k_scale") or name.endswith("_attn_v_scale"):
                params_paiton[name] = torch.tensor(
                    [fp8_scale_factor],
                    dtype=torch.float32,
                    device="cuda",
                )

    def _add_deepseek_router_defaults(
        self,
        params_paiton: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]],
    ) -> None:
        if expected_constant_names is None:
            return

        num_hash_layers = getattr(self.config, "num_hash_layers", 0)
        num_experts = getattr(self.config, "n_routed_experts",
                              getattr(self.config, "num_experts", 0))
        topk = getattr(self.config, "num_experts_per_tok", 0)
        vocab_size = getattr(self.config, "vocab_size", 0)

        for layer_id in range(num_hash_layers):
            name = f"layers_{layer_id}_ffn_experts_hash_indices_table"
            if name not in expected_constant_names or name in params_paiton:
                continue
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(os.getenv("PAITON_DEEPSEEK_V4_HASH_SEED", "0")) +
                            layer_id)
            table = torch.stack(
                [
                    torch.randperm(num_experts, generator=gen)[:topk]
                    for _ in range(vocab_size)
                ],
                dim=0,
            ).to(dtype=torch.int32)
            params_paiton[name] = table.cuda()

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            torch.cuda.set_device(int(local_rank_env))

        expected_all = set(self.model.get_constant_names(unbound_constants_only=False))
        loaded_names: Set[str] = set()
        global_params: Dict[str, Tensor] = {}
        layer_params: Dict[str, Tensor] = {}
        current_layer: Optional[int] = None
        layer_regex = re.compile(r"layers\.(\d+)\.")

        def set_mapped(mapped: Dict[str, Tensor]) -> None:
            if not mapped:
                return
            self.model.set_many_constants_with_tensors(mapped)
            loaded_names.update(mapped)

        def flush_layer() -> None:
            nonlocal layer_params
            if current_layer is None or not layer_params:
                return
            layer_prefix = f"layers_{current_layer}_"
            expected_layer = {
                name for name in expected_all if name.startswith(layer_prefix)
            }
            set_mapped(self.map_pt_params(
                layer_params,
                expected_constant_names=expected_layer,
            ))
            layer_params = {}

        for name, tensor in weights:
            if name.startswith("mtp."):
                continue
            param = tensor.detach().cpu()
            match = layer_regex.match(name)
            if match:
                layer_id = int(match[1])
                if current_layer is None:
                    current_layer = layer_id
                elif layer_id != current_layer:
                    if layer_id < current_layer:
                        raise RuntimeError(
                            "DeepSeek V4 weight stream is not layer-contiguous; "
                            f"saw layer {layer_id} after layer {current_layer}.")
                    flush_layer()
                    current_layer = layer_id
                layer_params[name] = param
            else:
                global_params[name] = param

        flush_layer()

        expected_global = {
            name for name in expected_all if not name.startswith("layers_")
        }
        set_mapped(self.map_pt_params(
            global_params,
            expected_constant_names=expected_global,
        ))

        missing = sorted(expected_all - loaded_names)
        if missing:
            raise RuntimeError(
                "Paiton DeepSeek V4 constants mismatch: missing expected "
                f"constants during load_weights() (mapped={len(loaded_names)}, "
                f"expected={len(expected_all)}). First 50 missing:\n- "
                + "\n- ".join(missing[:50]))
        return set()
