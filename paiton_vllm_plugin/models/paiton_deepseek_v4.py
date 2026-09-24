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

_INT32_PAITON_DTYPE = torch_dtype_to_string(torch.int32)
_FLOAT32_PAITON_DTYPE = torch_dtype_to_string(torch.float32)


@dataclass(slots=True)
class _DeepseekLayerInputBinding:
    layer_idx: int
    metadata_key: str
    sparse_alias_key: int
    ctx: object
    kv_cache: torch.Tensor
    kv_cache_view: torch.Tensor
    kv_cache_pdata: PData
    kv_cache_data_ptr: int
    context_sparse_kv: Optional[torch.Tensor]
    kv_cache_name: str
    sparse_kv_name: str
    compressor_state_name: str
    compressed_offset_name: str
    indexer_state_name: str
    indexer_kv_name: str
    indexer_q_fp8_name: str
    indexer_weights_name: str
    indexer_k_fp8_name: str
    indexer_k_scale_name: str
    cu_seqlen_ks_name: str
    cu_seqlen_ke_name: str
    indexer_num_existing_rows_name: str
    sparse_indices_name: str
    sparse_topk_name: str
    indexer_q_fp8_scratch_key: str
    indexer_weights_scratch_key: str
    cu_seqlen_ks_scratch_key: str
    cu_seqlen_ke_scratch_key: str
    indexer_num_existing_rows_scratch_key: str
    compiled_sparse_indices_scratch_key: str
    compiled_sparse_length_scratch_key: str
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
    # False when the compiled artifact drops the paged kv_cache_{i} input
    # (GLM MLA latent-only mode): nothing consumes that binding then.
    has_kv_cache_input: bool = True


@dataclass(slots=True)
class _DeepseekInputPlan:
    signature: tuple
    expected_inputs: Set[str]
    ordered_input_names: tuple[str, ...]
    input_name_to_position: Dict[str, int]
    layer_bindings: tuple[_DeepseekLayerInputBinding, ...]
    needs_sparse_any: bool
    first_kv_cache: Optional[torch.Tensor]
    compiled_indexer_outputs: bool
    sparse_mla_index_width: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    graph_max_seq_len: int
    static_indexer_num_existing_rows: bool
    static_compressed_offset: bool


class _TrackedInputs(dict[str, PData]):
    """Populate ordered runtime bindings without a per-forward closure class."""

    __slots__ = (
        "_positions",
        "ordered_inputs",
        "ordered_ptrs",
        "_seen_generation",
        "_generation",
        "_pdata_cache",
        "metadata_changed",
        "pointers_changed",
        "bound_count",
    )

    def __init__(self, plan: _DeepseekInputPlan):
        super().__init__()
        size = len(plan.ordered_input_names)
        self._positions = plan.input_name_to_position
        self.ordered_inputs: list[Optional[PData]] = [None] * size
        self.ordered_ptrs: list[int] = [0] * size
        self._seen_generation: list[int] = [0] * size
        self._generation = 0
        self._pdata_cache: dict[tuple, PData] = {}
        self.metadata_changed = False
        self.pointers_changed = False
        self.bound_count = 0

    def reset(self) -> None:
        self.clear()
        self._generation += 1
        self.metadata_changed = False
        self.pointers_changed = False
        self.bound_count = 0

    def __setitem__(self, key: str, value: PData) -> None:
        idx = self._positions.get(key)
        if idx is None:
            super().__setitem__(key, value)
            return
        if self._seen_generation[idx] != self._generation:
            self._seen_generation[idx] = self._generation
            self.bound_count += 1
        previous = self.ordered_inputs[idx]
        if (
            previous is None
            or previous.shape != value.shape
            or previous.dtype != value.dtype
        ):
            self.metadata_changed = True
        if self.ordered_ptrs[idx] != value.data_ptr:
            self.pointers_changed = True
        self.ordered_inputs[idx] = value
        self.ordered_ptrs[idx] = value.data_ptr

    def __contains__(self, key: object) -> bool:
        idx = self._positions.get(key) if isinstance(key, str) else None
        if idx is not None:
            return self.was_bound(idx)
        return super().__contains__(key)

    def __getitem__(self, key: str) -> PData:
        idx = self._positions.get(key)
        if idx is not None and self.was_bound(idx):
            value = self.ordered_inputs[idx]
            assert value is not None
            return value
        return super().__getitem__(key)

    def was_bound(self, idx: int) -> bool:
        return self._seen_generation[idx] == self._generation

    def bind_tensor(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        cache_descriptor: bool,
    ) -> None:
        self[name] = self.tensor_pdata(
            tensor, cache_descriptor=cache_descriptor
        )

    def tensor_pdata(
        self,
        tensor: torch.Tensor,
        *,
        cache_descriptor: bool,
    ) -> PData:
        if not cache_descriptor:
            return torch_to_paiton_data(tensor)
        data_ptr = tensor.data_ptr()
        shape = tensor.shape
        key = (data_ptr, shape, tensor.dtype)
        pdata = self._pdata_cache.get(key)
        if pdata is None:
            pdata = PData(
                data_ptr,
                list(shape),
                torch_dtype_to_string(tensor.dtype),
            )
            self._pdata_cache[key] = pdata
        return pdata

    def bind_raw(
        self,
        name: str,
        data_ptr: int,
        shape: tuple[int, ...],
        dtype: str,
        *,
        cache_descriptor: bool,
    ) -> None:
        key = (data_ptr, shape, dtype)
        pdata = self._pdata_cache.get(key) if cache_descriptor else None
        if pdata is None:
            pdata = PData(data_ptr, list(shape), dtype)
            if cache_descriptor:
                self._pdata_cache[key] = pdata
        self[name] = pdata


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

    # The compiled graph selects the final token of each request before the
    # LM head.  It can produce sampling logprobs for those rows, but it cannot
    # produce prompt logprobs, which require one LM-head row per prompt token.
    supports_prompt_logprobs = False

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__(vllm_config, prefix=prefix)
        self._deepseek_sliding_window = getattr(self.config, "sliding_window", None)
        self._disable_vllm_sliding_window_check()
        self._paiton_graph_mode = os.getenv("PAITON_ENABLE_GRAPHS", "1") == "1"
        self._paiton_graph_max_seq_len = int(
            vllm_config.model_config.max_model_len
        )
        # Graph mode is decode-only (one query token per active sequence), so
        # max_num_seqs is the tight upper bound rather than the much larger
        # max_num_batched_tokens prefill limit.
        self._paiton_graph_scratch_token_capacity = int(
            getattr(vllm_config.scheduler_config, "max_num_seqs", 0)
            or 0
        )
        self._paiton_physical_block_high_water: int = 0
        self._paiton_bound_run_enabled = (
            os.getenv("PAITON_DISABLE_BOUND_RUN", "0") != "1"
        )
        self._paiton_graph_debug = os.getenv("PAITON_GRAPH_DEBUG", "0") == "1"
        self._paiton_validate_kv_bindings = (
            os.getenv("PAITON_VALIDATE_KV_BINDINGS", "0") == "1"
        )
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

    def _graph_scratch_cache(self, nt: int) -> dict:
        """Return bounded graph scratch storage shared by every token count."""
        configured_capacity = int(
            getattr(self, "_paiton_graph_scratch_token_capacity", 0) or 0
        )
        if configured_capacity and nt > configured_capacity:
            raise RuntimeError(
                f"Paiton graph scratch needs {nt} tokens, but vLLM was "
                f"configured for {configured_capacity} max_num_seqs"
            )

        cache = getattr(self, "_paiton_graph_scratch_cache", None)
        if cache is None:
            capacity = configured_capacity or self._rounded_runtime_capacity(nt)
            cache = {"token_capacity": capacity, "retired_backings": []}
            self._paiton_graph_scratch_cache = cache
        elif not configured_capacity and nt > int(cache["token_capacity"]):
            cache["token_capacity"] = self._rounded_runtime_capacity(
                nt, current=int(cache["token_capacity"])
            )
        return cache

    @staticmethod
    def _graph_scratch_tensor(
        cache: dict,
        key: str,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Allocate stable storage, retaining backing used by captured graphs."""
        tensor = cache.get(key)
        if (
            tensor is None
            or tensor.ndim != len(shape)
            or tensor.shape[0] < shape[0]
            or tensor.shape[1:] != shape[1:]
            or tensor.dtype != dtype
            or tensor.device != device
        ):
            if tensor is not None:
                cache["retired_backings"].append(tensor)
            tensor = torch.empty(shape, dtype=dtype, device=device)
            cache[key] = tensor
        return tensor

    def _allocate_runtime_logits(
        self,
        rows: int,
        role: str,
        *,
        graph_mode: bool,
        device: torch.device,
    ) -> torch.Tensor:
        shape = (rows, self.config.vocab_size)
        if not graph_mode:
            return torch.empty(shape, dtype=torch.float32, device=device)
        cache = self._graph_scratch_cache(rows)
        backing = self._graph_scratch_tensor(
            cache,
            f"logits_output_{role}",
            (cache["token_capacity"], self.config.vocab_size),
            dtype=torch.float32,
            device=device,
        )
        return backing[:rows]

    def select_logits_for_sampling(
        self,
        logits: torch.Tensor,
        input_batch: object,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Return the compact per-request rows emitted by the artifact.

        The GLM artifact already selects each request's final token before its
        LM head.  vLLM normally performs that selection after ``forward`` by
        indexing a token-major output tensor.  Keeping a token-major
        ``[num_tokens, vocab]`` tensor solely for that second selection costs
        9.45 GiB at 16K tokens, so the model runner calls this hook instead.

        Multi-token speculative outputs are intentionally not accepted here:
        this artifact emits one row per request, while that path needs one row
        per sampled and draft token.
        """
        if int(getattr(input_batch, "num_draft_tokens", 0)) != 0:
            raise RuntimeError(
                "Paiton compact logits do not support speculative draft tokens."
            )
        rows = int(logits_indices.numel())
        if logits.shape[0] < rows:
            raise RuntimeError(
                "Paiton compact logits output has fewer rows than vLLM needs: "
                f"got {logits.shape[0]}, need {rows}."
            )
        return logits[:rows]

    def _prepare_runtime_inputs(
        self,
        specs: tuple[
            tuple[str, Optional[torch.Tensor], torch.dtype], ...
        ],
        *,
        graph_mode: bool,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], ...]:
        if not graph_mode:
            return tuple(
                tensor.to(device=device, dtype=dtype, copy=False).contiguous()
                if tensor is not None
                else None
                for _, tensor, dtype in specs
            )

        cache = getattr(self, "_paiton_graph_input_cache", None)
        if cache is None:
            cache = {}
            self._paiton_graph_input_cache = cache
        prepared: list[Optional[torch.Tensor]] = []
        copy_dst: list[torch.Tensor] = []
        copy_src: list[torch.Tensor] = []
        token_capacity = getattr(
            self, "_paiton_graph_scratch_token_capacity", 0
        )
        for name, tensor, dtype in specs:
            if tensor is None:
                prepared.append(None)
                continue
            if token_capacity and tensor.ndim > 0:
                capacity = token_capacity + (
                    1 if name == "query_start_locations" else 0
                )
                capacity = max(capacity, tensor.shape[0])
                backing_shape = (capacity, *tensor.shape[1:])
                key = (device, name, tensor.shape[1:], dtype)
            else:
                backing_shape = tensor.shape
                key = (device, name, tensor.shape, dtype)
            persistent = cache.get(key)
            if persistent is None:
                persistent = torch.empty(
                    backing_shape, dtype=dtype, device=device
                )
                cache[key] = persistent
            view = (
                persistent[: tensor.shape[0]]
                if persistent.shape != tensor.shape
                else persistent
            )
            prepared.append(view)
            copy_dst.append(view)
            copy_src.append(tensor)

        if copy_dst:
            torch._foreach_copy_(copy_dst, copy_src)
        return tuple(prepared)

    def _runtime_inputs(self, plan: _DeepseekInputPlan) -> _TrackedInputs:
        inputs = getattr(self, "_paiton_tracked_inputs", None)
        if inputs is None or inputs._positions is not plan.input_name_to_position:
            inputs = _TrackedInputs(plan)
            self._paiton_tracked_inputs = inputs
        inputs.reset()
        return inputs

    def _runtime_scratch_view(
        self,
        cache: dict,
        key: str,
        shape: tuple[int, ...],
        *,
        rows: int,
        graph_mode: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if graph_mode:
            return self._graph_scratch_tensor(
                cache, key, shape, dtype=dtype, device=device
            )[:rows]
        tensor = cache.get(key)
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            cache[key] = tensor
        return tensor

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
        compress_ratios = getattr(self.config, "compress_ratios", None)
        indexer_types = getattr(self.config, "indexer_types", None)
        return (
            ordered_input_names,
            id(self.model),
            id(static_context),
            self.num_layers,
            self.cache_dtype,
            id(compress_ratios),
            id(indexer_types),
        )

    def _refresh_deepseek_kv_binding(
        self,
        binding: _DeepseekLayerInputBinding,
        *,
        validate_context: bool = False,
    ) -> tuple[torch.Tensor, PData]:
        if not validate_context:
            return binding.kv_cache, binding.kv_cache_pdata

        kv_cache = binding.kv_cache
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

    def _indexer_num_existing_rows_is_static(self) -> bool:
        return False

    def _sparse_mla_compressed_offset_is_static(self) -> bool:
        return False

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

    def _indexer_all_short_flag_value(self, max_seq_len: int) -> int:
        """Per-step all-short flag for sparse-indexer emission reuse.

        Returns 1 iff every scheduled row's context length fits in the
        compiled artifact's ``index_topk`` so the selected key set is the
        whole mapped row. ``max_seq_len`` is the host-side bound of the
        batch's ``seq_lens`` (the tensor bound as ``context_lengths``), so
        subclasses must only return 1 when that bound cannot under-report
        a long row. The default disables the feature.
        """
        return 0

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
        static_context = getattr(
            self.compilation_config,
            "static_forward_context",
            {},
        )
        compress_ratios = getattr(self.config, "compress_ratios", None)
        indexer_types = getattr(self.config, "indexer_types", None)
        cached = getattr(self, "_deepseek_input_plan", None)
        if cached is not None and cached.signature[1:] == (
            id(self.model),
            id(static_context),
            self.num_layers,
            self.cache_dtype,
            id(compress_ratios),
            id(indexer_types),
        ):
            return cached

        input_name_to_index = self.model.get_input_name_to_index_map()
        ordered_input_names = self._ordered_input_names(input_name_to_index)
        signature = self._deepseek_input_plan_signature(ordered_input_names)

        expected_inputs = set(ordered_input_names)
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
            context_sparse_kv = self._first_tensor_attr(
                ctx,
                (
                    "sparse_mla_kv",
                    "sparse_mla_kv_cache",
                    "mla_kv_cache",
                    "kv_c_and_k_pe_cache",
                    "compressed_kv_cache",
                ),
            )
            sparse_alias_key = self._sparse_mla_runtime_alias_key(layer_idx)
            layer_bindings.append(
                _DeepseekLayerInputBinding(
                    layer_idx=layer_idx,
                    metadata_key=str(layer_idx),
                    sparse_alias_key=sparse_alias_key,
                    ctx=ctx,
                    kv_cache=kv_cache,
                    kv_cache_view=kv_cache_view,
                    kv_cache_pdata=torch_to_paiton_data(kv_cache_view),
                    kv_cache_data_ptr=kv_cache.data_ptr(),
                    context_sparse_kv=context_sparse_kv,
                    kv_cache_name=prefix["kv_cache"],
                    sparse_kv_name=prefix["sparse_kv"],
                    compressor_state_name=prefix["compressor_state"],
                    compressed_offset_name=prefix["compressed_offset"],
                    indexer_state_name=prefix["indexer_state"],
                    indexer_kv_name=prefix["indexer_kv"],
                    indexer_q_fp8_name=prefix["indexer_q_fp8"],
                    indexer_weights_name=prefix["indexer_weights"],
                    indexer_k_fp8_name=prefix["indexer_k_fp8"],
                    indexer_k_scale_name=prefix["indexer_k_scale"],
                    cu_seqlen_ks_name=prefix["cu_seqlen_ks"],
                    cu_seqlen_ke_name=prefix["cu_seqlen_ke"],
                    indexer_num_existing_rows_name=(
                        prefix["indexer_num_existing_rows"]
                    ),
                    sparse_indices_name=prefix["sparse_indices"],
                    sparse_topk_name=prefix["sparse_topk"],
                    indexer_q_fp8_scratch_key=f"iq8_{layer_idx}",
                    indexer_weights_scratch_key=f"iw_{layer_idx}",
                    cu_seqlen_ks_scratch_key=f"cs_{layer_idx}",
                    cu_seqlen_ke_scratch_key=f"ce_{layer_idx}",
                    indexer_num_existing_rows_scratch_key=f"inr_{layer_idx}",
                    compiled_sparse_indices_scratch_key=(
                        f"compiled_sparse_indices_{sparse_alias_key}"
                    ),
                    compiled_sparse_length_scratch_key=(
                        f"compiled_sparse_length_{sparse_alias_key}"
                    ),
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
                    has_kv_cache_input=prefix["kv_cache"] in expected_inputs,
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
            compiled_indexer_outputs=(
                self._compiled_indexer_overwrites_sparse_inputs()
            ),
            sparse_mla_index_width=self._sparse_mla_index_width(),
            index_n_heads=self._index_n_heads(),
            index_head_dim=self._index_head_dim(),
            index_topk=int(getattr(self.config, "index_topk", 2048)),
            graph_max_seq_len=(
                (int(getattr(self, "_paiton_graph_max_seq_len", 0)) + 255) // 256
            ) * 256,
            static_indexer_num_existing_rows=(
                self._indexer_num_existing_rows_is_static()
            ),
            static_compressed_offset=(
                self._sparse_mla_compressed_offset_is_static()
            ),
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
        graph_mode = self._paiton_graph_mode and max_query_len == 1
        num_runtime_rows = input_ids.shape[0]
        seq_lens = getattr(attn_metadata, "seq_lens", None)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        block_table = getattr(attn_metadata, "block_table", None)
        if seq_lens is not None and seq_lens.ndim == 1 and seq_lens.numel() > 0:
            num_runtime_rows = seq_lens.shape[0]
        elif query_start_loc is not None and query_start_loc.numel() >= 2:
            num_runtime_rows = query_start_loc.numel() - 1

        # The compiled LM head emits one row per request.  Keep that compact
        # representation and let vLLM consume it through
        # select_logits_for_sampling(), rather than allocating/scattering a
        # token-major [num_tokens, vocab] tensor just for vLLM to gather the
        # same final-token rows again.
        runtime_output = self._allocate_runtime_logits(
            num_runtime_rows,
            "compact",
            graph_mode=graph_mode,
            device=input_ids.device,
        )

        device = input_ids.device
        runtime_input_specs = (
            ("input_ids", input_ids, torch.int32),
            ("position_ids", positions, torch.int64),
            ("slot_mapping", slot_mapping, torch.int64),
            ("query_start_locations", query_start_loc, torch.int32),
            ("context_lengths", seq_lens, torch.int32),
            ("block_tables", block_table, torch.int32),
        )
        prepared_inputs = self._prepare_runtime_inputs(
            runtime_input_specs,
            graph_mode=graph_mode,
            device=device,
        )
        (
            input_ids_i32,
            position_ids_i64,
            slot_mapping_i64,
            query_start_loc_i32,
            seq_lens_i32,
            block_table_i32,
        ) = prepared_inputs
        assert input_ids_i32 is not None
        run_input_backings: list[torch.Tensor] = [
            tensor for tensor in prepared_inputs if tensor is not None
        ]
        input_plan = self._get_deepseek_input_plan()
        expected_inputs = input_plan.expected_inputs
        ordered_input_names = input_plan.ordered_input_names
        inputs = self._runtime_inputs(input_plan)
        ordered_inputs = inputs.ordered_inputs
        ordered_ptrs = inputs.ordered_ptrs
        for (name, _, _), tensor in zip(runtime_input_specs, prepared_inputs):
            if tensor is not None:
                inputs.bind_tensor(
                    name, tensor, cache_descriptor=graph_mode
                )

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
                    torch.full(
                        [1], max_query_len, dtype=torch.int32, device=device
                    ),
                    torch.full(
                        [1], max_seq_len, dtype=torch.int32, device=device
                    ),
                )
                scalar_cache[scalar_key] = scalar_pair
                scalar_cache[(device, "max_length_values")] = (
                    max_query_len,
                    max_seq_len,
                )
            max_query_len_backing, max_seq_len_backing = scalar_pair
        else:
            max_query_len_backing = torch.empty(
                [1], dtype=torch.int32, device=device
            )
            max_seq_len_backing = torch.empty(
                [1], dtype=torch.int32, device=device
            )
        if graph_mode:
            values_key = (device, "max_length_values")
            previous_max_query_len, previous_max_seq_len = scalar_cache[values_key]
            if previous_max_query_len != max_query_len:
                max_query_len_backing.fill_(max_query_len)
            if previous_max_seq_len != max_seq_len:
                max_seq_len_backing.fill_(max_seq_len)
            scalar_cache[values_key] = (max_query_len, max_seq_len)
        else:
            max_query_len_backing.fill_(max_query_len)
            max_seq_len_backing.fill_(max_seq_len)
        run_input_backings.extend([max_query_len_backing, max_seq_len_backing])
        # All-short emission-reuse flag for the sparse indexer: per-step
        # int32[1], host-computed (no device sync) from the attention
        # metadata's max-sequence-length bound. Compiled GLM artifacts built
        # with PAITON_INDEXER_ALL_SHORT_REUSE=1 declare this input; older
        # artifacts and other models do not, so bind only when present.
        if "indexer_all_short_flag" in expected_inputs:
            all_short_value = self._indexer_all_short_flag_value(max_seq_len)
            if graph_mode:
                flag_backing = scalar_cache.get((device, "indexer_all_short"))
                if flag_backing is None:
                    flag_backing = torch.full(
                        [1], all_short_value, dtype=torch.int32, device=device
                    )
                    scalar_cache[(device, "indexer_all_short")] = flag_backing
                    scalar_cache[(device, "indexer_all_short_value")] = (
                        all_short_value
                    )
                elif (
                    scalar_cache[(device, "indexer_all_short_value")]
                    != all_short_value
                ):
                    flag_backing.fill_(all_short_value)
                    scalar_cache[(device, "indexer_all_short_value")] = (
                        all_short_value
                    )
            else:
                flag_backing = torch.full(
                    [1], all_short_value, dtype=torch.int32, device=device
                )
            run_input_backings.append(flag_backing)
            inputs.bind_raw(
                "indexer_all_short_flag",
                flag_backing.data_ptr(),
                (1,),
                _INT32_PAITON_DTYPE,
                cache_descriptor=graph_mode,
            )
        # Match graph scalar shapes to the fixed indexer workspace.
        bound_max_seq_len = max_seq_len
        if graph_mode:
            bound_max_seq_len = input_plan.graph_max_seq_len
        inputs.bind_raw(
            "max_query_len",
            max_query_len_backing.data_ptr(),
            (max_query_len, 0),
            _INT32_PAITON_DTYPE,
            cache_descriptor=graph_mode,
        )
        inputs.bind_raw(
            "max_seq_len",
            max_seq_len_backing.data_ptr(),
            (bound_max_seq_len, 0),
            _INT32_PAITON_DTYPE,
            cache_descriptor=graph_mode,
        )

        needs_sparse_any = input_plan.needs_sparse_any
        first_kv_cache = input_plan.first_kv_cache
        compiled_indexer_outputs = input_plan.compiled_indexer_outputs
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
                        input_plan.sparse_mla_index_width,
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
                    compressed_slot_offset_tensor = torch.full(
                        [1],
                        compressed_slot_offset_input_value,
                        dtype=torch.int64,
                        device=device,
                    )
                    scalar_cache[scalar_key] = compressed_slot_offset_tensor
                elif not input_plan.static_compressed_offset:
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
        nt = input_ids.shape[0]
        if graph_mode:
            sc = self._graph_scratch_cache(nt)
            scratch_nt = int(sc["token_capacity"])
        else:
            if (
                not hasattr(self, "_scratch_cache")
                or self._scratch_cache.get("nt") != nt
            ):
                self._scratch_cache = {"nt": nt}
            sc = self._scratch_cache
            scratch_nt = nt

        # Full-indexer layers share one MFMA logits workspace.
        if "indexer_logits_workspace" in expected_inputs:
            logits_stride = int(max_seq_len)
            if graph_mode:
                # A fixed stride avoids graph recapture as context grows.
                logits_stride = input_plan.graph_max_seq_len
            index_topk = input_plan.index_topk
            required_logits = (
                scratch_nt * logits_stride if logits_stride > index_topk else 1
            )
            logits_capacity = 1 << max(0, required_logits - 1).bit_length()
            logits_scratch = sc.get("indexer_logits_workspace")
            if logits_scratch is None or logits_scratch.numel() < logits_capacity:
                if graph_mode and logits_scratch is not None:
                    sc["retired_backings"].append(logits_scratch)
                logits_scratch = torch.empty(
                    logits_capacity, dtype=torch.float32, device=device
                )
                sc["indexer_logits_workspace"] = logits_scratch
            run_input_backings.append(logits_scratch)
            inputs.bind_raw(
                "indexer_logits_workspace",
                logits_scratch.data_ptr(),
                (nt, logits_stride),
                _FLOAT32_PAITON_DTYPE,
                cache_descriptor=graph_mode,
            )
        validate_cached_kv = getattr(
            self, "_paiton_validate_kv_bindings", False
        )
        c128_sparse_input_cache: Dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        sparse_input_alias_cache: Dict[
            int,
            tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        ] = {}
        for layer_binding in input_plan.layer_bindings:
            i = layer_binding.layer_idx
            ctx = layer_binding.ctx
            layer_attn_metadata = (
                all_attn_metadata.get(layer_binding.metadata_key, attn_metadata)
                if isinstance(all_attn_metadata, dict)
                else attn_metadata
            )
            kv_cache, kv_cache_pdata = self._refresh_deepseek_kv_binding(
                layer_binding,
                validate_context=validate_cached_kv,
            )
            if layer_binding.has_kv_cache_input:
                # GLM MLA artifacts drop the paged kv_cache_{i} input: the
                # latent plane is bound as sparse_mla_kv_{i} below, so
                # nothing consumes this binding. Skip it rather than bind
                # the (flat-view) tensor under the dead name.
                run_input_backings.append(kv_cache)
                inputs[layer_binding.kv_cache_name] = kv_cache_pdata

            sparse_kv = None
            if layer_binding.has_sparse_kv:
                sparse_kv = layer_binding.context_sparse_kv
                if validate_cached_kv:
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
                    layer_binding.context_sparse_kv = sparse_kv
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
                    layer_binding.sparse_kv_name,
                    sparse_kv,
                    step_required_slots,
                )
                run_input_backings.append(sparse_kv)
                inputs.bind_tensor(
                    layer_binding.sparse_kv_name,
                    sparse_kv,
                    cache_descriptor=graph_mode,
                )

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
                inputs.bind_tensor(
                    layer_binding.compressor_state_name,
                    compressor_state_cache,
                    cache_descriptor=graph_mode,
                )

            if layer_binding.has_sparse_mla_compressed_offset:
                if compressed_slot_offset_tensor is None:
                    raise RuntimeError(
                        f"sparse_mla_compressed_offset_{i} is expected but "
                        "the sparse MLA runtime offset tensor is unavailable."
                    )
                run_input_backings.append(compressed_slot_offset_tensor)
                inputs.bind_tensor(
                    layer_binding.compressed_offset_name,
                    compressed_slot_offset_tensor,
                    cache_descriptor=graph_mode,
                )

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
                inputs.bind_tensor(
                    layer_binding.indexer_state_name,
                    indexer_state_cache,
                    cache_descriptor=graph_mode,
                )

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
                    layer_binding.indexer_kv_name,
                    sparse_mla_indexer_kv,
                    step_required_slots,
                )
                run_input_backings.append(sparse_mla_indexer_kv)
                inputs.bind_tensor(
                    layer_binding.indexer_kv_name,
                    sparse_mla_indexer_kv,
                    cache_descriptor=graph_mode,
                )

            # The compiled artifact produces the C4 sparse-indexer aux tensors
            # in-graph. The runtime only allocates storage and binds it here.
            if layer_binding.has_indexer_q_fp8:
                key = layer_binding.indexer_q_fp8_scratch_key
                indexer_q_fp8 = self._runtime_scratch_view(
                    sc,
                    key,
                    (
                        scratch_nt,
                        input_plan.index_n_heads,
                        input_plan.index_head_dim,
                    ),
                    rows=nt,
                    graph_mode=graph_mode,
                    dtype=torch.float8_e4m3fnuz,
                    device=device,
                )
                run_input_backings.append(indexer_q_fp8)
                inputs.bind_tensor(
                    layer_binding.indexer_q_fp8_name,
                    indexer_q_fp8,
                    cache_descriptor=graph_mode,
                )

            if layer_binding.has_indexer_weights:
                key = layer_binding.indexer_weights_scratch_key
                indexer_weights = self._runtime_scratch_view(
                    sc,
                    key,
                    (scratch_nt, input_plan.index_n_heads),
                    rows=nt,
                    graph_mode=graph_mode,
                    dtype=torch.float32,
                    device=device,
                )
                run_input_backings.append(indexer_weights)
                inputs.bind_tensor(
                    layer_binding.indexer_weights_name,
                    indexer_weights,
                    cache_descriptor=graph_mode,
                )

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
                    inputs.bind_tensor(
                        layer_binding.indexer_k_fp8_name,
                        indexer_k_fp8,
                        cache_descriptor=graph_mode,
                    )
                if layer_binding.has_indexer_k_scale:
                    run_input_backings.append(indexer_k_scale)
                    inputs.bind_tensor(
                        layer_binding.indexer_k_scale_name,
                        indexer_k_scale,
                        cache_descriptor=graph_mode,
                    )

            if (
                layer_binding.has_cu_seqlen_ks
                or layer_binding.has_cu_seqlen_ke
            ):
                cks_key = layer_binding.cu_seqlen_ks_scratch_key
                cke_key = layer_binding.cu_seqlen_ke_scratch_key
                cu_seqlen_ks = self._runtime_scratch_view(
                    sc,
                    cks_key,
                    (scratch_nt,),
                    rows=nt,
                    graph_mode=graph_mode,
                    dtype=torch.int32,
                    device=device,
                )
                cu_seqlen_ke = self._runtime_scratch_view(
                    sc,
                    cke_key,
                    (scratch_nt,),
                    rows=nt,
                    graph_mode=graph_mode,
                    dtype=torch.int32,
                    device=device,
                )
                if layer_binding.has_cu_seqlen_ks:
                    run_input_backings.append(cu_seqlen_ks)
                    inputs.bind_tensor(
                        layer_binding.cu_seqlen_ks_name,
                        cu_seqlen_ks,
                        cache_descriptor=graph_mode,
                    )
                if layer_binding.has_cu_seqlen_ke:
                    run_input_backings.append(cu_seqlen_ke)
                    inputs.bind_tensor(
                        layer_binding.cu_seqlen_ke_name,
                        cu_seqlen_ke,
                        cache_descriptor=graph_mode,
                    )

            if layer_binding.has_indexer_num_existing_rows:
                inr_key = (
                    "static_indexer_num_existing_rows"
                    if input_plan.static_indexer_num_existing_rows
                    else layer_binding.indexer_num_existing_rows_scratch_key
                )
                if inr_key not in sc:
                    sc[inr_key] = torch.zeros(
                        (1,), dtype=torch.int32, device=device
                    )
                if not input_plan.static_indexer_num_existing_rows:
                    # Quantize only newly appended K rows.
                    num_existing = self._indexer_num_existing_rows(
                        i,
                        sparse_mla_indexer_kv,
                    )
                    sc[inr_key].fill_(num_existing)
                run_input_backings.append(sc[inr_key])
                inputs.bind_tensor(
                    layer_binding.indexer_num_existing_rows_name,
                    sc[inr_key],
                    cache_descriptor=graph_mode,
                )

            if (
                layer_binding.has_sparse_mla_indices
                or layer_binding.has_sparse_mla_topk_length
            ):
                sparse_alias_key = layer_binding.sparse_alias_key
                cached_sparse_inputs = sparse_input_alias_cache.get(sparse_alias_key)
                if cached_sparse_inputs is not None:
                    sparse_indices, sparse_topk_length = cached_sparse_inputs
                    if sparse_indices is not None:
                        run_input_backings.append(sparse_indices)
                        inputs.bind_tensor(
                            layer_binding.sparse_indices_name,
                            sparse_indices,
                            cache_descriptor=graph_mode,
                        )
                    elif layer_binding.has_sparse_mla_indices:
                        raise RuntimeError(
                            f"sparse_mla_indices_{i} aliases sparse MLA group "
                            f"{sparse_alias_key}, but that group has no indices."
                        )
                    if sparse_topk_length is not None:
                        run_input_backings.append(sparse_topk_length)
                        inputs.bind_tensor(
                            layer_binding.sparse_topk_name,
                            sparse_topk_length,
                            cache_descriptor=graph_mode,
                        )
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
                    indices_key = (
                        layer_binding.compiled_sparse_indices_scratch_key
                    )
                    length_key = (
                        layer_binding.compiled_sparse_length_scratch_key
                    )
                    sparse_indices = self._runtime_scratch_view(
                        sc,
                        indices_key,
                        (scratch_nt, input_plan.sparse_mla_index_width),
                        rows=nt,
                        graph_mode=graph_mode,
                        dtype=torch.int32,
                        device=device,
                    )
                    sparse_topk_length = self._runtime_scratch_view(
                        sc,
                        length_key,
                        (scratch_nt,),
                        rows=nt,
                        graph_mode=graph_mode,
                        dtype=torch.int32,
                        device=device,
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
                    sparse_indices_backing = None
                    if graph_mode and sparse_inputs_need_copy:
                        backing_key = f"graph_sparse_indices_{sparse_alias_key}"
                        sparse_indices_backing = self._graph_scratch_tensor(
                            sc,
                            backing_key,
                            (scratch_nt, *sparse_indices.shape[1:]),
                            dtype=torch.int32,
                            device=device,
                        )[:nt]
                    if sparse_inputs_need_copy:
                        sparse_indices = self._layer_sparse_input_copy(
                            sparse_indices,
                            dtype=torch.int32,
                            out=sparse_indices_backing,
                        )
                    run_input_backings.append(sparse_indices)
                    inputs.bind_tensor(
                        layer_binding.sparse_indices_name,
                        sparse_indices,
                        cache_descriptor=graph_mode,
                    )
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
                        sparse_length_backing = self._graph_scratch_tensor(
                            sc,
                            backing_key,
                            (scratch_nt, *sparse_topk_length.shape[1:]),
                            dtype=torch.int32,
                            device=device,
                        )[:nt]
                    if sparse_inputs_need_copy:
                        sparse_topk_length = self._layer_sparse_input_copy(
                            sparse_topk_length,
                            dtype=torch.int32,
                            out=sparse_length_backing,
                        )
                    run_input_backings.append(sparse_topk_length)
                    inputs.bind_tensor(
                        layer_binding.sparse_topk_name,
                        sparse_topk_length,
                        cache_descriptor=graph_mode,
                    )
                elif layer_binding.has_sparse_mla_topk_length:
                    raise RuntimeError(
                        f"sparse_mla_topk_length_{i} is expected but could not be generated. "
                        "This indicates a bug in the sparse MLA input generation logic."
                    )
                sparse_input_alias_cache[sparse_alias_key] = (
                    sparse_indices,
                    sparse_topk_length,
                )

        if inputs.bound_count != len(ordered_input_names):
            self._ensure_sparse_mla_inputs(
                expected_inputs,
                inputs,
                input_ids,
                run_input_backings,
            )
        if inputs.bound_count != len(ordered_input_names):
            for idx, name in enumerate(ordered_input_names):
                if not inputs.was_bound(idx) and name in inputs:
                    pd = inputs[name]
                    ordered_inputs[idx] = pd
                    ordered_ptrs[idx] = pd.data_ptr

        missing_inputs = {
            name
            for idx, name in enumerate(ordered_input_names)
            if not inputs.was_bound(idx)
        }
        if missing_inputs:
            raise RuntimeError(
                "Paiton DeepSeek V4 artifact input mismatch. Missing inputs: "
                + ", ".join(sorted(missing_inputs)))

        outputs = {
            "logits": inputs.tensor_pdata(
                runtime_output, cache_descriptor=graph_mode
            )
        }

        moe_topk_capture = getattr(self, "_moe_topk_capture", None)
        if moe_topk_capture is None:
            self._init_moe_topk_capture(runtime_output.device)
            moe_topk_capture = self._moe_topk_capture
        if moe_topk_capture:
            self._bind_moe_topk_outputs(outputs, nt)
        stream_ptr = torch.cuda.current_stream().cuda_stream
        self._run_input_backings = run_input_backings

        # Stable decode skips all C-side input work when pointers and metadata
        # match the previous invocation.
        can_bind = (
            getattr(self, "_paiton_bound_run_enabled", True)
            and len(ordered_inputs) == len(expected_inputs)
            and getattr(self.model, "_bound_run_available", True)
        )
        use_bound = False
        if can_bind:
            try:
                if (
                    not getattr(self, "_paiton_inputs_bound", False)
                    or inputs.metadata_changed
                ):
                    self.model.bind_inputs(ordered_inputs)
                    self._paiton_inputs_bound = True
                    self._bound_input_names = list(ordered_input_names)
                    use_bound = True
                else:
                    if (
                        graph_mode
                        and getattr(self, "_paiton_graph_debug", False)
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
                    if inputs.pointers_changed:
                        self.model.update_input_pointers(ordered_ptrs)
                    use_bound = True
            except AttributeError:
                # Older artifacts do not export persistent binding.
                self.model._bound_run_available = False
                use_bound = False

        if use_bound:
            self.model.run_bound(
                outputs, stream_ptr=stream_ptr, sync=False,
                graph_mode=graph_mode, return_outputs=False,
            )
        else:
            filtered_inputs = {
                name: ordered_inputs[idx]
                for idx, name in enumerate(ordered_input_names)
            }
            self.model.run(
                filtered_inputs, outputs,
                stream_ptr=stream_ptr, sync=False,
                graph_mode=graph_mode,
            )
        if moe_topk_capture:
            self._record_moe_topk_capture(
                nt, self._moe_topk_step,
                metadata={
                    "graph_mode": graph_mode,
                    "use_bound": use_bound,
                },
            )

        self._replicate_logits_if_needed(runtime_output)
        return runtime_output
