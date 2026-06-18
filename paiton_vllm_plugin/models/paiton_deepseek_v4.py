"""vLLM wrapper for Paiton-compiled DeepSeek V4 models."""

from __future__ import annotations

import os
import re
import types
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from torch import Tensor

from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.models.paiton_qwen3_moe import PaitonQwen3MoeForCausalLM
from paiton_vllm_plugin.runtime.core import (
    PData,
    runtime_uses_fnuz_fp8,
    torch_dtype_to_string,
    torch_to_paiton_data,
)


class PaitonDeepseekV4ForCausalLM(PaitonQwen3MoeForCausalLM):
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

    def _debug_sparse_mla_inputs(
        self,
        expected_inputs: Set[str],
        inputs: Dict[str, PData],
        generated_sparse_indices: Optional[torch.Tensor],
        generated_sparse_topk_length: Optional[torch.Tensor],
        run_input_backings: list[torch.Tensor],
    ) -> None:
        if os.getenv("PAITON_DEBUG_SPARSE_MLA", "0") != "1":
            return
        sparse_expected = sorted(
            name for name in expected_inputs if name.startswith("sparse_mla_")
        )
        sparse_present = sorted(name for name in sparse_expected if name in inputs)
        sparse_missing = sorted(set(sparse_expected) - set(sparse_present))
        msg = (
            "[PAITON_DEBUG_SPARSE_MLA] "
            f"expected={len(sparse_expected)} present={len(sparse_present)} "
            f"missing={len(sparse_missing)}"
        )
        if generated_sparse_indices is not None:
            valid = generated_sparse_indices[generated_sparse_indices >= 0]
            max_idx = int(valid.max().item()) if valid.numel() else -1
            msg += (
                f" indices_shape={tuple(generated_sparse_indices.shape)}"
                f" valid={int(valid.numel())} max_idx={max_idx}"
            )
        if generated_sparse_topk_length is not None:
            msg += f" topk_length={generated_sparse_topk_length.detach().cpu().tolist()}"
        kv_shapes = [
            tuple(t.shape)
            for t in run_input_backings
            if t.ndim == 3
            and t.shape[-1] == int(getattr(self.config, "head_dim", 512))
            and t.dtype == self.dtype
        ]
        if kv_shapes:
            msg += f" sparse_kv_shapes={kv_shapes[:4]}"
            if len(kv_shapes) > 4:
                msg += f"+{len(kv_shapes) - 4}"
        if sparse_missing:
            msg += f" first_missing={sparse_missing[:8]}"
        print(msg, flush=True)

    @staticmethod
    def _build_recent_sparse_mla_indices(
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        num_tokens: int,
        index_topk: int,
        block_size: int,
        device: torch.device,
        recent_window: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build recent-token sparse MLA indices in physical KV-slot space."""
        query_start_cpu = query_start_loc.to(
            device="cpu", dtype=torch.int64).tolist()
        seq_lens_cpu = seq_lens.to(device="cpu", dtype=torch.int64).tolist()
        block_table_cpu = block_table.to(device="cpu", dtype=torch.int64)

        indices_cpu = torch.full(
            (num_tokens, index_topk),
            -1,
            dtype=torch.int32,
        )
        topk_lengths_cpu = torch.zeros((num_tokens,), dtype=torch.int32)
        if index_topk <= 0 or num_tokens <= 0:
            return indices_cpu.to(device=device), topk_lengths_cpu.to(device=device)

        if recent_window is None or recent_window <= 0:
            recent_window = index_topk
        recent_window = min(index_topk, int(recent_window))

        num_reqs = min(len(seq_lens_cpu), max(0, len(query_start_cpu) - 1))
        for req_idx in range(num_reqs):
            token_start = int(query_start_cpu[req_idx])
            token_end = int(query_start_cpu[req_idx + 1])
            query_len = max(0, token_end - token_start)
            seq_len = int(seq_lens_cpu[req_idx])
            context_before_query = max(0, seq_len - query_len)

            for local_pos, token_idx in enumerate(range(token_start, token_end)):
                if token_idx >= num_tokens:
                    break
                logical_pos = context_before_query + local_pos
                first_pos = max(0, logical_pos - recent_window + 1)
                out_col = 0
                for pos in range(first_pos, logical_pos + 1):
                    block_col = pos // block_size
                    if block_col >= block_table_cpu.shape[1]:
                        continue
                    block_id = int(block_table_cpu[req_idx, block_col].item())
                    if block_id < 0:
                        continue
                    indices_cpu[token_idx, out_col] = (
                        block_id * block_size + pos % block_size
                    )
                    out_col += 1
                    if out_col >= index_topk:
                        break
                topk_lengths_cpu[token_idx] = out_col

        return indices_cpu.to(device=device), topk_lengths_cpu.to(device=device)

    def _get_sparse_mla_kv_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        *,
        compressed_slot_offset: Optional[int] = None,
    ) -> torch.Tensor:
        caches = getattr(self, "_sparse_mla_kv_caches", None)
        if caches is None:
            caches = {}
            self._sparse_mla_kv_caches = caches
        offsets = getattr(self, "_sparse_mla_kv_offsets", None)
        if offsets is None:
            offsets = {}
            self._sparse_mla_kv_offsets = offsets

        valid_slots = slot_mapping[slot_mapping >= 0]
        required_slots = 1
        if valid_slots.numel() > 0:
            required_slots = int(valid_slots.max().item()) + 1
        if sparse_indices is not None:
            valid_indices = sparse_indices[sparse_indices >= 0]
            if valid_indices.numel() > 0:
                required_slots = max(
                    required_slots,
                    int(valid_indices.max().item()) + 1,
                )
        if compressed_slot_offset is not None:
            required_slots = max(
                required_slots,
                2 * compressed_slot_offset,
            )

        head_dim = int(kv_cache.shape[-1])
        device = kv_cache.device
        current = caches.get(layer_idx)
        previous_offset = offsets.get(layer_idx)
        if (
            current is None
            or current.device != device
            or current.dtype != self.dtype
            or current.shape[0] < required_slots
            or current.shape[2] != head_dim
            or previous_offset != compressed_slot_offset
        ):
            new_cache = torch.empty(
                (required_slots, 1, head_dim),
                dtype=self.dtype,
                device=device,
            )
            if current is not None and current.numel() > 0:
                if compressed_slot_offset is not None and previous_offset is not None:
                    copy_raw = min(previous_offset, compressed_slot_offset)
                    if copy_raw > 0:
                        new_cache[:copy_raw].copy_(current[:copy_raw])
                        new_cache[
                            compressed_slot_offset : compressed_slot_offset + copy_raw
                        ].copy_(
                            current[previous_offset : previous_offset + copy_raw]
                        )
                else:
                    copy_slots = min(current.shape[0], new_cache.shape[0])
                    new_cache[:copy_slots].copy_(current[:copy_slots])
            caches[layer_idx] = new_cache
            offsets[layer_idx] = compressed_slot_offset
            current = new_cache
        return current

    def _get_sparse_mla_indexer_kv_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        *,
        compressed_slot_offset: Optional[int] = None,
    ) -> torch.Tensor:
        caches = getattr(self, "_sparse_mla_indexer_kv_caches", None)
        if caches is None:
            caches = {}
            self._sparse_mla_indexer_kv_caches = caches
        offsets = getattr(self, "_sparse_mla_indexer_kv_offsets", None)
        if offsets is None:
            offsets = {}
            self._sparse_mla_indexer_kv_offsets = offsets

        valid_slots = slot_mapping[slot_mapping >= 0]
        required_slots = 1
        if valid_slots.numel() > 0:
            required_slots = int(valid_slots.max().item()) + 1
        if sparse_indices is not None:
            valid_indices = sparse_indices[sparse_indices >= 0]
            if valid_indices.numel() > 0:
                required_slots = max(required_slots,
                                     int(valid_indices.max().item()) + 1)
        if compressed_slot_offset is not None:
            required_slots = max(
                required_slots,
                2 * compressed_slot_offset,
            )

        head_dim = self._index_head_dim()
        device = kv_cache.device
        current = caches.get(layer_idx)
        previous_offset = offsets.get(layer_idx)
        if (
            current is None
            or current.device != device
            or current.dtype != self.dtype
            or current.shape[0] < required_slots
            or current.shape[2] != head_dim
            or previous_offset != compressed_slot_offset
        ):
            new_cache = torch.empty(
                (required_slots, 1, head_dim),
                dtype=self.dtype,
                device=device,
            )
            if current is not None and current.numel() > 0:
                if compressed_slot_offset is not None and previous_offset is not None:
                    copy_raw = min(previous_offset, compressed_slot_offset)
                    if copy_raw > 0:
                        new_cache[:copy_raw].copy_(current[:copy_raw])
                        new_cache[
                            compressed_slot_offset : compressed_slot_offset + copy_raw
                        ].copy_(
                            current[previous_offset : previous_offset + copy_raw]
                        )
                else:
                    copy_slots = min(current.shape[0], new_cache.shape[0])
                    new_cache[:copy_slots].copy_(current[:copy_slots])
            caches[layer_idx] = new_cache
            offsets[layer_idx] = compressed_slot_offset
            current = new_cache
        return current

    @staticmethod
    def _compressed_sparse_slot_offset(
        slot_mapping: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
    ) -> int:
        required_slots = 1
        valid_slots = slot_mapping[slot_mapping >= 0]
        if valid_slots.numel() > 0:
            required_slots = int(valid_slots.max().item()) + 1
        if sparse_indices is not None:
            valid_indices = sparse_indices[sparse_indices >= 0]
            if valid_indices.numel() > 0:
                required_slots = max(required_slots, int(valid_indices.max().item()) + 1)
        return required_slots

    def _state_cache_block_size(
        self,
        layer_idx: int,
        *,
        indexer: bool,
        fallback_block_size: int,
    ) -> int:
        input_name = (
            f"indexer_state_cache_{layer_idx}"
            if indexer
            else f"compressor_state_cache_{layer_idx}"
        )
        runtime_model = self.__dict__.get("model")
        get_shape = getattr(runtime_model, "get_input_maximum_shape", None)
        if get_shape is not None:
            try:
                shape = get_shape(input_name)
            except Exception:
                shape = None
            if shape is not None and len(shape) >= 2 and int(shape[1]) > 0:
                return int(shape[1])
        return int(fallback_block_size)

    @staticmethod
    def _combine_sparse_index_lists(
        compressed_indices: Optional[torch.Tensor],
        compressed_topk_length: Optional[torch.Tensor],
        recent_indices: Optional[torch.Tensor],
        recent_topk_length: Optional[torch.Tensor],
        *,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if compressed_indices is None and recent_indices is None:
            raise ValueError("at least one sparse index source is required")

        base = compressed_indices if compressed_indices is not None else recent_indices
        assert base is not None
        num_tokens = int(base.shape[0])
        device = base.device

        out = torch.full(
            (num_tokens, width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        out_len = torch.zeros((num_tokens,), dtype=torch.int32, device=device)

        if compressed_indices is not None:
            compressed_indices = compressed_indices.to(dtype=torch.int32, copy=False)
            comp_width = min(width, int(compressed_indices.shape[1]))
            out[:, :comp_width] = compressed_indices[:, :comp_width]
            if compressed_topk_length is None:
                compressed_topk_length = (compressed_indices >= 0).sum(dim=-1).to(
                    dtype=torch.int32
                )
            else:
                compressed_topk_length = compressed_topk_length.to(
                    dtype=torch.int32, copy=False
                )
            out_len = torch.clamp(compressed_topk_length, min=0, max=width)

        if recent_indices is not None:
            recent_indices = recent_indices.to(dtype=torch.int32, copy=False)
            if recent_topk_length is None:
                recent_topk_length = (recent_indices >= 0).sum(dim=-1).to(
                    dtype=torch.int32
                )
            else:
                recent_topk_length = recent_topk_length.to(dtype=torch.int32, copy=False)

            token_rows = torch.arange(num_tokens, device=device)
            for col in range(int(recent_indices.shape[1])):
                dst_col = out_len + col
                valid = (
                    (recent_indices[:, col] >= 0)
                    & (col < recent_topk_length)
                    & (dst_col < width)
                )
                if valid.any():
                    out[token_rows[valid], dst_col[valid]] = recent_indices[valid, col]
            out_len = torch.clamp(out_len + recent_topk_length, min=0, max=width)

        return out.contiguous(), out_len.contiguous()

    @staticmethod
    def _map_c128a_prefill_local_indices_to_slots(
        prefill_local_indices: torch.Tensor,
        block_table: torch.Tensor,
        query_start_loc: torch.Tensor,
        compress_ratio: int,
        compressed_slot_offset: int,
        *,
        num_decode_tokens: int = 0,
        block_size: int = 64,
    ) -> torch.Tensor:
        if prefill_local_indices.numel() == 0:
            return prefill_local_indices.to(dtype=torch.int32, copy=True).contiguous()

        num_prefill_tokens = int(prefill_local_indices.shape[0])
        device = prefill_local_indices.device
        token_indices = torch.arange(
            num_decode_tokens,
            num_decode_tokens + num_prefill_tokens,
            dtype=torch.int64,
            device=device,
        )
        req_idx = torch.searchsorted(
            query_start_loc[1:].to(device=device, dtype=torch.int64),
            token_indices,
            right=True,
        ).to(dtype=torch.int64)

        local_indices = prefill_local_indices.to(dtype=torch.int64, copy=False)
        safe_local = torch.clamp(local_indices, min=0)
        boundary_positions = (safe_local + 1) * int(compress_ratio) - 1
        block_cols = boundary_positions // int(block_size)
        block_offsets = boundary_positions % int(block_size)

        valid = (local_indices >= 0) & (block_cols < int(block_table.shape[1]))
        out = torch.full_like(prefill_local_indices, -1, dtype=torch.int32)
        if not valid.any():
            return out.contiguous()

        safe_blocks = torch.full_like(block_cols, 0)
        safe_blocks[valid] = block_table[
            req_idx.unsqueeze(1).expand_as(block_cols)[valid],
            block_cols[valid],
        ].to(dtype=torch.int64)
        valid &= safe_blocks >= 0
        mapped = (
            compressed_slot_offset
            + safe_blocks * int(block_size)
            + block_offsets
        ).to(dtype=torch.int32)
        out[valid] = mapped[valid]
        return out.contiguous()

    @staticmethod
    def _map_c128_dense_slots_to_compiler_slots(
        dense_indices: torch.Tensor,
        compressed_slot_offset: int,
        compress_ratio: int,
        *,
        block_size: int,
    ) -> torch.Tensor:
        if dense_indices.numel() == 0:
            return dense_indices.to(dtype=torch.int32, copy=True).contiguous()

        compressed_block_size = max(1, int(block_size) // int(compress_ratio))
        dense_i64 = dense_indices.to(dtype=torch.int64, copy=False)
        safe_dense = torch.clamp(dense_i64, min=0)
        physical_blocks = safe_dense // compressed_block_size
        compressed_offsets = safe_dense % compressed_block_size
        raw_block_offsets = compressed_offsets * int(compress_ratio) + (
            int(compress_ratio) - 1
        )

        out = torch.full_like(dense_indices, -1, dtype=torch.int32)
        valid = dense_i64 >= 0
        mapped = (
            int(compressed_slot_offset)
            + physical_blocks * int(block_size)
            + raw_block_offsets
        ).to(dtype=torch.int32)
        out[valid] = mapped[valid]
        return out.contiguous()

    def _build_c128a_sparse_inputs(
        self,
        layer_attn_metadata,
        query_start_loc: torch.Tensor,
        compressed_slot_offset: int,
        recent_indices: Optional[torch.Tensor],
        recent_topk_length: Optional[torch.Tensor],
        *,
        layer_idx: int,
        positions: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compress_ratio = self._layer_compress_ratio(layer_idx)
        if compress_ratio <= 1:
            raise ValueError("C128 sparse inputs are only valid for compressed layers")

        decode_indices = self._first_tensor_attr(
            layer_attn_metadata,
            ("c128a_global_decode_topk_indices",),
        )
        decode_topk_length = self._first_tensor_attr(
            layer_attn_metadata,
            ("c128a_decode_topk_lens",),
        )
        prefill_local = self._first_tensor_attr(
            layer_attn_metadata,
            ("c128a_prefill_topk_indices",),
        )

        if (
            decode_indices is None
            and decode_topk_length is None
            and prefill_local is None
            and positions is not None
        ):
            synth_metadata = self._synthesize_c128a_metadata(
                positions=positions,
                query_start_loc=query_start_loc,
                compress_ratio=compress_ratio,
                slot_mapping=slot_mapping,
            )
            layer_attn_metadata = types.SimpleNamespace(
                **getattr(layer_attn_metadata, "__dict__", {}),
                **synth_metadata,
            )
            decode_indices = self._first_tensor_attr(
                layer_attn_metadata,
                ("c128a_global_decode_topk_indices",),
            )
            decode_topk_length = self._first_tensor_attr(
                layer_attn_metadata,
                ("c128a_decode_topk_lens",),
            )
            prefill_local = self._first_tensor_attr(
                layer_attn_metadata,
                ("c128a_prefill_topk_indices",),
            )

        block_table = getattr(layer_attn_metadata, "block_table", None)
        block_size = int(getattr(layer_attn_metadata, "block_size", 64))

        mapped_decode = None
        mapped_prefill = None
        num_decode_tokens = 0
        num_prefill_tokens = 0
        if decode_indices is not None:
            mapped_decode = self._map_c128_dense_slots_to_compiler_slots(
                decode_indices.reshape(decode_indices.shape[0], -1),
                compressed_slot_offset,
                compress_ratio,
                block_size=block_size,
            )
            num_decode_tokens = int(mapped_decode.shape[0])
        if prefill_local is not None:
            if block_table is None:
                raise RuntimeError(
                    "C128 prefill sparse metadata is missing block_table."
                )
            num_prefill_tokens = int(prefill_local.shape[0])
            mapped_prefill = self._map_c128a_prefill_local_indices_to_slots(
                prefill_local,
                block_table,
                query_start_loc,
                compress_ratio,
                compressed_slot_offset,
                num_decode_tokens=num_decode_tokens,
                block_size=block_size,
            )

        if mapped_decode is None and mapped_prefill is None:
            if recent_indices is not None:
                if os.getenv("PAITON_DEBUG_SPARSE_MLA", "0") == "1":
                    print(
                        "[PAITON_DEBUG_SPARSE_MLA] "
                        f"layer={layer_idx} ratio={compress_ratio} "
                        "missing c128 metadata, falling back to recent-window indices only",
                        flush=True,
                    )
                return self._combine_sparse_index_lists(
                    None,
                    None,
                    recent_indices,
                    recent_topk_length,
                    width=self._sparse_mla_index_width(),
                )
            raise RuntimeError(
                "Compressed layer expected C128 sparse metadata, but none was found "
                "and no recent-window fallback was available."
            )

        compressed_width = max(
            int(mapped_decode.shape[1]) if mapped_decode is not None else 0,
            int(mapped_prefill.shape[1]) if mapped_prefill is not None else 0,
        )
        num_tokens = (
            int(recent_indices.shape[0])
            if recent_indices is not None
            else num_decode_tokens + num_prefill_tokens
        )
        compressed_indices = torch.full(
            (num_tokens, compressed_width),
            -1,
            dtype=torch.int32,
            device=(mapped_decode if mapped_decode is not None else mapped_prefill).device,
        )
        compressed_topk_length = torch.zeros(
            (num_tokens,),
            dtype=torch.int32,
            device=compressed_indices.device,
        )
        if mapped_decode is not None:
            compressed_indices[:num_decode_tokens, : mapped_decode.shape[1]] = mapped_decode
            if decode_topk_length is None:
                compressed_topk_length[:num_decode_tokens] = (
                    mapped_decode >= 0
                ).sum(dim=-1).to(dtype=torch.int32)
            else:
                compressed_topk_length[:num_decode_tokens] = decode_topk_length.to(
                    dtype=torch.int32,
                    copy=False,
                )
        if mapped_prefill is not None:
            start = num_decode_tokens
            end = start + num_prefill_tokens
            compressed_indices[start:end, : mapped_prefill.shape[1]] = mapped_prefill
            compressed_topk_length[start:end] = (mapped_prefill >= 0).sum(dim=-1).to(
                dtype=torch.int32
            )

        return self._combine_sparse_index_lists(
            compressed_indices,
            compressed_topk_length,
            recent_indices,
            recent_topk_length,
            width=self._sparse_mla_index_width(),
        )

    @staticmethod
    def _split_decode_and_prefill_tokens(
        query_start_loc: torch.Tensor,
    ) -> tuple[int, int]:
        if query_start_loc.numel() < 2:
            return 0, 0

        query_lens = (
            query_start_loc[1:].to(dtype=torch.int64)
            - query_start_loc[:-1].to(dtype=torch.int64)
        )
        is_prefill = query_lens > 1
        if not bool(is_prefill.any().item()):
            num_tokens = int(query_start_loc[-1].item())
            return num_tokens, 0

        first_prefill = int(torch.argmax(is_prefill.to(dtype=torch.int32)).item())
        num_decode_tokens = int(query_start_loc[first_prefill].item())
        num_prefill_tokens = int(query_start_loc[-1].item()) - num_decode_tokens
        return num_decode_tokens, num_prefill_tokens

    @staticmethod
    def _build_c128a_dense_topk_rows(
        positions: torch.Tensor,
        *,
        compress_ratio: int,
        width: int,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = int(positions.shape[0])
        device = positions.device
        rows = torch.full(
            (num_tokens, width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        lens = torch.clamp(
            (positions.to(dtype=torch.int64) + 1) // int(compress_ratio),
            min=0,
            max=width,
        ).to(dtype=torch.int32)

        if slot_mapping is not None:
            valid = slot_mapping[:num_tokens] >= 0
            lens = torch.where(valid, lens, torch.zeros_like(lens))

        token_rows = torch.arange(num_tokens, device=device)
        col_ids = torch.arange(width, device=device, dtype=torch.int32)
        valid_cols = col_ids.unsqueeze(0) < lens.unsqueeze(1)
        if valid_cols.any():
            rows[token_rows.unsqueeze(1).expand_as(rows)[valid_cols], col_ids.unsqueeze(0).expand_as(rows)[valid_cols]] = col_ids.unsqueeze(0).expand_as(rows)[valid_cols]
        return rows.contiguous(), lens.contiguous()

    def _synthesize_c128a_metadata(
        self,
        *,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        compress_ratio: int,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        num_decode_tokens, num_prefill_tokens = self._split_decode_and_prefill_tokens(
            query_start_loc
        )
        num_tokens = int(positions.shape[0])
        if num_tokens != num_decode_tokens + num_prefill_tokens:
            num_prefill_tokens = max(0, num_tokens - num_decode_tokens)

        max_compressed = int(
            (((max(0, int(positions.max().item()) + 1) // int(compress_ratio)) + 127) // 128)
            * 128
        ) if num_tokens > 0 else 128
        width = max(128, min(self._sparse_mla_index_width(), max_compressed))
        out: dict[str, torch.Tensor] = {}

        if num_decode_tokens > 0:
            decode_rows, decode_lens = self._build_c128a_dense_topk_rows(
                positions[:num_decode_tokens],
                compress_ratio=compress_ratio,
                width=width,
                slot_mapping=slot_mapping,
            )
            out["c128a_global_decode_topk_indices"] = decode_rows.view(
                num_decode_tokens, 1, width
            )
            out["c128a_decode_topk_lens"] = decode_lens

        if num_prefill_tokens > 0:
            prefill_rows, _ = self._build_c128a_dense_topk_rows(
                positions[num_decode_tokens:],
                compress_ratio=compress_ratio,
                width=width,
            )
            out["c128a_prefill_topk_indices"] = prefill_rows

        return out

    @staticmethod
    def _layer_sparse_input_copy(
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Make a per-layer mutable copy for sparse MLA inputs.

        The compiled C4 path mutates ``sparse_mla_indices`` and
        ``sparse_mla_topk_length`` in-place when it prepends compressed top-k
        slots ahead of the recent-window seed. Reusing the same backing across
        multiple layers in one forward lets earlier layers corrupt later ones.
        """
        return tensor.to(dtype=dtype, copy=True).contiguous()

    def _get_deepseek_v4_state_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        *,
        indexer: bool,
    ) -> torch.Tensor:
        caches_attr = (
            "_deepseek_v4_indexer_state_caches"
            if indexer
            else "_deepseek_v4_compressor_state_caches"
        )
        caches = getattr(self, caches_attr, None)
        if caches is None:
            caches = {}
            setattr(self, caches_attr, caches)

        kv_block_size = int(kv_cache.shape[2])
        max_num_blocks = int(kv_cache.shape[1])
        block_size = self._state_cache_block_size(
            layer_idx,
            indexer=indexer,
            fallback_block_size=kv_block_size,
        )
        required_blocks = 1
        if slot_mapping is not None:
            valid_slots = slot_mapping[slot_mapping >= 0]
            if valid_slots.numel() > 0:
                required_blocks = max(
                    required_blocks,
                    int(valid_slots.max().item()) // block_size + 1,
                )
        if block_tables is not None:
            valid_blocks = block_tables[block_tables >= 0]
            if valid_blocks.numel() > 0:
                required_blocks = max(
                    required_blocks,
                    int(valid_blocks.max().item()) + 1,
                )
        if required_blocks > max_num_blocks:
            raise RuntimeError(
                "DeepSeek V4 compressor metadata references block "
                f"{required_blocks - 1}, but the runtime KV cache only has "
                f"{max_num_blocks} blocks.")

        compress_ratio = self._layer_compress_ratio(layer_idx)
        coff = 1 + int(compress_ratio == 4)
        head_dim = self._index_head_dim() if indexer else int(
            getattr(self.config, "head_dim", 512))
        state_dim = 2 * coff * head_dim
        device = kv_cache.device

        current = caches.get(layer_idx)
        if (
            current is None
            or current.device != device
            or current.dtype != torch.float32
            or current.shape[0] < required_blocks
            or current.shape[1] != block_size
            or current.shape[2] != state_dim
        ):
            new_cache = torch.zeros(
                (required_blocks, block_size, state_dim),
                dtype=torch.float32,
                device=device,
            )
            if current is not None and current.numel() > 0:
                copy_blocks = min(current.shape[0], new_cache.shape[0])
                copy_tokens = min(current.shape[1], new_cache.shape[1])
                copy_width = min(current.shape[2], new_cache.shape[2])
                new_cache[:copy_blocks, :copy_tokens, :copy_width].copy_(
                    current[:copy_blocks, :copy_tokens, :copy_width])
            caches[layer_idx] = new_cache
            current = new_cache
        return current

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
            device="cuda",
        )
        if not all_attn_metadata:
            return output
        if os.getenv("PAITON_DEBUG_SPARSE_MLA", "0") == "1" and isinstance(all_attn_metadata, dict):
            print(
                "[PAITON_DEBUG_SPARSE_MLA] "
                f"attn_metadata_keys={list(all_attn_metadata.keys())[:32]}",
                flush=True,
            )

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
                device="cuda",
            )

        input_ids_i32 = input_ids.to(dtype=torch.int32, copy=False).contiguous()
        run_input_backings: list[torch.Tensor] = [input_ids_i32]
        inputs: Dict[str, PData] = {
            "input_ids": torch_to_paiton_data(input_ids_i32),
        }
        expected_inputs = set(self.model.get_input_name_to_index_map())

        slot_mapping_i64 = None
        query_start_loc_i32 = None
        seq_lens_i32 = None
        block_table_i32 = None

        # Capture context for the indexer FP8 MQA runtime hooks.
        def num_tokens_for_input_alloc() -> int:
            return int(input_ids.shape[0])

        def device_for_input_alloc() -> torch.device:
            return input_ids.device

        if positions is not None:
            position_ids_i64 = positions.to(dtype=torch.int64, copy=False).contiguous()
            run_input_backings.append(position_ids_i64)
            inputs["position_ids"] = torch_to_paiton_data(position_ids_i64)

        if hasattr(attn_metadata, "slot_mapping"):
            slot_mapping_i64 = attn_metadata.slot_mapping.to(
                dtype=torch.int64, copy=False).contiguous()
            run_input_backings.append(slot_mapping_i64)
            inputs["slot_mapping"] = torch_to_paiton_data(slot_mapping_i64)

        if hasattr(attn_metadata, "query_start_loc"):
            query_start_loc_i32 = attn_metadata.query_start_loc.to(
                dtype=torch.int32, copy=False).contiguous()
            run_input_backings.append(query_start_loc_i32)
            inputs["query_start_locations"] = torch_to_paiton_data(
                query_start_loc_i32)

        if hasattr(attn_metadata, "seq_lens"):
            seq_lens_i32 = attn_metadata.seq_lens.to(
                dtype=torch.int32, copy=False).contiguous()
            run_input_backings.append(seq_lens_i32)
            inputs["context_lengths"] = torch_to_paiton_data(seq_lens_i32)

        if hasattr(attn_metadata, "block_table"):
            block_table_i32 = attn_metadata.block_table.to(
                dtype=torch.int32, copy=False).contiguous()
            run_input_backings.append(block_table_i32)
            inputs["block_tables"] = torch_to_paiton_data(block_table_i32)

        generated_sparse_indices = None
        generated_sparse_topk_length = None

        max_query_len_backing = torch.empty([1], dtype=torch.int32, device="cuda")
        max_seq_len_backing = torch.empty([1], dtype=torch.int32, device="cuda")
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

        for i in range(self.num_layers):
            ctx = self.compilation_config.static_forward_context.get(str(i))
            if ctx is None:
                continue
            layer_attn_metadata = (
                all_attn_metadata.get(str(i), attn_metadata)
                if isinstance(all_attn_metadata, dict)
                else attn_metadata
            )
            if os.getenv("PAITON_DEBUG_SPARSE_MLA", "0") == "1" and i < 6:
                layer_sparse_attrs = sorted(
                    name
                    for name in dir(layer_attn_metadata)
                    if "topk" in name or "sparse" in name or name.startswith("c128a_")
                )
                print(
                    "[PAITON_DEBUG_SPARSE_MLA] "
                    f"layer={i} ratio={self._layer_compress_ratio(i)} "
                    f"attrs={layer_sparse_attrs[:16]}",
                    flush=True,
                )
            kv_cache = self._get_kv_cache_tensor(ctx)
            if kv_cache is None:
                continue
            run_input_backings.append(kv_cache)
            inputs[f"kv_cache_{i}"] = torch_to_paiton_data(
                kv_cache.view(self.cache_dtype))
            
            needs_sparse = (
                f"sparse_mla_kv_{i}" in expected_inputs
                or f"sparse_mla_indexer_kv_{i}" in expected_inputs
                or f"sparse_mla_indices_{i}" in expected_inputs
                or f"sparse_mla_topk_length_{i}" in expected_inputs
                or f"sparse_mla_compressed_offset_{i}" in expected_inputs
            )
            
            if (
                generated_sparse_indices is None
                and needs_sparse
            ):
                if (query_start_loc_i32 is not None
                    and seq_lens_i32 is not None
                    and block_table_i32 is not None):
                    generated_sparse_indices, generated_sparse_topk_length = (
                        self._build_recent_sparse_mla_indices(
                            query_start_loc_i32,
                            seq_lens_i32,
                            block_table_i32,
                            int(input_ids.shape[0]),
                            self._sparse_mla_index_width(),
                            int(kv_cache.shape[2]),
                            input_ids.device,
                            int(self._deepseek_sliding_window or 0),
                        )
                    )
                    run_input_backings.extend(
                        [generated_sparse_indices, generated_sparse_topk_length]
                    )
                else:
                    raise RuntimeError(
                        "DeepSeek V4 sparse MLA needs query_start_loc, seq_lens, "
                        "and block_table metadata to build sparse indices."
                    )
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
            if (
                sparse_kv is None
                and f"sparse_mla_kv_{i}" in expected_inputs
            ):
                if slot_mapping_i64 is not None and generated_sparse_indices is not None:
                    compressed_slot_offset = (
                        self._compressed_sparse_slot_offset(
                            slot_mapping_i64,
                            generated_sparse_indices,
                        )
                        if self._layer_compress_ratio(i) > 1
                        else None
                    )
                    sparse_kv = self._get_sparse_mla_kv_cache(
                        i,
                        kv_cache,
                        slot_mapping_i64,
                        generated_sparse_indices,
                        compressed_slot_offset=compressed_slot_offset,
                    )
                else:
                    raise RuntimeError(
                        f"sparse_mla_kv_{i} is expected but runtime sparse "
                        "KV cache metadata is unavailable."
                    )
            if sparse_kv is not None:
                run_input_backings.append(sparse_kv)
                inputs[f"sparse_mla_kv_{i}"] = torch_to_paiton_data(
                    sparse_kv.contiguous())

            if f"compressor_state_cache_{i}" in expected_inputs:
                compressor_state_cache = self._get_deepseek_v4_state_cache(
                    i,
                    kv_cache,
                    slot_mapping_i64,
                    block_table_i32,
                    indexer=False,
                )
                run_input_backings.append(compressor_state_cache)
                inputs[f"compressor_state_cache_{i}"] = torch_to_paiton_data(
                    compressor_state_cache.contiguous())

            if (
                f"sparse_mla_compressed_offset_{i}" in expected_inputs
                and slot_mapping_i64 is not None
            ):
                compressed_slot_offset_value = self._compressed_sparse_slot_offset(
                    slot_mapping_i64,
                    generated_sparse_indices,
                )
                compressed_slot_offset = torch.tensor(
                    [compressed_slot_offset_value],
                    dtype=torch.int64,
                    device=input_ids.device,
                )
                run_input_backings.append(compressed_slot_offset)
                inputs[f"sparse_mla_compressed_offset_{i}"] = torch_to_paiton_data(
                    compressed_slot_offset.contiguous())

            if f"indexer_state_cache_{i}" in expected_inputs:
                indexer_state_cache = self._get_deepseek_v4_state_cache(
                    i,
                    kv_cache,
                    slot_mapping_i64,
                    block_table_i32,
                    indexer=True,
                )
                run_input_backings.append(indexer_state_cache)
                inputs[f"indexer_state_cache_{i}"] = torch_to_paiton_data(
                    indexer_state_cache.contiguous())

            sparse_mla_indexer_kv = None
            if f"sparse_mla_indexer_kv_{i}" in expected_inputs:
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
                        self._compressed_sparse_slot_offset(
                            slot_mapping_i64,
                            generated_sparse_indices,
                        )
                        if self._layer_compress_ratio(i) > 1
                        else None
                    ),
                )
                run_input_backings.append(sparse_mla_indexer_kv)
                inputs[f"sparse_mla_indexer_kv_{i}"] = torch_to_paiton_data(
                    sparse_mla_indexer_kv.contiguous())

            # The compiled artifact produces the C4 sparse-indexer aux tensors
            # in-graph. The runtime only allocates storage and binds it here.
            if f"indexer_q_fp8_{i}" in expected_inputs:
                indexer_q_fp8 = torch.empty(
                    (num_tokens_for_input_alloc(),
                     self._index_n_heads(),
                     self._index_head_dim()),
                    dtype=torch.float8_e4m3fnuz,
                    device=device_for_input_alloc(),
                )
                run_input_backings.append(indexer_q_fp8)
                inputs[f"indexer_q_fp8_{i}"] = torch_to_paiton_data(
                    indexer_q_fp8.contiguous())

            if f"indexer_weights_{i}" in expected_inputs:
                indexer_weights = torch.empty(
                    (num_tokens_for_input_alloc(),
                     self._index_n_heads()),
                    dtype=torch.float32,
                    device=device_for_input_alloc(),
                )
                run_input_backings.append(indexer_weights)
                inputs[f"indexer_weights_{i}"] = torch_to_paiton_data(
                    indexer_weights.contiguous())

            if (
                f"indexer_k_fp8_{i}" in expected_inputs
                or f"indexer_k_scale_{i}" in expected_inputs
            ):
                if sparse_mla_indexer_kv is None:
                    raise RuntimeError(
                        f"indexer_k_fp8_{i} / indexer_k_scale_{i} require "
                        f"sparse_mla_indexer_kv_{i}, but it was not bound."
                    )
                num_sparse_rows = int(sparse_mla_indexer_kv.shape[0])
                indexer_k_fp8 = torch.empty(
                    (num_sparse_rows, self._index_head_dim()),
                    dtype=torch.float8_e4m3fnuz,
                    device=device_for_input_alloc(),
                )
                indexer_k_scale = torch.empty(
                    (num_sparse_rows,),
                    dtype=torch.float32,
                    device=device_for_input_alloc(),
                )
                if f"indexer_k_fp8_{i}" in expected_inputs:
                    run_input_backings.append(indexer_k_fp8)
                    inputs[f"indexer_k_fp8_{i}"] = torch_to_paiton_data(
                        indexer_k_fp8.contiguous())
                if f"indexer_k_scale_{i}" in expected_inputs:
                    run_input_backings.append(indexer_k_scale)
                    inputs[f"indexer_k_scale_{i}"] = torch_to_paiton_data(
                        indexer_k_scale.contiguous())

            if (
                f"cu_seqlen_ks_{i}" in expected_inputs
                or f"cu_seqlen_ke_{i}" in expected_inputs
            ):
                cu_seqlen_ks = torch.empty(
                    (num_tokens_for_input_alloc(),),
                    dtype=torch.int32,
                    device=device_for_input_alloc(),
                )
                cu_seqlen_ke = torch.empty(
                    (num_tokens_for_input_alloc(),),
                    dtype=torch.int32,
                    device=device_for_input_alloc(),
                )
                if f"cu_seqlen_ks_{i}" in expected_inputs:
                    run_input_backings.append(cu_seqlen_ks)
                    inputs[f"cu_seqlen_ks_{i}"] = torch_to_paiton_data(
                        cu_seqlen_ks.contiguous())
                if f"cu_seqlen_ke_{i}" in expected_inputs:
                    run_input_backings.append(cu_seqlen_ke)
                    inputs[f"cu_seqlen_ke_{i}"] = torch_to_paiton_data(
                        cu_seqlen_ke.contiguous())

            compressed_slot_offset_value = (
                self._compressed_sparse_slot_offset(
                    slot_mapping_i64,
                    generated_sparse_indices,
                )
                if (
                    slot_mapping_i64 is not None
                    and generated_sparse_indices is not None
                    and self._layer_compress_ratio(i) > 1
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
                self._layer_compress_ratio(i) == 128
                and compressed_slot_offset_value is not None
                and query_start_loc_i32 is not None
                and generated_sparse_indices is not None
            ):
                sparse_indices, sparse_topk_length = self._build_c128a_sparse_inputs(
                    layer_attn_metadata,
                    query_start_loc_i32,
                    compressed_slot_offset_value,
                    generated_sparse_indices,
                    generated_sparse_topk_length,
                    layer_idx=i,
                    positions=position_ids_i64 if positions is not None else None,
                    slot_mapping=slot_mapping_i64,
                )
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
            elif f"sparse_mla_indices_{i}" in expected_inputs:
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
            elif f"sparse_mla_topk_length_{i}" in expected_inputs:
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
        self._debug_sparse_mla_inputs(
            expected_inputs,
            inputs,
            generated_sparse_indices,
            generated_sparse_topk_length,
            run_input_backings,
        )
        filtered_inputs = {
            name: value for name, value in inputs.items() if name in expected_inputs
        }
        missing_inputs = expected_inputs - set(filtered_inputs)
        if missing_inputs:
            raise RuntimeError(
                "Paiton DeepSeek V4 artifact input mismatch. Missing inputs: "
                + ", ".join(sorted(missing_inputs)))

        outputs = {"logits": torch_to_paiton_data(runtime_output)}
        stream_ptr = torch.cuda.current_stream().cuda_stream
        self._run_input_backings = run_input_backings
        # Match the Qwen MoE runtime: the compiled runtime launches kernels
        # outside PyTorch's normal bookkeeping, so we synchronize here for
        # correctness.
        self.model.run(filtered_inputs, outputs, stream_ptr=stream_ptr, sync=True)
        if runtime_output is not output:
            if query_start_loc_i32 is None or query_start_loc_i32.numel() < num_runtime_rows + 1:
                raise RuntimeError(
                    "DeepSeek V4 compact logits output requires query_start_loc metadata "
                    "to expand per-request logits back into vLLM's per-token shape."
                )
            sample_rows = (query_start_loc_i32[1:] - 1).to(dtype=torch.int64)
            output.index_copy_(0, sample_rows, runtime_output)
        if os.getenv("PAITON_DEBUG_LOGITS", "0") == "1" and output.numel() > 0:
            debug_rows = [("row0", output[0].float())]
            if query_start_loc_i32 is not None and query_start_loc_i32.numel() >= 2:
                sampled_row_idx = int((query_start_loc_i32[1] - 1).item())
                if sampled_row_idx != 0:
                    debug_rows.append(
                        (f"sampled_row{sampled_row_idx}", output[sampled_row_idx].float())
                    )

            debug_parts = []
            for label, row in debug_rows:
                finite = torch.isfinite(row)
                top_vals, top_ids = torch.topk(row, k=min(10, row.numel()))
                debug_parts.append(
                    f"{label}: "
                    f"finite={int(finite.sum().item())}/{row.numel()} "
                    f"min={float(row[finite].min().item()) if finite.any() else float('nan')} "
                    f"max={float(row[finite].max().item()) if finite.any() else float('nan')} "
                    f"top_ids={top_ids.detach().cpu().tolist()} "
                    f"top_vals={[float(x) for x in top_vals.detach().cpu().tolist()]}"
                )
            print("[PAITON_DEBUG_LOGITS] " + " | ".join(debug_parts), flush=True)
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
            elif ".attn.wq_b.weight" in name or ".attn.wo_a.weight" in name:
                out_name = convert_name(name)
                value = self.get_rank_weight(param, dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
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
            if os.getenv("PAITON_DEBUG_ROUTER_DEFAULTS", "0") == "1":
                print(
                    "[PAITON_DEBUG_ROUTER_DEFAULTS] "
                    f"synthesizing missing hash_indices_table for layer={layer_id}",
                    flush=True,
                )
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
