from __future__ import annotations

import types
from typing import Optional

import torch


class DeepseekV4SparseRuntimeMixin:
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
        """Build recent-token sparse MLA indices in physical KV-slot space.

        GPU-vectorized implementation. Replaces the original CPU triple-loop
        (``.to("cpu").tolist()`` + per-block ``.item()``) which caused a major
        per-step latency spike. The algorithm is a standard stream-compaction:

        1. Map each token to its request via ``searchsorted`` on
           ``query_start_loc``.
        2. Compute per-token ``logical_pos`` and ``first_pos``.
        3. Build a ``[num_tokens, recent_window]`` candidate-position matrix.
        4. Vectorized block-table lookup -> physical slots + validity mask.
        5. ``cumsum`` the validity mask to get dense output columns.
        6. Scatter-write valid slots; ``topk_length`` = per-row valid count.
        """
        indices = torch.full(
            (num_tokens, index_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        topk_lengths = torch.zeros((num_tokens,), dtype=torch.int32, device=device)
        if index_topk <= 0 or num_tokens <= 0:
            return indices, topk_lengths

        if recent_window is None or recent_window <= 0:
            recent_window = index_topk
        recent_window = min(index_topk, int(recent_window))

        qsl = query_start_loc.to(device=device, dtype=torch.int64)
        sl = seq_lens.to(device=device, dtype=torch.int64)
        bt = block_table.to(device=device, dtype=torch.int64)
        max_blocks = int(bt.shape[1])

        token_idx = torch.arange(num_tokens, device=device, dtype=torch.int64)
        # req_idx: which request each token belongs to.
        req_idx = torch.searchsorted(qsl, token_idx, right=True) - 1
        req_idx = req_idx.clamp(min=0, max=int(qsl.numel()) - 2)

        token_start = qsl[req_idx]
        local_pos = token_idx - token_start
        query_len = qsl[req_idx + 1] - token_start
        seq_len = sl[req_idx]
        context_before_query = (seq_len - query_len).clamp(min=0)
        logical_pos = context_before_query + local_pos

        first_pos = (logical_pos - recent_window + 1).clamp(min=0)

        # candidate_positions[t, c] = first_pos[t] + c  (shape [N, recent_window])
        col_offset = torch.arange(recent_window, device=device, dtype=torch.int64)
        candidate_pos = first_pos.unsqueeze(1) + col_offset.unsqueeze(0)
        # Valid where candidate_pos <= logical_pos (the window upper bound).
        window_valid = candidate_pos <= logical_pos.unsqueeze(1)

        block_col = candidate_pos // block_size
        block_offset = candidate_pos % block_size
        # block_col out of range -> invalid.
        block_col_valid = block_col < max_blocks
        # Gather block_id for each (token, candidate): block_table[req_idx, block_col]
        safe_block_col = block_col.clamp(min=0, max=max(0, max_blocks - 1))
        block_id = bt[req_idx.unsqueeze(1), safe_block_col]
        block_id_valid = block_id >= 0

        valid = window_valid & block_col_valid & block_id_valid
        slot = block_id * block_size + block_offset

        # Stream-compaction: cumsum of valid mask gives dense output columns.
        valid_int = valid.to(dtype=torch.int32)
        out_col = torch.cumsum(valid_int, dim=1) - 1
        # Only write where valid AND out_col < index_topk.
        write_mask = valid & (out_col < index_topk)
        safe_out_col = out_col.clamp(min=0)
        # Scatter-write only valid slots (invalid entries leave the -1 fill).
        write_rows = token_idx.unsqueeze(1).expand(-1, recent_window)
        indices[write_rows[write_mask], safe_out_col[write_mask]] = \
            slot[write_mask].to(dtype=torch.int32)
        # topk_length = number of valid entries per row, capped at index_topk.
        topk_lengths = valid_int.sum(dim=1).clamp(max=index_topk).to(dtype=torch.int32)

        return indices.contiguous(), topk_lengths.contiguous()

    def _get_sparse_mla_kv_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        *,
        compressed_slot_offset: Optional[int] = None,
        required_slots: Optional[int] = None,
    ) -> torch.Tensor:
        caches = getattr(self, "_sparse_mla_kv_caches", None)
        if caches is None:
            caches = {}
            self._sparse_mla_kv_caches = caches
        offsets = getattr(self, "_sparse_mla_kv_offsets", None)
        if offsets is None:
            offsets = {}
            self._sparse_mla_kv_offsets = offsets

        if required_slots is None:
            required_slots = 1
            valid_slots = slot_mapping[slot_mapping >= 0]
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
        max_raw_slots = int(kv_cache.shape[1]) * int(kv_cache.shape[2])
        max_slots = 2 * max_raw_slots if compressed_slot_offset is not None else max_raw_slots
        capacity_slots = self._rounded_runtime_capacity(
            required_slots,
            current=int(current.shape[0]) if current is not None else None,
            quantum=256,
            maximum=max_slots,
        )
        if (
            current is None
            or current.device != device
            or current.dtype != self.dtype
            or current.shape[0] < required_slots
            or current.shape[2] != head_dim
            or previous_offset != compressed_slot_offset
        ):
            new_cache = torch.empty(
                (capacity_slots, 1, head_dim),
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
        required_slots: Optional[int] = None,
    ) -> torch.Tensor:
        caches = getattr(self, "_sparse_mla_indexer_kv_caches", None)
        if caches is None:
            caches = {}
            self._sparse_mla_indexer_kv_caches = caches
        offsets = getattr(self, "_sparse_mla_indexer_kv_offsets", None)
        if offsets is None:
            offsets = {}
            self._sparse_mla_indexer_kv_offsets = offsets

        if required_slots is None:
            required_slots = 1
            valid_slots = slot_mapping[slot_mapping >= 0]
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
        max_raw_slots = int(kv_cache.shape[1]) * int(kv_cache.shape[2])
        max_slots = 2 * max_raw_slots if compressed_slot_offset is not None else max_raw_slots
        capacity_slots = self._rounded_runtime_capacity(
            required_slots,
            current=int(current.shape[0]) if current is not None else None,
            quantum=256,
            maximum=max_slots,
        )
        if (
            current is None
            or current.device != device
            or current.dtype != self.dtype
            or current.shape[0] < required_slots
            or current.shape[2] != head_dim
            or previous_offset != compressed_slot_offset
        ):
            new_cache = torch.empty(
                (capacity_slots, 1, head_dim),
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
        *,
        required_slots: Optional[int] = None,
    ) -> int:
        if required_slots is not None:
            return required_slots
        required_slots = 1
        valid_slots = slot_mapping[slot_mapping >= 0]
        if valid_slots.numel() > 0:
            required_slots = int(valid_slots.max().item()) + 1
        if sparse_indices is not None:
            valid_indices = sparse_indices[sparse_indices >= 0]
            if valid_indices.numel() > 0:
                required_slots = max(required_slots, int(valid_indices.max().item()) + 1)
        return required_slots

    @staticmethod
    def _compute_step_slot_extents(
        slot_mapping: Optional[torch.Tensor],
        sparse_indices: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
    ) -> tuple[int, int, int, int]:
        """Compute per-step slot/block extents ONCE, avoiding per-layer .item() syncs.

        Returns (required_slots, max_slot, max_block, required_blocks_from_bt):
          - required_slots: max physical slot referenced by slot_mapping or
            sparse_indices (+1). Identical across all layers for a given step.
          - max_slot: max value in slot_mapping (or -1). Used by per-layer state
            caches to derive required_blocks = max_slot // block_size + 1 without
            another .item() (block_size can vary per layer).
          - max_block: max value in block_tables (or -1). Block-table-derived
            block extent, block-size-independent.
          - required_blocks_from_bt: max_block + 1 (or 1 if no block_tables).

        This collapses the ~150-170 per-layer .item() GPU->CPU syncs (43 layers
        x ~3-4 calls each) down to at most two .item() syncs per step.
        """
        max_slot = -1
        required_slots = 1
        if slot_mapping is not None:
            valid_slots = slot_mapping[slot_mapping >= 0]
            if valid_slots.numel() > 0:
                max_slot = int(valid_slots.max().item())
                required_slots = max(required_slots, max_slot + 1)
        if sparse_indices is not None:
            valid_indices = sparse_indices[sparse_indices >= 0]
            if valid_indices.numel() > 0:
                max_idx = int(valid_indices.max().item())
                required_slots = max(required_slots, max_idx + 1)

        max_block = -1
        required_blocks_from_bt = 1
        if block_tables is not None:
            valid_blocks = block_tables[block_tables >= 0]
            if valid_blocks.numel() > 0:
                max_block = int(valid_blocks.max().item())
                required_blocks_from_bt = max_block + 1

        return required_slots, max_slot, max_block, required_blocks_from_bt

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

            recent_width = int(recent_indices.shape[1])
            cols = torch.arange(recent_width, device=device, dtype=torch.int32)
            dst_col = out_len.unsqueeze(1) + cols.unsqueeze(0)
            valid = (
                (recent_indices >= 0)
                & (cols.unsqueeze(0) < recent_topk_length.unsqueeze(1))
                & (dst_col < width)
            )
            if valid.any():
                token_rows = torch.arange(num_tokens, device=device).unsqueeze(1)
                out[
                    token_rows.expand_as(recent_indices)[valid],
                    dst_col.to(dtype=torch.int64)[valid],
                ] = recent_indices[valid]
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
        block_table: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
        num_prefill_tokens: int = 0,
    ) -> torch.Tensor:
        """Map C128 dense compressed indices to physical KV-cache slots.

        When ``block_table`` and ``query_start_loc`` are provided, this is
        request-aware: each decode token is mapped to its request via
        ``searchsorted`` on ``query_start_loc``, and the compressed block
        index is translated through the per-request ``block_table`` row.
        This prevents different requests' compressed KV from aliasing the
        same physical slots in batched decode.

        When ``block_table`` / ``query_start_loc`` are ``None`` (the legacy
        path), the mapping is the original formula — correct only for
        single-request decode or when dense indices are globally unique.
        """
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

        if block_table is not None and query_start_loc is not None:
            # Request-aware path: translate physical_blocks through the
            # per-request block_table so different requests' compressed KV
            # lands at distinct physical slots.
            num_decode_tokens = int(dense_indices.shape[0])
            device = dense_indices.device
            bt = block_table.to(device=device, dtype=torch.int64)
            qsl = query_start_loc.to(device=device, dtype=torch.int64)
            max_blocks = int(bt.shape[1])

            # Map each decode token to its request. Decode tokens are the
            # first num_decode_tokens tokens in the batch (prefill tokens
            # follow, if any).
            token_idx = torch.arange(
                num_decode_tokens, dtype=torch.int64, device=device
            )
            req_idx = torch.searchsorted(
                qsl[1:], token_idx, right=True
            ).to(dtype=torch.int64)
            req_idx = req_idx.clamp(min=0, max=int(qsl.numel()) - 2)

            # Look up the actual KV block ID for each compressed block.
            # physical_blocks has shape [num_tokens, width]; we need
            # block_table[req_idx[t], physical_blocks[t, c]] for each (t, c).
            safe_block_cols = physical_blocks.clamp(min=0, max=max(0, max_blocks - 1))
            # Gather: req_idx is [num_tokens], physical_blocks is [num_tokens, width]
            req_expanded = req_idx.unsqueeze(1).expand_as(safe_block_cols)
            block_ids = bt[req_expanded, safe_block_cols]  # [num_tokens, width]

            block_valid = (physical_blocks < max_blocks) & (block_ids >= 0)
            valid = valid & block_valid

            mapped = (
                int(compressed_slot_offset)
                + block_ids * int(block_size)
                + raw_block_offsets
            ).to(dtype=torch.int32)
        else:
            # Legacy non-request-aware path (single request or globally unique).
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
                block_table=block_table,
                query_start_loc=query_start_loc,
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
        if tensor.dtype == dtype and tensor.is_contiguous():
            return tensor.clone()
        return tensor.to(dtype=dtype, copy=True).contiguous()

    def _get_deepseek_v4_state_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        slot_mapping: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        *,
        indexer: bool,
        required_blocks: Optional[int] = None,
        max_slot: Optional[int] = None,
        max_block: Optional[int] = None,
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
        if required_blocks is None:
            required_blocks = 1
            if max_slot is not None and max_slot >= 0:
                required_blocks = max(
                    required_blocks,
                    max_slot // block_size + 1,
                )
            if max_block is not None and max_block >= 0:
                required_blocks = max(required_blocks, max_block + 1)
            elif max_slot is None:
                # Fall back to per-call .item() only when no precomputed
                # extents were supplied (backward-compatible path).
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
        capacity_blocks = self._rounded_runtime_capacity(
            required_blocks,
            current=int(current.shape[0]) if current is not None else None,
            quantum=16,
            maximum=max_num_blocks,
        )
        if (
            current is None
            or current.device != device
            or current.dtype != torch.float32
            or current.shape[0] < required_blocks
            or current.shape[1] != block_size
            or current.shape[2] != state_dim
        ):
            new_cache = torch.zeros(
                (capacity_blocks, block_size, state_dim),
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
