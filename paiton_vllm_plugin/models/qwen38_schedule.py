# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral scheduled-token ABI translation for Qwen3.8."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

import torch

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from paiton_vllm_plugin.models.qwen38_contract import Qwen38ContractError


MAX_ACTIVE_SEQUENCES = 128
MAX_SCHEDULED_TOKENS = 4096
MAX_SEQUENCE_LENGTH = 4096
KV_BLOCK_SIZE = 784
MAX_BLOCKS_PER_SEQUENCE = 6
REQUIRED_PHYSICAL_KV_BLOCKS = 1153
REQUIRED_GDN_STATE_SLOTS = REQUIRED_PHYSICAL_KV_BLOCKS
GDN_STATE_PAGE_BYTES = KV_BLOCK_SIZE * 4 * 256 * 2 * 2
GDN_CONV_PAGE_STRIDE = GDN_STATE_PAGE_BYTES // 2
GDN_RECURRENT_PAGE_STRIDE = GDN_STATE_PAGE_BYTES // 4
ATTENTION_PLANE_STRIDE = KV_BLOCK_SIZE * 4 * 256
ATTENTION_BLOCK_STRIDE = 2 * ATTENTION_PLANE_STRIDE
TOKEN_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_PROVIDER_ROUTE = "default-v1"
M128_STOCK_EXACT_DECODE_ROUTE = "stock-exact128-decode-m128-v1"
M128_PAITON_GENERIC_ROUTE = "paiton-generic-m128-v1"
M128_STOCK_PROVIDER_ID = (
    "paiton-qwen38-stock-triton36-mfma-paged-attention-hsaco-gfx950-"
    "exact128-decode-m128-v2"
)
M128_GENERIC_PROVIDER_ID = (
    "paiton-qwen38-scheduled-gqa-two-pass-hip-partial-m128-v1"
)
M128_ROUTE_PREDICATE = {
    "id": "paiton.qwen38.m128-route.exact128-one-token-decode.v1",
    "version": 1,
    "dispatch": "host-selected-hash-bound-aot-artifact",
    "stock_route": M128_STOCK_EXACT_DECODE_ROUTE,
    "generic_route": M128_PAITON_GENERIC_ROUTE,
    "stock_conditions": {
        "token_bucket": 128,
        "actual_token_count": 128,
        "actual_sequence_count": 128,
        "scheduler_phase": {
            "num_prefills": 0,
            "num_prefill_tokens": 0,
            "num_decodes": 128,
            "num_decode_tokens": 128,
            "num_actual_tokens": 128,
        },
        "query_length_per_sequence": 1,
        "runtime_contracts_validated": [
            "gdn-state",
            "attention-kv",
            "query-layout",
            "block-table",
            "slot-mapping",
            "position-layout",
        ],
    },
    "otherwise": M128_PAITON_GENERIC_ROUTE,
}
M128_ROUTE_PREDICATE_SHA256 = hashlib.sha256(
    json.dumps(
        M128_ROUTE_PREDICATE, sort_keys=True, separators=(",", ":")
    ).encode()
).hexdigest()
LINEAR_ATTENTION_LAYERS = tuple(
    layer for layer in range(64) if (layer + 1) % 4
)
GDN_CACHE_GROUPS = tuple(
    LINEAR_ATTENTION_LAYERS[index::3] for index in range(3)
)


def scheduled_token_bucket(tokens: int) -> int:
    if not isinstance(tokens, int) or tokens < 1 or tokens > MAX_SCHEDULED_TOKENS:
        raise Qwen38ContractError(
            f"scheduled token count must be in [1,{MAX_SCHEDULED_TOKENS}], got {tokens!r}"
        )
    return next(bucket for bucket in TOKEN_BUCKETS if bucket >= tokens)


def scheduled_provider_route(
    *,
    token_count: int,
    sequence_count: int,
    token_bucket: int,
    scheduler_phase: tuple[int, int, int, int, int],
    runtime_contracts_validated: bool,
) -> str:
    """Select one hash-bound AOT provider without reading device metadata."""

    if not runtime_contracts_validated:
        raise Qwen38ContractError(
            "scheduled provider route requires validated state/KV/layout contracts"
        )
    exact_m128_decode = (
        token_bucket == 128
        and token_count == 128
        and sequence_count == 128
        and scheduler_phase == (0, 0, 128, 128, 128)
    )
    if exact_m128_decode:
        return M128_STOCK_EXACT_DECODE_ROUTE
    if token_bucket == 128:
        if not 65 <= token_count <= 127:
            raise Qwen38ContractError(
                "generic M128 provider route is restricted to 65-127 tokens"
            )
        return M128_PAITON_GENERIC_ROUTE
    return DEFAULT_PROVIDER_ROUTE


@dataclass(frozen=True)
class Qwen38ScheduledStep:
    token_count: int
    sequence_count: int
    token_bucket: int
    provider_route: str
    provider_route_predicate_sha256: str | None
    output_rows: int
    query_start_locations: torch.Tensor
    token_to_sequence: torch.Tensor
    context_lengths: torch.Tensor
    gdn_state_indices: Mapping[int, torch.Tensor]
    gdn_has_initial_state: torch.Tensor
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    logit_row_indices: torch.Tensor
    gdn_storage: Mapping[int, tuple[torch.Tensor, torch.Tensor]]
    attention_storage: Mapping[int, torch.Tensor]


class Qwen38ScheduledMetadataBridge:
    """Validate one vLLM scheduler step and expose one batched Paiton call.

    Tensor-content guards use asynchronous device assertions. Structural
    checks use only Python metadata and tensor shape/dtype, avoiding D2H reads
    in the request path.
    """

    def __init__(
        self,
        gdn_layers: Mapping[int, Any],
        attention_layers: Mapping[int, Any],
    ) -> None:
        if not gdn_layers or not attention_layers:
            raise Qwen38ContractError("scheduled bridge requires GDN and attention layers")
        self.gdn_layers = dict(gdn_layers)
        self.attention_layers = dict(attention_layers)

    @staticmethod
    def _async_assert(value: torch.Tensor, message: str) -> None:
        try:
            torch._assert_async(value, message)
        except TypeError:
            torch._assert_async(value)

    @staticmethod
    def _same_tensor_view(left: torch.Tensor, right: torch.Tensor) -> bool:
        """Return whether two scheduler tensors describe the same device bytes."""
        return left is right or (
            left.dtype == right.dtype
            and left.device == right.device
            and left.shape == right.shape
            and left.stride() == right.stride()
            and left.data_ptr() == right.data_ptr()
        )

    @classmethod
    def _assert_equal_or_same(
        cls,
        left: torch.Tensor,
        right: torch.Tensor,
        message: str,
    ) -> None:
        # vLLM shares schedule tensors across layers. Avoid launching a GPU
        # equality reduction when identity already proves the same bytes while
        # retaining the content guard for independently materialized metadata.
        if not cls._same_tensor_view(left, right):
            cls._async_assert(torch.all(left == right), message)

    @staticmethod
    def _require_tensor(
        value: object,
        *,
        name: str,
        dtype: torch.dtype,
        rank: int,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise Qwen38ContractError(f"{name} must be a tensor")
        if value.dtype != dtype or value.ndim != rank:
            raise Qwen38ContractError(
                f"{name} must have dtype={dtype} rank={rank}, "
                f"got dtype={value.dtype} shape={tuple(value.shape)}"
            )
        return value.contiguous()

    @staticmethod
    def _contiguous_descriptor_alias(
        tensor: torch.Tensor,
        *,
        name: str,
        physical_stride: tuple[int, ...],
        descriptor_stride: tuple[int, ...],
    ) -> torch.Tensor:
        if tensor.stride() != physical_stride:
            raise Qwen38ContractError(
                f"{name} physical stride differs: {tensor.stride()} != "
                f"{physical_stride}"
            )
        alias = torch.as_strided(
            tensor,
            size=tensor.shape,
            stride=descriptor_stride,
            storage_offset=tensor.storage_offset(),
        )
        if not alias.is_contiguous() or alias.data_ptr() != tensor.data_ptr():
            raise Qwen38ContractError(
                f"{name} cannot form a zero-copy contiguous runtime descriptor"
            )
        return alias

    def prepare(
        self,
        metadata: Mapping[str, Any],
        tokens: int,
        text_positions: torch.Tensor | None = None,
    ) -> Qwen38ScheduledStep:
        bucket = scheduled_token_bucket(tokens)
        first_gdn_layer = next(iter(self.gdn_layers))
        first_gdn_key = (
            f"model.language_model.layers.{first_gdn_layer}.linear_attn"
        )
        first_gdn = metadata.get(first_gdn_key)
        if not isinstance(first_gdn, GDNAttentionMetadata):
            raise Qwen38ContractError("missing scheduled GDN metadata")
        if first_gdn.num_spec_decodes or first_gdn.num_spec_decode_tokens:
            raise Qwen38ContractError("speculative GDN schedules are not admitted")
        if first_gdn.spec_sequence_masks is not None:
            raise Qwen38ContractError("speculative GDN masks are not admitted")
        if first_gdn.num_actual_tokens != tokens:
            raise Qwen38ContractError(
                f"GDN scheduled tokens {first_gdn.num_actual_tokens} != input {tokens}"
            )
        state_indices = self._require_tensor(
            first_gdn.non_spec_state_indices_tensor,
            name="GDN state indices",
            dtype=torch.int32,
            rank=1,
        )
        sequence_count = int(state_indices.shape[0])
        if sequence_count < 1 or sequence_count > MAX_ACTIVE_SEQUENCES:
            raise Qwen38ContractError(
                f"active sequence count must be in [1,{MAX_ACTIVE_SEQUENCES}], "
                f"got {sequence_count}"
            )
        query_starts = self._require_tensor(
            first_gdn.non_spec_query_start_loc,
            name="GDN query starts",
            dtype=torch.int32,
            rank=1,
        )
        if tuple(query_starts.shape) != (sequence_count + 1,):
            raise Qwen38ContractError(
                "GDN query starts must contain one boundary per active sequence"
            )
        raw_initial: torch.Tensor | None = None
        if first_gdn.has_initial_state is None:
            has_initial = torch.ones_like(state_indices, dtype=torch.int32)
        else:
            raw_initial = self._require_tensor(
                first_gdn.has_initial_state,
                name="GDN initial-state mask",
                dtype=torch.bool,
                rank=1,
            )
            if tuple(raw_initial.shape) != (sequence_count,):
                raise Qwen38ContractError(
                    "GDN initial-state mask differs from active sequence count"
                )
            has_initial = raw_initial.to(dtype=torch.int32).contiguous()

        phase = (
            first_gdn.num_prefills,
            first_gdn.num_prefill_tokens,
            first_gdn.num_decodes,
            first_gdn.num_decode_tokens,
            first_gdn.num_actual_tokens,
        )
        if (
            any(not isinstance(value, int) or value < 0 for value in phase)
            or phase[0] + phase[2] != sequence_count
            or phase[1] + phase[3] != tokens
            or phase[4] != tokens
        ):
            raise Qwen38ContractError(
                "GDN scheduler phase totals differ from active sequences/tokens"
            )
        exact_m128_decode = sequence_count == 128 and phase == (
            0,
            0,
            128,
            128,
            128,
        )
        # The embedded stock-MFMA provider is guarded by the exact 128-token
        # extent. Keep exact-128 prefills and mixed schedules on M256 unless
        # all 128 rows are one-token decodes. Sub-128 M128 buckets use only
        # the separately validated generic Paiton fallback.
        if bucket == 128 and tokens == 128 and not exact_m128_decode:
            bucket = 256
        gdn_storage: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        gdn_state_indices: dict[int, torch.Tensor] = {}
        for layer, holder in self.gdn_layers.items():
            key = f"model.language_model.layers.{layer}.linear_attn"
            layer_metadata = metadata.get(key)
            if not isinstance(layer_metadata, GDNAttentionMetadata):
                raise Qwen38ContractError(f"missing GDN metadata for layer {layer}")
            layer_phase = (
                layer_metadata.num_prefills,
                layer_metadata.num_prefill_tokens,
                layer_metadata.num_decodes,
                layer_metadata.num_decode_tokens,
                layer_metadata.num_actual_tokens,
            )
            if layer_phase != phase:
                raise Qwen38ContractError(
                    f"GDN scheduler phase differs for layer {layer}: "
                    f"{layer_phase} != {phase}"
                )
            layer_indices = self._require_tensor(
                layer_metadata.non_spec_state_indices_tensor,
                name=f"GDN state indices layer {layer}",
                dtype=torch.int32,
                rank=1,
            )
            if tuple(layer_indices.shape) != (sequence_count,):
                raise Qwen38ContractError(
                    f"GDN state-index count differs for layer {layer}"
                )
            layer_query_starts = self._require_tensor(
                layer_metadata.non_spec_query_start_loc,
                name=f"GDN query starts layer {layer}",
                dtype=torch.int32,
                rank=1,
            )
            if tuple(layer_query_starts.shape) != (sequence_count + 1,):
                raise Qwen38ContractError(
                    f"GDN query-start count differs for layer {layer}"
                )
            self._assert_equal_or_same(
                layer_query_starts,
                query_starts,
                f"GDN query starts differ for layer {layer}",
            )
            raw_layer_initial = layer_metadata.has_initial_state
            if raw_layer_initial is None and raw_initial is None:
                layer_initial = has_initial
            elif (
                raw_layer_initial is not None
                and raw_initial is not None
                and self._same_tensor_view(
                    self._require_tensor(
                        raw_layer_initial,
                        name=f"GDN initial-state mask layer {layer}",
                        dtype=torch.bool,
                        rank=1,
                    ),
                    raw_initial,
                )
            ):
                layer_initial = has_initial
            else:
                layer_initial = (
                    torch.ones_like(layer_indices, dtype=torch.int32)
                    if raw_layer_initial is None
                    else self._require_tensor(
                        raw_layer_initial,
                        name=f"GDN initial-state mask layer {layer}",
                        dtype=torch.bool,
                        rank=1,
                    ).to(dtype=torch.int32)
                )
            if tuple(layer_initial.shape) != (sequence_count,):
                raise Qwen38ContractError(
                    f"GDN initial-state count differs for layer {layer}"
                )
            self._assert_equal_or_same(
                layer_initial,
                has_initial,
                f"GDN initial-state mask differs for layer {layer}",
            )
            storage = getattr(holder, "kv_cache", None)
            if not isinstance(storage, (tuple, list)) or len(storage) != 2:
                raise Qwen38ContractError(f"GDN layer {layer} has no scheduler storage")
            conv, recurrent = storage
            if (
                not isinstance(conv, torch.Tensor)
                or conv.dtype != torch.bfloat16
                or conv.ndim != 3
                or tuple(conv.shape[1:]) != (3, 10240)
            ):
                raise Qwen38ContractError(
                    f"GDN convolution storage differs for layer {layer}: "
                    f"{getattr(conv, 'shape', None)}"
                )
            if (
                not isinstance(recurrent, torch.Tensor)
                or recurrent.dtype != torch.float32
                or recurrent.ndim != 4
                or tuple(recurrent.shape[1:]) != (48, 128, 128)
            ):
                raise Qwen38ContractError(
                    f"GDN recurrent storage differs for layer {layer}: "
                    f"{getattr(recurrent, 'shape', None)}"
                )
            if (
                conv.shape[0] != REQUIRED_GDN_STATE_SLOTS
                or recurrent.shape[0] != REQUIRED_GDN_STATE_SLOTS
            ):
                raise Qwen38ContractError(
                    f"GDN layer {layer} must expose exactly "
                    f"{REQUIRED_GDN_STATE_SLOTS} scheduler state slots"
                )
            conv_alias = self._contiguous_descriptor_alias(
                conv,
                name=f"GDN convolution storage layer {layer}",
                physical_stride=(GDN_CONV_PAGE_STRIDE, 10240, 1),
                descriptor_stride=(3 * 10240, 10240, 1),
            )
            recurrent_alias = self._contiguous_descriptor_alias(
                recurrent,
                name=f"GDN recurrent storage layer {layer}",
                physical_stride=(
                    GDN_RECURRENT_PAGE_STRIDE,
                    128 * 128,
                    128,
                    1,
                ),
                descriptor_stride=(48 * 128 * 128, 128 * 128, 128, 1),
            )
            gdn_state_indices[layer] = layer_indices
            gdn_storage[layer] = (
                conv_alias,
                recurrent_alias,
            )

        first_attention_layer = next(iter(self.attention_layers))
        first_attention_key = (
            f"model.language_model.layers.{first_attention_layer}.self_attn"
        )
        first_attention = metadata.get(first_attention_key)
        if first_attention is None:
            raise Qwen38ContractError("missing scheduled attention metadata")
        attention_query_starts = self._require_tensor(
            getattr(first_attention, "query_start_loc", None),
            name="attention query starts",
            dtype=torch.int32,
            rank=1,
        )
        context_lengths = self._require_tensor(
            getattr(first_attention, "seq_lens", None),
            name="attention context lengths",
            dtype=torch.int32,
            rank=1,
        )
        raw_block_tables = self._require_tensor(
            getattr(first_attention, "block_table", None),
            name="attention block tables",
            dtype=torch.int32,
            rank=2,
        )
        raw_slot_mapping = self._require_tensor(
            getattr(first_attention, "slot_mapping", None),
            name="attention slot mapping",
            dtype=torch.int64,
            rank=1,
        )
        if tuple(attention_query_starts.shape) != (sequence_count + 1,):
            raise Qwen38ContractError("attention query starts differ from sequence count")
        if tuple(context_lengths.shape) != (sequence_count,):
            raise Qwen38ContractError("attention context lengths differ from sequence count")
        if (
            raw_block_tables.shape[0] != sequence_count
            or raw_block_tables.shape[1] < MAX_BLOCKS_PER_SEQUENCE
        ):
            raise Qwen38ContractError(
                "attention block tables do not admit 128 sequences of length 4096"
            )
        if raw_slot_mapping.shape[0] < tokens:
            raise Qwen38ContractError("attention slot mapping is shorter than scheduled tokens")
        block_tables = raw_block_tables[:, :MAX_BLOCKS_PER_SEQUENCE].contiguous()
        slot_mapping = raw_slot_mapping[:tokens].contiguous()
        self._assert_equal_or_same(
            attention_query_starts,
            query_starts,
            "GDN and attention query starts differ",
        )

        attention_storage: dict[int, torch.Tensor] = {}
        for layer, holder in self.attention_layers.items():
            key = f"model.language_model.layers.{layer}.self_attn"
            layer_metadata = metadata.get(key)
            if layer_metadata is None:
                raise Qwen38ContractError(
                    f"missing attention metadata for layer {layer}"
                )
            if int(getattr(layer_metadata, "num_actual_tokens", -1)) != tokens:
                raise Qwen38ContractError(
                    f"attention token count differs for layer {layer}"
                )
            layer_query = self._require_tensor(
                getattr(layer_metadata, "query_start_loc", None),
                name=f"attention query starts layer {layer}",
                dtype=torch.int32,
                rank=1,
            )
            layer_context = self._require_tensor(
                getattr(layer_metadata, "seq_lens", None),
                name=f"attention context lengths layer {layer}",
                dtype=torch.int32,
                rank=1,
            )
            layer_blocks = self._require_tensor(
                getattr(layer_metadata, "block_table", None),
                name=f"attention block tables layer {layer}",
                dtype=torch.int32,
                rank=2,
            )
            layer_slots = self._require_tensor(
                getattr(layer_metadata, "slot_mapping", None),
                name=f"attention slot mapping layer {layer}",
                dtype=torch.int64,
                rank=1,
            )
            if (
                tuple(layer_query.shape) != (sequence_count + 1,)
                or tuple(layer_context.shape) != (sequence_count,)
                or layer_blocks.shape[0] != sequence_count
                or layer_blocks.shape[1] < MAX_BLOCKS_PER_SEQUENCE
                or layer_slots.shape[0] < tokens
            ):
                raise Qwen38ContractError(
                    f"attention scheduler shapes differ for layer {layer}"
                )
            self._assert_equal_or_same(
                layer_query,
                attention_query_starts,
                f"attention query starts differ for layer {layer}",
            )
            self._assert_equal_or_same(
                layer_context,
                context_lengths,
                f"attention context lengths differ for layer {layer}",
            )
            if not self._same_tensor_view(layer_blocks, raw_block_tables):
                self._async_assert(
                    torch.all(
                        layer_blocks[:, :MAX_BLOCKS_PER_SEQUENCE] == block_tables
                    ),
                    f"attention block tables differ for layer {layer}",
                )
            if not self._same_tensor_view(layer_slots, raw_slot_mapping):
                self._async_assert(
                    torch.all(layer_slots[:tokens] == slot_mapping),
                    f"attention slot mapping differs for layer {layer}",
                )
            physical = getattr(holder, "kv_cache", None)
            if (
                not isinstance(physical, torch.Tensor)
                or physical.dtype != torch.bfloat16
                or physical.ndim != 5
                or physical.shape[0] != 2
                or physical.shape[1] != REQUIRED_PHYSICAL_KV_BLOCKS
                or tuple(physical.shape[2:]) != (KV_BLOCK_SIZE, 4, 256)
            ):
                raise Qwen38ContractError(
                    f"attention layer {layer} lacks required physical KV capacity"
                )
            attention_storage[layer] = self._contiguous_descriptor_alias(
                physical,
                name=f"attention storage layer {layer}",
                physical_stride=(
                    ATTENTION_PLANE_STRIDE,
                    ATTENTION_BLOCK_STRIDE,
                    4 * 256,
                    256,
                    1,
                ),
                descriptor_stride=(
                    REQUIRED_PHYSICAL_KV_BLOCKS * ATTENTION_PLANE_STRIDE,
                    ATTENTION_PLANE_STRIDE,
                    4 * 256,
                    256,
                    1,
                ),
            )

        query_lengths = query_starts[1:] - query_starts[:-1]
        token_to_sequence = torch.repeat_interleave(
            torch.arange(sequence_count, dtype=torch.int32, device=query_starts.device),
            query_lengths.to(dtype=torch.int64),
        ).contiguous()
        if tuple(token_to_sequence.shape) != (tokens,):
            raise Qwen38ContractError(
                "scheduler query lengths do not sum to the scheduled token count"
            )
        logit_rows = (query_starts[1:] - 1).to(dtype=torch.int32).contiguous()

        self._async_assert(query_starts[0] == 0, "query starts must begin at zero")
        self._async_assert(query_starts[-1] == tokens, "query starts must end at tokens")
        self._async_assert(torch.all(query_lengths > 0), "empty query rows are not admitted")
        self._async_assert(
            torch.all((context_lengths >= 1) & (context_lengths <= MAX_SEQUENCE_LENGTH)),
            "context length outside the admitted range",
        )
        self._async_assert(
            torch.all(context_lengths >= query_lengths),
            "context length is shorter than the scheduled query",
        )
        self._async_assert(
            torch.all(
                (slot_mapping >= 0)
                & (slot_mapping < REQUIRED_PHYSICAL_KV_BLOCKS * KV_BLOCK_SIZE)
            ),
            "slot mapping outside physical KV capacity",
        )
        required_blocks = (context_lengths + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
        block_columns = torch.arange(
            MAX_BLOCKS_PER_SEQUENCE,
            dtype=torch.int32,
            device=block_tables.device,
        ).unsqueeze(0)
        referenced = block_columns < required_blocks.unsqueeze(1)
        self._async_assert(
            torch.all(
                (~referenced)
                | (
                    (block_tables >= 0)
                    & (block_tables < REQUIRED_PHYSICAL_KV_BLOCKS)
                )
            ),
            "referenced block table entry outside physical KV capacity",
        )
        representative_gdn_indices = []
        for group_index, group_layers in enumerate(GDN_CACHE_GROUPS):
            present = [layer for layer in group_layers if layer in gdn_state_indices]
            if not present:
                continue
            representative = gdn_state_indices[present[0]]
            representative_gdn_indices.append(representative)
            for layer in present[1:]:
                self._assert_equal_or_same(
                    gdn_state_indices[layer],
                    representative,
                    f"GDN cache-group {group_index} state indices differ at layer {layer}",
                )
            self._async_assert(
                torch.all(
                    (representative >= 0)
                    & (representative < REQUIRED_GDN_STATE_SLOTS)
                ),
                f"GDN state index outside the admitted pool for layer {present[0]}",
            )
            sorted_state_indices = torch.sort(representative).values
            self._async_assert(
                torch.all(
                    sorted_state_indices[1:] != sorted_state_indices[:-1]
                ),
                f"duplicate GDN slot ownership for layer {present[0]}",
            )
        active_allocations = [*representative_gdn_indices, block_tables[referenced]]
        if active_allocations:
            all_allocations = torch.cat(active_allocations)
            sorted_allocations = torch.sort(all_allocations).values
            self._async_assert(
                torch.all(sorted_allocations[1:] != sorted_allocations[:-1]),
                "hybrid-cache block ownership overlaps across active groups",
            )

        if text_positions is not None:
            positions = self._require_tensor(
                text_positions,
                name="scheduled text positions",
                dtype=torch.int64,
                rank=1,
            )
            if tuple(positions.shape) != (tokens,):
                raise Qwen38ContractError(
                    "scheduled text-position count differs from scheduled tokens"
                )
            sequence_context_starts = context_lengths - query_lengths
            token_context_starts = torch.repeat_interleave(
                sequence_context_starts,
                query_lengths.to(dtype=torch.int64),
            )
            token_query_starts = torch.repeat_interleave(
                query_starts[:-1],
                query_lengths.to(dtype=torch.int64),
            )
            expected_positions = token_context_starts.to(dtype=torch.int64) + (
                torch.arange(tokens, dtype=torch.int64, device=positions.device)
                - token_query_starts.to(dtype=torch.int64)
            )
            self._async_assert(
                torch.all(positions == expected_positions),
                "scheduled positions differ from query/context boundaries",
            )
            expected_blocks = block_tables[
                token_to_sequence.to(dtype=torch.int64),
                positions // KV_BLOCK_SIZE,
            ].to(dtype=torch.int64)
            expected_slots = (
                expected_blocks * KV_BLOCK_SIZE + positions % KV_BLOCK_SIZE
            )
            self._async_assert(
                torch.all(slot_mapping == expected_slots),
                "slot mapping differs from positions and block tables",
            )

        provider_route = scheduled_provider_route(
            token_count=tokens,
            sequence_count=sequence_count,
            token_bucket=bucket,
            scheduler_phase=phase,
            runtime_contracts_validated=True,
        )

        return Qwen38ScheduledStep(
            token_count=tokens,
            sequence_count=sequence_count,
            token_bucket=bucket,
            provider_route=provider_route,
            provider_route_predicate_sha256=(
                M128_ROUTE_PREDICATE_SHA256
                if bucket == 128
                else None
            ),
            output_rows=min(bucket, MAX_ACTIVE_SEQUENCES),
            query_start_locations=query_starts,
            token_to_sequence=token_to_sequence,
            context_lengths=context_lengths,
            gdn_state_indices=gdn_state_indices,
            gdn_has_initial_state=has_initial,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            logit_row_indices=logit_rows,
            gdn_storage=gdn_storage,
            attention_storage=attention_storage,
        )


__all__ = [
    "DEFAULT_PROVIDER_ROUTE",
    "MAX_ACTIVE_SEQUENCES",
    "MAX_SCHEDULED_TOKENS",
    "M128_GENERIC_PROVIDER_ID",
    "M128_PAITON_GENERIC_ROUTE",
    "M128_ROUTE_PREDICATE",
    "M128_ROUTE_PREDICATE_SHA256",
    "M128_STOCK_EXACT_DECODE_ROUTE",
    "M128_STOCK_PROVIDER_ID",
    "Qwen38ScheduledMetadataBridge",
    "Qwen38ScheduledStep",
    "TOKEN_BUCKETS",
    "scheduled_provider_route",
    "scheduled_token_bucket",
]
