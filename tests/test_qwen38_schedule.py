from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from paiton_vllm_plugin.models.qwen38_contract import Qwen38ContractError
from paiton_vllm_plugin.models.qwen38_schedule import (
    DEFAULT_PROVIDER_ROUTE,
    M128_PAITON_GENERIC_ROUTE,
    M128_ROUTE_PREDICATE_SHA256,
    M128_STOCK_EXACT_DECODE_ROUTE,
    Qwen38ScheduledMetadataBridge,
    TOKEN_BUCKETS,
    scheduled_provider_route,
    scheduled_token_bucket,
)


GDN_KEY = "model.language_model.layers.0.linear_attn"
ATTN_KEY = "model.language_model.layers.3.self_attn"


def _bridge():
    page_bytes = 784 * 4 * 256 * 2 * 2
    raw = torch.empty(1153 * page_bytes, dtype=torch.uint8, device="meta")
    conv = torch.as_strided(
        raw.view(torch.bfloat16),
        size=(1153, 3, 10240),
        stride=(page_bytes // 2, 10240, 1),
    )
    recurrent = torch.as_strided(
        raw.view(torch.float32),
        size=(1153, 48, 128, 128),
        stride=(page_bytes // 4, 128 * 128, 128, 1),
        storage_offset=(3 * 10240 * 2) // 4,
    )
    kv = torch.as_strided(
        raw.view(torch.bfloat16),
        size=(2, 1153, 784, 4, 256),
        stride=(784 * 4 * 256, 2 * 784 * 4 * 256, 4 * 256, 256, 1),
    )
    return Qwen38ScheduledMetadataBridge(
        {0: SimpleNamespace(kv_cache=(conv, recurrent))},
        {3: SimpleNamespace(kv_cache=kv)},
    )


def _metadata(
    query_starts,
    context_lengths,
    state_indices,
    *,
    has_initial=None,
    block_tables=None,
    slot_mapping=None,
):
    starts = torch.tensor(query_starts, dtype=torch.int32)
    contexts = torch.tensor(context_lengths, dtype=torch.int32)
    states = torch.tensor(state_indices, dtype=torch.int32)
    tokens = query_starts[-1]
    sequences = len(context_lengths)
    if block_tables is None:
        block_tables = torch.full((sequences, 6), -1, dtype=torch.int32)
        block_tables[:, 0] = torch.arange(100, 100 + sequences, dtype=torch.int32)
    if slot_mapping is None:
        query_lengths = starts[1:] - starts[:-1]
        context_starts = contexts - query_lengths
        positions = torch.cat(
            [
                torch.arange(
                    int(context_starts[index]),
                    int(contexts[index]),
                    dtype=torch.int64,
                )
                for index in range(sequences)
            ]
        )
        sequence_rows = torch.repeat_interleave(
            torch.arange(sequences, dtype=torch.int64),
            query_lengths.to(torch.int64),
        )
        selected_blocks = block_tables[
            sequence_rows, positions // 784
        ].to(torch.int64)
        slot_mapping = selected_blocks * 784 + positions % 784
    initial = (
        None
        if has_initial is None
        else torch.tensor(has_initial, dtype=torch.bool)
    )
    query_lengths = starts[1:] - starts[:-1]
    decodes = int(torch.count_nonzero(query_lengths == 1))
    prefills = sequences - decodes
    decode_tokens = decodes
    prefill_tokens = tokens - decode_tokens
    gdn = GDNAttentionMetadata(
        num_prefills=prefills,
        num_prefill_tokens=prefill_tokens,
        num_decodes=decodes,
        num_decode_tokens=decode_tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=tokens,
        has_initial_state=initial,
        non_spec_query_start_loc=starts,
        non_spec_state_indices_tensor=states,
    )
    attention = SimpleNamespace(
        num_actual_tokens=tokens,
        query_start_loc=starts.clone(),
        seq_lens=contexts,
        block_table=block_tables,
        slot_mapping=slot_mapping,
    )
    return {GDN_KEY: gdn, ATTN_KEY: attention}


def test_bucket_selection_is_deterministic_and_bounded():
    assert TOKEN_BUCKETS == (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    assert scheduled_token_bucket(1) == 1
    assert scheduled_token_bucket(3) == 4
    assert scheduled_token_bucket(129) == 256
    assert scheduled_token_bucket(4096) == 4096
    for invalid in (0, 4097, None):
        with pytest.raises(Qwen38ContractError, match="scheduled token count"):
            scheduled_token_bucket(invalid)


def test_homogeneous_decode_becomes_one_batched_step():
    bridge = _bridge()
    step = bridge.prepare(
        _metadata([0, 1, 2, 3, 4], [11, 21, 31, 41], [2, 5, 7, 9]),
        4,
    )
    assert step.token_count == 4
    assert step.sequence_count == 4
    assert step.token_bucket == 4
    assert step.output_rows == 4
    assert step.provider_route == DEFAULT_PROVIDER_ROUTE
    assert step.provider_route_predicate_sha256 is None
    assert torch.equal(step.token_to_sequence, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(step.logit_row_indices, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(step.gdn_has_initial_state, torch.ones(4, dtype=torch.int32))
    assert torch.equal(
        step.gdn_state_indices[0], torch.tensor([2, 5, 7, 9], dtype=torch.int32)
    )
    assert step.gdn_storage[0][0].device.type == "meta"
    assert step.attention_storage[3].device.type == "meta"
    assert step.attention_storage[3].is_contiguous()
    assert step.attention_storage[3].stride() == (
        1153 * 784 * 4 * 256,
        784 * 4 * 256,
        4 * 256,
        256,
        1,
    )


def test_exact_128_way_decode_selects_stock_only_m128_provider():
    step = _bridge().prepare(
        _metadata(
            list(range(129)),
            list(range(300, 428)),
            list(range(500, 628)),
        ),
        128,
    )
    assert step.sequence_count == 128
    assert step.token_bucket == 128
    assert step.provider_route == M128_STOCK_EXACT_DECODE_ROUTE
    assert step.provider_route_predicate_sha256 == M128_ROUTE_PREDICATE_SHA256


def test_exact_128_token_prefill_bypasses_stock_guarded_m128_provider():
    step = _bridge().prepare(
        _metadata([0, 128], [128], [500]),
        128,
    )
    assert step.sequence_count == 1
    assert step.token_bucket == 256
    assert step.provider_route == DEFAULT_PROVIDER_ROUTE
    assert step.provider_route_predicate_sha256 is None


def test_sub128_token_prefill_selects_generic_fallback_m128_provider():
    step = _bridge().prepare(
        _metadata([0, 70], [70], [500]),
        70,
    )
    assert step.sequence_count == 1
    assert step.token_bucket == 128
    assert step.provider_route == M128_PAITON_GENERIC_ROUTE
    assert step.provider_route_predicate_sha256 == M128_ROUTE_PREDICATE_SHA256


@pytest.mark.parametrize("sequences", [65, 79, 93, 106, 127])
def test_partial_decode_selects_paiton_only_m128_provider(sequences):
    step = _bridge().prepare(
        _metadata(
            list(range(sequences + 1)),
            list(range(300, 300 + sequences)),
            list(range(500, 500 + sequences)),
        ),
        sequences,
    )
    assert step.sequence_count == sequences
    assert step.token_bucket == 128
    assert step.provider_route == M128_PAITON_GENERIC_ROUTE
    assert step.provider_route_predicate_sha256 == M128_ROUTE_PREDICATE_SHA256


def test_exact_128_token_mixed_schedule_routes_to_m256():
    query_starts = [0, 2, *range(3, 129)]
    contexts = [2, *range(301, 427)]
    step = _bridge().prepare(
        _metadata(query_starts, contexts, list(range(500, 627))),
        128,
    )
    assert step.sequence_count == 127
    assert step.token_bucket == 256
    assert step.provider_route == DEFAULT_PROVIDER_ROUTE


@pytest.mark.parametrize("tokens", [1, 64])
def test_small_decode_buckets_retain_default_provider_route(tokens):
    step = _bridge().prepare(
        _metadata(
            list(range(tokens + 1)),
            list(range(300, 300 + tokens)),
            list(range(500, 500 + tokens)),
        ),
        tokens,
    )
    assert step.token_bucket == tokens
    assert step.provider_route == DEFAULT_PROVIDER_ROUTE


def test_provider_route_rejects_unvalidated_runtime_contracts():
    with pytest.raises(Qwen38ContractError, match="validated state/KV/layout"):
        scheduled_provider_route(
            token_count=128,
            sequence_count=128,
            token_bucket=128,
            scheduler_phase=(0, 0, 128, 128, 128),
            runtime_contracts_validated=False,
        )


def test_inconsistent_scheduler_phase_fails_before_provider_selection():
    metadata = _metadata(
        list(range(80)),
        list(range(300, 379)),
        list(range(500, 579)),
    )
    metadata[GDN_KEY] = replace(metadata[GDN_KEY], num_decode_tokens=78)
    with pytest.raises(Qwen38ContractError, match="phase totals"):
        _bridge().prepare(metadata, 79)


def test_non_unit_exact_decode_query_layout_fails_closed():
    metadata = _metadata(
        list(range(129)),
        list(range(300, 428)),
        list(range(500, 628)),
    )
    malformed = metadata[GDN_KEY].non_spec_query_start_loc.clone()
    malformed[1] = 0
    malformed[2] = 2
    metadata[GDN_KEY] = replace(
        metadata[GDN_KEY], non_spec_query_start_loc=malformed
    )
    metadata[ATTN_KEY].query_start_loc = malformed.clone()
    with pytest.raises((AssertionError, RuntimeError)):
        _bridge().prepare(metadata, 128)


def test_non_hybrid_attention_layout_fails_closed():
    bridge = _bridge()
    bridge.attention_layers[3] = SimpleNamespace(
        kv_cache=torch.empty(
            (2, 1153, 784, 4, 256), dtype=torch.bfloat16, device="meta"
        )
    )
    with pytest.raises(Qwen38ContractError, match="physical stride differs"):
        bridge.prepare(_metadata([0, 1], [1], [1]), 1)


def test_ragged_mixed_step_preserves_scheduler_order():
    step = _bridge().prepare(
        _metadata(
            [0, 1, 4, 6],
            [129, 3, 18],
            [4, 8, 12],
            has_initial=[True, False, True],
        ),
        6,
    )
    assert step.token_bucket == 8
    assert torch.equal(
        step.token_to_sequence,
        torch.tensor([0, 1, 1, 1, 2, 2], dtype=torch.int32),
    )
    assert torch.equal(
        step.logit_row_indices, torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    assert torch.equal(
        step.gdn_has_initial_state, torch.tensor([1, 0, 1], dtype=torch.int32)
    )


def test_each_gdn_hybrid_group_preserves_its_own_state_indices():
    bridge = _bridge()
    bridge.gdn_layers[1] = bridge.gdn_layers[0]
    metadata = _metadata([0, 1, 2], [8, 9], [3, 4])
    metadata["model.language_model.layers.1.linear_attn"] = replace(
        metadata[GDN_KEY],
        non_spec_state_indices_tensor=torch.tensor([17, 23], dtype=torch.int32),
    )
    step = bridge.prepare(metadata, 2)
    assert torch.equal(
        step.gdn_state_indices[0], torch.tensor([3, 4], dtype=torch.int32)
    )
    assert torch.equal(
        step.gdn_state_indices[1], torch.tensor([17, 23], dtype=torch.int32)
    )


def test_same_gdn_cache_group_must_share_state_indices():
    bridge = _bridge()
    bridge.gdn_layers[4] = bridge.gdn_layers[0]
    metadata = _metadata([0, 1, 2], [8, 9], [3, 4])
    metadata["model.language_model.layers.4.linear_attn"] = replace(
        metadata[GDN_KEY],
        non_spec_state_indices_tensor=torch.tensor([17, 23], dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="cache-group 0 state indices differ"):
        bridge.prepare(metadata, 2)


def test_shared_layer_metadata_uses_one_set_of_content_guards(monkeypatch):
    bridge = _bridge()
    bridge.gdn_layers[4] = bridge.gdn_layers[0]
    bridge.attention_layers[7] = bridge.attention_layers[3]
    metadata = _metadata([0, 1, 2], [8, 9], [3, 4])
    metadata["model.language_model.layers.4.linear_attn"] = metadata[GDN_KEY]
    metadata["model.language_model.layers.7.self_attn"] = metadata[ATTN_KEY]

    messages = []
    original = Qwen38ScheduledMetadataBridge._async_assert

    def counting_assert(value, message):
        messages.append(message)
        original(value, message)

    monkeypatch.setattr(
        Qwen38ScheduledMetadataBridge,
        "_async_assert",
        staticmethod(counting_assert),
    )
    bridge.prepare(metadata, 2)

    assert len(messages) <= 12
    assert not any("layer 4" in message for message in messages)
    assert not any("layer 7" in message for message in messages)


def test_positions_and_slot_mapping_are_cross_validated():
    bridge = _bridge()
    metadata = _metadata([0, 1, 3], [11, 2], [3, 4])
    positions = torch.tensor([10, 0, 1], dtype=torch.int64)
    step = bridge.prepare(metadata, 3, positions)
    assert step.sequence_count == 2
    with pytest.raises(RuntimeError, match="positions differ"):
        bridge.prepare(metadata, 3, positions + 1)
    broken = dict(metadata)
    broken[ATTN_KEY] = SimpleNamespace(
        **{
            **vars(metadata[ATTN_KEY]),
            "slot_mapping": metadata[ATTN_KEY].slot_mapping + 1,
        }
    )
    with pytest.raises(RuntimeError, match="slot mapping differs"):
        bridge.prepare(broken, 3, positions)


def test_cross_group_hybrid_block_overlap_fails_closed():
    bridge = _bridge()
    bridge.gdn_layers[1] = bridge.gdn_layers[0]
    metadata = _metadata([0, 1], [1], [3])
    metadata["model.language_model.layers.1.linear_attn"] = replace(
        metadata[GDN_KEY],
        non_spec_state_indices_tensor=torch.tensor([3], dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="ownership overlaps"):
        bridge.prepare(metadata, 1)


def test_duplicate_state_slot_fails_closed():
    with pytest.raises(RuntimeError, match="duplicate GDN slot ownership"):
        _bridge().prepare(
            _metadata([0, 1, 2], [8, 9], [3, 3]),
            2,
        )


def test_invalid_referenced_block_fails_closed():
    blocks = torch.zeros((1, 6), dtype=torch.int32)
    blocks[0, 0] = 1153
    with pytest.raises(RuntimeError, match="referenced block table"):
        _bridge().prepare(
            _metadata(
                [0, 1],
                [1],
                [1],
                block_tables=blocks,
                slot_mapping=torch.tensor([0], dtype=torch.int64),
            ),
            1,
        )


def test_more_than_128_sequences_fails_before_storage_access():
    starts = list(range(130))
    with pytest.raises(Qwen38ContractError, match="active sequence count"):
        _bridge().prepare(
            _metadata(starts, [1] * 129, list(range(129))),
            129,
        )
