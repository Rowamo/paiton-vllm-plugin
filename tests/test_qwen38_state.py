from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from paiton_vllm_plugin.models.qwen38_contract import Qwen38ContractError
from paiton_vllm_plugin.models.qwen38_state import Qwen38SchedulerStateBridge


GDN_KEY = "model.language_model.layers.0.linear_attn"
ATTN_KEY = "model.language_model.layers.3.self_attn"


def _gdn_metadata(
    *,
    prefill: bool,
    tokens: int,
    state_index: int = 2,
    index_dtype: torch.dtype = torch.int64,
):
    return GDNAttentionMetadata(
        num_prefills=1 if prefill else 0,
        num_prefill_tokens=tokens if prefill else 0,
        num_decodes=0 if prefill else 1,
        num_decode_tokens=0 if prefill else tokens,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=tokens,
        non_spec_state_indices_tensor=torch.tensor([state_index], dtype=index_dtype),
    )


def _metadata(
    *,
    prefill: bool,
    tokens: int,
    max_seq_len: int,
    index_dtype: torch.dtype = torch.int64,
):
    return {
        GDN_KEY: _gdn_metadata(
            prefill=prefill, tokens=tokens, index_dtype=index_dtype
        ),
        ATTN_KEY: SimpleNamespace(
            num_actual_tokens=tokens,
            max_seq_len=max_seq_len,
            block_table=torch.tensor(
                [[5, 9, 11, 13, 15, 17, 3, 7, 1, 2, 4, 6, 8, 10, 12, 14]],
                dtype=index_dtype,
            ),
        ),
    }


def _bridge() -> tuple[Qwen38SchedulerStateBridge, object, object]:
    bridge = Qwen38SchedulerStateBridge.__new__(Qwen38SchedulerStateBridge)
    nn.Module.__init__(bridge)
    conv = torch.full((4, 3, 10240), 7, dtype=torch.bfloat16)
    recurrent = torch.full((4, 48, 128, 128), 9, dtype=torch.float32)
    gdn = SimpleNamespace(kv_cache=(conv, recurrent))
    kv = torch.full((2, 20, 16, 4, 256), 5, dtype=torch.bfloat16)
    attention = SimpleNamespace(kv_cache=kv)
    bridge.gdn_layers = {0: gdn}
    bridge.attention_layers = {3: attention}
    bridge._active = False
    bridge._decode_steps = 0
    bridge._sequence_length = 0
    bridge._persistent_inputs = None
    bridge._persistent_dirty = False
    return bridge, gdn, attention


def test_fresh_prefill_resets_selected_state_and_preserves_others():
    bridge, gdn, attention = _bridge()
    before_conv = gdn.kv_cache[0].clone()
    before_kv = attention.kv_cache.clone()
    prepared = bridge.prepare(
        _metadata(prefill=True, tokens=128, max_seq_len=128), 128, torch.device("cpu")
    )
    assert torch.count_nonzero(prepared.inputs["gdn_conv_state_0"]) == 0
    assert torch.count_nonzero(prepared.inputs["gdn_recurrent_state_0"]) == 0
    assert torch.count_nonzero(prepared.inputs["kv_cache_3"]) == 0

    prepared.inputs["gdn_conv_state_0"].fill_(1)
    prepared.inputs["gdn_recurrent_state_0"].fill_(2)
    prepared.inputs["kv_cache_3"][:, :8].fill_(3)
    bridge.commit(prepared)

    assert torch.equal(gdn.kv_cache[0][2], torch.ones_like(gdn.kv_cache[0][2]))
    assert torch.equal(gdn.kv_cache[1][2], torch.full_like(gdn.kv_cache[1][2], 2))
    assert torch.equal(gdn.kv_cache[0][0], before_conv[0])
    assert torch.equal(gdn.kv_cache[0][1], before_conv[1])
    assert torch.equal(gdn.kv_cache[0][3], before_conv[3])
    assert torch.equal(attention.kv_cache[:, 5], torch.full_like(attention.kv_cache[:, 5], 3))
    untouched = set(range(20)) - {5, 9, 11, 13, 15, 17, 3, 7}
    for block in untouched:
        assert torch.equal(attention.kv_cache[:, block], before_kv[:, block])


def test_decode_gathers_request_owned_state_in_logical_order():
    bridge, gdn, attention = _bridge()
    prefill = bridge.prepare(
        _metadata(prefill=True, tokens=1, max_seq_len=1), 1, torch.device("cpu")
    )
    prefill.inputs["gdn_conv_state_0"].fill_(4)
    prefill.inputs["gdn_recurrent_state_0"].fill_(6)
    prefill.inputs["kv_cache_3"][:, 0].fill_(8)
    bridge.commit(prefill)

    decode = bridge.prepare(
        _metadata(prefill=False, tokens=1, max_seq_len=2), 1, torch.device("cpu")
    )
    assert torch.equal(
        decode.inputs["gdn_conv_state_0"],
        torch.full_like(decode.inputs["gdn_conv_state_0"], 4),
    )
    assert torch.equal(
        decode.inputs["gdn_recurrent_state_0"],
        torch.full_like(decode.inputs["gdn_recurrent_state_0"], 6),
    )
    logical = decode.inputs["kv_cache_3"].reshape(2, 4096, 4, 256)
    assert torch.equal(logical[:, 0], torch.full_like(logical[:, 0], 8))
    assert torch.count_nonzero(logical[:, 1]) == 0


def test_hybrid_scheduler_page_translates_to_logical_blocks_and_preserves_tail():
    bridge, _, attention = _bridge()
    attention.kv_cache = torch.full(
        (2, 20, 32, 4, 256), 5, dtype=torch.bfloat16
    )

    prefill = bridge.prepare(
        _metadata(prefill=True, tokens=16, max_seq_len=16),
        16,
        torch.device("cpu"),
    )
    assert prefill.physical_block_count == 1
    assert prefill.physical_block_size == 32
    prefill.inputs["kv_cache_3"].reshape(2, 4096, 4, 256)[:, :16].fill_(8)
    bridge.commit(prefill)

    assert torch.equal(
        attention.kv_cache[:, 5, :16],
        torch.full_like(attention.kv_cache[:, 5, :16], 8),
    )
    assert torch.equal(
        attention.kv_cache[:, 5, 16:],
        torch.full_like(attention.kv_cache[:, 5, 16:], 5),
    )

    decode = bridge.prepare(
        _metadata(prefill=False, tokens=1, max_seq_len=17),
        1,
        torch.device("cpu"),
    )
    logical = decode.inputs["kv_cache_3"].reshape(2, 4096, 4, 256)
    assert torch.equal(logical[:, :16], torch.full_like(logical[:, :16], 8))
    assert torch.count_nonzero(logical[:, 16]) == 0
    logical[:, 16].fill_(9)
    bridge.commit(decode)

    assert torch.equal(
        attention.kv_cache[:, 5, :16],
        torch.full_like(attention.kv_cache[:, 5, :16], 8),
    )
    assert torch.equal(
        attention.kv_cache[:, 5, 16],
        torch.full_like(attention.kv_cache[:, 5, 16], 9),
    )
    assert torch.equal(
        attention.kv_cache[:, 5, 17:],
        torch.full_like(attention.kv_cache[:, 5, 17:], 5),
    )


def test_scheduler_int32_indices_are_normalized_for_torch_gather_and_scatter():
    bridge, _, _ = _bridge()
    prepared = bridge.prepare(
        _metadata(
            prefill=True,
            tokens=1,
            max_seq_len=1,
            index_dtype=torch.int32,
        ),
        1,
        torch.device("cpu"),
    )
    assert prepared.state_indices[0].dtype == torch.int64
    assert prepared.block_ids[3].dtype == torch.int64
    bridge.commit(prepared)


def test_hybrid_cache_groups_use_each_gdn_layers_scheduler_index():
    bridge, _, attention = _bridge()
    conv = torch.zeros((8, 3, 10240), dtype=torch.bfloat16)
    recurrent = torch.zeros((8, 48, 128, 128), dtype=torch.float32)
    bridge.gdn_layers = {
        0: SimpleNamespace(kv_cache=(conv, recurrent)),
        1: SimpleNamespace(kv_cache=(conv, recurrent)),
    }
    bridge.attention_layers = {3: attention}
    metadata = _metadata(prefill=True, tokens=1, max_seq_len=1)
    metadata["model.language_model.layers.1.linear_attn"] = _gdn_metadata(
        prefill=True, tokens=1, state_index=6
    )

    prefill = bridge.prepare(metadata, 1, torch.device("cpu"))
    prefill.inputs["gdn_conv_state_0"].fill_(1)
    prefill.inputs["gdn_recurrent_state_0"].fill_(2)
    prefill.inputs["gdn_conv_state_1"].fill_(3)
    prefill.inputs["gdn_recurrent_state_1"].fill_(4)
    bridge.commit(prefill)

    assert torch.equal(conv[2], torch.ones_like(conv[2]))
    assert torch.equal(conv[6], torch.full_like(conv[6], 3))
    assert torch.equal(recurrent[2], torch.full_like(recurrent[2], 2))
    assert torch.equal(recurrent[6], torch.full_like(recurrent[6], 4))

    decode_metadata = _metadata(prefill=False, tokens=1, max_seq_len=2)
    decode_metadata["model.language_model.layers.1.linear_attn"] = _gdn_metadata(
        prefill=False, tokens=1, state_index=6
    )
    decode = bridge.prepare(decode_metadata, 1, torch.device("cpu"))
    assert torch.equal(
        decode.inputs["gdn_conv_state_0"],
        torch.ones_like(decode.inputs["gdn_conv_state_0"]),
    )
    assert torch.equal(
        decode.inputs["gdn_conv_state_1"],
        torch.full_like(decode.inputs["gdn_conv_state_1"], 3),
    )


def test_gdn_layers_with_inconsistent_scheduler_phase_fail_closed():
    bridge, gdn, attention = _bridge()
    bridge.gdn_layers = {0: gdn, 1: gdn}
    bridge.attention_layers = {3: attention}
    metadata = _metadata(prefill=True, tokens=1, max_seq_len=1)
    metadata["model.language_model.layers.1.linear_attn"] = _gdn_metadata(
        prefill=False, tokens=1, state_index=6
    )
    with pytest.raises(Qwen38ContractError, match="phase differs"):
        bridge.prepare(metadata, 1, torch.device("cpu"))


def test_single_token_fresh_request_uses_scheduler_sequence_boundary():
    bridge, _, _ = _bridge()
    metadata = _metadata(prefill=False, tokens=1, max_seq_len=1)
    prepared = bridge.prepare(metadata, 1, torch.device("cpu"))
    assert prepared.fresh_prefill is True
    assert torch.count_nonzero(prepared.inputs["gdn_conv_state_0"]) == 0
    bridge.commit(prepared)

    decode = bridge.prepare(
        _metadata(prefill=False, tokens=1, max_seq_len=2),
        1,
        torch.device("cpu"),
    )
    assert decode.fresh_prefill is False


def test_single_token_decode_without_active_request_or_fresh_boundary_fails():
    bridge, _, _ = _bridge()
    with pytest.raises(Qwen38ContractError, match="no active"):
        bridge.prepare(
            _metadata(prefill=False, tokens=1, max_seq_len=2),
            1,
            torch.device("cpu"),
        )


def test_completed_request_replacement_and_long_decode_are_admitted():
    bridge, _, _ = _bridge()
    prepared = bridge.prepare(
        _metadata(prefill=True, tokens=1, max_seq_len=1), 1, torch.device("cpu")
    )
    bridge.commit(prepared)
    replacement = bridge.prepare(
        _metadata(prefill=True, tokens=128, max_seq_len=128),
        128,
        torch.device("cpu"),
    )
    bridge.commit(replacement)
    for step in range(12):
        decode = bridge.prepare(
            _metadata(prefill=False, tokens=1, max_seq_len=129 + step),
            1,
            torch.device("cpu"),
        )
        bridge.commit(decode)
    assert bridge._decode_steps == 12


def test_persistent_arena_owns_state_and_resets_only_on_fresh_request():
    bridge, gdn, attention = _bridge()
    arena = bridge.initialize_persistent_arena(torch.device("cpu"))
    assert bridge.persistent_ready
    assert set(arena) == {
        "gdn_conv_state_0",
        "gdn_recurrent_state_0",
        "kv_cache_3",
    }

    prefill = bridge.prepare(
        _metadata(prefill=True, tokens=128, max_seq_len=128),
        128,
        torch.device("cpu"),
    )
    assert prefill.persistent
    assert prefill.inputs is arena
    assert sorted(len(group) for group in bridge._persistent_reset_groups) == [1, 2]
    prefill.inputs["gdn_conv_state_0"].fill_(3)
    prefill.inputs["kv_cache_3"].fill_(4)
    bridge.commit(prefill)

    decode = bridge.prepare(
        _metadata(prefill=False, tokens=1, max_seq_len=129),
        1,
        torch.device("cpu"),
    )
    assert torch.count_nonzero(decode.inputs["gdn_conv_state_0"])
    assert torch.count_nonzero(decode.inputs["kv_cache_3"])
    bridge.commit(decode)

    replacement = bridge.prepare(
        _metadata(prefill=True, tokens=128, max_seq_len=128),
        128,
        torch.device("cpu"),
    )
    assert torch.count_nonzero(replacement.inputs["gdn_conv_state_0"]) == 0
    assert torch.count_nonzero(replacement.inputs["kv_cache_3"]) == 0
    assert torch.count_nonzero(gdn.kv_cache[0])
    assert torch.count_nonzero(attention.kv_cache)


def test_decode_sequence_discontinuity_fails_closed():
    bridge, _, _ = _bridge()
    prefill = bridge.prepare(
        _metadata(prefill=True, tokens=128, max_seq_len=128),
        128,
        torch.device("cpu"),
    )
    bridge.commit(prefill)
    with pytest.raises(Qwen38ContractError, match="does not continue"):
        bridge.prepare(
            _metadata(prefill=False, tokens=1, max_seq_len=130),
            1,
            torch.device("cpu"),
        )
