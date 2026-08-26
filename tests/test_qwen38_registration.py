import pytest
import torch

from paiton_vllm_plugin import register_paiton_models
from paiton_vllm_plugin.models.qwen38_contract import Qwen38ContractError
from paiton_vllm_plugin.models.paiton_qwen38 import PaitonQwen38ForCausalLM


def test_qwen38_registration_is_explicit_and_does_not_replace_stock():
    from vllm import ModelRegistry

    register_paiton_models()
    supported = set(ModelRegistry.get_supported_archs())
    assert "PaitonQwen38ForCausalLM" in supported
    assert "Qwen3_5ForConditionalGeneration" in supported


def test_allocation_only_metadata_is_distinct_from_microbatch_metadata():
    classify = PaitonQwen38ForCausalLM._is_allocation_only_metadata
    assert classify(None)
    assert classify({})
    assert classify({"attention": None})
    assert not classify({"attention": object()})
    assert not classify([])


def test_text_only_mrope_positions_match_stock_text_semantics():
    model = PaitonQwen38ForCausalLM.__new__(PaitonQwen38ForCausalLM)
    positions, delta = model.get_mrope_input_positions([91, 92, 93, 94], [])
    expected = torch.arange(4, dtype=torch.int64).expand(3, -1)
    assert torch.equal(positions, expected)
    assert delta == 0
    with pytest.raises(Qwen38ContractError, match="vision features"):
        model.get_mrope_input_positions([91], [object()])


def test_text_position_assertion_covers_all_mrope_planes_and_sequence():
    expected = torch.arange(4, dtype=torch.int64)
    positions = expected.expand(3, -1).clone()
    PaitonQwen38ForCausalLM._assert_text_positions(positions, expected)

    wrong_plane = positions.clone()
    wrong_plane[2, 1] = 9
    with pytest.raises(RuntimeError):
        PaitonQwen38ForCausalLM._assert_text_positions(wrong_plane, expected)

    wrong_sequence = expected.clone()
    wrong_sequence[3] = 8
    with pytest.raises(RuntimeError):
        PaitonQwen38ForCausalLM._assert_text_positions(wrong_sequence, expected)


def _scheduled_logits_model(*, vocab_size=7):
    model = PaitonQwen38ForCausalLM.__new__(PaitonQwen38ForCausalLM)
    model._scheduled_enabled = True
    model._scheduled_pending_logits = None
    model._scheduled_allocation_only_logits_pending = False
    model.config = type("Config", (), {"vocab_size": vocab_size})()
    model.runtime_trace = type(
        "Trace", (), {"mark": lambda *args, **kwargs: None, "finish": lambda *args: None}
    )()
    model.logits_processor = lambda _lm_head, logits: logits
    model.greedy_output_provider = None
    return model


def test_scheduled_compute_logits_accepts_only_marked_allocation_profile():
    model = _scheduled_logits_model()
    profile_logits = torch.randn(3, 7)
    model._scheduled_allocation_only_logits_pending = True

    result = model.compute_logits(profile_logits)

    assert result is profile_logits
    assert not model._scheduled_allocation_only_logits_pending
    with pytest.raises(Qwen38ContractError, match="no pending logits"):
        model.compute_logits(profile_logits)


def test_scheduled_compute_logits_rejects_malformed_allocation_profile():
    model = _scheduled_logits_model()
    model._scheduled_allocation_only_logits_pending = True

    with pytest.raises(Qwen38ContractError, match="already contain logits"):
        model.compute_logits(torch.randn(3, 6))


def test_scheduled_compute_logits_consumes_real_result_not_placeholder():
    model = _scheduled_logits_model()
    real_logits = torch.randn(2, 7)
    model._scheduled_pending_logits = real_logits

    result = model.compute_logits(torch.full((8, 7), -1.0))

    assert result is real_logits
    assert model._scheduled_pending_logits is None
