from __future__ import annotations

from types import SimpleNamespace
import weakref

import torch

import paiton_vllm_plugin.models.qwen38_greedy as greedy
from paiton_vllm_plugin.models.qwen38_greedy import (
    Qwen38GreedyOutputProvider,
    install_qwen38_greedy_sampler_hook,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
    ThinkingTokenBudgetLogitsProcessor,
)
from vllm.v1.sample.logits_processor.state import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler


def _builtin_processors() -> list[object]:
    min_tokens = MinTokensLogitsProcessor.__new__(MinTokensLogitsProcessor)
    min_tokens.min_toks = {}
    logit_bias = LogitBiasLogitsProcessor.__new__(LogitBiasLogitsProcessor)
    logit_bias.biases = {}
    thinking = ThinkingTokenBudgetLogitsProcessor.__new__(
        ThinkingTokenBudgetLogitsProcessor
    )
    thinking.is_enabled = False
    thinking._state = {}
    return [min_tokens, logit_bias, thinking]


def _metadata(processors: list[object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        max_num_logprobs=None,
        all_greedy=True,
        all_random=False,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        spec_token_ids=None,
        output_token_ids=[[]],
        logitsprocs=SimpleNamespace(
            non_argmax_invariant=(
                _builtin_processors() if processors is None else processors
            )
        ),
    )


def test_ordinary_greedy_metadata_is_eligible_and_sensitive_modes_fall_back(
    monkeypatch,
):
    provider = Qwen38GreedyOutputProvider.__new__(Qwen38GreedyOutputProvider)
    monkeypatch.setattr(
        Qwen38GreedyOutputProvider,
        "_logits_contract_failure",
        lambda self, logits, **kwargs: None,
    )
    logits = torch.zeros((1, 1), dtype=torch.bfloat16)
    metadata = _metadata()
    # This pinned runner may retain a populated metadata field even though its
    # non-speculative control flow called Sampler.forward directly.  The
    # predict_bonus_token argument is the relevant Sampler ABI gate.
    metadata.spec_token_ids = [[47]]
    metadata.output_token_ids = None
    assert (
        provider.eligibility_failure(
            logits,
            metadata,
            predict_bonus_token=False,
            logprobs_mode_override=None,
        )
        is None
    )

    metadata.max_num_logprobs = 1
    assert provider.eligibility_failure(
        logits, metadata, predict_bonus_token=False, logprobs_mode_override=None
    ) == "logprobs-requested"
    metadata.max_num_logprobs = None
    metadata.no_penalties = False
    assert provider.eligibility_failure(
        logits, metadata, predict_bonus_token=False, logprobs_mode_override=None
    ) == "active-penalties"


def test_active_and_unknown_non_argmax_processors_fail_closed():
    processors = _builtin_processors()
    processors[1].biases = {0: {47: 1.0}}
    assert Qwen38GreedyOutputProvider._processors_failure(
        _metadata(processors)
    ) == "active-logit-bias"
    assert Qwen38GreedyOutputProvider._processors_failure(
        _metadata([object()])
    ) == "custom-or-unknown-logits-processor"


def test_exact_hook_preserves_unregistered_stock_sampling_and_dispatches_registered():
    status = install_qwen38_greedy_sampler_hook()
    assert status["installed"] is True

    logits = torch.tensor([[1.0, 5.0, 5.0]], dtype=torch.bfloat16)
    metadata = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.empty(0),
        presence_penalties=torch.empty(0),
        repetition_penalties=torch.empty(0),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
        spec_token_ids=None,
    )
    ordinary = Sampler()(logits, metadata)
    assert ordinary.sampled_token_ids.tolist() == [[1]]

    sentinel = object()

    class FakeProvider:
        def eligibility_failure(self, *args, **kwargs):
            return None

        def sample(self, incoming):
            assert incoming is logits
            return sentinel

        def note_fallback(self, reason):
            raise AssertionError(reason)

    provider = FakeProvider()
    greedy._PROVIDERS_BY_DATA_PTR[logits.data_ptr()] = weakref.ref(provider)
    try:
        assert Sampler()(logits, metadata) is sentinel
    finally:
        greedy._PROVIDERS_BY_DATA_PTR.pop(logits.data_ptr(), None)
