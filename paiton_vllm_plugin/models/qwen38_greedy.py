# SPDX-License-Identifier: Apache-2.0
"""Fail-closed device-resident greedy sampling for the pinned Qwen3.8 lane."""

from __future__ import annotations

import ctypes
from functools import wraps
import hashlib
import inspect
import weakref
from typing import Any

import torch

from paiton_vllm_plugin.models.qwen38_contract import (
    GreedyOutputProviderRecord,
    Qwen38ContractError,
)
from paiton_vllm_plugin.models.runtime_compat import current_device_stream_ptr


VOCAB_SIZE = 248320
PINNED_VLLM_VERSION = "0.19.1.dev3+g72ed2b398.d20260513"
PINNED_SAMPLER_FORWARD_SHA256 = (
    "e41775e291fbbecfa87122e3b3b7ae0741a015d3841d8c785ac445a637ac96c4"
)

_PROVIDERS_BY_DATA_PTR: dict[
    int, weakref.ReferenceType[Qwen38GreedyOutputProvider]
] = {}
_PROVIDER_INSTANCES: weakref.WeakSet[Qwen38GreedyOutputProvider] = weakref.WeakSet()
_HOOK_STATUS: dict[str, object] = {
    "installed": False,
    "reason": "not-requested",
    "vllm_version": None,
    "sampler_forward_sha256": None,
}


def _source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def _lookup_provider(logits: torch.Tensor) -> Qwen38GreedyOutputProvider | None:
    reference = _PROVIDERS_BY_DATA_PTR.get(logits.data_ptr())
    if reference is None:
        return None
    provider = reference()
    if provider is None:
        _PROVIDERS_BY_DATA_PTR.pop(logits.data_ptr(), None)
        return None
    return provider


def install_qwen38_greedy_sampler_hook() -> dict[str, object]:
    """Install one exact-version sampler dispatch; mismatches retain vLLM."""

    from vllm import __version__ as vllm_version
    from vllm.v1.sample.sampler import Sampler

    if getattr(Sampler, "_paiton_qwen38_greedy_hook_v1", False):
        return dict(_HOOK_STATUS)
    original = Sampler.forward
    source_sha = _source_sha256(original)
    _HOOK_STATUS.update(
        {
            "vllm_version": vllm_version,
            "sampler_forward_sha256": source_sha,
        }
    )
    if vllm_version != PINNED_VLLM_VERSION:
        _HOOK_STATUS.update({"installed": False, "reason": "vllm-version-mismatch"})
        return dict(_HOOK_STATUS)
    if source_sha != PINNED_SAMPLER_FORWARD_SHA256:
        _HOOK_STATUS.update(
            {"installed": False, "reason": "sampler-forward-source-mismatch"}
        )
        return dict(_HOOK_STATUS)

    @wraps(original)
    def forward(
        self: Any,
        logits: torch.Tensor,
        sampling_metadata: Any,
        predict_bonus_token: bool = False,
        logprobs_mode_override: str | None = None,
    ) -> Any:
        provider = _lookup_provider(logits)
        if provider is not None:
            failure = provider.eligibility_failure(
                logits,
                sampling_metadata,
                predict_bonus_token=predict_bonus_token,
                logprobs_mode_override=logprobs_mode_override,
            )
            if failure is None:
                return provider.sample(logits)
            provider.note_fallback(failure)
        return original(
            self,
            logits,
            sampling_metadata,
            predict_bonus_token=predict_bonus_token,
            logprobs_mode_override=logprobs_mode_override,
        )

    Sampler.forward = forward
    Sampler._paiton_qwen38_greedy_hook_v1 = True
    _HOOK_STATUS.update({"installed": True, "reason": "exact-abi-match"})
    return dict(_HOOK_STATUS)


class Qwen38GreedyOutputProvider:
    """One-launch BF16 argmax for one manifest-identified logits buffer."""

    def __init__(self, record: GreedyOutputProviderRecord) -> None:
        if record.vllm_version != PINNED_VLLM_VERSION:
            raise Qwen38ContractError(
                "greedy output provider manifest vLLM identity differs"
            )
        if record.sampler_forward_sha256 != PINNED_SAMPLER_FORWARD_SHA256:
            raise Qwen38ContractError(
                "greedy output provider manifest sampler ABI differs"
            )
        hook_status = install_qwen38_greedy_sampler_hook()
        if hook_status.get("installed") is not True:
            raise Qwen38ContractError(
                "greedy output provider cannot install exact sampler ABI: "
                f"{hook_status.get('reason')}"
            )
        self.record = record
        self._library = ctypes.CDLL(str(record.path))
        abi_version = self._library.paiton_qwen38_greedy_argmax_abi_version
        abi_version.argtypes = []
        abi_version.restype = ctypes.c_uint32
        vocab_size = self._library.paiton_qwen38_greedy_argmax_vocab_size
        vocab_size.argtypes = []
        vocab_size.restype = ctypes.c_uint32
        workspace_bytes = self._library.paiton_qwen38_greedy_argmax_workspace_bytes
        workspace_bytes.argtypes = []
        workspace_bytes.restype = ctypes.c_uint64
        if int(abi_version()) != record.abi_version:
            raise Qwen38ContractError("greedy output provider ABI export differs")
        if int(vocab_size()) != VOCAB_SIZE:
            raise Qwen38ContractError("greedy output provider vocabulary export differs")
        if int(workspace_bytes()) != record.workspace_bytes:
            raise Qwen38ContractError("greedy output provider workspace export differs")
        self._invoke = getattr(self._library, record.symbol)
        self._invoke.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._invoke.restype = ctypes.c_int
        self._sampled_token_ids: torch.Tensor | None = None
        self._workspace: torch.Tensor | None = None
        self._device: torch.device | None = None
        self._registered_data_ptr: int | None = None
        self._registered_storage_ptr: int | None = None
        self._self_reference = weakref.ref(self)
        self.activation: dict[str, object] = {
            "id": record.id,
            "path": str(record.path),
            "sha256": record.sha256,
            "abi_version": record.abi_version,
            "workspace_bytes": record.workspace_bytes,
            "hook": hook_status,
            "initialized": False,
            "registered_logits_calls": 0,
            "fast_path_calls": 0,
            "fallback_calls": 0,
            "fallback_reasons": {},
        }
        _PROVIDER_INSTANCES.add(self)

    def initialize(self, device: torch.device | str) -> None:
        device = torch.device(device)
        if device.type != "cuda":
            raise Qwen38ContractError(
                f"greedy output provider requires HIP, got {device}"
            )
        active_index = torch.cuda.current_device()
        expected_index = active_index if device.index is None else device.index
        if expected_index != active_index:
            raise Qwen38ContractError(
                f"greedy output provider device {device} is not active cuda:{active_index}"
            )
        properties = torch.cuda.get_device_properties(active_index)
        architecture = str(getattr(properties, "gcnArchName", ""))
        if not architecture.startswith("gfx950"):
            raise Qwen38ContractError(
                f"greedy output provider requires gfx950, got {architecture!r}"
            )
        canonical_device = torch.device("cuda", active_index)
        if self._device is not None:
            if self._device != canonical_device:
                raise Qwen38ContractError("greedy output provider device changed")
            return
        self._sampled_token_ids = torch.empty(
            (1, 1), dtype=torch.int32, device=canonical_device
        )
        self._workspace = torch.zeros(
            self.record.workspace_bytes,
            dtype=torch.uint8,
            device=canonical_device,
        )
        self._device = canonical_device
        self.activation.update(
            {
                "initialized": True,
                "device": str(canonical_device),
                "architecture": architecture,
                "sampled_token_ids_data_ptr": self._sampled_token_ids.data_ptr(),
                "workspace_data_ptr": self._workspace.data_ptr(),
            }
        )

    def register_logits(self, logits: torch.Tensor) -> None:
        if self._device is None:
            self.initialize(logits.device)
        failure = self._logits_contract_failure(logits, require_registration=False)
        if failure is not None:
            raise Qwen38ContractError(
                f"cannot register greedy output logits: {failure}"
            )
        old_pointer = self._registered_data_ptr
        if old_pointer is not None and old_pointer != logits.data_ptr():
            old_reference = _PROVIDERS_BY_DATA_PTR.get(old_pointer)
            if old_reference is not None and old_reference() is self:
                _PROVIDERS_BY_DATA_PTR.pop(old_pointer, None)
        self._registered_data_ptr = logits.data_ptr()
        self._registered_storage_ptr = logits.untyped_storage().data_ptr()
        _PROVIDERS_BY_DATA_PTR[self._registered_data_ptr] = self._self_reference
        self.activation["registered_logits_calls"] = (
            int(self.activation["registered_logits_calls"]) + 1
        )
        self.activation["registered_logits_data_ptr"] = self._registered_data_ptr
        self.activation["registered_logits_storage_ptr"] = (
            self._registered_storage_ptr
        )

    def _logits_contract_failure(
        self, logits: torch.Tensor, *, require_registration: bool = True
    ) -> str | None:
        if self._device is None:
            return "provider-not-initialized"
        if logits.device != self._device:
            return "logits-device"
        if logits.dtype is not torch.bfloat16:
            return "logits-dtype"
        if tuple(logits.shape) != (1, VOCAB_SIZE):
            return "logits-shape"
        if tuple(logits.stride()) != (VOCAB_SIZE, 1):
            return "logits-stride"
        if logits.storage_offset() != 0:
            return "logits-storage-offset"
        if logits.requires_grad:
            return "logits-requires-grad"
        if require_registration:
            if logits.data_ptr() != self._registered_data_ptr:
                return "logits-data-pointer"
            if logits.untyped_storage().data_ptr() != self._registered_storage_ptr:
                return "logits-storage-pointer"
        return None

    @staticmethod
    def _processors_failure(sampling_metadata: Any) -> str | None:
        from vllm.v1.sample.logits_processor.builtin import (
            LogitBiasLogitsProcessor,
            MinTokensLogitsProcessor,
            ThinkingTokenBudgetLogitsProcessor,
        )

        processors = getattr(sampling_metadata, "logitsprocs", None)
        non_invariant = getattr(processors, "non_argmax_invariant", None)
        if not isinstance(non_invariant, list):
            return "logits-processors-contract"
        for processor in non_invariant:
            if type(processor) is LogitBiasLogitsProcessor:
                if processor.biases:
                    return "active-logit-bias"
            elif type(processor) is MinTokensLogitsProcessor:
                if processor.min_toks:
                    return "active-min-tokens"
            elif type(processor) is ThinkingTokenBudgetLogitsProcessor:
                if processor.is_enabled and processor._state:
                    return "active-thinking-budget"
            else:
                return "custom-or-unknown-logits-processor"
        return None

    def eligibility_failure(
        self,
        logits: torch.Tensor,
        sampling_metadata: Any,
        *,
        predict_bonus_token: bool,
        logprobs_mode_override: str | None,
    ) -> str | None:
        failure = self._logits_contract_failure(logits)
        if failure is not None:
            return failure
        if predict_bonus_token:
            return "predict-bonus-token"
        if logprobs_mode_override is not None:
            return "logprobs-mode-override"
        if getattr(sampling_metadata, "max_num_logprobs", object()) is not None:
            return "logprobs-requested"
        if getattr(sampling_metadata, "all_greedy", False) is not True:
            return "not-all-greedy"
        if getattr(sampling_metadata, "all_random", True) is not False:
            return "random-sampling"
        if getattr(sampling_metadata, "no_penalties", False) is not True:
            return "active-penalties"
        if getattr(sampling_metadata, "allowed_token_ids_mask", object()) is not None:
            return "allowed-token-mask"
        if bool(getattr(sampling_metadata, "bad_words_token_ids", None)):
            return "bad-words"
        return self._processors_failure(sampling_metadata)

    def note_fallback(self, reason: str) -> None:
        self.activation["fallback_calls"] = int(self.activation["fallback_calls"]) + 1
        reasons = self.activation["fallback_reasons"]
        assert isinstance(reasons, dict)
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    def sample(self, logits: torch.Tensor) -> Any:
        from vllm.v1.outputs import SamplerOutput

        assert self._sampled_token_ids is not None
        assert self._workspace is not None
        stream_ptr = current_device_stream_ptr(logits.device)
        status = self._invoke(
            ctypes.c_void_p(logits.data_ptr()),
            ctypes.c_void_p(self._sampled_token_ids.data_ptr()),
            ctypes.c_void_p(self._workspace.data_ptr()),
            ctypes.c_uint64(self._workspace.numel()),
            ctypes.c_void_p(stream_ptr),
        )
        if status != 0:
            raise Qwen38ContractError(
                f"greedy output provider launch failed with HIP error {status}"
            )
        self.activation["fast_path_calls"] = int(self.activation["fast_path_calls"]) + 1
        return SamplerOutput(
            sampled_token_ids=self._sampled_token_ids,
            logprobs_tensors=None,
        )


def qwen38_greedy_output_snapshot() -> dict[str, object]:
    return {
        "hook": dict(_HOOK_STATUS),
        "providers": [dict(provider.activation) for provider in _PROVIDER_INSTANCES],
    }


__all__ = [
    "Qwen38GreedyOutputProvider",
    "install_qwen38_greedy_sampler_hook",
    "qwen38_greedy_output_snapshot",
]
