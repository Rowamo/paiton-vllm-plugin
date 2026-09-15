import types

from paiton_vllm_plugin.runtime.core.model import (
    PaitonModelCapability,
    _query_model_capabilities,
)


class _CapabilityLoader:
    def __init__(self, value=None):
        self.lib = types.SimpleNamespace()
        if value is not None:
            self.lib.PaitonModelContainerGetCapabilities = object()
            self._value = value

    def PaitonModelContainerGetCapabilities(self, _handle, capabilities_out):
        capabilities_out._obj.value = self._value


def test_legacy_artifact_without_capability_symbol_defaults_to_none():
    capabilities = _query_model_capabilities(_CapabilityLoader(), None)
    assert capabilities == PaitonModelCapability.NONE


def test_artifact_capability_symbol_is_authoritative():
    capabilities = _query_model_capabilities(_CapabilityLoader(1), None)
    assert capabilities == (
        PaitonModelCapability.LOGITS_REPLICATED_ON_ALL_TP_RANKS
    )
