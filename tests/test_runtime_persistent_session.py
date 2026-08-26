from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from paiton_vllm_plugin.runtime.core import model as runtime_model


class _RecordingLoader:
    def __init__(self, *, bind: bool = True, run: bool = True) -> None:
        symbols = {}
        if bind:
            symbols["PaitonModelContainerBindInputs"] = object()
        if run:
            symbols["PaitonModelContainerRunBound"] = object()
        self.lib = SimpleNamespace(**symbols)
        self.bind_calls: list[tuple[object, ...]] = []
        self.run_calls: list[tuple[object, ...]] = []

    def PaitonModelContainerBindInputs(self, *args: object) -> None:
        self.bind_calls.append(args)

    def PaitonModelContainerRunBound(self, *args: object) -> None:
        self.run_calls.append(args)


class _FakeModel:
    def __init__(self, loader: _RecordingLoader) -> None:
        self.memloader = loader
        self.handle = object()
        self._input_name_to_index = {"input_ids": 0}
        self._output_name_to_index = {"logits": 0}
        self._output_ndims = [2]

    def _dict_to_ordered_list(self, values, *, is_inputs: bool):
        mapping = (
            self._input_name_to_index if is_inputs else self._output_name_to_index
        )
        result = [None] * len(mapping)
        for name, value in values.items():
            result[mapping[name]] = value
        return result

    @staticmethod
    def _convert_params_to_c_format(values):
        return tuple(values)


def _tensors() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {"input_ids": torch.zeros(1, dtype=torch.int64)},
        {"logits": torch.zeros((1, 4), dtype=torch.bfloat16)},
    )


def test_missing_persistent_abi_fails_closed_before_binding():
    inputs, outputs = _tensors()
    model = _FakeModel(_RecordingLoader(run=False))
    with pytest.raises(RuntimeError, match="PaitonModelContainerRunBound"):
        runtime_model.PersistentTensorSession(model, inputs, outputs)


def test_persistent_session_binds_once_and_reuses_stream_objects(monkeypatch):
    monkeypatch.setattr(
        runtime_model,
        "_check_tensors_contiguous_and_on_gpu",
        lambda *args, **kwargs: None,
    )
    inputs, outputs = _tensors()
    loader = _RecordingLoader()
    model = _FakeModel(loader)
    session = runtime_model.PersistentTensorSession(
        model,
        inputs,
        outputs,
        static_output_shapes={"logits": (1, 4)},
    )

    first = session.run(stream_ptr=17, sync=False, graph_mode=True)
    second = session.run(stream_ptr=17, sync=True, graph_mode=False)
    default = session.run(stream_ptr=None, sync=False, graph_mode=True)

    assert len(loader.bind_calls) == 1
    assert len(loader.run_calls) == 3
    assert first["logits"] is outputs["logits"]
    assert second["logits"] is outputs["logits"]
    assert default["logits"] is outputs["logits"]
    assert loader.run_calls[0][3] is loader.run_calls[1][3]
    assert loader.run_calls[0][3].value == 17
    assert loader.run_calls[2][3].value is None
    assert loader.run_calls[0][4].value is False
    assert loader.run_calls[0][5].value is True
    assert loader.run_calls[1][4].value is True
    assert loader.run_calls[1][5].value is False


def test_static_output_contract_requires_exact_names_and_shapes(monkeypatch):
    monkeypatch.setattr(
        runtime_model,
        "_check_tensors_contiguous_and_on_gpu",
        lambda *args, **kwargs: None,
    )
    inputs, outputs = _tensors()
    model = _FakeModel(_RecordingLoader())
    with pytest.raises(ValueError, match="names must exactly match"):
        runtime_model.PersistentTensorSession(
            model,
            inputs,
            outputs,
            static_output_shapes={"other": (1, 4)},
        )
    with pytest.raises(ValueError, match=r"expected \(1, 5\)"):
        runtime_model.PersistentTensorSession(
            model,
            inputs,
            outputs,
            static_output_shapes={"logits": (1, 5)},
        )
