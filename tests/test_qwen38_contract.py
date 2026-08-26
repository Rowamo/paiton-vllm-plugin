from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from paiton_vllm_plugin.models.qwen38_contract import (
    COMPILER_COMMIT,
    COMPILER_TREE,
    MANIFEST_SHA256_ENV,
    Qwen38ContractError,
    load_qwen38_artifact_family,
)
from paiton_vllm_plugin.models.paiton_qwen38 import PaitonQwen38ForCausalLM


PRIOR_ENV = "PAITON_QWEN38_TEST_EVIDENCE_ROOT"
CHECKPOINT_ROOT_ENV = "PAITON_QWEN38_TEST_CHECKPOINT_ROOT"


def _external_test_root(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to run Qwen3.8 artifact-contract tests")
    return Path(value).resolve()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding_sha(plan: dict[str, object]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    prior = _external_test_root(PRIOR_ENV)
    checkpoint_root = _external_test_root(CHECKPOINT_ROOT_ENV)
    plugin_root = Path(__file__).parents[1]
    artifacts = []
    for tokens in (1, 128, 512, 2048):
        report = json.loads(
            (
                prior
                / "evidence"
                / f"full-decoder-compile-m{tokens}-v1.json"
            ).read_text()
        )
        artifacts.append(
            {
                "tokens": tokens,
                "path": report["shared_object"],
                "sha256": report["shared_object_sha256"],
                "size": report["shared_object_size"],
                "input_count": report["input_count"],
                "state_count": report["state_count"],
                "output_shape": report["output_shape"],
                "states": report["states"],
            }
        )
    mapping = prior / "evidence/full-checkpoint-mapping-v1.json"
    sources = [
        plugin_root / "paiton_vllm_plugin/models/qwen38_contract.py",
        plugin_root / "paiton_vllm_plugin/models/qwen38_state.py",
        plugin_root / "paiton_vllm_plugin/models/paiton_qwen38.py",
    ]
    payload = {
        "schema": "paiton.qwen38.fixed-m-artifact-family.v1",
        "target": {
            "repository": "amd/Qwen3.8-27B-Quark-AWQ-MXFP4",
            "revision": "156be69f9cac862a41d8b32e773ea2d2754341e8",
            "checkpoint_path": str(checkpoint_root / "model.safetensors"),
            "checkpoint_sha256": "be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272",
            "checkpoint_size": 19798196184,
            "config_path": str(checkpoint_root / "config.json"),
        },
        "compiler": {
            "commit": COMPILER_COMMIT,
            "tree": COMPILER_TREE,
        },
        "plugin": {
            "base_commit": "3cbb75d0e1f4ce95ca00d257342e82735fdeafa2",
            "base_tree": "f1ec0ca7a284baf63d0730125e5f33162b95b723",
            "commit": "0" * 40,
            "tree": "1" * 40,
            "source_files": [
                {"path": str(source), "sha256": _sha(source)} for source in sources
            ],
        },
        "weight_mapping": {"path": str(mapping), "sha256": _sha(mapping)},
        "artifacts": artifacts,
    }
    path = tmp_path / "artifact-family.json"
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    return path


def test_exact_fixed_m_family_loads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = _manifest(tmp_path)
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    family = load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)
    assert tuple(sorted(family.artifacts)) == (1, 128, 512, 2048)
    assert len(family.required_records) == 1347
    assert len(family.deferred_records) == 348
    assert len(family.state_records) == 112


def test_manifest_can_freeze_exactly_once_derived_constant_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    plan: dict[str, object] = {
        "schema": "paiton.qwen38.artifact-constant-materialization.v1",
        "required_runtime_constant_count": 1346,
        "direct_runtime_constant_count": 1345,
        "derived_runtime_constant_count": 1,
        "logical_checkpoint_source_count": 1347,
        "logical_source_coverage": "exactly-once",
        "derived_constants": [
            {
                "runtime_name": "model_language_model_layers_0_linear_attn_a_log_dt_bias_fused",
                "source_names": [
                    "model.language_model.layers.0.linear_attn.A_log",
                    "model.language_model.layers.0.linear_attn.dt_bias",
                ],
                "concatenate_dimension": 0,
                "dtype": "BF16",
                "shape": [96],
                "provider_id": "paiton-test-concatenation-v1",
            }
        ],
    }
    plan["binding_plan_sha256"] = _binding_sha(plan)
    payload["artifacts"][-1]["constant_materialization"] = plan
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    family = load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)
    artifact = family.artifacts[2048]
    assert len(artifact.physical_runtime_names) == 1346
    assert len(artifact.derived_constants) == 1
    assert artifact.binding_plan_sha256 == plan["binding_plan_sha256"]


def test_derived_constant_source_reuse_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    derived = {
        "runtime_name": "model_language_model_layers_0_linear_attn_a_log_dt_bias_fused",
        "source_names": [
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.linear_attn.dt_bias",
        ],
        "concatenate_dimension": 0,
        "dtype": "BF16",
        "shape": [96],
        "provider_id": "paiton-test-concatenation-v1",
    }
    second = dict(derived)
    second["runtime_name"] = "model_language_model_layers_0_linear_attn_reused"
    plan: dict[str, object] = {
        "schema": "paiton.qwen38.artifact-constant-materialization.v1",
        "required_runtime_constant_count": 1347,
        "direct_runtime_constant_count": 1345,
        "derived_runtime_constant_count": 2,
        "logical_checkpoint_source_count": 1347,
        "logical_source_coverage": "exactly-once",
        "derived_constants": [derived, second],
    }
    plan["binding_plan_sha256"] = _binding_sha(plan)
    payload["artifacts"][-1]["constant_materialization"] = plan
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    with pytest.raises(Qwen38ContractError, match="not singleton"):
        load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)


def test_manifest_hash_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    path = _manifest(tmp_path)
    monkeypatch.delenv(MANIFEST_SHA256_ENV, raising=False)
    with pytest.raises(Qwen38ContractError, match=MANIFEST_SHA256_ENV):
        load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)


def test_artifact_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _manifest(tmp_path)
    payload = json.loads(path.read_text())
    payload["artifacts"][0]["sha256"] = "f" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    with pytest.raises(Qwen38ContractError, match="artifact SHA256 M=1"):
        load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)


def test_unsupported_m_never_buckets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _manifest(tmp_path)
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    family = load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)
    with pytest.raises(Qwen38ContractError, match="no padding or redirection"):
        family.artifact_for_tokens(127)


def test_optional_greedy_output_provider_is_byte_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path = _manifest(tmp_path)
    provider_path = tmp_path / "greedy-output.so"
    provider_path.write_bytes(b"bounded-gfx950-provider")
    payload = json.loads(path.read_text())
    payload["greedy_output_provider"] = {
        "id": "paiton-qwen38-bf16-greedy-argmax-gfx950-v1",
        "target": "gfx950",
        "path": str(provider_path),
        "sha256": _sha(provider_path),
        "size": provider_path.stat().st_size,
        "input_dtype": "bfloat16",
        "input_shape": [1, 248320],
        "output_dtype": "int32",
        "output_shape": [1, 1],
        "abi_version": 1,
        "symbol": "paiton_qwen38_greedy_argmax_bf16_v1",
        "workspace_bytes": 2064,
        "launches": 1,
        "runtime_jit_or_tuning": False,
        "vllm_version": "0.19.1.dev3+g72ed2b398.d20260513",
        "sampler_forward_sha256": (
            "e41775e291fbbecfa87122e3b3b7ae0741a015d3841d8c785ac445a637ac96c4"
        ),
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    family = load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)
    assert family.greedy_output_provider is not None
    assert family.greedy_output_provider.path == provider_path.resolve()

    payload["greedy_output_provider"]["sha256"] = "f" * 64
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(MANIFEST_SHA256_ENV, _sha(path))
    with pytest.raises(Qwen38ContractError, match="provider SHA256"):
        load_qwen38_artifact_family(path, verify_checkpoint_bytes=False)


def _serving_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1,
            max_num_batched_tokens=4096,
            enable_chunked_prefill=False,
        ),
        cache_config=SimpleNamespace(
            block_size=784,
            num_gpu_blocks_override=None,
            cache_dtype="auto",
            mamba_cache_mode="none",
            mamba_block_size=4096,
            enable_prefix_caching=False,
        ),
        model_config=SimpleNamespace(max_model_len=4096),
        speculative_config=None,
        lora_config=None,
    )


def test_fixed_m_serving_config_uses_one_scheduler_state_block_without_chunking():
    PaitonQwen38ForCausalLM._validate_serving_config(_serving_config())


def test_scheduled_serving_config_admits_the_bounded_token_factory_envelope():
    config = _serving_config()
    config.scheduler_config.max_num_seqs = 128
    config.scheduler_config.enable_chunked_prefill = True
    config.cache_config.block_size = 784
    config.cache_config.num_gpu_blocks_override = 1153
    PaitonQwen38ForCausalLM._validate_serving_config(config, scheduled=True)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("scheduler_config", "max_num_seqs", 127, "max_num_seqs"),
        ("scheduler_config", "max_num_batched_tokens", 2048, "exactly 4096"),
        ("scheduler_config", "enable_chunked_prefill", False, "chunked prefill"),
        ("cache_config", "block_size", 16, "exactly 784"),
        ("cache_config", "num_gpu_blocks_override", None, "1153"),
    ],
)
def test_scheduled_serving_config_fails_outside_bounded_envelope(
    section: str, field: str, value: object, message: str
):
    config = _serving_config()
    config.scheduler_config.max_num_seqs = 128
    config.scheduler_config.enable_chunked_prefill = True
    config.cache_config.block_size = 784
    config.cache_config.num_gpu_blocks_override = 1153
    setattr(getattr(config, section), field, value)
    with pytest.raises(Qwen38ContractError, match=message):
        PaitonQwen38ForCausalLM._validate_serving_config(config, scheduled=True)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("scheduler_config", "enable_chunked_prefill", True, "chunked prefill"),
        ("cache_config", "mamba_cache_mode", "align", "mamba_cache_mode"),
        ("cache_config", "enable_prefix_caching", True, "prefix caching"),
        ("cache_config", "mamba_block_size", 784, "mamba_block_size"),
    ],
)
def test_chunked_or_multiblock_mamba_serving_modes_fail_closed(
    section: str, field: str, value: object, message: str
):
    config = _serving_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(Qwen38ContractError, match=message):
        PaitonQwen38ForCausalLM._validate_serving_config(config)
