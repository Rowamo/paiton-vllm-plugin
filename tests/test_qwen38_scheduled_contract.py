from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from paiton_vllm_plugin.models.paiton_qwen38 import (
    _materialize_artifact_constant_maps,
)
from paiton_vllm_plugin.models.qwen38_contract import (
    Qwen38ContractError,
    TARGET_CHECKPOINT_SHA256,
    TARGET_REPOSITORY,
    TARGET_REVISION,
    TensorRecord,
)
from paiton_vllm_plugin.models.qwen38_schedule import (
    DEFAULT_PROVIDER_ROUTE,
    M128_PAITON_GENERIC_ROUTE,
    M128_ROUTE_PREDICATE_SHA256,
    M128_STOCK_EXACT_DECODE_ROUTE,
    TOKEN_BUCKETS,
)
from paiton_vllm_plugin.models.qwen38_scheduled_contract import (
    BASE_ARTIFACT_MANIFEST_SHA256,
    SCHEDULED_MANIFEST_SHA256_ENV,
    SCHEDULED_PROVIDER_ROUTE_CONTRACT,
    STARTING_COMPILER_COMMIT,
    STARTING_PLUGIN_COMMIT,
    load_qwen38_scheduled_artifact_family,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding_sha(plan: dict[str, object]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _required_records() -> dict[str, TensorRecord]:
    return {
        "left": TensorRecord(
            checkpoint_name="left",
            runtime_name="left_runtime",
            category="text",
            disposition="required",
            dtype="BF16",
            shape=(2,),
            sha256="0" * 64,
        ),
        "right": TensorRecord(
            checkpoint_name="right",
            runtime_name="right_runtime",
            category="text",
            disposition="required",
            dtype="BF16",
            shape=(2,),
            sha256="1" * 64,
        ),
        "other": TensorRecord(
            checkpoint_name="other",
            runtime_name="other_runtime",
            category="text",
            disposition="required",
            dtype="BF16",
            shape=(3,),
            sha256="2" * 64,
        ),
    }


def _derived_plan() -> dict[str, object]:
    plan: dict[str, object] = {
        "schema": "paiton.qwen38.artifact-constant-materialization.v1",
        "required_runtime_constant_count": 2,
        "direct_runtime_constant_count": 1,
        "derived_runtime_constant_count": 1,
        "logical_checkpoint_source_count": 3,
        "logical_source_coverage": "exactly-once",
        "derived_constants": [
            {
                "runtime_name": "left_right_fused",
                "source_names": ["left", "right"],
                "concatenate_dimension": 0,
                "dtype": "BF16",
                "shape": [4],
                "provider_id": "paiton-test-concatenation-v1",
            }
        ],
    }
    plan["binding_plan_sha256"] = _binding_sha(plan)
    return plan


def _manifest(tmp_path: Path) -> tuple[Path, SimpleNamespace]:
    abi = tmp_path / "abi.json"
    abi.write_text("{}\n")
    source = tmp_path / "source.py"
    source.write_text("# source identity\n")
    artifacts = []
    for tokens in TOKEN_BUCKETS:
        routes = (
            (M128_STOCK_EXACT_DECODE_ROUTE, M128_PAITON_GENERIC_ROUTE)
            if tokens == 128
            else (DEFAULT_PROVIDER_ROUTE,)
        )
        for route in routes:
            artifact = tmp_path / f"m{tokens}-{route}.so"
            artifact.write_bytes(f"artifact-{tokens}-{route}".encode())
            record: dict[str, object] = {
                "tokens": tokens,
                "provider_route": route,
                "provider_id": None,
                "provider_route_predicate_sha256": None,
                "path": str(artifact),
                "sha256": _sha(artifact),
                "size": artifact.stat().st_size,
                "input_count": 172,
                "state_count": 112,
                "output_shape": [min(tokens, 128), 248320],
                "max_active_sequences": 128,
                "max_scheduled_tokens": 4096,
                "runtime_jit_or_tuning": False,
                "stock_model_fallback": False,
            }
            if tokens == 128:
                record.update(SCHEDULED_PROVIDER_ROUTE_CONTRACT["routes"][route])
                record["provider_route_predicate_sha256"] = (
                    M128_ROUTE_PREDICATE_SHA256
                )
                record["constant_materialization"] = _derived_plan()
            artifacts.append(record)
    payload = {
        "schema": "paiton.qwen38.scheduled-artifact-family.v2",
        "provider_route_contract": SCHEDULED_PROVIDER_ROUTE_CONTRACT,
        "base_artifact_manifest_sha256": BASE_ARTIFACT_MANIFEST_SHA256,
        "target": {
            "repository": TARGET_REPOSITORY,
            "revision": TARGET_REVISION,
            "checkpoint_sha256": TARGET_CHECKPOINT_SHA256,
        },
        "compiler": {
            "parent_commit": STARTING_COMPILER_COMMIT,
            "commit": "3" * 40,
            "tree": "4" * 40,
        },
        "plugin": {
            "parent_commit": STARTING_PLUGIN_COMMIT,
            "commit": "5" * 40,
            "tree": "6" * 40,
            "source_files": [{"path": str(source), "sha256": _sha(source)}],
        },
        "abi": {"path": str(abi), "sha256": _sha(abi)},
        "artifacts": artifacts,
    }
    manifest = tmp_path / "scheduled.json"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    base_family = SimpleNamespace(required_records=_required_records())
    return manifest, base_family


def test_scheduled_family_preserves_per_bucket_constant_abis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest, base_family = _manifest(tmp_path)
    monkeypatch.setenv(SCHEDULED_MANIFEST_SHA256_ENV, _sha(manifest))
    family = load_qwen38_scheduled_artifact_family(base_family, manifest)

    assert family.artifacts[(1, DEFAULT_PROVIDER_ROUTE)].physical_runtime_names == {
        "left_runtime",
        "right_runtime",
        "other_runtime",
    }
    assert family.artifacts[
        (128, M128_PAITON_GENERIC_ROUTE)
    ].physical_runtime_names == {
        "left_right_fused",
        "other_runtime",
    }
    assert len(
        family.artifacts[(128, M128_STOCK_EXACT_DECODE_ROUTE)].derived_constants
    ) == 1


def test_scheduled_per_bucket_constant_maps_materialize_exact_physical_abis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest, base_family = _manifest(tmp_path)
    monkeypatch.setenv(SCHEDULED_MANIFEST_SHA256_ENV, _sha(manifest))
    scheduled = load_qwen38_scheduled_artifact_family(base_family, manifest)
    logical = {
        "left": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "right": torch.tensor([3.0, 4.0], dtype=torch.bfloat16),
        "other": torch.tensor([5.0, 6.0, 7.0], dtype=torch.bfloat16),
    }

    runtime_maps, plans = _materialize_artifact_constant_maps(
        base_family,
        logical,
        torch.device("cpu"),
        artifacts=scheduled.artifacts,
    )

    assert set(runtime_maps[(1, DEFAULT_PROVIDER_ROUTE)]) == scheduled.artifacts[
        (1, DEFAULT_PROVIDER_ROUTE)
    ].physical_runtime_names
    assert set(runtime_maps[(128, M128_PAITON_GENERIC_ROUTE)]) == scheduled.artifacts[
        (128, M128_PAITON_GENERIC_ROUTE)
    ].physical_runtime_names
    assert torch.equal(
        runtime_maps[(128, M128_PAITON_GENERIC_ROUTE)]["left_right_fused"],
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16),
    )
    assert next(
        plan
        for plan in plans
        if plan["tokens"] == 128
        and plan["provider_route"] == M128_PAITON_GENERIC_ROUTE
    )[
        "derived_runtime_constant_count"
    ] == 1


def test_scheduled_bucket_plan_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest, base_family = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"][7]["constant_materialization"][
        "direct_runtime_constant_count"
    ] = 2
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(SCHEDULED_MANIFEST_SHA256_ENV, _sha(manifest))

    with pytest.raises(Qwen38ContractError, match="direct_runtime_constant_count"):
        load_qwen38_scheduled_artifact_family(base_family, manifest)


def test_scheduled_route_predicate_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest, base_family = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["provider_route_contract"]["predicate"]["stock_conditions"][
        "actual_sequence_count"
    ] = 127
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(SCHEDULED_MANIFEST_SHA256_ENV, _sha(manifest))

    with pytest.raises(Qwen38ContractError, match="provider-route contract"):
        load_qwen38_scheduled_artifact_family(base_family, manifest)


def test_scheduled_route_duplicate_or_missing_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest, base_family = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    generic = next(
        record
        for record in payload["artifacts"]
        if record["tokens"] == 128
        and record["provider_route"] == M128_PAITON_GENERIC_ROUTE
    )
    payload["artifacts"].remove(generic)
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(SCHEDULED_MANIFEST_SHA256_ENV, _sha(manifest))

    with pytest.raises(Qwen38ContractError, match="bucket family is incomplete"):
        load_qwen38_scheduled_artifact_family(base_family, manifest)


def test_generic_m128_cannot_claim_stock_launch_or_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    manifest, base_family = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    generic = next(
        record
        for record in payload["artifacts"]
        if record["tokens"] == 128
        and record["provider_route"] == M128_PAITON_GENERIC_ROUTE
    )
    generic["stock_provider_launches_allowed"] = True
    generic["workspace_bytes_per_attention_op"] = 512
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    monkeypatch.setenv(SCHEDULED_MANIFEST_SHA256_ENV, _sha(manifest))

    with pytest.raises(
        Qwen38ContractError,
        match="workspace_bytes_per_attention_op|stock_provider_launches_allowed",
    ):
        load_qwen38_scheduled_artifact_family(base_family, manifest)
