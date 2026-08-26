# SPDX-License-Identifier: Apache-2.0
"""Hash-bound artifact contract for Qwen3.8 scheduled-token buckets."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Mapping

from paiton_vllm_plugin.models.qwen38_contract import (
    DerivedConstantRecord,
    Qwen38ArtifactFamily,
    Qwen38ContractError,
    TARGET_CHECKPOINT_SHA256,
    TARGET_REPOSITORY,
    TARGET_REVISION,
    _validate_constant_materialization,
    sha256_file,
)
from paiton_vllm_plugin.models.qwen38_schedule import (
    DEFAULT_PROVIDER_ROUTE,
    M128_GENERIC_PROVIDER_ID,
    M128_PAITON_GENERIC_ROUTE,
    M128_ROUTE_PREDICATE,
    M128_ROUTE_PREDICATE_SHA256,
    M128_STOCK_EXACT_DECODE_ROUTE,
    M128_STOCK_PROVIDER_ID,
    TOKEN_BUCKETS,
)


SCHEDULED_MANIFEST_ENV = "PAITON_QWEN38_SCHEDULED_ARTIFACT_MANIFEST"
SCHEDULED_MANIFEST_SHA256_ENV = (
    "PAITON_QWEN38_SCHEDULED_ARTIFACT_MANIFEST_SHA256"
)
STARTING_COMPILER_COMMIT = "a28ff291b363d369fcecba58f98260cf055fefb2"
STARTING_PLUGIN_COMMIT = "e41c53b7ca4bbbf22b67ed1f314e120c314abee0"
BASE_ARTIFACT_MANIFEST_SHA256 = (
    "22b7b9ebe5bd4d767f64c36c4071662a5becd8bfd99bc80478ab07aa346b876c"
)
STOCK_M128_HSACO_SHA256 = (
    "42b8916924a1959d326934041214d4bbe1ba9afca6570a2803ca77e51f7eea27"
)
SCHEDULED_PROVIDER_ROUTE_CONTRACT = {
    "schema": "paiton.qwen38.scheduled-provider-routes.v1",
    "predicate": M128_ROUTE_PREDICATE,
    "predicate_sha256": M128_ROUTE_PREDICATE_SHA256,
    "routes": {
        M128_STOCK_EXACT_DECODE_ROUTE: {
            "provider_id": f"{M128_STOCK_PROVIDER_ID}-packed-projection",
            "workspace_bytes_per_attention_op": 512,
            "stock_provider_launches_allowed": True,
            "embedded_stock_hsaco_sha256": STOCK_M128_HSACO_SHA256,
        },
        M128_PAITON_GENERIC_ROUTE: {
            "provider_id": f"{M128_GENERIC_PROVIDER_ID}-packed-projection",
            "workspace_bytes_per_attention_op": 0,
            "stock_provider_launches_allowed": False,
            "embedded_stock_hsaco_sha256": None,
        },
    },
}


@dataclass(frozen=True)
class ScheduledArtifactRecord:
    tokens: int
    provider_route: str
    provider_id: str | None
    provider_route_predicate_sha256: str | None
    output_rows: int
    path: Path
    sha256: str
    size: int
    physical_runtime_names: frozenset[str]
    derived_constants: tuple[DerivedConstantRecord, ...]
    binding_plan_sha256: str


@dataclass(frozen=True)
class Qwen38ScheduledArtifactFamily:
    manifest_path: Path
    manifest_sha256: str
    abi_path: Path
    abi_sha256: str
    compiler_commit: str
    compiler_tree: str
    plugin_commit: str
    plugin_tree: str
    artifacts: Mapping[tuple[int, str], ScheduledArtifactRecord]

    def artifact_for_route(
        self,
        token_bucket: int,
        provider_route: str,
    ) -> ScheduledArtifactRecord:
        try:
            return self.artifacts[(token_bucket, provider_route)]
        except KeyError as error:
            raise Qwen38ContractError(
                "scheduled artifact route is missing: "
                f"M={token_bucket} route={provider_route}"
            ) from error

    def artifact_for_tokens(
        self,
        tokens: int,
        *,
        provider_route: str = DEFAULT_PROVIDER_ROUTE,
    ) -> ScheduledArtifactRecord:
        for bucket in TOKEN_BUCKETS:
            if tokens <= bucket:
                return self.artifact_for_route(bucket, provider_route)
        raise Qwen38ContractError(
            f"scheduled token count {tokens} exceeds {TOKEN_BUCKETS[-1]}"
        )


def _full_identity(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise Qwen38ContractError(f"{name} must be a full lowercase Git identity")
    return value


def load_qwen38_scheduled_artifact_family(
    base_family: Qwen38ArtifactFamily,
    manifest_path: str | Path | None = None,
) -> Qwen38ScheduledArtifactFamily:
    raw_path = manifest_path or os.environ.get(SCHEDULED_MANIFEST_ENV)
    if not raw_path:
        raise Qwen38ContractError(f"{SCHEDULED_MANIFEST_ENV} is required")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise Qwen38ContractError(f"scheduled artifact manifest is missing: {path}")
    expected_sha = os.environ.get(SCHEDULED_MANIFEST_SHA256_ENV)
    if not expected_sha:
        raise Qwen38ContractError(f"{SCHEDULED_MANIFEST_SHA256_ENV} is required")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise Qwen38ContractError(
            f"scheduled manifest SHA256 mismatch: {actual_sha} != {expected_sha}"
        )
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen38ContractError(
            f"scheduled artifact manifest is not parseable JSON: {path}"
        ) from error
    if payload.get("schema") != "paiton.qwen38.scheduled-artifact-family.v2":
        raise Qwen38ContractError("scheduled artifact manifest schema differs")
    if payload.get("provider_route_contract") != SCHEDULED_PROVIDER_ROUTE_CONTRACT:
        raise Qwen38ContractError("scheduled provider-route contract differs")
    if payload.get("base_artifact_manifest_sha256") != BASE_ARTIFACT_MANIFEST_SHA256:
        raise Qwen38ContractError("scheduled family does not bind the rollback family")
    target = payload.get("target")
    if not isinstance(target, dict) or (
        target.get("repository"),
        target.get("revision"),
        target.get("checkpoint_sha256"),
    ) != (TARGET_REPOSITORY, TARGET_REVISION, TARGET_CHECKPOINT_SHA256):
        raise Qwen38ContractError("scheduled family checkpoint identity differs")
    compiler = payload.get("compiler")
    plugin = payload.get("plugin")
    if not isinstance(compiler, dict) or not isinstance(plugin, dict):
        raise Qwen38ContractError("scheduled family source identities are missing")
    if compiler.get("parent_commit") != STARTING_COMPILER_COMMIT:
        raise Qwen38ContractError("scheduled compiler parent differs")
    if plugin.get("parent_commit") != STARTING_PLUGIN_COMMIT:
        raise Qwen38ContractError("scheduled plugin parent differs")
    compiler_commit = _full_identity(compiler.get("commit"), "compiler commit")
    compiler_tree = _full_identity(compiler.get("tree"), "compiler tree")
    plugin_commit = _full_identity(plugin.get("commit"), "plugin commit")
    plugin_tree = _full_identity(plugin.get("tree"), "plugin tree")
    source_files = plugin.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise Qwen38ContractError("scheduled plugin source ledger is empty")
    for raw in source_files:
        if not isinstance(raw, dict):
            raise Qwen38ContractError("scheduled plugin source ledger is malformed")
        source = Path(str(raw.get("path"))).resolve()
        if not source.is_file() or sha256_file(source) != raw.get("sha256"):
            raise Qwen38ContractError(
                f"scheduled plugin source identity differs: {source}"
            )
    abi = payload.get("abi")
    if not isinstance(abi, dict):
        raise Qwen38ContractError("scheduled ABI identity is missing")
    abi_path = Path(str(abi.get("path"))).resolve()
    abi_sha = str(abi.get("sha256"))
    if not abi_path.is_file() or sha256_file(abi_path) != abi_sha:
        raise Qwen38ContractError("scheduled ABI bytes differ")
    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise Qwen38ContractError("scheduled artifacts must be an array")
    artifacts: dict[tuple[int, str], ScheduledArtifactRecord] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise Qwen38ContractError("scheduled artifact record is malformed")
        tokens = raw.get("tokens")
        provider_route = raw.get("provider_route")
        key = (tokens, provider_route)
        if (
            not isinstance(tokens, int)
            or not isinstance(provider_route, str)
            or key in artifacts
        ):
            raise Qwen38ContractError(f"invalid scheduled artifact bucket {tokens!r}")
        expected_route = (
            provider_route in (
                M128_STOCK_EXACT_DECODE_ROUTE,
                M128_PAITON_GENERIC_ROUTE,
            )
            if tokens == 128
            else provider_route == DEFAULT_PROVIDER_ROUTE
        )
        if not expected_route:
            raise Qwen38ContractError(
                f"invalid scheduled artifact route M={tokens} {provider_route!r}"
            )
        route_contract = (
            SCHEDULED_PROVIDER_ROUTE_CONTRACT["routes"].get(provider_route)
            if tokens == 128
            else None
        )
        provider_id = raw.get("provider_id")
        predicate_sha256 = raw.get("provider_route_predicate_sha256")
        if tokens == 128:
            assert isinstance(route_contract, dict)
            for field, value in route_contract.items():
                if raw.get(field) != value:
                    raise Qwen38ContractError(
                        f"scheduled artifact M=128 route={provider_route} {field} differs"
                    )
            if predicate_sha256 != M128_ROUTE_PREDICATE_SHA256:
                raise Qwen38ContractError(
                    "scheduled M128 artifact predicate identity differs"
                )
        elif predicate_sha256 is not None:
            raise Qwen38ContractError(
                f"non-M128 artifact M={tokens} has an M128 route predicate"
            )
        artifact_path = Path(str(raw.get("path"))).resolve()
        size = raw.get("size")
        digest = raw.get("sha256")
        output_rows = min(tokens, 128)
        physical_names, derived_constants, binding_plan_sha256 = (
            _validate_constant_materialization(
                raw.get("constant_materialization"),
                tokens=tokens,
                tensor_records=base_family.required_records,
            )
        )
        expected = {
            "input_count": 172,
            "state_count": 112,
            "output_shape": [output_rows, 248320],
            "max_active_sequences": 128,
            "max_scheduled_tokens": 4096,
            "runtime_jit_or_tuning": False,
            "stock_model_fallback": False,
        }
        for field, value in expected.items():
            if raw.get(field) != value:
                raise Qwen38ContractError(
                    f"scheduled artifact M={tokens} {field} differs"
                )
        if (
            not isinstance(size, int)
            or size <= 0
            or not artifact_path.is_file()
            or artifact_path.stat().st_size != size
            or sha256_file(artifact_path) != digest
        ):
            raise Qwen38ContractError(
                f"scheduled artifact M={tokens} path/bytes differ"
            )
        artifacts[key] = ScheduledArtifactRecord(
            tokens=tokens,
            provider_route=provider_route,
            provider_id=(str(provider_id) if provider_id is not None else None),
            provider_route_predicate_sha256=(
                str(predicate_sha256) if predicate_sha256 is not None else None
            ),
            output_rows=output_rows,
            path=artifact_path,
            sha256=str(digest),
            size=size,
            physical_runtime_names=physical_names,
            derived_constants=derived_constants,
            binding_plan_sha256=binding_plan_sha256,
        )
    expected_keys = {
        (tokens, DEFAULT_PROVIDER_ROUTE)
        for tokens in TOKEN_BUCKETS
        if tokens != 128
    }
    expected_keys.update(
        {
            (128, M128_STOCK_EXACT_DECODE_ROUTE),
            (128, M128_PAITON_GENERIC_ROUTE),
        }
    )
    if set(artifacts) != expected_keys:
        raise Qwen38ContractError("scheduled artifact bucket family is incomplete")
    return Qwen38ScheduledArtifactFamily(
        manifest_path=path,
        manifest_sha256=actual_sha,
        abi_path=abi_path,
        abi_sha256=abi_sha,
        compiler_commit=compiler_commit,
        compiler_tree=compiler_tree,
        plugin_commit=plugin_commit,
        plugin_tree=plugin_tree,
        artifacts=artifacts,
    )


__all__ = [
    "Qwen38ScheduledArtifactFamily",
    "SCHEDULED_PROVIDER_ROUTE_CONTRACT",
    "ScheduledArtifactRecord",
    "load_qwen38_scheduled_artifact_family",
]
