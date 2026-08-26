# SPDX-License-Identifier: Apache-2.0
"""Fail-closed contract for the pinned Qwen3.8-27B MXFP4 vertical slice.

This is intentionally not a generic Qwen, AWQ, or Quark resolver.  The model
adapter accepts one immutable checkpoint, one compiler identity, one plugin
source set, and four exact-shape artifacts described by a signed-by-hash local
manifest.  The manifest remains outside the package because compiled artifacts
are experimental products with their own provenance and retention lifecycle.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


TARGET_REPOSITORY = "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"
TARGET_REVISION = "156be69f9cac862a41d8b32e773ea2d2754341e8"
TARGET_CHECKPOINT_SHA256 = (
    "be1d745bc7312fdf1486059ec57cdeb514cc4d1aa06528c6677a0ebc0a0e1272"
)
TARGET_CHECKPOINT_SIZE = 19_798_196_184
TARGET_CONFIG_SHA256 = (
    "04c9b07a3a9260cbc8a2ea5b5e5f84ced8274cf412deb9895e91204383ed20e3"
)
COMPILER_COMMIT = "a28ff291b363d369fcecba58f98260cf055fefb2"
COMPILER_TREE = "bfc327454d2ce29bb2e1a39ba9f394432723a7c0"
PLUGIN_BASE_COMMIT = "3cbb75d0e1f4ce95ca00d257342e82735fdeafa2"
PLUGIN_BASE_TREE = "f1ec0ca7a284baf63d0730125e5f33162b95b723"

ADMITTED_TOKEN_COUNTS = (1, 128, 512, 2048)
REQUIRED_TEXT_TENSORS = 1347
DEFERRED_VISION_TENSORS = 333
DEFERRED_MTP_TENSORS = 15
TOTAL_CHECKPOINT_TENSORS = 1695

MANIFEST_ENV = "PAITON_QWEN38_ARTIFACT_MANIFEST"
MANIFEST_SHA256_ENV = "PAITON_QWEN38_ARTIFACT_MANIFEST_SHA256"


class Qwen38ContractError(RuntimeError):
    """The requested model or artifact family violates the frozen contract."""


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise Qwen38ContractError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}"
        )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Qwen38ContractError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class TensorRecord:
    checkpoint_name: str
    runtime_name: str | None
    category: str
    disposition: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str


@dataclass(frozen=True)
class DerivedConstantRecord:
    runtime_name: str
    source_names: tuple[str, ...]
    concatenate_dimension: int
    dtype: str
    shape: tuple[int, ...]
    provider_id: str


@dataclass(frozen=True)
class ArtifactRecord:
    tokens: int
    path: Path
    sha256: str
    size: int
    physical_runtime_names: frozenset[str]
    derived_constants: tuple[DerivedConstantRecord, ...]
    binding_plan_sha256: str


@dataclass(frozen=True)
class GreedyOutputProviderRecord:
    id: str
    path: Path
    sha256: str
    size: int
    abi_version: int
    symbol: str
    workspace_bytes: int
    vllm_version: str
    sampler_forward_sha256: str


@dataclass(frozen=True)
class Qwen38ArtifactFamily:
    manifest_path: Path
    manifest_sha256: str
    checkpoint_path: Path
    config_path: Path
    mapping_path: Path
    mapping_sha256: str
    artifacts: Mapping[int, ArtifactRecord]
    tensor_records: Mapping[str, TensorRecord]
    state_records: tuple[Mapping[str, Any], ...]
    plugin_commit: str
    plugin_tree: str
    greedy_output_provider: GreedyOutputProviderRecord | None

    def artifact_for_tokens(self, tokens: int) -> ArtifactRecord:
        if tokens not in self.artifacts:
            raise Qwen38ContractError(
                f"unsupported fixed token count M={tokens}; admitted counts are "
                f"{list(ADMITTED_TOKEN_COUNTS)} and no padding or redirection is allowed"
            )
        return self.artifacts[tokens]

    @property
    def required_records(self) -> dict[str, TensorRecord]:
        return {
            name: record
            for name, record in self.tensor_records.items()
            if record.disposition == "required"
        }

    @property
    def deferred_records(self) -> dict[str, TensorRecord]:
        return {
            name: record
            for name, record in self.tensor_records.items()
            if record.disposition == "deferred"
        }


def _validate_tensor_mapping(
    mapping_path: Path, expected_sha256: str
) -> tuple[dict[str, TensorRecord], tuple[Mapping[str, Any], ...]]:
    if not mapping_path.is_file():
        raise Qwen38ContractError(f"weight mapping does not exist: {mapping_path}")
    _require_equal(
        "weight mapping SHA256", sha256_file(mapping_path), expected_sha256
    )
    try:
        payload = json.loads(mapping_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen38ContractError(
            f"weight mapping is not parseable JSON: {mapping_path}"
        ) from error
    _require_equal(
        "weight mapping schema",
        payload.get("schema"),
        "paiton.qwen38.full-checkpoint-mapping.v1",
    )
    _require_equal("weight mapping passed", payload.get("passed"), True)
    target = _require_mapping(payload.get("target"), "weight mapping target")
    _require_equal("checkpoint repository", target.get("repository"), TARGET_REPOSITORY)
    _require_equal("checkpoint revision", target.get("revision"), TARGET_REVISION)
    checkpoint = _require_mapping(
        payload.get("checkpoint"), "weight mapping checkpoint"
    )
    _require_equal(
        "checkpoint SHA256", checkpoint.get("sha256"), TARGET_CHECKPOINT_SHA256
    )
    _require_equal("checkpoint size", checkpoint.get("size"), TARGET_CHECKPOINT_SIZE)
    config = _require_mapping(payload.get("config"), "weight mapping config")
    _require_equal("config SHA256", config.get("sha256"), TARGET_CONFIG_SHA256)
    counts = _require_mapping(payload.get("counts"), "weight mapping counts")
    _require_equal(
        "checkpoint tensor count",
        counts.get("checkpoint_tensors"),
        TOTAL_CHECKPOINT_TENSORS,
    )
    _require_equal(
        "required text tensor count",
        counts.get("required_text_tensors"),
        REQUIRED_TEXT_TENSORS,
    )
    _require_equal("deferred tensor count", counts.get("deferred_tensors"), 348)

    records: dict[str, TensorRecord] = {}
    raw_tensors = payload.get("tensors")
    if not isinstance(raw_tensors, list):
        raise Qwen38ContractError("weight mapping tensors must be an array")
    for raw in raw_tensors:
        item = _require_mapping(raw, "weight mapping tensor")
        name = item.get("checkpoint_name")
        if not isinstance(name, str) or not name:
            raise Qwen38ContractError("weight mapping tensor has invalid name")
        if name in records:
            raise Qwen38ContractError(f"duplicate weight mapping tensor: {name}")
        shape = item.get("shape")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(dim, int) and dim >= 0 for dim in shape)
        ):
            raise Qwen38ContractError(f"invalid shape for {name}: {shape!r}")
        record = TensorRecord(
            checkpoint_name=name,
            runtime_name=item.get("runtime_name"),
            category=str(item.get("category")),
            disposition=str(item.get("disposition")),
            dtype=str(item.get("dtype")),
            shape=tuple(shape),
            sha256=str(item.get("sha256")),
        )
        if record.disposition == "required" and not record.runtime_name:
            raise Qwen38ContractError(f"required tensor lacks runtime name: {name}")
        if record.disposition == "deferred" and record.runtime_name is not None:
            raise Qwen38ContractError(f"deferred tensor has runtime name: {name}")
        records[name] = record

    if len(records) != TOTAL_CHECKPOINT_TENSORS:
        raise Qwen38ContractError(
            f"weight mapping has {len(records)} tensors, expected "
            f"{TOTAL_CHECKPOINT_TENSORS}"
        )
    required = [r for r in records.values() if r.disposition == "required"]
    vision = [r for r in records.values() if r.category == "vision"]
    mtp = [r for r in records.values() if r.category == "mtp"]
    _require_equal("required tensor count", len(required), REQUIRED_TEXT_TENSORS)
    _require_equal("deferred vision count", len(vision), DEFERRED_VISION_TENSORS)
    _require_equal("deferred MTP count", len(mtp), DEFERRED_MTP_TENSORS)
    if any(r.disposition != "deferred" for r in (*vision, *mtp)):
        raise Qwen38ContractError("vision and MTP tensors must be explicitly deferred")
    runtime_names = [r.runtime_name for r in required]
    if len(runtime_names) != len(set(runtime_names)):
        raise Qwen38ContractError("required runtime names are not one-to-one")
    return records, tuple(payload.get("metadata_only", ()))


def _binding_plan_sha256(plan: Mapping[str, Any]) -> str:
    encoded = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_constant_materialization(
    value: object,
    *,
    tokens: int,
    tensor_records: Mapping[str, TensorRecord],
) -> tuple[frozenset[str], tuple[DerivedConstantRecord, ...], str]:
    required = {
        name: record
        for name, record in tensor_records.items()
        if record.disposition == "required"
    }
    logical_runtime_names = {
        record.runtime_name for record in required.values()
    }
    if None in logical_runtime_names or len(logical_runtime_names) != len(required):
        raise Qwen38ContractError("required runtime-name mapping is not one-to-one")

    # Older all-direct families predate an explicit materialization sidecar.
    # Their physical ABI is still unambiguous and remains fail-closed.
    if value is None:
        plan: dict[str, Any] = {
            "schema": "paiton.qwen38.artifact-constant-materialization.v1",
            "required_runtime_constant_count": len(required),
            "direct_runtime_constant_count": len(required),
            "derived_runtime_constant_count": 0,
            "logical_checkpoint_source_count": len(required),
            "logical_source_coverage": "exactly-once",
            "derived_constants": [],
        }
        return (
            frozenset(str(name) for name in logical_runtime_names),
            (),
            _binding_plan_sha256(plan),
        )

    raw_plan = _require_mapping(value, f"artifact M={tokens} materialization")
    _require_equal(
        f"artifact M={tokens} materialization schema",
        raw_plan.get("schema"),
        "paiton.qwen38.artifact-constant-materialization.v1",
    )
    raw_derived = raw_plan.get("derived_constants")
    if not isinstance(raw_derived, list):
        raise Qwen38ContractError(
            f"artifact M={tokens} derived_constants must be an array"
        )

    derived: list[DerivedConstantRecord] = []
    derived_names: set[str] = set()
    source_coverage: Counter[str] = Counter()
    normalized_derived: list[dict[str, Any]] = []
    for raw in raw_derived:
        item = _require_mapping(raw, f"artifact M={tokens} derived constant")
        runtime_name = item.get("runtime_name")
        source_names = item.get("source_names")
        dimension = item.get("concatenate_dimension")
        dtype = item.get("dtype")
        shape = item.get("shape")
        provider_id = item.get("provider_id")
        if not isinstance(runtime_name, str) or not runtime_name:
            raise Qwen38ContractError(
                f"artifact M={tokens} derived runtime name is invalid"
            )
        if runtime_name in derived_names or runtime_name in logical_runtime_names:
            raise Qwen38ContractError(
                f"artifact M={tokens} derived runtime name overlaps: {runtime_name}"
            )
        if (
            not isinstance(source_names, list)
            or len(source_names) < 2
            or not all(isinstance(name, str) and name for name in source_names)
            or len(source_names) != len(set(source_names))
        ):
            raise Qwen38ContractError(
                f"artifact M={tokens} derived sources are invalid for {runtime_name}"
            )
        if not isinstance(dimension, int) or dimension < 0:
            raise Qwen38ContractError(
                f"artifact M={tokens} concatenate dimension is invalid for "
                f"{runtime_name}"
            )
        if dtype not in ("U8", "BF16"):
            raise Qwen38ContractError(
                f"artifact M={tokens} derived dtype is invalid for {runtime_name}"
            )
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(size, int) and size >= 0 for size in shape)
            or dimension >= len(shape)
        ):
            raise Qwen38ContractError(
                f"artifact M={tokens} derived shape is invalid for {runtime_name}"
            )
        if not isinstance(provider_id, str) or not provider_id:
            raise Qwen38ContractError(
                f"artifact M={tokens} provider id is invalid for {runtime_name}"
            )
        try:
            sources = [required[name] for name in source_names]
        except KeyError as error:
            raise Qwen38ContractError(
                f"artifact M={tokens} derived source is not required: {error.args[0]}"
            ) from error
        if any(source.dtype != dtype for source in sources):
            raise Qwen38ContractError(
                f"artifact M={tokens} derived source dtype differs for {runtime_name}"
            )
        source_shapes = [source.shape for source in sources]
        if any(len(source_shape) != len(shape) for source_shape in source_shapes):
            raise Qwen38ContractError(
                f"artifact M={tokens} derived source rank differs for {runtime_name}"
            )
        expected_shape = list(source_shapes[0])
        expected_shape[dimension] = sum(
            source_shape[dimension] for source_shape in source_shapes
        )
        for source_shape in source_shapes[1:]:
            if any(
                source_shape[axis] != expected_shape[axis]
                for axis in range(len(shape))
                if axis != dimension
            ):
                raise Qwen38ContractError(
                    f"artifact M={tokens} derived source shape differs for "
                    f"{runtime_name}"
                )
        if tuple(expected_shape) != tuple(shape):
            raise Qwen38ContractError(
                f"artifact M={tokens} derived result shape differs for {runtime_name}"
            )

        record = DerivedConstantRecord(
            runtime_name=runtime_name,
            source_names=tuple(source_names),
            concatenate_dimension=dimension,
            dtype=dtype,
            shape=tuple(shape),
            provider_id=provider_id,
        )
        derived.append(record)
        derived_names.add(runtime_name)
        source_coverage.update(source_names)
        normalized_derived.append(
            {
                "runtime_name": runtime_name,
                "source_names": list(source_names),
                "concatenate_dimension": dimension,
                "dtype": dtype,
                "shape": list(shape),
                "provider_id": provider_id,
            }
        )

    duplicate_sources = sorted(
        name for name, count in source_coverage.items() if count != 1
    )
    if duplicate_sources:
        raise Qwen38ContractError(
            f"artifact M={tokens} derived source coverage is not singleton: "
            f"{duplicate_sources}"
        )
    direct_sources = set(required).difference(source_coverage)
    direct_runtime_names = {
        str(required[name].runtime_name) for name in direct_sources
    }
    physical_names = direct_runtime_names | derived_names
    plan: dict[str, Any] = {
        "schema": "paiton.qwen38.artifact-constant-materialization.v1",
        "required_runtime_constant_count": len(physical_names),
        "direct_runtime_constant_count": len(direct_sources),
        "derived_runtime_constant_count": len(derived),
        "logical_checkpoint_source_count": len(required),
        "logical_source_coverage": "exactly-once",
        "derived_constants": normalized_derived,
    }
    for field, expected in plan.items():
        _require_equal(
            f"artifact M={tokens} materialization {field}",
            raw_plan.get(field),
            expected,
        )
    binding_sha = _binding_plan_sha256(plan)
    _require_equal(
        f"artifact M={tokens} materialization binding SHA256",
        raw_plan.get("binding_plan_sha256"),
        binding_sha,
    )
    return frozenset(physical_names), tuple(derived), binding_sha


def load_qwen38_artifact_family(
    manifest_path: str | Path | None = None,
    *,
    verify_checkpoint_bytes: bool = True,
) -> Qwen38ArtifactFamily:
    """Load and physically verify the exact local artifact family.

    The caller must provide the manifest hash out of band.  Merely pointing at
    a JSON file is insufficient because the file controls artifact paths and
    identities.
    """

    raw_path = manifest_path or os.environ.get(MANIFEST_ENV)
    if not raw_path:
        raise Qwen38ContractError(f"{MANIFEST_ENV} is required")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise Qwen38ContractError(f"artifact-family manifest does not exist: {path}")
    expected_manifest_sha = os.environ.get(MANIFEST_SHA256_ENV)
    if not expected_manifest_sha:
        raise Qwen38ContractError(f"{MANIFEST_SHA256_ENV} is required")
    actual_manifest_sha = sha256_file(path)
    _require_equal(
        "artifact-family manifest SHA256", actual_manifest_sha, expected_manifest_sha
    )
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Qwen38ContractError(f"invalid artifact-family manifest: {path}") from error

    _require_equal(
        "artifact-family schema",
        payload.get("schema"),
        "paiton.qwen38.fixed-m-artifact-family.v1",
    )
    target = _require_mapping(payload.get("target"), "artifact-family target")
    for field, expected in (
        ("repository", TARGET_REPOSITORY),
        ("revision", TARGET_REVISION),
        ("checkpoint_sha256", TARGET_CHECKPOINT_SHA256),
        ("checkpoint_size", TARGET_CHECKPOINT_SIZE),
    ):
        _require_equal(f"artifact-family target {field}", target.get(field), expected)
    compiler = _require_mapping(payload.get("compiler"), "artifact-family compiler")
    _require_equal("compiler commit", compiler.get("commit"), COMPILER_COMMIT)
    _require_equal("compiler tree", compiler.get("tree"), COMPILER_TREE)
    plugin = _require_mapping(payload.get("plugin"), "artifact-family plugin")
    _require_equal("plugin base commit", plugin.get("base_commit"), PLUGIN_BASE_COMMIT)
    _require_equal("plugin base tree", plugin.get("base_tree"), PLUGIN_BASE_TREE)
    plugin_commit = plugin.get("commit")
    plugin_tree = plugin.get("tree")
    if not all(isinstance(value, str) and len(value) == 40 for value in (plugin_commit, plugin_tree)):
        raise Qwen38ContractError("plugin commit and tree must be full 40-hex identities")

    source_files = plugin.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise Qwen38ContractError("plugin source_files must be a non-empty array")
    for raw in source_files:
        item = _require_mapping(raw, "plugin source file")
        source = Path(str(item.get("path"))).resolve()
        if not source.is_file():
            raise Qwen38ContractError(f"plugin source file is missing: {source}")
        _require_equal(
            f"plugin source SHA256 {source}",
            sha256_file(source),
            item.get("sha256"),
        )

    checkpoint_path = Path(str(target.get("checkpoint_path"))).resolve()
    config_path = Path(str(target.get("config_path"))).resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size != TARGET_CHECKPOINT_SIZE:
        raise Qwen38ContractError(
            f"checkpoint path/size violates pinned identity: {checkpoint_path}"
        )
    if not config_path.is_file():
        raise Qwen38ContractError(f"checkpoint config is missing: {config_path}")
    _require_equal("checkpoint config SHA256", sha256_file(config_path), TARGET_CONFIG_SHA256)
    if verify_checkpoint_bytes:
        _require_equal(
            "checkpoint file SHA256",
            sha256_file(checkpoint_path),
            TARGET_CHECKPOINT_SHA256,
        )

    mapping = _require_mapping(payload.get("weight_mapping"), "weight mapping")
    mapping_path = Path(str(mapping.get("path"))).resolve()
    mapping_sha = str(mapping.get("sha256"))
    tensor_records, _ = _validate_tensor_mapping(mapping_path, mapping_sha)

    artifacts_payload = payload.get("artifacts")
    if not isinstance(artifacts_payload, list):
        raise Qwen38ContractError("artifacts must be an array")
    artifacts: dict[int, ArtifactRecord] = {}
    state_records: tuple[Mapping[str, Any], ...] | None = None
    for raw in artifacts_payload:
        item = _require_mapping(raw, "artifact")
        tokens = item.get("tokens")
        if not isinstance(tokens, int) or tokens in artifacts:
            raise Qwen38ContractError(f"invalid or duplicate artifact M={tokens!r}")
        artifact_path = Path(str(item.get("path"))).resolve()
        physical_names, derived_constants, binding_plan_sha256 = (
            _validate_constant_materialization(
                item.get("constant_materialization"),
                tokens=tokens,
                tensor_records=tensor_records,
            )
        )
        record = ArtifactRecord(
            tokens=tokens,
            path=artifact_path,
            sha256=str(item.get("sha256")),
            size=int(item.get("size", -1)),
            physical_runtime_names=physical_names,
            derived_constants=derived_constants,
            binding_plan_sha256=binding_plan_sha256,
        )
        if not artifact_path.is_file() or artifact_path.stat().st_size != record.size:
            raise Qwen38ContractError(
                f"artifact path/size violates manifest for M={tokens}: {artifact_path}"
            )
        _require_equal(
            f"artifact SHA256 M={tokens}", sha256_file(artifact_path), record.sha256
        )
        _require_equal(f"artifact input count M={tokens}", item.get("input_count"), 116)
        _require_equal(f"artifact state count M={tokens}", item.get("state_count"), 112)
        _require_equal(
            f"artifact output shape M={tokens}", item.get("output_shape"), [1, 248320]
        )
        raw_states = item.get("states")
        if not isinstance(raw_states, list) or len(raw_states) != 112:
            raise Qwen38ContractError(f"artifact M={tokens} has invalid state contract")
        normalized_states = tuple(dict(state) for state in raw_states)
        if state_records is None:
            state_records = normalized_states
        elif normalized_states != state_records:
            raise Qwen38ContractError(
                f"artifact M={tokens} state ABI differs from the family"
            )
        artifacts[tokens] = record
    _require_equal("artifact token set", tuple(sorted(artifacts)), ADMITTED_TOKEN_COUNTS)
    assert state_records is not None

    provider_record: GreedyOutputProviderRecord | None = None
    raw_provider = payload.get("greedy_output_provider")
    if raw_provider is not None:
        provider = _require_mapping(raw_provider, "greedy output provider")
        expected_provider_values = {
            "id": "paiton-qwen38-bf16-greedy-argmax-gfx950-v1",
            "target": "gfx950",
            "input_dtype": "bfloat16",
            "input_shape": [1, 248320],
            "output_dtype": "int32",
            "output_shape": [1, 1],
            "abi_version": 1,
            "symbol": "paiton_qwen38_greedy_argmax_bf16_v1",
            "workspace_bytes": 2064,
            "launches": 1,
            "runtime_jit_or_tuning": False,
        }
        for field, expected in expected_provider_values.items():
            _require_equal(
                f"greedy output provider {field}", provider.get(field), expected
            )
        provider_path = Path(str(provider.get("path"))).resolve()
        provider_sha = str(provider.get("sha256"))
        provider_size = provider.get("size")
        if not isinstance(provider_size, int) or provider_size <= 0:
            raise Qwen38ContractError("greedy output provider size must be positive")
        if (
            not provider_path.is_file()
            or provider_path.stat().st_size != provider_size
        ):
            raise Qwen38ContractError(
                "greedy output provider path/size violates manifest: "
                f"{provider_path}"
            )
        _require_equal(
            "greedy output provider SHA256",
            sha256_file(provider_path),
            provider_sha,
        )
        vllm_version = provider.get("vllm_version")
        sampler_forward_sha = provider.get("sampler_forward_sha256")
        if not isinstance(vllm_version, str) or not vllm_version:
            raise Qwen38ContractError(
                "greedy output provider vllm_version must be a non-empty string"
            )
        if not isinstance(sampler_forward_sha, str) or len(sampler_forward_sha) != 64:
            raise Qwen38ContractError(
                "greedy output provider sampler_forward_sha256 must be full SHA256"
            )
        provider_record = GreedyOutputProviderRecord(
            id=str(provider["id"]),
            path=provider_path,
            sha256=provider_sha,
            size=provider_size,
            abi_version=int(provider["abi_version"]),
            symbol=str(provider["symbol"]),
            workspace_bytes=int(provider["workspace_bytes"]),
            vllm_version=vllm_version,
            sampler_forward_sha256=sampler_forward_sha,
        )

    return Qwen38ArtifactFamily(
        manifest_path=path,
        manifest_sha256=actual_manifest_sha,
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        mapping_path=mapping_path,
        mapping_sha256=mapping_sha,
        artifacts=artifacts,
        tensor_records=tensor_records,
        state_records=state_records,
        plugin_commit=plugin_commit,
        plugin_tree=plugin_tree,
        greedy_output_provider=provider_record,
    )


__all__ = [
    "ADMITTED_TOKEN_COUNTS",
    "ArtifactRecord",
    "DerivedConstantRecord",
    "GreedyOutputProviderRecord",
    "Qwen38ArtifactFamily",
    "Qwen38ContractError",
    "TensorRecord",
    "load_qwen38_artifact_family",
    "sha256_file",
]
