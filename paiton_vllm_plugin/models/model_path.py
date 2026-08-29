"""Helpers for resolving target-qualified compiled Paiton artifacts."""

from __future__ import annotations

import re
from pathlib import Path

from paiton_vllm_plugin.artifact_manifest import (
    ArtifactCompatibilityError,
    detect_runtime_gpu_arch,
    load_and_validate_artifact_manifest,
    normalize_gpu_arch,
)


_ARTIFACT_RE = re.compile(
    r"^(?P<prefix>.+?)(?:_(?P<arch>gfx[0-9a-f]+))?_tp(?P<tp>\d+)"
    r"(?:_mt(?P<mt>\d+))?(?:_ps(?P<ps>\d+))?\.so$",
    re.IGNORECASE,
)


def _list_artifacts(model_path: Path) -> list[tuple[Path, re.Match[str]]]:
    artifacts: list[tuple[Path, re.Match[str]]] = []
    for path in model_path.glob("*.so"):
        match = _ARTIFACT_RE.match(path.name)
        if match is not None:
            artifacts.append((path, match))
    return artifacts


def _resolve_artifact_prefix(
    model_path: Path,
    artifact_prefix: str | None,
    artifacts: list[tuple[Path, re.Match[str]]],
) -> str:
    available_prefixes = sorted({match.group("prefix") for _, match in artifacts})
    for candidate in (artifact_prefix, model_path.name):
        if candidate and candidate in available_prefixes:
            return candidate
    if len(available_prefixes) == 1:
        return available_prefixes[0]
    if not available_prefixes:
        return artifact_prefix or model_path.name
    raise FileNotFoundError(
        f"Found multiple compiled model prefixes in {model_path}: {available_prefixes}. "
        "Use a model directory or repo containing artifacts for exactly one model."
    )


def resolve_model_so_path(
    model_path: Path,
    artifact_prefix: str | None,
    tp_size: int,
    max_input_tokens: int | None = None,
    decode_partition_size: int | None = None,
    target_arch: str | None = None,
) -> Path:
    """Select an artifact for the active GPU and validate it before ``dlopen``.

    Architecture-qualified artifacts always require manifest v1. Legacy
    unqualified artifacts remain available only on pre-gfx12 targets so the
    existing CDNA deployment path remains usable during migration.
    """
    selected_arch = normalize_gpu_arch(target_arch or detect_runtime_gpu_arch())
    all_artifacts = _list_artifacts(model_path)
    matching_arch = [
        item for item in all_artifacts if item[1].group("arch") == selected_arch
    ]
    versioned = bool(matching_arch)
    if versioned:
        eligible = matching_arch
    elif selected_arch.startswith("gfx12"):
        available = sorted(
            {match.group("arch") for _, match in all_artifacts if match.group("arch")}
        )
        raise ArtifactCompatibilityError(
            f"no {selected_arch}-qualified Paiton artifact exists in {model_path}; "
            f"available targets: {available or ['legacy-unqualified']}"
        )
    else:
        eligible = [item for item in all_artifacts if item[1].group("arch") is None]

    resolved_prefix = _resolve_artifact_prefix(model_path, artifact_prefix, eligible)
    candidates = [
        (path, match)
        for path, match in eligible
        if match.group("prefix") == resolved_prefix
        and int(match.group("tp")) == tp_size
    ]

    def finish(path: Path) -> Path:
        if versioned:
            load_and_validate_artifact_manifest(
                path, expected_arch=selected_arch, expected_tp_size=tp_size
            )
        return path

    def capacity(item: tuple[Path, re.Match[str]]) -> int | None:
        value = item[1].group("mt")
        return int(value) if value is not None else None

    def partition(item: tuple[Path, re.Match[str]]) -> int | None:
        value = item[1].group("ps")
        return int(value) if value is not None else None

    plain = [item for item in candidates if capacity(item) is None]
    token_capped = sorted(
        (item for item in candidates if capacity(item) is not None),
        key=lambda item: (capacity(item) or -1, item[0].name),
    )

    if max_input_tokens is not None:
        compatible = [
            item for item in token_capped if (capacity(item) or -1) >= max_input_tokens
        ]
        if decode_partition_size is not None:
            exact_partition = [
                item for item in compatible if partition(item) == decode_partition_size
            ]
            if exact_partition:
                compatible = exact_partition
            else:
                compatible = [item for item in compatible if partition(item) is None]

        exact_capacity = [
            item for item in compatible if capacity(item) == max_input_tokens
        ]
        if len(exact_capacity) > 1 and decode_partition_size is None:
            raise FileNotFoundError(
                "Found multiple exact-capacity artifacts without a "
                "decode_partition_size to disambiguate them: "
                f"{[item[0].name for item in exact_capacity]}"
            )
        if exact_capacity:
            return finish(exact_capacity[0][0])
        if len(compatible) == 1:
            return finish(compatible[0][0])
        if len(compatible) > 1 and decode_partition_size is None:
            raise FileNotFoundError(
                "Found multiple compatible compiled artifacts without a "
                "decode_partition_size to disambiguate them: "
                f"{[item[0].name for item in compatible]}"
            )
        if compatible:
            return finish(compatible[0][0])

        partition_plain = [
            item for item in plain if partition(item) == decode_partition_size
        ]
        if decode_partition_size is not None and partition_plain:
            return finish(partition_plain[0][0])
        if plain and not token_capped:
            no_partition = [item for item in plain if partition(item) is None]
            if no_partition:
                return finish(no_partition[0][0])
        raise FileNotFoundError(
            f"Could not find a compatible {selected_arch} artifact for "
            f"tp_size={tp_size}, max_input_tokens={max_input_tokens} in {model_path}"
        )

    if decode_partition_size is not None:
        partition_plain = [
            item for item in plain if partition(item) == decode_partition_size
        ]
        if partition_plain:
            return finish(partition_plain[0][0])
    no_partition = [item for item in plain if partition(item) is None]
    if no_partition:
        return finish(no_partition[0][0])
    if token_capped:
        return finish(token_capped[-1][0])
    raise FileNotFoundError(
        f"Could not find a compiled {selected_arch} model artifact for "
        f"prefix={resolved_prefix!r}, tp_size={tp_size} in {model_path}"
    )
