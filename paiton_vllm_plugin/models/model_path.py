"""Helpers for resolving compiled Paiton model artifacts."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

_ARTIFACT_RE = re.compile(
    r"^(?P<prefix>.+)_tp(?P<tp>\d+)(?:_mt(?P<mt>\d+))?(?:_ps(?P<ps>\d+))?\.so$"
)


def _list_artifacts(model_path: Path) -> list[tuple[Path, re.Match[str]]]:
    artifacts: list[tuple[Path, re.Match[str]]] = []
    for path in model_path.glob("*.so"):
        match = _ARTIFACT_RE.match(path.name)
        if match is not None:
            artifacts.append((path, match))
    return artifacts


def _resolve_artifact_prefix(model_path: Path, artifact_prefix: str | None) -> str:
    artifacts = _list_artifacts(model_path)
    available_prefixes = sorted({match.group("prefix") for _, match in artifacts})

    for candidate in (artifact_prefix, model_path.name):
        if candidate and candidate in available_prefixes:
            return candidate

    if len(available_prefixes) == 1:
        return available_prefixes[0]

    if not available_prefixes:
        return artifact_prefix or model_path.name

    raise FileNotFoundError(
        "Found multiple compiled model prefixes in "
        f"{model_path}: {available_prefixes}. "
        "Use a model directory or repo that contains artifacts for exactly one model."
    )


def resolve_model_so_path(
    model_path: Path,
    artifact_prefix: str | None,
    tp_size: int,
    max_input_tokens: int | None = None,
    decode_partition_size: int | None = None,
) -> Path:
    resolved_prefix = _resolve_artifact_prefix(model_path, artifact_prefix)

    plain_candidate = model_path / f"{resolved_prefix}_tp{tp_size}.so"
    plain_partition_candidate = (
        model_path / f"{resolved_prefix}_tp{tp_size}_ps{decode_partition_size}.so"
        if decode_partition_size is not None
        else None
    )

    def _mt_sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_mt(\d+)(?:_ps(\d+))?\.so$", path.name)
        if match is None:
            return (-1, path.name)
        return (int(match.group(1)), path.name)

    def _mt_capacity(path: Path) -> int | None:
        match = re.search(r"_mt(\d+)(?:_ps(\d+))?\.so$", path.name)
        if match is None:
            return None
        return int(match.group(1))

    def _partition_size(path: Path) -> int | None:
        match = re.search(r"_ps(\d+)\.so$", path.name)
        if match is None:
            return None
        return int(match.group(1))

    def _warn_if_oversized(selected: Path) -> None:
        selected_capacity = _mt_capacity(selected)
        if selected_capacity is not None and selected_capacity > max_input_tokens:
            warnings.warn(
                "Falling back to a larger compiled Paiton artifact because "
                f"no exact _mt{max_input_tokens} .so was found. "
                f"Requested max_input_tokens={max_input_tokens}, selected "
                f"{selected.name}. Rebuild the exact-capacity artifact if "
                "generation quality or memory usage differs across builds.",
                stacklevel=2,
            )

    def _resolve_compatible_mt_candidates(
        candidates: list[Path],
        requested_max_input_tokens: int,
        requested_partition_size: int | None,
    ) -> list[Path]:
        filtered = [
            path
            for path in candidates
            if (_mt_capacity(path) or -1) >= requested_max_input_tokens
        ]
        if requested_partition_size is None:
            return filtered

        exact_partition = [
            path for path in filtered if _partition_size(path) == requested_partition_size
        ]
        if exact_partition:
            return exact_partition

        return [path for path in filtered if _partition_size(path) is None]

    mt_candidates = sorted(
        model_path.glob(f"{resolved_prefix}_tp{tp_size}_mt*.so"),
        key=_mt_sort_key,
    )
    if max_input_tokens is not None:
        exact_candidates = []
        if decode_partition_size is not None:
            exact_candidates.append(
                model_path / (
                    f"{resolved_prefix}_tp{tp_size}_mt{max_input_tokens}"
                    f"_ps{decode_partition_size}.so"
                )
            )
        exact_candidates.append(
            model_path / f"{resolved_prefix}_tp{tp_size}_mt{max_input_tokens}.so"
        )
        for exact_candidate in exact_candidates:
            if exact_candidate.exists():
                return exact_candidate

        compatible_mt_candidates = _resolve_compatible_mt_candidates(
            mt_candidates,
            max_input_tokens,
            decode_partition_size,
        )
        if len(compatible_mt_candidates) == 1:
            selected = compatible_mt_candidates[0]
            _warn_if_oversized(selected)
            return selected

        if len(compatible_mt_candidates) > 1 and decode_partition_size is None:
            available_mt = [path.name for path in compatible_mt_candidates]
            raise FileNotFoundError(
                "Found multiple compatible compiled model .so artifacts but no "
                "decode_partition_size was specified to disambiguate them. "
                f"Set decode_partition_size in config.json or remove extras. Available: {available_mt}"
            )

        if compatible_mt_candidates:
            selected = compatible_mt_candidates[0]
            _warn_if_oversized(selected)
            return selected

        if plain_partition_candidate is not None and plain_partition_candidate.exists():
            return plain_partition_candidate
        if plain_candidate.exists() and not mt_candidates:
            return plain_candidate
        tried = [candidate.name for candidate in exact_candidates] + [plain_candidate.name]
        if plain_partition_candidate is not None:
            tried.insert(len(exact_candidates), plain_partition_candidate.name)
        available_mt = [path.name for path in mt_candidates]
        raise FileNotFoundError(
            "Could not find a compatible compiled model .so for "
            f"max_input_tokens={max_input_tokens}. "
            f"Tried: {tried} in {model_path}. "
            f"Available token-capped artifacts: {available_mt}"
        )

    if plain_partition_candidate is not None and plain_partition_candidate.exists():
        return plain_partition_candidate
    if plain_candidate.exists():
        return plain_candidate
    if mt_candidates:
        return mt_candidates[-1]

    tried = [plain_candidate.name]
    raise FileNotFoundError(
        f"Could not find compiled model .so. Tried: {tried} in {model_path}"
    )
