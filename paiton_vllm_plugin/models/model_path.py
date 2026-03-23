"""Helpers for resolving compiled Paiton model artifacts."""

from __future__ import annotations

import re
from pathlib import Path


def resolve_model_so_path(
    model_path: Path,
    tp_size: int,
    max_input_tokens: int | None = None,
) -> Path:
    plain_candidate = model_path / f"{model_path.name}_tp{tp_size}.so"

    def _mt_sort_key(path: Path) -> tuple[int, str]:
        match = re.search(r"_mt(\d+)\.so$", path.name)
        if match is None:
            return (-1, path.name)
        return (int(match.group(1)), path.name)

    def _mt_capacity(path: Path) -> int | None:
        match = re.search(r"_mt(\d+)\.so$", path.name)
        if match is None:
            return None
        return int(match.group(1))

    mt_candidates = sorted(
        model_path.glob(f"{model_path.name}_tp{tp_size}_mt*.so"),
        key=_mt_sort_key,
    )
    if max_input_tokens is not None:
        exact_candidate = model_path / f"{model_path.name}_tp{tp_size}_mt{max_input_tokens}.so"
        if exact_candidate.exists():
            return exact_candidate

        compatible_mt_candidates = [
            path
            for path in mt_candidates
            if (_mt_capacity(path) or -1) >= max_input_tokens
        ]
        if compatible_mt_candidates:
            return compatible_mt_candidates[0]

        if plain_candidate.exists() and not mt_candidates:
            return plain_candidate

        tried = [exact_candidate.name, plain_candidate.name]
        available_mt = [path.name for path in mt_candidates]
        raise FileNotFoundError(
            "Could not find a compatible compiled model .so for "
            f"max_input_tokens={max_input_tokens}. "
            f"Tried: {tried} in {model_path}. "
            f"Available token-capped artifacts: {available_mt}"
        )

    if plain_candidate.exists():
        return plain_candidate
    if mt_candidates:
        return mt_candidates[-1]

    tried = [plain_candidate.name]
    raise FileNotFoundError(
        f"Could not find compiled model .so. Tried: {tried} in {model_path}"
    )
