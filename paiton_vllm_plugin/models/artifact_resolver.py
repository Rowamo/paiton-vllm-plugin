# SPDX-License-Identifier: Apache-2.0
"""Helpers for resolving where compiled Paiton artifacts live locally."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_artifact_dir(
    model_ref: str,
    revision: str | None = None,
    token: str | bool | None = None,
    download_dir: str | None = None,
) -> Path:
    """Resolve the local directory containing compiled Paiton `.so` artifacts.

    `model_ref` may be either a local path or a Hugging Face repo id. For
    remote repos we download shared libraries and their mandatory manifests
    into the normal Hugging Face
    cache and return the resolved snapshot directory.
    """
    local_path = Path(model_ref)
    if local_path.exists():
        return local_path

    resolved_path = snapshot_download(
        repo_id=model_ref,
        repo_type="model",
        revision=revision,
        token=token,
        cache_dir=download_dir,
        allow_patterns=["*.so", "*.manifest.json", "config.json"],
    )
    return Path(resolved_path)
