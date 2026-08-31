#!/usr/bin/env python3
"""Create a directly servable Paiton model directory without copying 19.9 GB."""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

from verify_release import (
    ReleaseVerificationError,
    load_release_manifest,
    verify_bundle,
    verify_model_directory,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the verified Paiton artifact on the exact AMD checkpoint. "
            "The output directory must not already exist."
        )
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing the Paiton release bundle.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--base-model-dir",
        type=Path,
        help="Existing directory containing the exact AMD model snapshot.",
    )
    source.add_argument(
        "--download",
        action="store_true",
        help="Resolve the pinned AMD model revision through Hugging Face Hub.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "hardlink", "symlink", "copy"),
        default="auto",
        help=(
            "How to populate base-model files. auto tries hardlinks and falls "
            "back to symlinks, avoiding another 19.9 GB checkpoint copy."
        ),
    )
    parser.add_argument(
        "--hf-token",
        help="Optional Hugging Face token used only with --download.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only the local Hugging Face cache with --download.",
    )
    parser.add_argument(
        "--verify-checkpoint-sha256",
        action="store_true",
        help="Hash the 19.9 GB checkpoint rather than checking only its byte size.",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"expected a JSON object in {path}")
    return value


def _resolve_base_model(
    base_model_dir: Path | None,
    *,
    download: bool,
    release: Mapping[str, Any],
    token: str | None,
    local_files_only: bool,
) -> Path:
    if base_model_dir is not None:
        path = base_model_dir.resolve()
        if not path.is_dir():
            raise ReleaseVerificationError(f"base model directory does not exist: {path}")
        return path
    if not download:
        raise ReleaseVerificationError(
            "select --base-model-dir PATH or --download for the pinned Hugging Face snapshot"
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ReleaseVerificationError(
            "--download requires huggingface_hub; install it with "
            "`python3 -m pip install huggingface_hub==1.6.0`"
        ) from error
    base_model = release["base_model"]
    print(
        f"Resolving {base_model['repo_id']} at immutable revision "
        f"{base_model['revision']}...",
        flush=True,
    )
    return Path(
        snapshot_download(
            repo_id=base_model["repo_id"],
            revision=base_model["revision"],
            token=token,
            local_files_only=local_files_only,
        )
    ).resolve()


def _validate_base_model(
    base_dir: Path,
    release: Mapping[str, Any],
) -> None:
    checkpoint = release["base_model"]["checkpoint"]
    checkpoint_path = base_dir / checkpoint["file"]
    if not checkpoint_path.is_file():
        raise ReleaseVerificationError(
            f"base snapshot is missing {checkpoint['file']}: {base_dir}"
        )
    if checkpoint_path.stat().st_size != checkpoint["size_bytes"]:
        raise ReleaseVerificationError(
            f"checkpoint size mismatch: expected {checkpoint['size_bytes']}, "
            f"got {checkpoint_path.stat().st_size}"
        )
    config_path = base_dir / "config.json"
    if not config_path.is_file():
        raise ReleaseVerificationError(f"base snapshot is missing config.json: {base_dir}")
    config = _read_json(config_path)
    if config.get("model_type") != "qwen3_5":
        raise ReleaseVerificationError(
            f"base config model_type must be qwen3_5, got {config.get('model_type')!r}"
        )
    quantization = config.get("quantization_config")
    if not isinstance(quantization, Mapping) or quantization.get("quant_method") != "quark":
        raise ReleaseVerificationError("base model is not the expected Quark checkpoint")
    weight = (quantization.get("global_quant_config") or {}).get("weight") or {}
    if weight.get("dtype") != "int4" or weight.get("group_size") != 128:
        raise ReleaseVerificationError(
            "base model is not the expected Quark INT4 group-128 checkpoint"
        )


def _ensure_safe_output(base_dir: Path, bundle_dir: Path, output_dir: Path) -> None:
    output = output_dir.absolute()
    if output.exists():
        raise ReleaseVerificationError(
            f"output directory already exists; choose a new path: {output}"
        )
    for label, source in (("base model", base_dir), ("bundle", bundle_dir)):
        source = source.resolve()
        try:
            output.relative_to(source)
        except ValueError:
            continue
        raise ReleaseVerificationError(
            f"output directory must not be inside the {label} directory: {output}"
        )


def _populate_file(source: Path, destination: Path, mode: str) -> str:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode in ("auto", "hardlink"):
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError as error:
            if mode == "hardlink" or error.errno not in (
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
                errno.ENOTSUP,
            ):
                raise
    if mode in ("auto", "symlink"):
        os.symlink(source, destination)
        return "symlink"
    shutil.copy2(source, destination)
    return "copy"


def _populate_base_model(
    base_dir: Path,
    output_dir: Path,
    release: Mapping[str, Any],
    mode: str,
) -> dict[str, int]:
    overlay_names = {
        declaration["file"] for declaration in release["files"].values()
    }
    overlay_names.update({"bundle-manifest.json", "paiton-release.json"})
    counts = {"hardlink": 0, "symlink": 0, "copy": 0}
    for source in sorted(path for path in base_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(base_dir)
        if relative.parts[0] in (".git", ".cache"):
            continue
        if relative.as_posix() in overlay_names:
            continue
        destination = output_dir / relative
        used = _populate_file(source, destination, mode)
        counts[used] += 1
    return counts


def _copy_release_files(
    bundle_dir: Path,
    output_dir: Path,
    release: Mapping[str, Any],
) -> None:
    for declaration in release["files"].values():
        source = bundle_dir / declaration["file"]
        destination = output_dir / declaration["file"]
        shutil.copy2(source, destination)
        destination.chmod(0o644)
    shutil.copy2(bundle_dir / "bundle-manifest.json", output_dir / "paiton-release.json")
    (output_dir / "paiton-release.json").chmod(0o644)


def _write_installation_record(
    output_dir: Path,
    release: Mapping[str, Any],
    counts: Mapping[str, int],
) -> None:
    record = {
        "schema_version": 1,
        "release_id": release["release_id"],
        "base_model": release["base_model"]["repo_id"],
        "base_revision": release["base_model"]["revision"],
        "population": dict(counts),
    }
    (output_dir / "paiton-installation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = build_parser().parse_args()
    bundle_dir = args.bundle_dir.resolve()
    release = verify_bundle(bundle_dir)
    # Re-open through the public parser so a copied/renamed manifest cannot be
    # substituted between bundle verification and installation.
    release = load_release_manifest(bundle_dir / "bundle-manifest.json")
    base_dir = _resolve_base_model(
        args.base_model_dir,
        download=args.download,
        release=release,
        token=args.hf_token,
        local_files_only=args.local_files_only,
    )
    _validate_base_model(base_dir, release)
    output_dir = args.output_dir.absolute()
    _ensure_safe_output(base_dir, bundle_dir, output_dir)
    output_dir.mkdir(parents=True)
    counts = _populate_base_model(base_dir, output_dir, release, args.link_mode)
    _copy_release_files(bundle_dir, output_dir, release)
    _write_installation_record(output_dir, release, counts)
    verify_model_directory(
        output_dir,
        release,
        verify_checkpoint_sha256=args.verify_checkpoint_sha256,
    )

    print(f"Installed verified Paiton model directory: {output_dir}")
    print(
        "Base files: "
        f"{counts['hardlink']} hardlinks, {counts['symlink']} symlinks, "
        f"{counts['copy']} copies"
    )
    print("Serve with the pinned runtime:")
    print("  export VLLM_USE_PAITON_PLATFORM=1 PAITON_GPU_ARCH=gfx1201")
    print(
        f"  vllm serve {output_dir} --tensor-parallel-size 1 "
        "--served-model-name qwen38 "
        "--max-model-len 8192 --max-num-batched-tokens 8192 "
        "--max-num-seqs 1 --kv-cache-dtype auto --enforce-eager "
        "--no-enable-prefix-caching --reasoning-parser qwen3 --port 8000"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
