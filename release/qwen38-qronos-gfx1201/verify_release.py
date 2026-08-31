#!/usr/bin/env python3
"""Verify the Paiton Qwen3.8 RDNA4 release before loading its shared library."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class ReleaseVerificationError(RuntimeError):
    """Raised when release content does not match its declared contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"could not read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseVerificationError(f"expected a JSON object in {path}")
    return value


def load_release_manifest(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema_version") != 1:
        raise ReleaseVerificationError(
            f"unsupported release schema {manifest.get('schema_version')!r}"
        )
    if not manifest.get("release_id"):
        raise ReleaseVerificationError("release manifest is missing release_id")
    for key in ("base_model", "files", "runtime", "contract", "source"):
        if not isinstance(manifest.get(key), Mapping):
            raise ReleaseVerificationError(f"release manifest is missing {key}")
    return manifest


def _safe_file(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise ReleaseVerificationError(f"invalid release file name: {name!r}")
    relative = Path(name)
    if relative.is_absolute() or relative.name != name or ".." in relative.parts:
        raise ReleaseVerificationError(f"unsafe release file name: {name!r}")
    resolved_root = root.resolve()
    candidate = resolved_root / relative
    if candidate.parent != resolved_root:
        raise ReleaseVerificationError(f"release file escapes bundle: {name!r}")
    return candidate


def _verify_declared_file(root: Path, declaration: Mapping[str, Any]) -> Path:
    path = _safe_file(root, declaration.get("file"))
    if not path.is_file():
        raise ReleaseVerificationError(f"missing release file: {path}")
    expected_size = declaration.get("size_bytes")
    if not isinstance(expected_size, int) or path.stat().st_size != expected_size:
        raise ReleaseVerificationError(
            f"size mismatch for {path.name}: expected {expected_size}, "
            f"got {path.stat().st_size}"
        )
    expected_sha256 = declaration.get("sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ReleaseVerificationError(f"invalid SHA256 declaration for {path.name}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ReleaseVerificationError(
            f"SHA256 mismatch for {path.name}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return path


def _require_equal(actual: object, expected: object, description: str) -> None:
    if actual != expected:
        raise ReleaseVerificationError(
            f"{description} mismatch: expected {expected!r}, got {actual!r}"
        )


def _verify_artifact_contract(
    artifact_path: Path,
    artifact_manifest_path: Path,
    release: Mapping[str, Any],
) -> None:
    artifact_manifest = _load_json(artifact_manifest_path)
    for key in ("manifest_version", "binary_abi_version", "capability_version"):
        _require_equal(artifact_manifest.get(key), 1, key)

    artifact = artifact_manifest.get("artifact")
    target = artifact_manifest.get("target")
    parallelism = artifact_manifest.get("parallelism")
    limits = artifact_manifest.get("limits")
    model = artifact_manifest.get("model")
    contract = artifact_manifest.get("paiton_qwen38_contract")
    for name, value in (
        ("artifact", artifact),
        ("target", target),
        ("parallelism", parallelism),
        ("limits", limits),
        ("model", model),
        ("paiton_qwen38_contract", contract),
    ):
        if not isinstance(value, Mapping):
            raise ReleaseVerificationError(
                f"compiled artifact manifest is missing {name}"
            )

    artifact_declaration = release["files"]["artifact"]
    _require_equal(artifact.get("filename"), artifact_path.name, "artifact filename")
    _require_equal(
        artifact.get("size_bytes"), artifact_declaration["size_bytes"], "artifact size"
    )
    _require_equal(
        artifact.get("sha256"), artifact_declaration["sha256"], "artifact SHA256"
    )

    runtime = release["runtime"]
    release_contract = release["contract"]
    base_model = release["base_model"]
    _require_equal(target.get("arch"), runtime["gpu_arch"], "GPU architecture")
    _require_equal(target.get("family"), "rdna4", "GPU family")
    _require_equal(target.get("wave_size"), 32, "wave size")
    _require_equal(
        parallelism.get("tp_size"),
        release_contract["tensor_parallel_size"],
        "tensor parallel size",
    )
    _require_equal(
        limits.get("max_context_length"),
        release_contract["max_model_len"],
        "maximum context length",
    )
    _require_equal(
        limits.get("max_batched_tokens"),
        release_contract["max_num_batched_tokens"],
        "maximum batched tokens",
    )
    _require_equal(model.get("repository"), base_model["repo_id"], "base model")
    _require_equal(model.get("revision"), base_model["revision"], "base revision")
    _require_equal(model.get("scope"), release_contract["scope"], "model scope")
    _require_equal(contract.get("version"), 3, "Qwen3.8 contract version")
    _require_equal(
        contract.get("max_batch_size"),
        release_contract["max_batch_size"],
        "maximum batch size",
    )
    _require_equal(contract.get("quark_group_size"), 128, "INT4 group size")
    _require_equal(contract.get("quark_algorithm"), "qronos", "Quark algorithm")
    _require_equal(contract.get("multimodal"), False, "multimodal scope")


def _verify_generated_config(config_path: Path, release: Mapping[str, Any]) -> None:
    config = _load_json(config_path)
    architectures = config.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise ReleaseVerificationError("generated config has no architectures")
    _require_equal(
        architectures[0], "PaitonQwen38ForCausalLM", "Paiton architecture"
    )
    _require_equal(config.get("language_model_only"), True, "text-only config")
    _require_equal(config.get("quantization_config"), None, "runtime quant config")
    source_quant = config.get("paiton_source_quantization_config")
    if not isinstance(source_quant, Mapping):
        raise ReleaseVerificationError(
            "generated config is missing paiton_source_quantization_config"
        )
    _require_equal(source_quant.get("quant_method"), "quark", "source quant method")
    contract = config.get("paiton_qwen38_contract")
    if not isinstance(contract, Mapping):
        raise ReleaseVerificationError(
            "generated config is missing paiton_qwen38_contract"
        )
    release_contract = release["contract"]
    _require_equal(contract.get("version"), 3, "config contract version")
    _require_equal(contract.get("scope"), release_contract["scope"], "config scope")
    _require_equal(
        contract.get("tp_size"),
        release_contract["tensor_parallel_size"],
        "config tensor parallel size",
    )
    _require_equal(
        contract.get("max_context_length"),
        release_contract["max_model_len"],
        "config maximum context length",
    )


def verify_bundle(bundle_dir: Path, manifest_path: Path | None = None) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = manifest_path or bundle_dir / "bundle-manifest.json"
    release = load_release_manifest(manifest_path)
    declared_files = release["files"]
    for key in ("artifact", "artifact_manifest", "config"):
        if not isinstance(declared_files.get(key), Mapping):
            raise ReleaseVerificationError(f"release manifest is missing files.{key}")
    artifact_path = _verify_declared_file(bundle_dir, declared_files["artifact"])
    artifact_manifest_path = _verify_declared_file(
        bundle_dir, declared_files["artifact_manifest"]
    )
    config_path = _verify_declared_file(bundle_dir, declared_files["config"])
    _verify_artifact_contract(
        artifact_path, artifact_manifest_path, release
    )
    _verify_generated_config(config_path, release)
    return release


def verify_model_directory(
    model_dir: Path,
    release: Mapping[str, Any],
    *,
    verify_checkpoint_sha256: bool,
) -> None:
    model_dir = model_dir.resolve()
    for declaration in release["files"].values():
        _verify_declared_file(model_dir, declaration)
    checkpoint = release["base_model"]["checkpoint"]
    checkpoint_path = _safe_file(model_dir, checkpoint.get("file"))
    if not checkpoint_path.is_file():
        raise ReleaseVerificationError(f"missing checkpoint: {checkpoint_path}")
    _require_equal(
        checkpoint_path.stat().st_size,
        checkpoint["size_bytes"],
        "checkpoint size",
    )
    if verify_checkpoint_sha256:
        _require_equal(
            sha256_file(checkpoint_path),
            checkpoint["sha256"],
            "checkpoint SHA256",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify a Paiton Qwen3.8 RDNA4 release before dlopen."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing bundle-manifest.json and release files.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="Optional complete/installed model directory to verify.",
    )
    parser.add_argument(
        "--verify-checkpoint-sha256",
        action="store_true",
        help="Hash the 19.9 GB checkpoint in addition to checking its byte size.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    release = verify_bundle(args.bundle_dir)
    if args.model_dir is not None:
        verify_model_directory(
            args.model_dir,
            release,
            verify_checkpoint_sha256=args.verify_checkpoint_sha256,
        )
    print(f"Verified release: {release['release_id']}")
    print(f"Base revision: {release['base_model']['revision']}")
    print(f"Target: {release['runtime']['gpu_arch']}")
    if args.model_dir is not None:
        checkpoint_mode = "SHA256" if args.verify_checkpoint_sha256 else "byte size"
        print(f"Verified model directory ({checkpoint_mode} checkpoint check): {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
