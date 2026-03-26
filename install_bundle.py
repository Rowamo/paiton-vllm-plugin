#!/usr/bin/env python3
"""Install a Paiton overlay bundle into a Hugging Face model snapshot.

Expected bundle layout:

  <bundle>/
    install_bundle.py
    manifest.json
    <one or more>.so

Minimal manifest.json:

{
  "model_id": "meta-llama/Llama-3.1-8B-Instruct",
  "revision": "main",
  "paiton_architecture": "PaitonLlamaForCausalLM",
  "decode_partition_size": 256,
  "kv_cache_dtype": "fp8",
  "artifacts": [
    {
      "file": "Llama-3.1-8B-Instruct-FP8-KV_tp1_mt16384_ps256.so",
      "sha256": "<optional sha256>"
    }
  ]
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - user environment issue
    raise SystemExit(
        "huggingface_hub is required to run install_bundle.py. "
        "Install it with `pip install huggingface_hub`."
    ) from exc


DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.safetensors",
    "*.bin",
    "tokenizer*",
]

MODEL_TYPE_TO_ARCH = {
    "llama": "PaitonLlamaForCausalLM",
    "qwen2": "PaitonQwen2ForCausalLM",
    "qwen3": "PaitonQwen3ForCausalLM",
    "qwen3_moe": "PaitonQwen3MoeForCausalLM",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a base Hugging Face model, patch config.json, and install Paiton .so artifacts.",
    )
    parser.add_argument(
        "--bundle",
        default=None,
        help="Bundle directory. Defaults to the directory containing this script.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Target model directory to create or update.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated/private models.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision override. Defaults to manifest.json.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload of Hugging Face files.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--skip-checksums",
        action="store_true",
        help="Skip SHA256 verification for bundle artifacts.",
    )
    return parser


def load_manifest(bundle_dir: Path) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "model_id" not in manifest:
        raise ValueError("manifest.json must contain `model_id`.")

    return manifest


def normalize_artifacts(manifest: dict[str, Any], bundle_dir: Path) -> list[dict[str, Any]]:
    artifacts = manifest.get("artifacts")
    if artifacts is None:
        so_files = manifest.get("so_files")
        if so_files is not None:
            artifacts = [{"file": file_name} for file_name in so_files]
        else:
            artifacts = [{"file": path.name} for path in sorted(bundle_dir.glob("*.so"))]

    if not artifacts:
        raise ValueError(
            "No bundle artifacts found. Add `artifacts` to manifest.json or place at least one `.so` next to the script."
        )

    normalized: list[dict[str, Any]] = []
    for artifact in artifacts:
        if isinstance(artifact, str):
            artifact = {"file": artifact}
        file_name = artifact.get("file")
        if not file_name:
            raise ValueError("Each artifact entry must contain `file`.")
        artifact_path = bundle_dir / file_name
        if not artifact_path.exists():
            raise FileNotFoundError(f"Bundle artifact not found: {artifact_path}")
        normalized.append(
            {
                "file": file_name,
                "path": artifact_path,
                "sha256": artifact.get("sha256"),
            }
        )
    return normalized


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(artifacts: list[dict[str, Any]], skip_checksums: bool) -> None:
    if skip_checksums:
        return

    for artifact in artifacts:
        expected = artifact.get("sha256")
        if not expected:
            continue
        actual = sha256sum(artifact["path"])
        if actual != expected:
            raise ValueError(
                f"Checksum mismatch for {artifact['file']}: expected {expected}, got {actual}"
            )


def infer_paiton_architecture(
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    explicit_arch = manifest.get("paiton_architecture")
    if explicit_arch:
        return explicit_arch

    architectures = config.get("architectures") or []
    if architectures:
        first_arch = architectures[0]
        if first_arch.startswith("Paiton"):
            return first_arch
        return f"Paiton{first_arch}"

    model_type = config.get("model_type")
    if model_type in MODEL_TYPE_TO_ARCH:
        return MODEL_TYPE_TO_ARCH[model_type]

    raise ValueError(
        "Could not infer Paiton architecture from config.json. "
        "Set `paiton_architecture` in manifest.json."
    )


def patch_config(output_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Downloaded model is missing config.json: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    paiton_arch = infer_paiton_architecture(config, manifest)
    current_architectures = config.get("architectures") or []
    rewritten_architectures = [paiton_arch]
    rewritten_architectures.extend(
        arch for arch in current_architectures if arch != paiton_arch
    )
    config["architectures"] = rewritten_architectures

    if "decode_partition_size" in manifest:
        config["decode_partition_size"] = int(manifest["decode_partition_size"])

    if "rope_scaling" in manifest:
        config["rope_scaling"] = manifest["rope_scaling"]

    config_overrides = manifest.get("config_overrides") or {}
    for key, value in config_overrides.items():
        config[key] = value

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    return config


def copy_artifacts(artifacts: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    copied_paths: list[Path] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        destination = output_dir / artifact["file"]
        shutil.copy2(artifact["path"], destination)
        copied_paths.append(destination)
    return copied_paths


def validate_output(output_dir: Path, copied_paths: list[Path]) -> None:
    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in installed model directory: {config_path}")

    if not copied_paths:
        raise ValueError("No bundle artifacts were copied.")

    missing = [path for path in copied_paths if not path.exists()]
    if missing:
        missing_display = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing installed artifacts: {missing_display}")


def render_serve_command(output_dir: Path, manifest: dict[str, Any]) -> str:
    command = ["vllm", "serve", str(output_dir)]
    kv_cache_dtype = manifest.get("kv_cache_dtype")
    if kv_cache_dtype:
        command.extend(["--kv-cache-dtype", str(kv_cache_dtype)])
    command.extend(["--port", "8000"])
    return " ".join(command)


def main() -> int:
    args = build_parser().parse_args()

    bundle_dir = (
        Path(args.bundle).resolve()
        if args.bundle is not None
        else Path(__file__).resolve().parent
    )
    output_dir = Path(args.output_dir).resolve()

    manifest = load_manifest(bundle_dir)
    artifacts = normalize_artifacts(manifest, bundle_dir)
    verify_artifacts(artifacts, skip_checksums=args.skip_checksums)

    output_dir.mkdir(parents=True, exist_ok=True)
    revision = args.revision or manifest.get("revision")
    allow_patterns = manifest.get("allow_patterns", DEFAULT_ALLOW_PATTERNS)

    print(
        f"Downloading Hugging Face model {manifest['model_id']} "
        f"to {output_dir}..."
    )
    snapshot_download(
        repo_id=manifest["model_id"],
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        token=args.hf_token,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )

    config = patch_config(output_dir, manifest)
    copied_paths = copy_artifacts(artifacts, output_dir)
    validate_output(output_dir, copied_paths)

    print("Installed Paiton bundle successfully.")
    print(f"Model directory: {output_dir}")
    print(f"Paiton architecture: {config['architectures'][0]}")
    if "decode_partition_size" in config:
        print(f"decode_partition_size: {config['decode_partition_size']}")
    print("Installed artifacts:")
    for path in copied_paths:
        print(f"  - {path.name}")
    print("Serve with:")
    print(f"  {render_serve_command(output_dir, manifest)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
