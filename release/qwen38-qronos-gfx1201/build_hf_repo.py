#!/usr/bin/env python3
"""Build a verified local Hugging Face model tree without network access.

This release-specific helper only reads local files.  It never imports a Hub
client, reads authentication state, creates a repository, or uploads content.
The output is assembled in a sibling temporary directory and renamed into
place only after every validation succeeds.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from verify_release import (
    ReleaseVerificationError,
    load_release_manifest,
    sha256_file,
    validate_final_release_manifest,
    verify_bundle,
    verify_model_directory,
    verify_publication_metadata,
    verify_release_source_checkout,
)


RELEASE_ID = "paiton-qwen38-qronos-w4a16-gfx1201-v1"
BASE_REPOSITORY = "amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16"
BASE_REVISION = "649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2"
HF_REPOSITORY = "EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4"
HF_TAG = "paiton-rdna4-v1"

CHECKPOINT_NAME = "model.safetensors"
CHECKPOINT_SIZE = 19_893_384_832
CHECKPOINT_SHA256 = "32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454"

ARTIFACT_NAME = (
    "Qwen3.8-27B-Quark-Qronos-INT4-W4A16_"
    "gfx1201_tp1_mt8192_ctx8192.so"
)
ARTIFACT_MANIFEST_NAME = ARTIFACT_NAME.removesuffix(".so") + ".manifest.json"

EXPECTED_DECLARATIONS = {
    "artifact": {
        "file": ARTIFACT_NAME,
        "size_bytes": 6_699_696,
        "sha256": "b3b24c9341c2d28b849e06f5842cfb607af82bb208dea49ef72bffabbe933761",
    },
    "artifact_manifest": {
        "file": ARTIFACT_MANIFEST_NAME,
        "size_bytes": 789_856,
        "sha256": "174fc432a779a8fade9ce97082ccecef4a019f0202f02e1db164b6b3bb6c901b",
    },
    "config": {
        "file": "config.json",
        "size_bytes": 13_647,
        "sha256": "1974563798648387a8c63130906dd08b1451083e7f273c14116840637511fb85",
    },
}

# Exact unchanged files from the pinned AMD snapshot.  The checkpoint is
# declared separately in the release manifest because hashing it dominates
# the build time.
UPSTREAM_OVERLAY_FILES = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "crc32.txt": "0193833c505e6e4b57530d50ce524f5c8d88120b2dd76257f1ccba14bf26a92d",
    "generation_config.json": "07f857aba5260b2ea2513f80de8062d086661f00eada0a3794964e665ba680f5",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "preprocessor_config.json": "957eb01d1ea45341a92d543daec95857a7cbeff5803834bc0603b27ba7b41b3f",
    "processor_config.json": "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}

UPSTREAM_METADATA_FILES = {
    ".gitattributes": (
        ".gitattributes",
        "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930",
    ),
    "LICENSE": (
        "LICENSES/AMD-Qwen3.8-Apache-2.0.txt",
        "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    ),
    "README.md": (
        "README.amd.md",
        "bdd894b8a3c1e11eb2b23daf568bb9ee640b63df1f0d5cbb5df4b1e965f85f74",
    ),
}

RELEASE_FILES = {
    "MODEL_CARD.md": ("README.md", 0o644),
    "MODIFICATIONS.md": ("MODIFICATIONS.md", 0o644),
    "THIRD_PARTY_NOTICES.md": ("THIRD_PARTY_NOTICES.md", 0o644),
    "paiton-release.spdx.json": ("paiton-release.spdx.json", 0o644),
    "provenance.intoto.jsonl": ("provenance.intoto.jsonl", 0o644),
    "verify_release.py": ("verify_release.py", 0o755),
    "LICENSES/AITER-MIT.txt": ("LICENSES/AITER-MIT.txt", 0o644),
    "LICENSES/Composable-Kernel-MIT.txt": (
        "LICENSES/Composable-Kernel-MIT.txt",
        0o644,
    ),
    "LICENSES/Triton-MIT.txt": ("LICENSES/Triton-MIT.txt", 0o644),
    "LICENSES/flash-linear-attention-MIT.txt": (
        "LICENSES/flash-linear-attention-MIT.txt",
        0o644,
    ),
}

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay-dir",
        type=Path,
        required=True,
        help="Verified complete local overlay containing the checkpoint and release artifact.",
    )
    parser.add_argument(
        "--upstream-metadata-dir",
        type=Path,
        required=True,
        help="Local directory containing the pinned AMD .gitattributes, LICENSE, and README.md.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Approved release manifest; status must be exactly 'released'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New local directory to create. It must not already exist.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("auto", "hardlink", "copy"),
        default="auto",
        help=(
            "Checkpoint placement. 'auto' hardlinks only on the same filesystem "
            "and otherwise copies; default: auto."
        ),
    )
    return parser


def _require_regular_file(path: Path, description: str) -> Path:
    try:
        file_stat = path.lstat()
    except OSError as error:
        raise ReleaseVerificationError(
            f"missing {description}: {path}: {error}"
        ) from error
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
        raise ReleaseVerificationError(
            f"{description} must be a regular file, not a symlink: {path}"
        )
    return path


def _require_directory(path: Path, description: str) -> Path:
    try:
        directory_stat = path.lstat()
    except OSError as error:
        raise ReleaseVerificationError(
            f"missing {description}: {path}: {error}"
        ) from error
    if path.is_symlink() or not stat.S_ISDIR(directory_stat.st_mode):
        raise ReleaseVerificationError(
            f"{description} must be a directory, not a symlink: {path}"
        )
    return path.resolve()


def _expect_equal(actual: object, expected: object, description: str) -> None:
    if actual != expected:
        raise ReleaseVerificationError(
            f"{description} mismatch: expected {expected!r}, got {actual!r}"
        )


def _validate_sha(value: object, description: str, pattern: re.Pattern[str]) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ReleaseVerificationError(f"invalid {description}: {value!r}")


def _validate_released_manifest(manifest_path: Path) -> dict[str, Any]:
    release = load_release_manifest(manifest_path)
    validate_final_release_manifest(release)
    _expect_equal(release.get("release_id"), RELEASE_ID, "release ID")
    _expect_equal(release.get("status"), "released", "release status")

    base_model = release["base_model"]
    _expect_equal(base_model.get("repo_id"), BASE_REPOSITORY, "base repository")
    _expect_equal(base_model.get("revision"), BASE_REVISION, "base revision")
    _expect_equal(
        base_model.get("checkpoint"),
        {
            "file": CHECKPOINT_NAME,
            "size_bytes": CHECKPOINT_SIZE,
            "sha256": CHECKPOINT_SHA256,
        },
        "checkpoint declaration",
    )

    declarations = release["files"]
    _expect_equal(set(declarations), set(EXPECTED_DECLARATIONS), "declared file keys")
    for key, expected in EXPECTED_DECLARATIONS.items():
        _expect_equal(declarations.get(key), expected, f"files.{key}")

    runtime = release["runtime"]
    _expect_equal(runtime.get("gpu_arch"), "gfx1201", "GPU architecture")
    _validate_sha(runtime.get("vllm_revision"), "vLLM revision", GIT_SHA_PATTERN)

    contract = release["contract"]
    for key, expected in (
        ("tensor_parallel_size", 1),
        ("max_batch_size", 1),
        ("max_model_len", 8192),
        ("max_num_batched_tokens", 8192),
        ("scope", "text-only"),
        ("multimodal", False),
        ("speculative_decoding", False),
    ):
        _expect_equal(contract.get(key), expected, f"contract.{key}")

    source = release["source"]
    _validate_sha(
        source.get("artifact_compiler_revision"),
        "artifact compiler revision",
        GIT_SHA_PATTERN,
    )
    _validate_sha(
        source.get("artifact_plugin_revision"),
        "artifact plugin revision",
        GIT_SHA_PATTERN,
    )
    _expect_equal(
        source.get("artifact_built_at"),
        None,
        "unrecorded artifact build timestamp",
    )
    _expect_equal(
        source.get("release_source_ref"),
        f"refs/tags/{RELEASE_ID}",
        "release source ref",
    )
    _validate_sha(
        source.get("release_source_revision"),
        "release source revision",
        GIT_SHA_PATTERN,
    )
    _validate_sha(source.get("packaged_at"), "packaging timestamp", TIMESTAMP_PATTERN)

    publication = release.get("publication")
    if not isinstance(publication, Mapping):
        raise ReleaseVerificationError("released manifest is missing publication metadata")
    _expect_equal(
        publication.get("huggingface_repository"),
        HF_REPOSITORY,
        "Hugging Face repository",
    )
    _expect_equal(
        publication.get("huggingface_revision"), HF_TAG, "Hugging Face tag"
    )
    if "release-candidate" in json.dumps(release, sort_keys=True).lower():
        raise ReleaseVerificationError(
            "released manifest still contains a release-candidate marker"
        )
    return release


def _validate_exact_file(path: Path, expected_sha256: str, description: str) -> None:
    _require_regular_file(path, description)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ReleaseVerificationError(
            f"{description} SHA256 mismatch: expected {expected_sha256}, got {actual}: {path}"
        )


def _copy_regular(source: Path, destination: Path, mode: int) -> None:
    _require_regular_file(source, "copy source")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    if os.path.lexists(destination):
        raise ReleaseVerificationError(f"refusing to overwrite staged file: {destination}")
    shutil.copyfile(source, destination, follow_symlinks=False)
    destination.chmod(mode)


def _place_checkpoint(source: Path, destination: Path, mode: str) -> str:
    source_stat = _require_regular_file(source, "checkpoint").stat()
    destination_device = destination.parent.stat().st_dev
    same_filesystem = source_stat.st_dev == destination_device

    if mode in ("auto", "hardlink") and same_filesystem:
        try:
            os.link(source, destination, follow_symlinks=False)
            return "hardlink"
        except OSError as error:
            if mode == "hardlink" or error.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
                errno.ENOTSUP,
            }:
                raise ReleaseVerificationError(
                    f"could not hardlink checkpoint: {error}"
                ) from error
    elif mode == "hardlink":
        raise ReleaseVerificationError(
            "checkpoint and output are on different filesystems; hardlink is unsafe"
        )

    _copy_regular(source, destination, 0o644)
    return "copy"


def _read_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(
            f"invalid {description}: {path}: {error}"
        ) from error


def _require_text(path: Path, terms: tuple[str, ...], description: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ReleaseVerificationError(f"could not read {description}: {error}") from error
    lowered = text.lower()
    for term in terms:
        if term.lower() not in lowered:
            raise ReleaseVerificationError(
                f"{description} is missing required text {term!r}: {path}"
            )
    for forbidden in ("do_not_publish", "do not publish", "review draft"):
        if forbidden in lowered:
            raise ReleaseVerificationError(
                f"{description} contains forbidden publication marker {forbidden!r}"
            )


def _validate_release_metadata(root: Path, release: Mapping[str, Any]) -> None:
    verify_publication_metadata(root, release)
    _require_text(
        root / "README.md",
        (
            f"base_model: {BASE_REPOSITORY}",
            "license: other",
            "license_name: Apache-2.0 AND MIT",
            "library_name: paiton-vllm-plugin",
            "pipeline_tag: text-generation",
            "executable",
            "gfx1201",
            "text-only",
            CHECKPOINT_SHA256,
            EXPECTED_DECLARATIONS["artifact"]["sha256"],
        ),
        "model card",
    )
    _require_text(
        root / "MODIFICATIONS.md",
        (
            BASE_REPOSITORY,
            BASE_REVISION,
            CHECKPOINT_SHA256,
            str(release["source"]["artifact_compiler_revision"]),
            str(release["source"]["artifact_plugin_revision"]),
        ),
        "modifications record",
    )
    _require_text(
        root / "THIRD_PARTY_NOTICES.md",
        (BASE_REPOSITORY, BASE_REVISION, "LICENSES/Apache-2.0.txt"),
        "third-party notices",
    )

    spdx = _read_json(root / "paiton-release.spdx.json", "SPDX document")
    if not isinstance(spdx, Mapping):
        raise ReleaseVerificationError("SPDX document must be a JSON object")
    _expect_equal(spdx.get("spdxVersion"), "SPDX-2.3", "SPDX version")
    serialized_spdx = json.dumps(spdx, sort_keys=True)
    for expected in (
        RELEASE_ID,
        BASE_REPOSITORY,
        BASE_REVISION,
        EXPECTED_DECLARATIONS["artifact"]["sha256"],
    ):
        if expected not in serialized_spdx:
            raise ReleaseVerificationError(
                f"SPDX document does not bind required release value: {expected}"
            )

    provenance_path = root / "provenance.intoto.jsonl"
    _require_regular_file(provenance_path, "provenance statement")
    statements = []
    line_number = 0
    try:
        for line_number, line in enumerate(
            provenance_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError("statement is not a JSON object")
            statements.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ReleaseVerificationError(
            f"invalid provenance statement at or before line {line_number}: {error}"
        ) from error
    if not statements:
        raise ReleaseVerificationError("provenance file contains no statements")
    serialized_provenance = json.dumps(statements, sort_keys=True)
    for expected in (
        ARTIFACT_NAME,
        EXPECTED_DECLARATIONS["artifact"]["sha256"],
        BASE_REPOSITORY,
        BASE_REVISION,
        CHECKPOINT_SHA256,
        str(release["source"]["artifact_compiler_revision"]),
    ):
        if expected not in serialized_provenance:
            raise ReleaseVerificationError(
                f"provenance does not bind required release value: {expected}"
            )


def _expected_inventory(release: Mapping[str, Any]) -> set[str]:
    inventory = set(UPSTREAM_OVERLAY_FILES)
    inventory.add(CHECKPOINT_NAME)
    inventory.update(declaration["file"] for declaration in release["files"].values())
    inventory.update(destination for destination, _ in UPSTREAM_METADATA_FILES.values())
    inventory.update(destination for destination, _ in RELEASE_FILES.values())
    inventory.update(
        {
            "LICENSE",
            "LICENSES/Apache-2.0.txt",
            "bundle-manifest.json",
            "paiton-release.json",
            "SHA256SUMS",
        }
    )
    return inventory


def _inventory(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ReleaseVerificationError(f"staged tree contains a symlink: {path}")
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ReleaseVerificationError(f"staged tree contains a special file: {path}")
    return files


def _write_checksums(root: Path, checkpoint_digest: str) -> None:
    checksum_path = root / "SHA256SUMS"
    if checksum_path.exists():
        raise ReleaseVerificationError(f"checksum file already exists: {checksum_path}")
    lines = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix()
        digest = checkpoint_digest if relative == CHECKPOINT_NAME else sha256_file(path)
        _validate_sha(digest, f"SHA256 for {relative}", SHA256_PATTERN)
        lines.append(f"{digest}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_path.chmod(0o644)


def _source_fingerprint(path: Path) -> tuple[int, int, int, int, int, int]:
    value = path.stat()
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_nlink,
    )


def _assert_safe_output(
    output_dir: Path,
    source_roots: tuple[Path, ...],
) -> Path:
    output_dir = output_dir.absolute()
    if os.path.lexists(output_dir):
        raise ReleaseVerificationError(
            f"output directory already exists; choose a new path: {output_dir}"
        )
    parent = _require_directory(output_dir.parent, "output parent directory")
    output_dir = parent / output_dir.name
    for source in source_roots:
        try:
            output_dir.relative_to(source)
        except ValueError:
            pass
        else:
            raise ReleaseVerificationError(
                f"output directory must not be inside an input tree: {source}"
            )
        try:
            source.relative_to(output_dir)
        except ValueError:
            pass
        else:
            raise ReleaseVerificationError(
                f"input tree must not be inside the output directory: {source}"
            )
    return output_dir


def _assemble(
    *,
    overlay_dir: Path,
    upstream_metadata_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    checkpoint_mode: str,
) -> tuple[str, str]:
    release_dir = Path(__file__).resolve().parent
    repository_root = release_dir.parents[1]

    overlay_dir = _require_directory(overlay_dir.absolute(), "overlay directory")
    upstream_metadata_dir = _require_directory(
        upstream_metadata_dir.absolute(), "upstream metadata directory"
    )
    manifest_path = _require_regular_file(
        manifest_path.absolute(), "released manifest"
    ).resolve()
    output_dir = _assert_safe_output(
        output_dir,
        (overlay_dir, upstream_metadata_dir, release_dir, repository_root),
    )

    release = _validate_released_manifest(manifest_path)
    verify_release_source_checkout(release, repository_root)
    verify_bundle(overlay_dir, manifest_path=manifest_path)

    for name, expected_sha256 in UPSTREAM_OVERLAY_FILES.items():
        _validate_exact_file(
            overlay_dir / name, expected_sha256, f"pinned upstream file {name}"
        )
    checkpoint_source = _require_regular_file(
        overlay_dir / CHECKPOINT_NAME, "checkpoint"
    )
    _expect_equal(
        checkpoint_source.stat().st_size, CHECKPOINT_SIZE, "checkpoint byte size"
    )
    for name, (_, expected_sha256) in UPSTREAM_METADATA_FILES.items():
        _validate_exact_file(
            upstream_metadata_dir / name,
            expected_sha256,
            f"pinned upstream metadata {name}",
        )

    repository_license = _require_regular_file(
        repository_root / "LICENSE", "repository license"
    )
    for source_relative in RELEASE_FILES:
        _require_regular_file(release_dir / source_relative, "approved release file")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    checkpoint_placement = "unknown"
    checkpoint_fingerprint: tuple[int, int, int, int, int, int] | None = None
    try:
        temporary.chmod(0o755)
        for name in sorted(UPSTREAM_OVERLAY_FILES):
            _copy_regular(overlay_dir / name, temporary / name, 0o644)

        for declaration in release["files"].values():
            name = declaration["file"]
            _copy_regular(overlay_dir / name, temporary / name, 0o644)

        checkpoint_placement = _place_checkpoint(
            checkpoint_source,
            temporary / CHECKPOINT_NAME,
            checkpoint_mode,
        )
        checkpoint_fingerprint = _source_fingerprint(checkpoint_source)

        for source_name, (destination_name, _) in UPSTREAM_METADATA_FILES.items():
            _copy_regular(
                upstream_metadata_dir / source_name,
                temporary / destination_name,
                0o644,
            )

        for source_name, (destination_name, file_mode) in RELEASE_FILES.items():
            _copy_regular(
                release_dir / source_name,
                temporary / destination_name,
                file_mode,
            )

        _copy_regular(repository_license, temporary / "LICENSE", 0o644)
        _copy_regular(
            temporary / "LICENSE", temporary / "LICENSES/Apache-2.0.txt", 0o644
        )
        _copy_regular(manifest_path, temporary / "bundle-manifest.json", 0o644)
        _copy_regular(
            temporary / "bundle-manifest.json",
            temporary / "paiton-release.json",
            0o644,
        )

        for name, expected_sha256 in UPSTREAM_OVERLAY_FILES.items():
            _validate_exact_file(
                temporary / name,
                expected_sha256,
                f"staged pinned upstream file {name}",
            )
        for _, (destination_name, expected_sha256) in UPSTREAM_METADATA_FILES.items():
            _validate_exact_file(
                temporary / destination_name,
                expected_sha256,
                f"staged pinned upstream metadata {destination_name}",
            )
        staged_release = _validate_released_manifest(
            temporary / "bundle-manifest.json"
        )
        _expect_equal(staged_release, release, "staged release manifest")
        if (temporary / "bundle-manifest.json").read_bytes() != (
            temporary / "paiton-release.json"
        ).read_bytes():
            raise ReleaseVerificationError(
                "bundle-manifest.json and paiton-release.json must be byte-identical"
            )

        _validate_release_metadata(temporary, release)
        verify_bundle(temporary)
        verify_model_directory(
            temporary,
            release,
            verify_checkpoint_sha256=True,
        )
        _write_checksums(temporary, CHECKPOINT_SHA256)

        actual_inventory = _inventory(temporary)
        expected_inventory = _expected_inventory(release)
        if actual_inventory != expected_inventory:
            missing = sorted(expected_inventory - actual_inventory)
            unexpected = sorted(actual_inventory - expected_inventory)
            raise ReleaseVerificationError(
                f"staged inventory mismatch; missing={missing}, unexpected={unexpected}"
            )

        if checkpoint_fingerprint != _source_fingerprint(checkpoint_source):
            raise ReleaseVerificationError(
                "checkpoint source changed while the staging tree was assembled"
            )
        staged_checkpoint = temporary / CHECKPOINT_NAME
        if checkpoint_placement == "hardlink":
            source_stat = checkpoint_source.stat()
            staged_stat = staged_checkpoint.stat()
            if (source_stat.st_dev, source_stat.st_ino) != (
                staged_stat.st_dev,
                staged_stat.st_ino,
            ):
                raise ReleaseVerificationError(
                    "checkpoint was expected to be a hardlink but inode identity differs"
                )

        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return checkpoint_placement, CHECKPOINT_SHA256


def main() -> int:
    args = build_parser().parse_args()
    try:
        checkpoint_placement, checkpoint_digest = _assemble(
            overlay_dir=args.overlay_dir,
            upstream_metadata_dir=args.upstream_metadata_dir,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            checkpoint_mode=args.checkpoint_mode,
        )
    except (ReleaseVerificationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.absolute()
    print(f"Assembled verified local Hugging Face tree: {output_dir}")
    print(f"Checkpoint placement: {checkpoint_placement}")
    print(f"Checkpoint SHA256: {checkpoint_digest}")
    print(f"Checksum manifest: {output_dir / 'SHA256SUMS'}")
    if checkpoint_placement == "hardlink":
        print(
            "Checkpoint storage is shared with the overlay; keep both paths "
            "read-only and run `sha256sum -c SHA256SUMS` immediately before use."
        )
    print("No network, authentication, repository, or upload operation was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
