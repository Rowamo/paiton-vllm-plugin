#!/usr/bin/env python3
"""Assemble and optionally archive the exact Qwen3.8 RDNA4 release bundle."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from verify_release import (
    ReleaseVerificationError,
    load_release_manifest,
    sha256_file,
    verify_bundle,
)


ARCHIVE_NAME = (
    "paiton-qwen38-qronos-w4a16-v1-linux-x86_64-rocm7.14-"
    "gfx1201-tp1-ctx8192.tar.zst"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="Compiler output directory containing the declared .so/config/manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory in which to assemble the bundle.",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Create the deterministic public .tar.zst after license gates pass.",
    )
    return parser


def _copy_template(template_dir: Path, output_dir: Path) -> None:
    for source in sorted(path for path in template_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(template_dir)
        if "__pycache__" in relative.parts or source.suffix in (".pyc", ".pyo"):
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _copy_declared_artifacts(
    artifact_dir: Path,
    output_dir: Path,
    release: dict,
) -> None:
    for declaration in release["files"].values():
        name = declaration["file"]
        source = artifact_dir / name
        if not source.is_file():
            raise ReleaseVerificationError(f"compiler output is missing {source}")
        if source.stat().st_size != declaration["size_bytes"]:
            raise ReleaseVerificationError(f"compiler output size mismatch: {source}")
        if sha256_file(source) != declaration["sha256"]:
            raise ReleaseVerificationError(f"compiler output SHA256 mismatch: {source}")
        shutil.copy2(source, output_dir / name)


def _normalize_modes(output_dir: Path) -> None:
    for path in output_dir.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
        elif path.suffix == ".py":
            path.chmod(0o755)
        else:
            path.chmod(0o644)


def _write_candidate_warning(output_dir: Path) -> None:
    license_path = output_dir / "LICENSE"
    if license_path.is_file():
        return
    (output_dir / "DO_NOT_PUBLISH.txt").write_text(
        "This is an internal release candidate. Paiton has no approved "
        "repository-level LICENSE yet. Complete PUBLICATION_CHECKLIST.md "
        "before distributing this executable.\n",
        encoding="utf-8",
    )


def _write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS"
    lines = []
    for path in sorted(candidate for candidate in output_dir.rglob("*") if candidate.is_file()):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_path.chmod(0o644)


def _assert_publication_material(output_dir: Path) -> None:
    if not (output_dir / "LICENSE").is_file():
        raise ReleaseVerificationError(
            "refusing to create a public archive without an approved LICENSE"
        )
    notices = (output_dir / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    if "review draft" in notices.lower() or "do not distribute" in notices.lower():
        raise ReleaseVerificationError(
            "refusing to create a public archive from draft third-party notices"
        )
    if (output_dir / "DO_NOT_PUBLISH.txt").exists():
        raise ReleaseVerificationError("candidate is explicitly marked DO_NOT_PUBLISH")


def _create_archive(output_dir: Path) -> tuple[Path, Path]:
    archive = output_dir.parent / ARCHIVE_NAME
    if archive.exists():
        raise ReleaseVerificationError(f"archive already exists: {archive}")
    subprocess.run(
        [
            "tar",
            "--sort=name",
            "--mtime=@0",
            "--owner=0",
            "--group=0",
            "--numeric-owner",
            "--mode=u+rwX,go+rX,go-w",
            "--zstd",
            "-cf",
            str(archive),
            "-C",
            str(output_dir.parent),
            output_dir.name,
        ],
        check=True,
    )
    digest_path = archive.with_suffix(archive.suffix + ".sha256")
    digest_path.write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8"
    )
    return archive, digest_path


def main() -> int:
    args = build_parser().parse_args()
    template_dir = Path(__file__).resolve().parent
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.absolute()
    if output_dir.exists():
        raise ReleaseVerificationError(
            f"output directory already exists; choose a new path: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    release = load_release_manifest(template_dir / "bundle-manifest.json")
    _copy_template(template_dir, output_dir)
    _copy_declared_artifacts(artifact_dir, output_dir, release)
    _write_candidate_warning(output_dir)
    _normalize_modes(output_dir)
    _write_checksums(output_dir)
    verify_bundle(output_dir)
    print(f"Assembled and verified release candidate: {output_dir}")
    print(f"Bundle SHA256 list: {output_dir / 'SHA256SUMS'}")
    if args.archive:
        _assert_publication_material(output_dir)
        archive, digest = _create_archive(output_dir)
        print(f"Public archive: {archive}")
        print(f"Archive checksum: {digest}")
    else:
        print("No public archive created; complete the license/publication gates first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
