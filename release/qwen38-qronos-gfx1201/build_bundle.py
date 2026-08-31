#!/usr/bin/env python3
"""Assemble and optionally archive the exact Qwen3.8 RDNA4 release bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from verify_release import (
    ReleaseVerificationError,
    load_release_manifest,
    sha256_file,
    validate_final_release_manifest,
    verify_bundle,
    verify_publication_metadata,
    verify_release_source_checkout,
)


ARCHIVE_NAME = (
    "paiton-qwen38-qronos-w4a16-v1-linux-x86_64-rocm7.14-"
    "gfx1201-tp1-ctx8192.tar.zst"
)
RELEASE_ROOT_NAME = "paiton-qwen38-qronos-w4a16-gfx1201-v1"


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
    parser.add_argument(
        "--release-source-revision",
        help=(
            "Exact clean tagged source commit to inject into final release assets. "
            "Must be supplied together with --packaged-at."
        ),
    )
    parser.add_argument(
        "--packaged-at",
        help=(
            "Actual UTC packaging timestamp (YYYY-MM-DDTHH:MM:SSZ). Must be "
            "supplied together with --release-source-revision."
        ),
    )
    return parser


def _git_output(repository_root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseVerificationError(
            f"could not read exact release source from Git: {error}"
        ) from error


def _tracked_blobs(
    repository_root: Path, revision: str, pathspec: str
) -> list[tuple[str, str]]:
    raw = _git_output(
        repository_root, "ls-tree", "-r", "-z", revision, "--", pathspec
    )
    blobs: list[tuple[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            metadata, raw_path = record.split(b"\t", 1)
            mode, object_type, object_id = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as error:
            raise ReleaseVerificationError(
                f"invalid Git tree record for {pathspec!r}"
            ) from error
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ReleaseVerificationError(
                f"release source contains a non-regular tracked entry: {path}"
            )
        blobs.append((path, object_id))
    if not blobs:
        raise ReleaseVerificationError(
            f"release source has no tracked files for {pathspec!r}"
        )
    return blobs


def _copy_template(
    template_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
    source_revision: str | None,
) -> None:
    if source_revision is not None:
        template_relative = template_dir.relative_to(repository_root).as_posix()
        prefix = f"{template_relative}/"
        for tracked_path, object_id in _tracked_blobs(
            repository_root, source_revision, template_relative
        ):
            if not tracked_path.startswith(prefix):
                raise ReleaseVerificationError(
                    f"tracked release path escaped its prefix: {tracked_path}"
                )
            relative = Path(tracked_path.removeprefix(prefix))
            if relative.as_posix() == "bundle-manifest.json":
                continue
            if "__pycache__" in relative.parts or relative.suffix in (".pyc", ".pyo"):
                continue
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(
                _git_output(repository_root, "cat-file", "blob", object_id)
            )
        return

    for source in sorted(path for path in template_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(template_dir)
        if "__pycache__" in relative.parts or source.suffix in (".pyc", ".pyo"):
            continue
        if relative.as_posix() == "bundle-manifest.json":
            continue
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _prepare_release_manifest(
    template_dir: Path,
    *,
    release_source_revision: str | None,
    packaged_at: str | None,
) -> dict:
    if (release_source_revision is None) != (packaged_at is None):
        raise ReleaseVerificationError(
            "--release-source-revision and --packaged-at must be supplied together"
        )
    release = load_release_manifest(template_dir / "bundle-manifest.json")
    if release_source_revision is None:
        return release
    if release.get("status") != "release-candidate":
        raise ReleaseVerificationError(
            "tracked manifest must remain a release-candidate template"
        )
    release = json.loads(json.dumps(release))
    release["status"] = "released"
    release["source"]["release_source_revision"] = release_source_revision
    release["source"]["packaged_at"] = packaged_at
    validate_final_release_manifest(release)
    return release


def _write_release_manifest(output_dir: Path, release: dict) -> None:
    destination = output_dir / "bundle-manifest.json"
    destination.write_text(
        json.dumps(release, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


def _copy_repository_license(
    template_dir: Path,
    output_dir: Path,
    *,
    repository_root: Path,
    source_revision: str | None,
) -> None:
    repository_license = template_dir.parents[1] / "LICENSE"
    if source_revision is None:
        if not repository_license.is_file():
            raise ReleaseVerificationError(
                f"repository-level license is missing: {repository_license}"
            )
        shutil.copy2(repository_license, output_dir / "LICENSE")
    else:
        entries = _tracked_blobs(repository_root, source_revision, "LICENSE")
        if len(entries) != 1 or entries[0][0] != "LICENSE":
            raise ReleaseVerificationError(
                "release source must contain exactly one tracked root LICENSE"
            )
        (output_dir / "LICENSE").write_bytes(
            _git_output(repository_root, "cat-file", "blob", entries[0][1])
        )
    licenses_dir = output_dir / "LICENSES"
    licenses_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_dir / "LICENSE", licenses_dir / "Apache-2.0.txt")


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


def _write_candidate_warning(output_dir: Path, release: dict) -> None:
    if release.get("status") == "released":
        return
    (output_dir / "DO_NOT_PUBLISH.txt").write_text(
        "This bundle is an internal release candidate. Its manifest is not "
        "marked released. Complete PUBLICATION_CHECKLIST.md before "
        "distributing this executable.\n",
        encoding="utf-8",
    )


def _write_checksums(output_dir: Path) -> None:
    checksum_path = output_dir / "SHA256SUMS"
    lines = []
    for path in sorted(
        candidate for candidate in output_dir.rglob("*") if candidate.is_file()
    ):
        if path == checksum_path:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_path.chmod(0o644)


def _assert_publication_material(output_dir: Path, repository_root: Path) -> None:
    release = load_release_manifest(output_dir / "bundle-manifest.json")
    validate_final_release_manifest(release)
    verify_release_source_checkout(release, repository_root)
    verify_publication_metadata(output_dir, release)
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
    for required in ("paiton-release.spdx.json", "provenance.intoto.jsonl"):
        if not (output_dir / required).is_file():
            raise ReleaseVerificationError(
                f"refusing to publish without required metadata: {required}"
            )


def _create_archive(output_dir: Path) -> tuple[Path, Path]:
    if output_dir.name != RELEASE_ROOT_NAME:
        raise ReleaseVerificationError(
            f"public archive root must be {RELEASE_ROOT_NAME!r}, "
            f"got {output_dir.name!r}"
        )
    archive = output_dir.parent / ARCHIVE_NAME
    digest_path = archive.with_suffix(archive.suffix + ".sha256")
    if os.path.lexists(archive) or os.path.lexists(digest_path):
        raise ReleaseVerificationError(f"archive already exists: {archive}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{archive.name}.", dir=archive.parent
    )
    os.close(descriptor)
    temporary_archive = Path(temporary_name)
    temporary_digest = temporary_archive.with_suffix(".sha256")
    try:
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
                str(temporary_archive),
                "-C",
                str(output_dir.parent),
                RELEASE_ROOT_NAME,
            ],
            check=True,
        )
        temporary_digest.write_text(
            f"{sha256_file(temporary_archive)}  {archive.name}\n",
            encoding="utf-8",
        )
        os.replace(temporary_archive, archive)
        os.replace(temporary_digest, digest_path)
    except BaseException:
        temporary_archive.unlink(missing_ok=True)
        temporary_digest.unlink(missing_ok=True)
        raise
    return archive, digest_path


def main() -> int:
    args = build_parser().parse_args()
    template_dir = Path(__file__).resolve().parent
    repository_root = template_dir.parents[1]
    artifact_dir = args.artifact_dir.resolve()
    output_dir = args.output_dir.absolute()
    if os.path.lexists(output_dir):
        raise ReleaseVerificationError(
            f"output directory already exists; choose a new path: {output_dir}"
        )
    if args.archive and output_dir.name != RELEASE_ROOT_NAME:
        raise ReleaseVerificationError(
            f"--archive requires --output-dir to end in {RELEASE_ROOT_NAME!r}"
        )
    release = _prepare_release_manifest(
        template_dir,
        release_source_revision=args.release_source_revision,
        packaged_at=args.packaged_at,
    )
    if release.get("status") == "released":
        verify_release_source_checkout(release, repository_root)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        source_revision = release["source"].get("release_source_revision")
        _copy_template(
            template_dir,
            temporary,
            repository_root=repository_root,
            source_revision=source_revision,
        )
        _write_release_manifest(temporary, release)
        _copy_repository_license(
            template_dir,
            temporary,
            repository_root=repository_root,
            source_revision=source_revision,
        )
        _copy_declared_artifacts(artifact_dir, temporary, release)
        _write_candidate_warning(temporary, release)
        _normalize_modes(temporary)
        _write_checksums(temporary)
        verify_bundle(temporary)
        if args.archive:
            _assert_publication_material(temporary, repository_root)
        os.replace(temporary, output_dir)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    label = (
        "release" if release.get("status") == "released" else "release candidate"
    )
    print(f"Assembled and verified {label}: {output_dir}")
    print(f"Bundle SHA256 list: {output_dir / 'SHA256SUMS'}")
    if args.archive:
        archive, digest = _create_archive(output_dir)
        print(f"Public archive: {archive}")
        print(f"Archive checksum: {digest}")
    else:
        print("No public archive created; complete the license/publication gates first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
