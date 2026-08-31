"""Zero-configuration server entrypoint for the public RDNA4 Qwen3.8 image."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shlex
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MODEL = (
    "EliovpAI/"
    "Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4"
)
DEFAULT_REVISION = "paiton-rdna4-v1"

BASE_MODEL = "amd/Qwen3.8-27B-Quark-Qronos-INT4-W4A16"
BASE_REVISION = "649ca9d47a7de5364c6fcccc0c1b4f6e542e15e2"

RELEASE_ID = "paiton-qwen38-qronos-w4a16-gfx1201-v1"
RELEASE_METADATA_NAME = "paiton-release.json"
CHECKPOINT_NAME = "model.safetensors"
CHECKPOINT_SIZE = 19_893_384_832
CHECKPOINT_SHA256 = "32190ba51af3e048f927b446f251a171e475cc91456a831e374709e74a8f0454"

ARTIFACT_NAME = (
    "Qwen3.8-27B-Quark-Qronos-INT4-W4A16_"
    "gfx1201_tp1_mt8192_ctx8192.so"
)
ARTIFACT_MANIFEST_NAME = ARTIFACT_NAME.removesuffix(".so") + ".manifest.json"

EXPECTED_OVERLAY_FILES = {
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

BASE_RUNTIME_FILES = (
    "chat_template.jinja",
    "crc32.txt",
    "generation_config.json",
    "merges.txt",
    CHECKPOINT_NAME,
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

BASE_RUNTIME_SHA256 = {
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "crc32.txt": "0193833c505e6e4b57530d50ce524f5c8d88120b2dd76257f1ccba14bf26a92d",
    "generation_config.json": "07f857aba5260b2ea2513f80de8062d086661f00eada0a3794964e665ba680f5",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    "preprocessor_config.json": "957eb01d1ea45341a92d543daec95857a7cbeff5803834bc0603b27ba7b41b3f",
    "processor_config.json": "14932921ca485d458a04dafd8069fbb0a4505622a48208d19ed247115801385b",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
}


class ReleaseModelError(RuntimeError):
    """Raised when a cached-base model cannot be assembled safely."""


@dataclass(frozen=True)
class FileFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    links: int


@dataclass(frozen=True)
class ValidatedBaseModel:
    directory: Path
    fingerprints: Mapping[str, FileFingerprint]


def _snapshot_download(**kwargs: Any) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(**kwargs)


def _hf_hub_download(**kwargs: Any) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(**kwargs)


def _release_revision() -> str:
    """Return the immutable release revision baked into the runtime image."""
    return os.environ.get("PAITON_MODEL_REVISION", DEFAULT_REVISION)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> FileFingerprint:
    value = path.stat()
    return FileFingerprint(
        device=value.st_dev,
        inode=value.st_ino,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
        links=value.st_nlink,
    )


def _regular_file(path: Path, description: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        file_stat = resolved.stat()
    except OSError as error:
        raise ReleaseModelError(f"missing {description}: {path}: {error}") from error
    if not stat.S_ISREG(file_stat.st_mode):
        raise ReleaseModelError(f"{description} is not a regular file: {path}")
    return resolved


def _safe_filename(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseModelError(f"invalid {description}: {value!r}")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value or ".." in relative.parts:
        raise ReleaseModelError(f"unsafe {description}: {value!r}")
    return value


def _validate_release_metadata(path: Path) -> dict[str, Any]:
    metadata_path = _regular_file(path, RELEASE_METADATA_NAME)
    try:
        release = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseModelError(
            f"could not read release metadata {metadata_path}: {error}"
        ) from error
    if not isinstance(release, dict):
        raise ReleaseModelError("release metadata must be a JSON object")
    if release.get("schema_version") != 1:
        raise ReleaseModelError(
            f"unsupported release schema: {release.get('schema_version')!r}"
        )
    if release.get("release_id") != RELEASE_ID:
        raise ReleaseModelError(
            f"release ID mismatch: expected {RELEASE_ID!r}, "
            f"got {release.get('release_id')!r}"
        )
    if release.get("status") != "released":
        raise ReleaseModelError(
            f"release status must be 'released', got {release.get('status')!r}"
        )

    base = release.get("base_model")
    if not isinstance(base, Mapping):
        raise ReleaseModelError("release metadata is missing base_model")
    if base.get("repo_id") != BASE_MODEL or base.get("revision") != BASE_REVISION:
        raise ReleaseModelError("release metadata does not pin the qualified AMD base")
    if base.get("checkpoint") != {
        "file": CHECKPOINT_NAME,
        "size_bytes": CHECKPOINT_SIZE,
        "sha256": CHECKPOINT_SHA256,
    }:
        raise ReleaseModelError("release checkpoint declaration mismatch")

    files = release.get("files")
    if not isinstance(files, Mapping) or set(files) != set(EXPECTED_OVERLAY_FILES):
        raise ReleaseModelError("release metadata must declare exactly three overlay files")
    seen: set[str] = set()
    for key, expected in EXPECTED_OVERLAY_FILES.items():
        declaration = files.get(key)
        if not isinstance(declaration, Mapping):
            raise ReleaseModelError(f"release metadata is missing files.{key}")
        name = _safe_filename(declaration.get("file"), f"files.{key}.file")
        if name in seen:
            raise ReleaseModelError(f"duplicate release filename: {name}")
        seen.add(name)
        if dict(declaration) != expected:
            raise ReleaseModelError(f"release declaration mismatch for files.{key}")

    publication = release.get("publication")
    if not isinstance(publication, Mapping):
        raise ReleaseModelError("release metadata is missing publication")
    if publication.get("huggingface_repository") != DEFAULT_MODEL:
        raise ReleaseModelError("release metadata Hugging Face repository mismatch")
    if publication.get("huggingface_revision") != DEFAULT_REVISION:
        raise ReleaseModelError("release metadata Hugging Face revision mismatch")
    return release


def _verify_declared_file(path: Path, declaration: Mapping[str, Any]) -> Path:
    expected_name = _safe_filename(declaration.get("file"), "release filename")
    resolved = _regular_file(path, expected_name)
    if path.name != expected_name:
        raise ReleaseModelError(
            f"downloaded filename mismatch: expected {expected_name!r}, got {path.name!r}"
        )
    expected_size = declaration.get("size_bytes")
    if type(expected_size) is not int or resolved.stat().st_size != expected_size:
        raise ReleaseModelError(
            f"size mismatch for {expected_name}: expected {expected_size}, "
            f"got {resolved.stat().st_size}"
        )
    expected_sha256 = declaration.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ReleaseModelError(f"invalid SHA256 declaration for {expected_name}")
    actual_sha256 = _sha256_file(resolved)
    if actual_sha256 != expected_sha256:
        raise ReleaseModelError(
            f"SHA256 mismatch for {expected_name}: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    return resolved


def _download_overlay_files() -> dict[str, Path]:
    revision = _release_revision()
    metadata_path = Path(
        _hf_hub_download(
            repo_id=DEFAULT_MODEL,
            filename=RELEASE_METADATA_NAME,
            revision=revision,
            repo_type="model",
        )
    )
    release = _validate_release_metadata(metadata_path)
    resolved = {
        RELEASE_METADATA_NAME: _regular_file(metadata_path, RELEASE_METADATA_NAME)
    }
    for key in ("config", "artifact", "artifact_manifest"):
        declaration = release["files"][key]
        name = declaration["file"]
        downloaded = Path(
            _hf_hub_download(
                repo_id=DEFAULT_MODEL,
                filename=name,
                revision=revision,
                repo_type="model",
            )
        )
        resolved[name] = _verify_declared_file(downloaded, declaration)
    return resolved


def _validate_base_model_dir(
    path: Path, *, explicit: bool
) -> ValidatedBaseModel | None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        if explicit:
            raise ReleaseModelError(
                f"base model does not exist: {path}: {error}"
            ) from error
        return None
    if not resolved.is_dir():
        if explicit:
            raise ReleaseModelError(f"base model is not a directory: {path}")
        return None
    fingerprints: dict[str, FileFingerprint] = {}
    verify_checkpoint = os.environ.get("PAITON_VERIFY_BASE_SHA256", "1") != "0"
    for name in BASE_RUNTIME_FILES:
        candidate = resolved / name
        try:
            source = _regular_file(candidate, f"base model file {name}")
        except ReleaseModelError:
            if explicit:
                raise
            return None
        before = _fingerprint(source)
        if name == CHECKPOINT_NAME:
            if before.size != CHECKPOINT_SIZE:
                if explicit:
                    raise ReleaseModelError(
                        f"checkpoint size mismatch: expected {CHECKPOINT_SIZE}, "
                        f"got {before.size}"
                    )
                return None
            if verify_checkpoint:
                actual_sha256 = _sha256_file(source)
                if actual_sha256 != CHECKPOINT_SHA256:
                    if explicit:
                        raise ReleaseModelError(
                            "base checkpoint SHA256 mismatch: expected "
                            f"{CHECKPOINT_SHA256}, got {actual_sha256}"
                        )
                    return None
        else:
            actual_sha256 = _sha256_file(source)
            expected_sha256 = BASE_RUNTIME_SHA256[name]
            if actual_sha256 != expected_sha256:
                if explicit:
                    raise ReleaseModelError(
                        f"base model SHA256 mismatch for {name}: expected "
                        f"{expected_sha256}, got {actual_sha256}"
                    )
                return None
        after = _fingerprint(source)
        if before != after:
            if explicit:
                raise ReleaseModelError(
                    f"base model file changed while it was verified: {name}"
                )
            return None
        fingerprints[name] = after
    return ValidatedBaseModel(resolved, fingerprints)


def _is_local_cache_miss(error: Exception) -> bool:
    try:
        from huggingface_hub.errors import LocalEntryNotFoundError
    except ImportError:
        return False
    return isinstance(error, LocalEntryNotFoundError)


def _resolve_base_model() -> ValidatedBaseModel | None:
    explicit_base = os.environ.get("PAITON_BASE_MODEL")
    if explicit_base is not None:
        candidate = Path(explicit_base)
        if candidate.exists():
            return _validate_base_model_dir(candidate, explicit=True)
        downloaded = Path(
            _snapshot_download(
                repo_id=explicit_base,
                repo_type="model",
                revision=BASE_REVISION,
            )
        )
        return _validate_base_model_dir(downloaded, explicit=True)

    if os.environ.get("PAITON_REUSE_HF_CACHE", "1") == "0":
        return None
    download_args: dict[str, Any] = {
        "repo_id": BASE_MODEL,
        "repo_type": "model",
        "revision": BASE_REVISION,
        "local_files_only": True,
    }
    base_cache = os.environ.get("PAITON_BASE_HF_CACHE")
    if base_cache:
        download_args["cache_dir"] = base_cache
    try:
        cached = Path(_snapshot_download(**download_args))
    except Exception as error:
        if _is_local_cache_miss(error):
            return None
        raise
    return _validate_base_model_dir(cached, explicit=False)


def _link_file(source: Path, destination: Path) -> str:
    resolved = _regular_file(source, f"link source for {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise ReleaseModelError(f"refusing to overwrite model file: {destination}")
    try:
        os.link(resolved, destination)
        return "hardlink"
    except OSError as error:
        if error.errno not in {
            errno.EXDEV,
            errno.EPERM,
            errno.EACCES,
            errno.EMLINK,
            errno.ENOTSUP,
        }:
            raise ReleaseModelError(
                f"could not link model file {destination.name}: {error}"
            ) from error
    os.symlink(resolved, destination)
    return "symlink"


def _build_reused_model_dir(
    base: ValidatedBaseModel, overlay_files: Mapping[str, Path]
) -> Path:
    runtime_root = Path(tempfile.mkdtemp(prefix="paiton-qwen38-reused-"))
    staging = runtime_root / ".staging"
    final = runtime_root / "model"
    try:
        staging.mkdir(mode=0o755)
        assembled_fingerprints: dict[str, FileFingerprint] = {}
        for name in BASE_RUNTIME_FILES:
            source = _regular_file(
                base.directory / name, f"validated base model file {name}"
            )
            expected = base.fingerprints[name]
            if _fingerprint(source) != expected:
                raise ReleaseModelError(
                    f"base model file changed before assembly: {name}"
                )
            method = _link_file(source, staging / name)
            after = _fingerprint(source)
            if method == "hardlink":
                if (
                    after.device != expected.device
                    or after.inode != expected.inode
                    or after.size != expected.size
                    or after.mtime_ns != expected.mtime_ns
                    or after.links != expected.links + 1
                ):
                    raise ReleaseModelError(
                        f"base model file changed while it was linked: {name}"
                    )
            elif after != expected:
                raise ReleaseModelError(
                    f"base model file changed while it was linked: {name}"
                )
            assembled = _regular_file(staging / name, f"assembled file {name}")
            if _fingerprint(assembled) != after:
                raise ReleaseModelError(
                    f"assembled base model link does not match its source: {name}"
                )
            assembled_fingerprints[name] = after

        expected_overlay_names = {RELEASE_METADATA_NAME} | {
            declaration["file"] for declaration in EXPECTED_OVERLAY_FILES.values()
        }
        if set(overlay_files) != expected_overlay_names:
            raise ReleaseModelError(
                "overlay file set changed before assembly: expected "
                f"{sorted(expected_overlay_names)}, got {sorted(overlay_files)}"
            )
        for name, source in sorted(overlay_files.items()):
            safe_name = _safe_filename(name, "overlay filename")
            _link_file(source, staging / safe_name)

        for name, expected in assembled_fingerprints.items():
            source = _regular_file(
                base.directory / name, f"validated base model file {name}"
            )
            assembled = _regular_file(staging / name, f"assembled file {name}")
            if (
                _fingerprint(source) != expected
                or _fingerprint(assembled) != expected
            ):
                raise ReleaseModelError(
                    f"base model file changed after assembly: {name}"
                )
            if name != CHECKPOINT_NAME:
                actual_sha256 = _sha256_file(assembled)
                if actual_sha256 != BASE_RUNTIME_SHA256[name]:
                    raise ReleaseModelError(
                        f"assembled base model SHA256 mismatch for {name}"
                    )
        checkpoint = _regular_file(
            staging / CHECKPOINT_NAME, "assembled base model checkpoint"
        )
        if checkpoint.stat().st_size != CHECKPOINT_SIZE:
            raise ReleaseModelError("linked checkpoint changed during assembly")

        release = _validate_release_metadata(staging / RELEASE_METADATA_NAME)
        for declaration in release["files"].values():
            _verify_declared_file(staging / declaration["file"], declaration)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(runtime_root, ignore_errors=True)
        raise
    return final


def _resolve_reused_model() -> Path | None:
    base = _resolve_base_model()
    if base is None:
        return None
    overlay_files = _download_overlay_files()
    return _build_reused_model_dir(base, overlay_files)


def _select_model() -> tuple[str, str | None]:
    revision = _release_revision()
    if "PAITON_MODEL" in os.environ:
        return os.environ.get("PAITON_MODEL", DEFAULT_MODEL), revision
    reused = _resolve_reused_model()
    if reused is not None:
        return str(reused), None
    # Preflight the immutable release metadata and executable overlay even when
    # vLLM will download the complete model repository itself. The individual
    # files then come from the same Hub cache during the full snapshot fetch.
    _download_overlay_files()
    return DEFAULT_MODEL, revision


def build_server_command(extra_args: list[str] | None = None) -> list[str]:
    model, revision = _select_model()
    command = ["vllm", "serve", model]
    if revision and not Path(model).exists():
        command.extend(["--revision", revision])
    command.extend(
        [
            "--served-model-name",
            os.environ.get("PAITON_SERVED_MODEL_NAME", "qwen38"),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            "8192",
            "--max-num-seqs",
            "1",
            "--kv-cache-dtype",
            "auto",
            "--kv-cache-memory-bytes",
            "2G",
            "--load-format",
            "safetensors",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("PAITON_PORT", "8000"),
        ]
    )
    command.extend(extra_args or [])
    return command


def main() -> None:
    command = build_server_command(sys.argv[1:])
    print("Starting the pinned Paiton RDNA4 server:", flush=True)
    print("  " + shlex.join(command), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
