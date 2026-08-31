#!/usr/bin/env python3
"""Verify the Paiton Qwen3.8 RDNA4 release before loading its shared library."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


RELEASE_ID = "paiton-qwen38-qronos-w4a16-gfx1201-v1"
RELEASE_SOURCE_REF = f"refs/tags/{RELEASE_ID}"
PUBLICATION = {
    "github_repository": "Eliovp-BV/paiton-vllm-plugin",
    "huggingface_repository": (
        "EliovpAI/Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4"
    ),
    "huggingface_revision": "paiton-rdna4-v1",
    "runtime_image": (
        "ghcr.io/eliovp-bv/paiton-vllm-plugin:qwen38-qronos-rdna4-v1"
    ),
}
PUBLICATION_METADATA_SHA256 = {
    "paiton-release.spdx.json": (
        "fa38316d1e115e73640037de8d62e45d569b6b26220a3566f19fe388ee36bafa"
    ),
    "provenance.intoto.jsonl": (
        "78311374c74e04b05e62bdd09a534ba096f60c0ae55a3f8b2d6b38b4594e2de5"
    ),
}
CANONICAL_RELEASE_CLAIMS_SHA256 = (
    "95e0eb75db330518c8567ed7af8e35eb6be809e22c712e0fcb497e12e9f8f6d7"
)
RETAINED_LICENSE_SHA256 = {
    "Apache-2.0.txt": (
        "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
    ),
    "AITER-MIT.txt": (
        "0baca809927cad448401b252ce3bae12687c2987597ad7088cf3423f872d5bcd"
    ),
    "Composable-Kernel-MIT.txt": (
        "20f3b83dfda01bd18d285e0edbb9d044be6dde04cdf76a1ec18d9fc7127936bf"
    ),
    "Triton-MIT.txt": (
        "92640fb97222fd0a698ff28ce0c3782c172623f8d6c609b557636a80f28fb946"
    ),
    "flash-linear-attention-MIT.txt": (
        "1350bfbef13ce4d3d3bdaa2f1fc4b1d1117d846732a2a62f891a67f3d5356d0d"
    ),
}
GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
UTC_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


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


def validate_final_release_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject a release manifest that is not bound to final public inputs."""
    normalized = json.loads(json.dumps(manifest))
    if not isinstance(normalized.get("source"), dict):
        raise ReleaseVerificationError("released manifest is missing source")
    normalized["status"] = "release-candidate"
    normalized["source"]["release_source_revision"] = None
    normalized["source"]["packaged_at"] = None
    claims_digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    _require_equal(
        claims_digest,
        CANONICAL_RELEASE_CLAIMS_SHA256,
        "canonical release claims SHA256",
    )
    _require_equal(manifest.get("release_id"), RELEASE_ID, "release ID")
    _require_equal(manifest.get("status"), "released", "release status")

    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise ReleaseVerificationError("released manifest is missing source")
    _require_equal(
        source.get("artifact_built_at"),
        None,
        "unrecorded artifact build timestamp",
    )
    _require_equal(
        source.get("release_source_ref"), RELEASE_SOURCE_REF, "release source ref"
    )
    release_revision = source.get("release_source_revision")
    if (
        not isinstance(release_revision, str)
        or GIT_COMMIT_PATTERN.fullmatch(release_revision) is None
    ):
        raise ReleaseVerificationError(
            f"invalid release source revision: {release_revision!r}"
        )
    packaged_at = source.get("packaged_at")
    if (
        not isinstance(packaged_at, str)
        or UTC_TIMESTAMP_PATTERN.fullmatch(packaged_at) is None
    ):
        raise ReleaseVerificationError(
            f"invalid packaging timestamp: {packaged_at!r}"
        )
    try:
        datetime.strptime(packaged_at, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ReleaseVerificationError(
            f"invalid packaging timestamp: {packaged_at!r}"
        ) from error

    publication = manifest.get("publication")
    if publication != PUBLICATION:
        raise ReleaseVerificationError(
            f"publication metadata mismatch: expected {PUBLICATION!r}, "
            f"got {publication!r}"
        )
    if "release-candidate" in json.dumps(manifest, sort_keys=True).lower():
        raise ReleaseVerificationError(
            "released manifest still contains a release-candidate marker"
        )


def verify_release_source_checkout(
    manifest: Mapping[str, Any], repository_root: Path
) -> None:
    """Bind final assets to a clean checkout and the declared release tag."""
    validate_final_release_manifest(manifest)
    repository_root = repository_root.resolve()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        head = git("rev-parse", "HEAD")
        tagged = git("rev-parse", f"{RELEASE_SOURCE_REF}^{{commit}}")
        dirty = git("status", "--porcelain", "--untracked-files=all")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ReleaseVerificationError(
            f"could not verify release source checkout: {error}"
        ) from error
    expected = manifest["source"]["release_source_revision"]
    _require_equal(head, expected, "release checkout revision")
    _require_equal(tagged, expected, "release tag revision")
    if dirty:
        raise ReleaseVerificationError(
            "release source checkout is dirty; build from the exact clean tag"
        )


def verify_publication_metadata(
    metadata_dir: Path, manifest: Mapping[str, Any]
) -> None:
    """Verify the exact schema-audited SPDX and in-toto release records."""
    validate_final_release_manifest(manifest)
    metadata_dir = metadata_dir.resolve()
    for name, expected_sha256 in PUBLICATION_METADATA_SHA256.items():
        path = _safe_file(metadata_dir, name)
        if not path.is_file():
            raise ReleaseVerificationError(f"missing publication metadata: {path}")
        _require_equal(
            sha256_file(path), expected_sha256, f"publication metadata SHA256 ({name})"
        )
    for name, expected_sha256 in RETAINED_LICENSE_SHA256.items():
        path = metadata_dir / "LICENSES" / name
        if not path.is_file():
            raise ReleaseVerificationError(f"missing retained license: {path}")
        _require_equal(
            sha256_file(path), expected_sha256, f"retained license SHA256 ({name})"
        )
    root_license = metadata_dir / "LICENSE"
    if not root_license.is_file():
        raise ReleaseVerificationError(f"missing root license: {root_license}")
    _require_equal(
        sha256_file(root_license),
        RETAINED_LICENSE_SHA256["Apache-2.0.txt"],
        "root Apache-2.0 license SHA256",
    )

    spdx = _load_json(metadata_dir / "paiton-release.spdx.json")
    _require_equal(spdx.get("spdxVersion"), "SPDX-2.3", "SPDX version")
    _require_equal(spdx.get("dataLicense"), "CC0-1.0", "SPDX data license")
    _require_equal(
        spdx.get("documentDescribes"),
        ["SPDXRef-Package-PaitonArtifact"],
        "SPDX described package",
    )
    creation = spdx.get("creationInfo")
    if not isinstance(creation, Mapping):
        raise ReleaseVerificationError("SPDX document is missing creationInfo")
    _require_equal(
        creation.get("licenseListVersion"), "3.28.0", "SPDX license-list version"
    )
    packages = spdx.get("packages")
    relationships = spdx.get("relationships")
    if not isinstance(packages, list) or not isinstance(relationships, list):
        raise ReleaseVerificationError("SPDX packages/relationships must be arrays")
    package_by_id: dict[str, Mapping[str, Any]] = {}
    for package in packages:
        if not isinstance(package, Mapping):
            raise ReleaseVerificationError("SPDX package must be an object")
        package_id = package.get("SPDXID")
        if not isinstance(package_id, str) or package_id in package_by_id:
            raise ReleaseVerificationError(f"invalid/duplicate SPDXID: {package_id!r}")
        package_by_id[package_id] = package
    artifact = package_by_id.get("SPDXRef-Package-PaitonArtifact")
    if artifact is None:
        raise ReleaseVerificationError("SPDX artifact package is missing")
    _require_equal(
        artifact.get("licenseConcluded"),
        "Apache-2.0 AND MIT",
        "artifact concluded license",
    )
    checksums = artifact.get("checksums")
    _require_equal(
        checksums,
        [
            {
                "algorithm": "SHA256",
                "checksumValue": manifest["files"]["artifact"]["sha256"],
            }
        ],
        "SPDX artifact checksum",
    )
    valid_ids = set(package_by_id) | {"SPDXRef-DOCUMENT"}
    relationship_set: set[tuple[str, str, str]] = set()
    for relationship in relationships:
        if not isinstance(relationship, Mapping):
            raise ReleaseVerificationError("SPDX relationship must be an object")
        source = relationship.get("spdxElementId")
        kind = relationship.get("relationshipType")
        target = relationship.get("relatedSpdxElement")
        if source not in valid_ids or target not in valid_ids or not isinstance(kind, str):
            raise ReleaseVerificationError(
                f"invalid SPDX relationship endpoint/type: {relationship!r}"
            )
        relationship_set.add((source, kind, target))
    for required in (
        (
            "SPDXRef-Package-PaitonArtifact",
            "GENERATED_FROM",
            "SPDXRef-Package-PaitonCompiler",
        ),
        (
            "SPDXRef-Package-PaitonArtifact",
            "GENERATED_FROM",
            "SPDXRef-Package-AMDModel",
        ),
        (
            "SPDXRef-Package-PaitonArtifact",
            "DYNAMIC_LINK",
            "SPDXRef-Package-ROCmRuntime",
        ),
        (
            "SPDXRef-Package-PaitonArtifact",
            "DYNAMIC_LINK",
            "SPDXRef-Package-UbuntuHostRuntime",
        ),
    ):
        if required not in relationship_set:
            raise ReleaseVerificationError(
                f"SPDX document is missing required relationship: {required!r}"
            )

    provenance_path = metadata_dir / "provenance.intoto.jsonl"
    try:
        lines = [
            line
            for line in provenance_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        statements = [json.loads(line) for line in lines]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReleaseVerificationError(f"invalid in-toto provenance: {error}") from error
    if len(statements) != 1 or not isinstance(statements[0], Mapping):
        raise ReleaseVerificationError("provenance must contain exactly one statement")
    statement = statements[0]
    _require_equal(
        statement.get("_type"),
        "https://in-toto.io/Statement/v1",
        "in-toto statement type",
    )
    _require_equal(
        statement.get("predicateType"),
        "https://slsa.dev/provenance/v1",
        "provenance predicate type",
    )
    expected_subjects = {
        declaration["file"]: declaration["sha256"]
        for declaration in manifest["files"].values()
    }
    actual_subjects: dict[str, str] = {}
    subjects = statement.get("subject")
    if not isinstance(subjects, list):
        raise ReleaseVerificationError("provenance subjects must be an array")
    for subject in subjects:
        if not isinstance(subject, Mapping):
            raise ReleaseVerificationError("provenance subject must be an object")
        digest = subject.get("digest")
        if not isinstance(subject.get("name"), str) or not isinstance(digest, Mapping):
            raise ReleaseVerificationError("invalid provenance subject")
        actual_subjects[subject["name"]] = digest.get("sha256")
    _require_equal(actual_subjects, expected_subjects, "provenance subjects")
    serialized = json.dumps(statement, sort_keys=True)
    if '"sha1"' in serialized:
        raise ReleaseVerificationError("provenance uses ambiguous sha1 git revisions")
    for expected in (
        manifest["source"]["artifact_compiler_revision"],
        manifest["source"]["artifact_plugin_revision"],
        manifest["base_model"]["revision"],
        manifest["base_model"]["checkpoint"]["sha256"],
        "3df477638d05169dbe73f4d58e3a13ad3ca7b5da",
        "7.14.60850",
    ):
        if expected not in serialized:
            raise ReleaseVerificationError(
                f"provenance does not bind required input: {expected}"
            )


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
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help=(
            "Allow the tracked internal release-candidate manifest. Never use "
            "this flag to approve or publish executable release bytes."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    release = verify_bundle(args.bundle_dir)
    if release.get("status") == "release-candidate" and args.allow_candidate:
        verification_label = "internal release candidate"
    else:
        validate_final_release_manifest(release)
        verify_publication_metadata(args.bundle_dir, release)
        verification_label = "release"
    if args.model_dir is not None:
        verify_model_directory(
            args.model_dir,
            release,
            verify_checkpoint_sha256=args.verify_checkpoint_sha256,
        )
    print(f"Verified {verification_label}: {release['release_id']}")
    print(f"Base revision: {release['base_model']['revision']}")
    print(f"Target: {release['runtime']['gpu_arch']}")
    if args.model_dir is not None:
        checkpoint_mode = "SHA256" if args.verify_checkpoint_sha256 else "byte size"
        print(f"Verified model directory ({checkpoint_mode} checkpoint check): {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
