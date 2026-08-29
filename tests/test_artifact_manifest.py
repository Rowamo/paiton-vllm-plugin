import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from paiton_vllm_plugin.artifact_manifest import (
    ArtifactCompatibilityError,
    load_and_validate_artifact_manifest,
    manifest_path_for,
)
from paiton_vllm_plugin.models.model_path import resolve_model_so_path


def write_artifact(directory: Path, arch: str, tp_size: int = 1) -> Path:
    artifact = directory / f"Qwen3.8_{arch}_tp{tp_size}_mt8192_ps512.so"
    payload = f"binary-for-{arch}".encode()
    artifact.write_bytes(payload)
    manifest = {
        "manifest_version": 1,
        "binary_abi_version": 1,
        "capability_version": 1,
        "artifact": {
            "filename": artifact.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        },
        "target": {
            "arch": arch,
            "family": "rdna4" if arch.startswith("gfx12") else "cdna4",
            "wave_size": 32 if arch.startswith("gfx12") else 64,
        },
        "parallelism": {"tp_size": tp_size},
    }
    manifest_path_for(artifact).write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


class ArtifactManifestTests(unittest.TestCase):
    def test_resolver_selects_and_validates_matching_rdna4_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            expected = write_artifact(directory, "gfx1201")
            write_artifact(directory, "gfx950")

            result = resolve_model_so_path(
                directory,
                "Qwen3.8",
                1,
                max_input_tokens=8192,
                decode_partition_size=512,
                target_arch="gfx1201:sramecc+",
            )
            self.assertEqual(result, expected)

    def test_rdna4_rejects_legacy_unqualified_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            (directory / "Qwen3.8_tp1_mt8192.so").touch()
            with self.assertRaisesRegex(ArtifactCompatibilityError, "qualified"):
                resolve_model_so_path(
                    directory, "Qwen3.8", 1, target_arch="gfx1201"
                )

    def test_wrong_arch_is_rejected_before_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            write_artifact(directory, "gfx950")
            with self.assertRaises(ArtifactCompatibilityError):
                resolve_model_so_path(
                    directory, "Qwen3.8", 1, target_arch="gfx1201"
                )

    def test_missing_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            artifact = directory / "Qwen3.8_gfx1201_tp1.so"
            artifact.touch()
            with self.assertRaisesRegex(ArtifactCompatibilityError, "mandatory manifest"):
                resolve_model_so_path(
                    directory, "Qwen3.8", 1, target_arch="gfx1201"
                )

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = write_artifact(Path(tmpdir), "gfx1201")
            artifact.write_bytes(b"tampered-binary")
            with self.assertRaises(ArtifactCompatibilityError):
                load_and_validate_artifact_manifest(
                    artifact, expected_arch="gfx1201", expected_tp_size=1
                )


if __name__ == "__main__":
    unittest.main()
