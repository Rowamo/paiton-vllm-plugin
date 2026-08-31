import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_TOOLS = REPOSITORY_ROOT / "release" / "qwen38-qronos-gfx1201"


def load_verify_module():
    spec = importlib.util.spec_from_file_location(
        "qwen38_release_verify", RELEASE_TOOLS / "verify_release.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VERIFY = load_verify_module()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return payload


def file_declaration(name: str, payload: bytes):
    return {"file": name, "size_bytes": len(payload), "sha256": sha256(payload)}


def build_fixture(root: Path):
    bundle = root / "bundle"
    base = root / "base"
    bundle.mkdir()
    base.mkdir()

    artifact_name = "Qwen3.8-test_gfx1201_tp1_mt8192_ctx8192.so"
    artifact_payload = b"test-paiton-artifact"
    (bundle / artifact_name).write_bytes(artifact_payload)

    base_model = "amd/Qwen3.8-test"
    revision = "a" * 40
    artifact_manifest_name = Path(artifact_name).with_suffix(".manifest.json").name
    artifact_manifest = {
        "manifest_version": 1,
        "binary_abi_version": 1,
        "capability_version": 1,
        "artifact": {
            "filename": artifact_name,
            "size_bytes": len(artifact_payload),
            "sha256": sha256(artifact_payload),
        },
        "target": {"arch": "gfx1201", "family": "rdna4", "wave_size": 32},
        "parallelism": {"tp_size": 1},
        "limits": {"max_context_length": 8192, "max_batched_tokens": 8192},
        "model": {"repository": base_model, "revision": revision, "scope": "text-only"},
        "paiton_qwen38_contract": {
            "version": 3,
            "max_batch_size": 1,
            "quark_group_size": 128,
            "quark_algorithm": "qronos",
            "multimodal": False,
        },
    }
    artifact_manifest_payload = write_json(
        bundle / artifact_manifest_name, artifact_manifest
    )

    generated_config = {
        "architectures": ["PaitonQwen38ForCausalLM", "Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "language_model_only": True,
        "quantization_config": None,
        "paiton_source_quantization_config": {"quant_method": "quark"},
        "paiton_qwen38_contract": {
            "version": 3,
            "scope": "text-only",
            "tp_size": 1,
            "max_context_length": 8192,
        },
    }
    config_payload = write_json(bundle / "config.json", generated_config)

    checkpoint_payload = b"packed-int4-checkpoint"
    (base / "model.safetensors").write_bytes(checkpoint_payload)
    write_json(base / "tokenizer.json", {"version": "test"})
    write_json(
        base / "config.json",
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "quantization_config": {
                "quant_method": "quark",
                "global_quant_config": {
                    "weight": {"dtype": "int4", "group_size": 128}
                },
            },
        },
    )

    release = {
        "schema_version": 1,
        "release_id": "test-release",
        "status": "test",
        "base_model": {
            "repo_id": base_model,
            "revision": revision,
            "checkpoint": file_declaration("model.safetensors", checkpoint_payload),
        },
        "files": {
            "artifact": file_declaration(artifact_name, artifact_payload),
            "artifact_manifest": file_declaration(
                artifact_manifest_name, artifact_manifest_payload
            ),
            "config": file_declaration("config.json", config_payload),
        },
        "runtime": {"gpu_arch": "gfx1201"},
        "contract": {
            "tensor_parallel_size": 1,
            "max_batch_size": 1,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "scope": "text-only",
        },
        "source": {"compiler_revision": "b" * 40, "plugin_revision": "c" * 40},
    }
    write_json(bundle / "bundle-manifest.json", release)
    return bundle, base, release, artifact_name


class Qwen38ReleaseBundleTests(unittest.TestCase):
    def test_verifier_accepts_consistent_bundle_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, _, _, artifact_name = build_fixture(Path(temporary))
            verified = VERIFY.verify_bundle(bundle)
            self.assertEqual(verified["release_id"], "test-release")

            with (bundle / artifact_name).open("ab") as artifact:
                artifact.write(b"tamper")
            with self.assertRaisesRegex(VERIFY.ReleaseVerificationError, "size mismatch"):
                VERIFY.verify_bundle(bundle)

    def test_verifier_rejects_manifest_path_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, _, release, _ = build_fixture(Path(temporary))
            release["files"]["artifact"]["file"] = "../artifact.so"
            write_json(bundle / "bundle-manifest.json", release)
            with self.assertRaisesRegex(VERIFY.ReleaseVerificationError, "unsafe"):
                VERIFY.verify_bundle(bundle)

    def test_installer_hardlinks_base_and_copies_verified_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, base, release, artifact_name = build_fixture(root)
            output = root / "installed"
            result = subprocess.run(
                [
                    sys.executable,
                    str(RELEASE_TOOLS / "install_overlay.py"),
                    "--bundle-dir",
                    str(bundle),
                    "--base-model-dir",
                    str(base),
                    "--output-dir",
                    str(output),
                    "--link-mode",
                    "hardlink",
                    "--verify-checkpoint-sha256",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (base / "model.safetensors").stat().st_ino,
                (output / "model.safetensors").stat().st_ino,
            )
            self.assertNotEqual(
                (bundle / artifact_name).stat().st_ino,
                (output / artifact_name).stat().st_ino,
            )
            installed_config = json.loads((output / "config.json").read_text())
            self.assertEqual(
                installed_config["architectures"][0], "PaitonQwen38ForCausalLM"
            )
            VERIFY.verify_model_directory(
                output, release, verify_checkpoint_sha256=True
            )


if __name__ == "__main__":
    unittest.main()
