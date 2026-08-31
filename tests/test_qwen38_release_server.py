import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paiton_vllm_plugin import qwen38_release_server as release_server
from paiton_vllm_plugin.qwen38_release_server import (
    BASE_MODEL,
    BASE_REVISION,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    RELEASE_METADATA_NAME,
    ReleaseModelError,
    build_server_command,
)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def base_runtime_hashes(base: Path) -> dict[str, str]:
    return {
        name: sha256((base / name).read_bytes())
        for name in release_server.BASE_RUNTIME_FILES
        if name != release_server.CHECKPOINT_NAME
    }


class Qwen38ReleaseServerTests(unittest.TestCase):
    def build_reuse_fixture(self, root: Path, *, status: str = "released"):
        base = root / "base"
        overlay = root / "overlay"
        runtime = root / "runtime"
        base.mkdir()
        overlay.mkdir()
        runtime.mkdir()

        checkpoint_payload = b"test-checkpoint"
        for name in release_server.BASE_RUNTIME_FILES:
            payload = (
                checkpoint_payload
                if name == release_server.CHECKPOINT_NAME
                else name.encode()
            )
            (base / name).write_bytes(payload)

        payloads = {
            "config.json": b"test-config",
            release_server.ARTIFACT_NAME: b"test-artifact",
            release_server.ARTIFACT_MANIFEST_NAME: b"test-artifact-manifest",
        }
        for name, payload in payloads.items():
            (overlay / name).write_bytes(payload)
        declarations = {
            "artifact": {
                "file": release_server.ARTIFACT_NAME,
                "size_bytes": len(payloads[release_server.ARTIFACT_NAME]),
                "sha256": sha256(payloads[release_server.ARTIFACT_NAME]),
            },
            "artifact_manifest": {
                "file": release_server.ARTIFACT_MANIFEST_NAME,
                "size_bytes": len(payloads[release_server.ARTIFACT_MANIFEST_NAME]),
                "sha256": sha256(
                    payloads[release_server.ARTIFACT_MANIFEST_NAME]
                ),
            },
            "config": {
                "file": "config.json",
                "size_bytes": len(payloads["config.json"]),
                "sha256": sha256(payloads["config.json"]),
            },
        }
        release = {
            "schema_version": 1,
            "release_id": release_server.RELEASE_ID,
            "status": status,
            "base_model": {
                "repo_id": BASE_MODEL,
                "revision": BASE_REVISION,
                "checkpoint": {
                    "file": release_server.CHECKPOINT_NAME,
                    "size_bytes": len(checkpoint_payload),
                    "sha256": sha256(checkpoint_payload),
                },
            },
            "files": declarations,
            "publication": {
                "huggingface_repository": DEFAULT_MODEL,
                "huggingface_revision": DEFAULT_REVISION,
            },
        }
        write_json(overlay / RELEASE_METADATA_NAME, release)
        return base, overlay, runtime, checkpoint_payload, declarations, release

    def test_default_command_is_complete_and_pinned(self):
        with patch.dict(
            os.environ, {"PAITON_REUSE_HF_CACHE": "0"}, clear=True
        ), patch.object(release_server, "_download_overlay_files") as preflight:
            command = build_server_command()
        preflight.assert_called_once_with()
        self.assertEqual(command[:3], ["vllm", "serve", DEFAULT_MODEL])
        self.assertIn(DEFAULT_REVISION, command)
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "1")
        self.assertEqual(command[command.index("--max-model-len") + 1], "8192")
        self.assertEqual(
            command[command.index("--kv-cache-memory-bytes") + 1], "2G"
        )
        self.assertIn("--no-enable-prefix-caching", command)

    def test_local_model_omits_remote_revision_and_allows_extra_args(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = str(Path(temporary))
            with patch.dict(
                os.environ,
                {"PAITON_MODEL": model, "PAITON_MODEL_REVISION": "ignored"},
                clear=True,
            ), patch.object(release_server, "_resolve_reused_model") as reuse:
                command = build_server_command(["--port", "9000"])
        reuse.assert_not_called()
        self.assertEqual(command[:3], ["vllm", "serve", model])
        self.assertNotIn("--revision", command)
        self.assertEqual(command[-2:], ["--port", "9000"])

    def test_explicit_remote_model_preserves_revision_and_skips_reuse(self):
        with patch.dict(
            os.environ,
            {"PAITON_MODEL": "example/model", "PAITON_MODEL_REVISION": "abc123"},
            clear=True,
        ), patch.object(release_server, "_resolve_reused_model") as reuse:
            command = build_server_command()
        reuse.assert_not_called()
        self.assertEqual(command[:3], ["vllm", "serve", "example/model"])
        self.assertEqual(command[3:5], ["--revision", "abc123"])

    def test_cache_reuse_can_be_disabled(self):
        with patch.dict(
            os.environ, {"PAITON_REUSE_HF_CACHE": "0"}, clear=True
        ), patch.object(release_server, "_snapshot_download") as snapshot, patch.object(
            release_server, "_download_overlay_files"
        ) as preflight:
            command = build_server_command()
        snapshot.assert_not_called()
        preflight.assert_called_once_with()
        self.assertEqual(command[:3], ["vllm", "serve", DEFAULT_MODEL])

    def test_incomplete_cached_snapshot_falls_back_to_full_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {}, clear=True), patch.object(
                release_server, "_snapshot_download", return_value=temporary
            ) as snapshot, patch.object(
                release_server, "_download_overlay_files"
            ) as preflight:
                command = build_server_command()
        snapshot.assert_called_once_with(
            repo_id=BASE_MODEL,
            repo_type="model",
            revision=BASE_REVISION,
            local_files_only=True,
        )
        preflight.assert_called_once_with()
        self.assertEqual(command[:3], ["vllm", "serve", DEFAULT_MODEL])
        self.assertIn(DEFAULT_REVISION, command)

    def test_read_only_host_cache_can_be_separate_from_download_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(
                os.environ,
                {"PAITON_BASE_HF_CACHE": "/models/base-cache"},
                clear=True,
            ), patch.object(
                release_server, "_snapshot_download", return_value=temporary
            ) as snapshot, patch.object(
                release_server, "_download_overlay_files"
            ) as preflight:
                command = build_server_command()
        snapshot.assert_called_once_with(
            repo_id=BASE_MODEL,
            repo_type="model",
            revision=BASE_REVISION,
            local_files_only=True,
            cache_dir="/models/base-cache",
        )
        preflight.assert_called_once_with()
        self.assertEqual(command[:3], ["vllm", "serve", DEFAULT_MODEL])

    def test_complete_cached_snapshot_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, overlay, _, checkpoint, _, _ = self.build_reuse_fixture(root)
            reused = root / "reused"
            reused.mkdir()
            overlay_files = {"config.json": overlay / "config.json"}
            with patch.dict(os.environ, {}, clear=True), patch.object(
                release_server, "CHECKPOINT_SIZE", len(checkpoint)
            ), patch.object(
                release_server, "CHECKPOINT_SHA256", sha256(checkpoint)
            ), patch.object(
                release_server, "BASE_RUNTIME_SHA256", base_runtime_hashes(base)
            ), patch.object(
                release_server, "_snapshot_download", return_value=str(base)
            ) as snapshot, patch.object(
                release_server,
                "_download_overlay_files",
                return_value=overlay_files,
            ) as overlay_download, patch.object(
                release_server,
                "_build_reused_model_dir",
                return_value=reused,
            ) as build_reused:
                command = build_server_command()

        snapshot.assert_called_once_with(
            repo_id=BASE_MODEL,
            repo_type="model",
            revision=BASE_REVISION,
            local_files_only=True,
        )
        overlay_download.assert_called_once_with()
        build_reused.assert_called_once()
        validated_base, actual_overlay = build_reused.call_args.args
        self.assertEqual(validated_base.directory, base)
        self.assertEqual(actual_overlay, overlay_files)
        self.assertEqual(command[:3], ["vllm", "serve", str(reused)])
        self.assertNotIn("--revision", command)

    def test_explicit_remote_base_uses_pinned_base_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _, _, checkpoint, _, _ = self.build_reuse_fixture(root)
            with patch.dict(
                os.environ, {"PAITON_BASE_MODEL": "example/base"}, clear=True
            ), patch.object(
                release_server, "CHECKPOINT_SIZE", len(checkpoint)
            ), patch.object(
                release_server, "CHECKPOINT_SHA256", sha256(checkpoint)
            ), patch.object(
                release_server, "BASE_RUNTIME_SHA256", base_runtime_hashes(base)
            ), patch.object(
                release_server, "_snapshot_download", return_value=str(base)
            ) as snapshot:
                resolved = release_server._resolve_base_model()
        snapshot.assert_called_once_with(
            repo_id="example/base",
            repo_type="model",
            revision=BASE_REVISION,
        )
        self.assertEqual(resolved.directory, base)

    def test_explicit_base_rejects_same_size_wrong_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, _, _, checkpoint, _, _ = self.build_reuse_fixture(root)
            (base / release_server.CHECKPOINT_NAME).write_bytes(
                b"X" + checkpoint[1:]
            )
            with patch.dict(
                os.environ, {"PAITON_BASE_MODEL": str(base)}, clear=True
            ), patch.object(
                release_server, "CHECKPOINT_SIZE", len(checkpoint)
            ), patch.object(
                release_server, "CHECKPOINT_SHA256", sha256(checkpoint)
            ), patch.object(
                release_server, "BASE_RUNTIME_SHA256", base_runtime_hashes(base)
            ):
                with self.assertRaisesRegex(ReleaseModelError, "checkpoint SHA256"):
                    release_server._resolve_base_model()

    def test_reuse_rejects_base_file_changed_after_hashing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, overlay, _, checkpoint, declarations, _ = (
                self.build_reuse_fixture(root)
            )
            overlay_files = {
                path.name: path
                for path in overlay.iterdir()
                if path.is_file()
            }
            with patch.dict(os.environ, {}, clear=True), patch.multiple(
                release_server,
                CHECKPOINT_SIZE=len(checkpoint),
                CHECKPOINT_SHA256=sha256(checkpoint),
                BASE_RUNTIME_SHA256=base_runtime_hashes(base),
                EXPECTED_OVERLAY_FILES=declarations,
            ):
                validated = release_server._validate_base_model_dir(
                    base, explicit=True
                )
                changed = base / "tokenizer_config.json"
                changed.write_bytes(changed.read_bytes() + b"changed")
                with self.assertRaisesRegex(
                    ReleaseModelError, "changed before assembly"
                ):
                    release_server._build_reused_model_dir(
                        validated, overlay_files
                    )

    def test_explicit_local_base_builds_atomic_linked_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, overlay, runtime, checkpoint, declarations, _ = (
                self.build_reuse_fixture(root)
            )
            download_calls = []

            def fake_download(**kwargs):
                download_calls.append(kwargs)
                return str(overlay / kwargs["filename"])

            real_mkdtemp = tempfile.mkdtemp

            def runtime_mkdtemp(*, prefix):
                return real_mkdtemp(prefix=prefix, dir=runtime)

            with patch.dict(
                os.environ,
                {
                    "PAITON_BASE_MODEL": str(base),
                    "PAITON_REUSE_HF_CACHE": "0",
                    "PAITON_MODEL_REVISION": "0123456789abcdef0123456789abcdef01234567",
                },
                clear=True,
            ), patch.multiple(
                release_server,
                CHECKPOINT_SIZE=len(checkpoint),
                CHECKPOINT_SHA256=sha256(checkpoint),
                BASE_RUNTIME_SHA256=base_runtime_hashes(base),
                EXPECTED_OVERLAY_FILES=declarations,
            ), patch.object(
                release_server, "_hf_hub_download", side_effect=fake_download
            ), patch.object(
                release_server.tempfile, "mkdtemp", side_effect=runtime_mkdtemp
            ):
                command = build_server_command()

            model = Path(command[2])
            self.assertTrue(model.is_dir())
            self.assertNotIn("--revision", command)
            self.assertEqual(
                {path.name for path in model.iterdir()},
                set(release_server.BASE_RUNTIME_FILES)
                | {RELEASE_METADATA_NAME}
                | {value["file"] for value in declarations.values()},
            )
            self.assertEqual(
                (model / release_server.CHECKPOINT_NAME).stat().st_ino,
                (base / release_server.CHECKPOINT_NAME).stat().st_ino,
            )
            self.assertEqual(
                [item["filename"] for item in download_calls],
                [
                    RELEASE_METADATA_NAME,
                    declarations["config"]["file"],
                    declarations["artifact"]["file"],
                    declarations["artifact_manifest"]["file"],
                ],
            )
            for item in download_calls:
                self.assertEqual(item["repo_id"], DEFAULT_MODEL)
                self.assertEqual(
                    item["revision"],
                    "0123456789abcdef0123456789abcdef01234567",
                )
                self.assertEqual(item["repo_type"], "model")

    def test_release_metadata_rejects_candidate_and_unsafe_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, overlay, _, checkpoint, declarations, release = (
                self.build_reuse_fixture(root, status="release-candidate")
            )
            with patch.multiple(
                release_server,
                CHECKPOINT_SIZE=len(checkpoint),
                CHECKPOINT_SHA256=sha256(checkpoint),
                EXPECTED_OVERLAY_FILES=declarations,
            ):
                with self.assertRaisesRegex(ReleaseModelError, "status"):
                    release_server._validate_release_metadata(
                        overlay / RELEASE_METADATA_NAME
                    )

                release["status"] = "released"
                release["files"]["artifact"]["file"] = "../artifact.so"
                write_json(overlay / RELEASE_METADATA_NAME, release)
                with self.assertRaisesRegex(ReleaseModelError, "unsafe"):
                    release_server._validate_release_metadata(
                        overlay / RELEASE_METADATA_NAME
                    )

    def test_overlay_download_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, overlay, _, checkpoint, declarations, _ = self.build_reuse_fixture(root)
            config = overlay / declarations["config"]["file"]
            config.write_bytes(b"X" + config.read_bytes()[1:])

            def fake_download(**kwargs):
                return str(overlay / kwargs["filename"])

            with patch.multiple(
                release_server,
                CHECKPOINT_SIZE=len(checkpoint),
                CHECKPOINT_SHA256=sha256(checkpoint),
                EXPECTED_OVERLAY_FILES=declarations,
            ), patch.object(
                release_server, "_hf_hub_download", side_effect=fake_download
            ):
                with self.assertRaisesRegex(ReleaseModelError, "SHA256 mismatch"):
                    release_server._download_overlay_files()


if __name__ == "__main__":
    unittest.main()
