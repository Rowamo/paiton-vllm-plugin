import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = REPOSITORY_ROOT / "release" / "qwen38-qronos-gfx1201"


def load_release_module(name: str, filename: str):
    if str(RELEASE_DIR) not in sys.path:
        sys.path.insert(0, str(RELEASE_DIR))
    spec = importlib.util.spec_from_file_location(name, RELEASE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verify_release = load_release_module("qwen38_verify_release", "verify_release.py")
# The publisher scripts import `verify_release` by its script-local name.
sys.modules["verify_release"] = verify_release
build_bundle = load_release_module("qwen38_build_bundle", "build_bundle.py")


class Qwen38ReleasePackagingTests(unittest.TestCase):
    def test_final_manifest_and_metadata_are_strictly_bound(self):
        release = build_bundle._prepare_release_manifest(
            RELEASE_DIR,
            release_source_revision=(
                "0123456789abcdef0123456789abcdef01234567"
            ),
            packaged_at="2026-08-31T12:00:00Z",
        )
        verify_release.validate_final_release_manifest(release)
        with tempfile.TemporaryDirectory() as temporary:
            staged = Path(temporary)
            for name in verify_release.PUBLICATION_METADATA_SHA256:
                shutil.copy2(RELEASE_DIR / name, staged / name)
            (staged / "LICENSES").mkdir()
            for name in verify_release.RETAINED_LICENSE_SHA256:
                if name != "Apache-2.0.txt":
                    shutil.copy2(
                        RELEASE_DIR / "LICENSES" / name,
                        staged / "LICENSES" / name,
                    )
            shutil.copy2(REPOSITORY_ROOT / "LICENSE", staged / "LICENSE")
            shutil.copy2(
                staged / "LICENSE", staged / "LICENSES" / "Apache-2.0.txt"
            )
            verify_release.verify_publication_metadata(staged, release)

    def test_final_manifest_arguments_must_be_paired(self):
        with self.assertRaisesRegex(
            verify_release.ReleaseVerificationError, "must be supplied together"
        ):
            build_bundle._prepare_release_manifest(
                RELEASE_DIR,
                release_source_revision=(
                    "0123456789abcdef0123456789abcdef01234567"
                ),
                packaged_at=None,
            )

    def test_final_manifest_rejects_modified_fixed_claims(self):
        release = build_bundle._prepare_release_manifest(
            RELEASE_DIR,
            release_source_revision=(
                "0123456789abcdef0123456789abcdef01234567"
            ),
            packaged_at="2026-08-31T12:00:00Z",
        )
        changed = json.loads(json.dumps(release))
        changed["runtime"]["vllm_revision"] = "0" * 40
        with self.assertRaisesRegex(
            verify_release.ReleaseVerificationError,
            "canonical release claims SHA256",
        ):
            verify_release.validate_final_release_manifest(changed)

    def test_verifier_cli_rejects_candidate_without_explicit_internal_flag(self):
        candidate = verify_release.load_release_manifest(
            RELEASE_DIR / "bundle-manifest.json"
        )
        with patch.object(
            sys, "argv", ["verify_release.py", "--bundle-dir", str(RELEASE_DIR)]
        ), patch.object(
            verify_release, "verify_bundle", return_value=candidate
        ):
            with self.assertRaisesRegex(
                verify_release.ReleaseVerificationError, "release status"
            ):
                verify_release.main()

        with patch.object(
            sys,
            "argv",
            [
                "verify_release.py",
                "--bundle-dir",
                str(RELEASE_DIR),
                "--allow-candidate",
            ],
        ), patch.object(verify_release, "verify_bundle", return_value=candidate):
            self.assertEqual(verify_release.main(), 0)

    def test_hf_builder_rejects_candidate_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            overlay = root / "overlay"
            metadata = root / "metadata"
            output = root / "output"
            overlay.mkdir()
            metadata.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    str(RELEASE_DIR / "build_hf_repo.py"),
                    "--overlay-dir",
                    str(overlay),
                    "--upstream-metadata-dir",
                    str(metadata),
                    "--manifest",
                    str(RELEASE_DIR / "bundle-manifest.json"),
                    "--output-dir",
                    str(output),
                    "--checkpoint-mode",
                    "copy",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("release status", result.stderr)
            self.assertFalse(output.exists())

    def test_archive_rejects_noncanonical_root_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            wrong_root = Path(temporary) / "caller-selected-name"
            wrong_root.mkdir()
            with self.assertRaisesRegex(
                verify_release.ReleaseVerificationError,
                "public archive root must be",
            ):
                build_bundle._create_archive(wrong_root)

    def test_release_source_must_be_clean_and_tag_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Release Test"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "release-test@example.invalid"],
                cwd=repository,
                check=True,
            )
            tracked = repository / "tracked.txt"
            tracked.write_text("release\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-q", "-m", "release"],
                cwd=repository,
                check=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "tag", verify_release.RELEASE_ID],
                cwd=repository,
                check=True,
            )
            release = build_bundle._prepare_release_manifest(
                RELEASE_DIR,
                release_source_revision=revision,
                packaged_at="2026-08-31T12:00:00Z",
            )
            verify_release.verify_release_source_checkout(release, repository)

            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(
                verify_release.ReleaseVerificationError, "checkout is dirty"
            ):
                verify_release.verify_release_source_checkout(release, repository)

    def test_public_template_copy_ignores_untracked_ignored_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            template = repository / "release" / "example"
            output = repository / "output"
            template.mkdir(parents=True)
            output.mkdir()
            (repository / ".gitignore").write_text(
                ".env\n*.bak\n", encoding="utf-8"
            )
            (template / "README.md").write_text("tracked\n", encoding="utf-8")
            (template / "bundle-manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.name", "Release Test"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.email", "release-test@example.invalid"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                [
                    "git",
                    "add",
                    ".gitignore",
                    "release/example/README.md",
                    "release/example/bundle-manifest.json",
                ],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-q", "-m", "release"],
                cwd=repository,
                check=True,
            )
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (template / ".env").write_text("SECRET=ignored\n", encoding="utf-8")
            (template / "backup.bak").write_text("ignored\n", encoding="utf-8")

            build_bundle._copy_template(
                template,
                output,
                repository_root=repository,
                source_revision=revision,
            )

            self.assertEqual((output / "README.md").read_text(), "tracked\n")
            self.assertFalse((output / "bundle-manifest.json").exists())
            self.assertFalse((output / ".env").exists())
            self.assertFalse((output / "backup.bak").exists())

    def test_retained_license_files_match_audited_bytes(self):
        expected = {
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
        for name, digest in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    verify_release.sha256_file(
                        REPOSITORY_ROOT / "LICENSE"
                        if name == "Apache-2.0.txt"
                        else RELEASE_DIR / "LICENSES" / name
                    ),
                    digest,
                )

    def test_bundle_apache_license_copy_matches_tagged_root_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            build_bundle._copy_repository_license(
                RELEASE_DIR,
                output,
                repository_root=REPOSITORY_ROOT,
                source_revision=revision,
            )
            self.assertEqual(
                (output / "LICENSE").read_bytes(),
                (output / "LICENSES" / "Apache-2.0.txt").read_bytes(),
            )
            self.assertEqual(
                verify_release.sha256_file(output / "LICENSE"),
                verify_release.RETAINED_LICENSE_SHA256["Apache-2.0.txt"],
            )


if __name__ == "__main__":
    unittest.main()
