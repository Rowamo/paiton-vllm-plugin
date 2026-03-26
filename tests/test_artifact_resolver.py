import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paiton_vllm_plugin.models.artifact_resolver import resolve_artifact_dir


class ResolveArtifactDirTests(unittest.TestCase):
    def test_returns_local_path_when_model_ref_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "Llama-3.1-8B-Instruct-FP8-KV"
            model_dir.mkdir()

            resolved = resolve_artifact_dir(str(model_dir))

            self.assertEqual(resolved, model_dir)

    @patch("paiton_vllm_plugin.models.artifact_resolver.snapshot_download")
    def test_downloads_so_artifacts_for_remote_repo(self, snapshot_download_mock) -> None:
        snapshot_download_mock.return_value = "/tmp/hf-cache/snapshots/123abc"

        resolved = resolve_artifact_dir(
            "eliovpai/Llama-3.1-8B-Instruct-FP8-KV",
            revision="main",
            token="hf_token",
            download_dir="/tmp/hf-cache",
        )

        self.assertEqual(resolved, Path("/tmp/hf-cache/snapshots/123abc"))
        snapshot_download_mock.assert_called_once_with(
            repo_id="eliovpai/Llama-3.1-8B-Instruct-FP8-KV",
            repo_type="model",
            revision="main",
            token="hf_token",
            cache_dir="/tmp/hf-cache",
            allow_patterns=["*.so"],
        )


if __name__ == "__main__":
    unittest.main()
