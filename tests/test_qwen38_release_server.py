import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paiton_vllm_plugin.qwen38_release_server import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    build_server_command,
)


class Qwen38ReleaseServerTests(unittest.TestCase):
    def test_default_command_is_complete_and_pinned(self):
        with patch.dict(os.environ, {}, clear=True):
            command = build_server_command()
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
            ):
                command = build_server_command(["--port", "9000"])
        self.assertEqual(command[:3], ["vllm", "serve", model])
        self.assertNotIn("--revision", command)
        self.assertEqual(command[-2:], ["--port", "9000"])


if __name__ == "__main__":
    unittest.main()
