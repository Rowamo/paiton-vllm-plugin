import tempfile
import unittest
from pathlib import Path

from paiton_vllm_plugin.models.model_path import resolve_model_so_path


class ResolveModelSoPathTests(unittest.TestCase):
    def test_prefers_exact_chunked_prefill_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "Meta-Llama-3.1-8B-Instruct"
            model_path.mkdir()
            exact = model_path / "Meta-Llama-3.1-8B-Instruct_tp2_mt8192.so"
            fallback = model_path / "Meta-Llama-3.1-8B-Instruct_tp2.so"
            exact.touch()
            fallback.touch()

            resolved = resolve_model_so_path(model_path, tp_size=2, max_input_tokens=8192)

            self.assertEqual(resolved, exact)

    def test_falls_back_to_plain_tp_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "Meta-Llama-3.1-8B-Instruct"
            model_path.mkdir()
            fallback = model_path / "Meta-Llama-3.1-8B-Instruct_tp2.so"
            fallback.touch()

            resolved = resolve_model_so_path(model_path, tp_size=2, max_input_tokens=8192)

            self.assertEqual(resolved, fallback)

    def test_falls_back_to_latest_available_chunked_prefill_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "Meta-Llama-3.1-8B-Instruct"
            model_path.mkdir()
            older = model_path / "Meta-Llama-3.1-8B-Instruct_tp2_mt4096.so"
            newer = model_path / "Meta-Llama-3.1-8B-Instruct_tp2_mt16384.so"
            older.touch()
            newer.touch()

            resolved = resolve_model_so_path(model_path, tp_size=2, max_input_tokens=8192)

            self.assertEqual(resolved, newer)

    def test_prefers_compatible_mt_artifact_over_plain_legacy_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "Meta-Llama-3.1-8B-Instruct"
            model_path.mkdir()
            plain = model_path / "Meta-Llama-3.1-8B-Instruct_tp2.so"
            compatible = model_path / "Meta-Llama-3.1-8B-Instruct_tp2_mt16384.so"
            plain.touch()
            compatible.touch()

            resolved = resolve_model_so_path(model_path, tp_size=2, max_input_tokens=8192)

            self.assertEqual(resolved, compatible)

    def test_raises_for_too_small_chunked_prefill_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "Meta-Llama-3.1-8B-Instruct"
            model_path.mkdir()
            plain = model_path / "Meta-Llama-3.1-8B-Instruct_tp2.so"
            too_small = model_path / "Meta-Llama-3.1-8B-Instruct_tp2_mt4096.so"
            plain.touch()
            too_small.touch()

            with self.assertRaises(FileNotFoundError):
                resolve_model_so_path(model_path, tp_size=2, max_input_tokens=8192)


if __name__ == "__main__":
    unittest.main()
