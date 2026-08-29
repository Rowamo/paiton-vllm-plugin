import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paiton_vllm_plugin.artifact_manifest import ArtifactCompatibilityError
from paiton_vllm_plugin.runtime.core.utils.lib_wrapper import MemLoader


class FakeFunction:
    def __init__(self, value):
        self.value = value
        self.restype = None

    def __call__(self):
        return self.value


class FakeLibrary:
    _handle = 1

    def __init__(self, abi=1, arch=b"gfx1201", wave=32):
        self.PaitonArtifactGetBinaryAbiVersion = FakeFunction(abi)
        self.PaitonArtifactGetTargetArch = FakeFunction(arch)
        self.PaitonArtifactGetWaveSize = FakeFunction(wave)


def write_manifest(artifact: Path):
    payload = artifact.read_bytes()
    artifact.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "binary_abi_version": 1,
                "manifest_version": 1,
                "capability_version": 1,
                "artifact": {
                    "filename": artifact.name,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                "target": {"arch": "gfx1201", "wave_size": 32},
                "parallelism": {"tp_size": 1},
            }
        ),
        encoding="utf-8",
    )


class BinaryDescriptorTests(unittest.TestCase):
    @patch("paiton_vllm_plugin.runtime.core.utils.lib_wrapper.ctypes.cdll.LoadLibrary")
    @patch(
        "paiton_vllm_plugin.runtime.core.utils.lib_wrapper.detect_runtime_gpu_arch",
        return_value="gfx1201",
    )
    def test_rdna4_missing_manifest_is_rejected_before_dlopen(
        self, _detect_arch, load_library
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "legacy_tp1.so"
            artifact.touch()
            with self.assertRaisesRegex(ArtifactCompatibilityError, "without manifest"):
                MemLoader(str(artifact))
        load_library.assert_not_called()

    @patch("paiton_vllm_plugin.runtime.core.utils.lib_wrapper.ctypes.cdll.LoadLibrary")
    @patch("paiton_vllm_plugin.runtime.core.utils.lib_wrapper.detect_runtime_gpu_arch", return_value="gfx1201")
    def test_binary_descriptor_matches_manifest(self, _detect_arch, load_library):
        load_library.return_value = FakeLibrary()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "model_gfx1201_tp1.so"
            artifact.touch()
            write_manifest(artifact)
            loader = MemLoader(str(artifact))
            self.assertTrue(loader.is_open)

    @patch("paiton_vllm_plugin.runtime.core.utils.lib_wrapper._dlclose")
    @patch("paiton_vllm_plugin.runtime.core.utils.lib_wrapper.ctypes.cdll.LoadLibrary")
    @patch("paiton_vllm_plugin.runtime.core.utils.lib_wrapper.detect_runtime_gpu_arch", return_value="gfx1201")
    def test_binary_descriptor_mismatch_closes_and_rejects(self, _detect_arch, load_library, dlclose):
        load_library.return_value = FakeLibrary(arch=b"gfx950", wave=64)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "model_gfx1201_tp1.so"
            artifact.touch()
            write_manifest(artifact)
            with self.assertRaises(ArtifactCompatibilityError):
                MemLoader(str(artifact))
        dlclose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
