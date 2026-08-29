import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paiton_vllm_plugin import paiton_platform_plugin, register_paiton_models


class PlatformRegistrationTests(unittest.TestCase):
    def test_registers_product_facing_qwen38_architecture(self):
        class Registry:
            registered = {}

            @classmethod
            def get_supported_archs(cls):
                return list(cls.registered)

            @classmethod
            def register_model(cls, architecture, model_path):
                cls.registered[architecture] = model_path

        with patch.dict(sys.modules, {"vllm": SimpleNamespace(ModelRegistry=Registry)}):
            register_paiton_models()
        self.assertEqual(
            Registry.registered["PaitonQwen38ForCausalLM"],
            "paiton_vllm_plugin.models.paiton_qwen38:PaitonQwen38ForCausalLM",
        )

    @patch.dict(os.environ, {"PAITON_GPU_ARCH": "gfx1201"}, clear=False)
    def test_enables_platform_for_gfx1201(self):
        self.assertEqual(
            paiton_platform_plugin(),
            "paiton_vllm_plugin.paiton_platform.PaitonPlatform",
        )

    @patch.dict(
        os.environ,
        {"PAITON_GPU_ARCH": "gfx1201", "VLLM_DISABLE_PAITON_PLATFORM": "1"},
        clear=False,
    )
    def test_explicit_disable_wins_on_gfx1201(self):
        self.assertIsNone(paiton_platform_plugin())

    @patch.dict(
        os.environ,
        {
            "PAITON_GPU_ARCH": "gfx1201",
            "VLLM_PAITON_VANILLA_ROCM_PLATFORM": "1",
        },
        clear=False,
    )
    def test_reference_harness_selects_unmodified_rocm_platform(self):
        self.assertEqual(
            paiton_platform_plugin(),
            "vllm.platforms.rocm.RocmPlatform",
        )


if __name__ == "__main__":
    unittest.main()
