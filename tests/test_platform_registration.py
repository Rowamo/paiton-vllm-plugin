import os
import unittest
from unittest.mock import patch

from paiton_vllm_plugin import paiton_platform_plugin


class PlatformRegistrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
