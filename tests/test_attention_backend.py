import importlib
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class PaitonAttentionBackendTests(unittest.TestCase):
    def test_shape_and_stride_contract_have_matching_rank(self) -> None:
        module_name = "vllm.v1.attention.backends.triton_attn"
        triton_module = ModuleType(module_name)
        triton_module.TritonAttentionBackend = type(
            "TritonAttentionBackend", (), {}
        )
        parents = {
            name: ModuleType(name)
            for name in (
                "vllm",
                "vllm.v1",
                "vllm.v1.attention",
                "vllm.v1.attention.backends",
            )
        }
        sys.modules.pop("paiton_vllm_plugin.paiton_attention_backend", None)
        with patch.dict(sys.modules, {**parents, module_name: triton_module}):
            backend = importlib.import_module(
                "paiton_vllm_plugin.paiton_attention_backend"
            ).PaitonTritonAttentionBackend
            self.assertEqual(backend.get_supported_kernel_block_sizes(), [16])
            shape = backend.get_kv_cache_shape(
                num_blocks=7,
                block_size=16,
                num_kv_heads=4,
                head_size=256,
            )
            self.assertEqual(shape, (7, 2, 16, 4, 256))
            self.assertEqual(
                backend.get_kv_cache_stride_order(),
                (0, 1, 2, 3, 4),
            )
            self.assertEqual(
                backend.get_kv_cache_stride_order(True),
                (0, 1, 2, 3, 4, 5),
            )


if __name__ == "__main__":
    unittest.main()
