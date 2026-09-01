import unittest

from paiton_vllm_plugin.runtime.core.utils.qwen38_memory import (
    estimate_qwen38_memory,
)
from tests.test_qronos_loader import (
    qwen38_bf16_kernel_scale_manifest_fixture,
    qwen38_manifest_fixture,
)


class Qwen38MemoryEstimatorTests(unittest.TestCase):
    @staticmethod
    def manifest():
        manifest = qwen38_manifest_fixture()
        manifest["memory_planning"] = {
            "activation_blob_bytes": 1000,
            "compiler_owned_constant_bytes": 128,
            "shared_workspace_bytes": 64,
            "unique_workspace_bytes": 32,
        }
        return manifest

    def test_counts_every_manifest_owned_memory_class(self) -> None:
        estimate = estimate_qwen38_memory(self.manifest())
        self.assertEqual(estimate.compiled_unbound_constants_bytes, 130560)
        self.assertEqual(
            estimate.runtime_shell_parameters_bytes,
            2 * 248320 * 5120 * 2,
        )
        self.assertEqual(estimate.compiler_owned_constants_bytes, 128)
        self.assertEqual(estimate.kv_cache_bytes, 0)
        self.assertEqual(
            estimate.gdn_state_bytes,
            3 * 10240 * 2 + 48 * 128 * 128 * 4,
        )
        self.assertEqual(estimate.activation_blob_bytes, 1000)
        self.assertEqual(estimate.workspace_bytes, 96)
        self.assertEqual(estimate.allocator_headroom_bytes, 2 * 1024**3)

        exact = estimate_qwen38_memory(
            self.manifest(), available_bytes=estimate.required_bytes
        )
        short = estimate_qwen38_memory(
            self.manifest(), available_bytes=estimate.required_bytes - 1
        )
        self.assertTrue(exact.fits)
        self.assertFalse(short.fits)

    def test_bf16_kernel_scale_manifest_counts_only_two_bytes_per_scale(self) -> None:
        legacy = self.manifest()
        candidate = qwen38_bf16_kernel_scale_manifest_fixture()
        candidate["memory_planning"] = dict(legacy["memory_planning"])

        legacy_estimate = estimate_qwen38_memory(legacy)
        candidate_estimate = estimate_qwen38_memory(candidate)
        self.assertEqual(
            legacy_estimate.compiled_unbound_constants_bytes
            - candidate_estimate.compiled_unbound_constants_bytes,
            48 * 40 * 2,
        )

    def test_8192_context_formula_is_manifest_driven(self) -> None:
        manifest = self.manifest()
        contract = manifest["paiton_qwen38_contract"]
        contract["num_full_attention_layers"] = 16
        contract["num_gdn_layers"] = 48
        estimate = estimate_qwen38_memory(manifest)
        self.assertEqual(estimate.kv_cache_bytes, 512 * 1024**2)
        self.assertEqual(
            estimate.gdn_state_bytes,
            48 * (3 * 10240 * 2 + 48 * 128 * 128 * 4),
        )

    def test_explicit_hybrid_cache_reservation_is_charged_once(self) -> None:
        reservation = 3 * 1024**3
        estimate = estimate_qwen38_memory(
            self.manifest(), hybrid_cache_reservation_bytes=reservation
        )
        self.assertEqual(estimate.hybrid_cache_bytes, reservation)
        self.assertLess(
            estimate.kv_cache_bytes + estimate.gdn_state_bytes,
            estimate.hybrid_cache_bytes,
        )

    def test_rejects_missing_planning_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "memory-planning metadata"):
            estimate_qwen38_memory(qwen38_manifest_fixture())

    def test_multimodal_contract_charges_exact_vision_tower_bytes(self) -> None:
        text_manifest = self.manifest()
        multimodal_manifest = self.manifest()
        contract = multimodal_manifest["paiton_qwen38_contract"]
        contract.update(
            {
                "version": 4,
                "scope": "multimodal",
                "multimodal": True,
                "position_ids_layout": "3_tokens_interleaved_thw",
                "runtime_shell_parameters": [
                    *contract["runtime_shell_parameters"],
                    {
                        "name": "model.visual.*",
                        "dtype": "uint8",
                        "shape": [921460192],
                    },
                ],
            }
        )
        text = estimate_qwen38_memory(text_manifest)
        multimodal = estimate_qwen38_memory(multimodal_manifest)
        self.assertEqual(
            multimodal.runtime_shell_parameters_bytes
            - text.runtime_shell_parameters_bytes,
            921460192,
        )
        self.assertGreater(multimodal.required_bytes, text.required_bytes)


if __name__ == "__main__":
    unittest.main()
