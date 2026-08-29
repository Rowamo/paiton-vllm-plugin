"""Reduced contract-v4 Qwen3.8 backbone gate on gfx1201."""

import json
import os
from pathlib import Path
import unittest

import torch
from safetensors import safe_open

from paiton_vllm_plugin.runtime.core import Model
from paiton_vllm_plugin.runtime.core.utils.qronos_loader import (
    QronosStreamingTransformer,
    qwen38_specs_from_manifest,
)
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import Qwen38UnquantizedLoader
from tests.test_qwen38_reduced_hardware import exact_runtime_inputs, runtime_inputs


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_MULTIMODAL_REDUCED") == "1",
    "set PAITON_RUN_QWEN38_MULTIMODAL_REDUCED=1 with artifact/checkpoint paths",
)
class Qwen38MultimodalReducedHardwareTest(unittest.TestCase):
    def test_three_axis_prefill_matches_stateful_decode(self):
        artifact = Path(os.environ["PAITON_QWEN38_MULTIMODAL_REDUCED_ARTIFACT"])
        checkpoint = Path(os.environ["PAITON_QWEN38_CHECKPOINT"])
        manifest = json.loads(artifact.with_suffix(".manifest.json").read_text())
        contract = manifest["paiton_qwen38_contract"]
        self.assertEqual(contract["version"], 4)
        self.assertEqual(contract["scope"], "multimodal")
        self.assertEqual(contract["position_ids_layout"], "3_tokens_interleaved_thw")
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1201"
        )

        with Model(str(artifact)) as model, safe_open(
            str(checkpoint), framework="pt", device="cpu"
        ) as source:
            unquantized = Qwen38UnquantizedLoader(manifest)
            for name, tensor in unquantized.iter_from_random_access_source(source):
                model.set_constant_with_tensor(name, tensor.cuda().contiguous())
            qronos = QronosStreamingTransformer(
                qwen38_specs_from_manifest(manifest),
                max_pending_linears=1,
                allowed_extra_layer_range=(4, 64),
            )
            for transformed in qronos.iter_from_random_access_source(source):
                for name, tensor in transformed.constants():
                    model.set_constant_with_tensor(name, tensor.cuda().contiguous())
            qronos.finish()

            torch.manual_seed(3804)
            embeds = torch.randn(
                (4, 5120), dtype=torch.bfloat16, device="cuda"
            ) * 0.01
            positions = torch.tensor(
                [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]],
                dtype=torch.int64,
                device="cuda",
            )

            def states():
                return (
                    torch.zeros((2, 30720), dtype=torch.bfloat16, device="cuda"),
                    torch.zeros(
                        (2, 48, 128, 128), dtype=torch.float32, device="cuda"
                    ),
                )

            def inputs(local_embeds, local_positions, conv, recurrent, kv, initial, slot):
                values = runtime_inputs(
                    local_embeds,
                    local_positions[0],
                    kv,
                    conv,
                    recurrent,
                    initial,
                )
                values["position_ids"] = local_positions.contiguous()
                values["slot_mapping"] = torch.tensor(
                    slot, dtype=torch.int64, device="cuda"
                )
                return exact_runtime_inputs(model, values)

            prefill_conv, prefill_recurrent = zip(*(states() for _ in range(3)))
            prefill_kv = torch.zeros(
                (3, 2, 16, 4, 256), dtype=torch.bfloat16, device="cuda"
            )
            prefill = torch.empty((4, 5120), dtype=torch.bfloat16, device="cuda")
            model.run_with_tensors(
                inputs(
                    embeds, positions, prefill_conv, prefill_recurrent,
                    prefill_kv, 0, range(4),
                ),
                {"hidden_states": prefill},
                sync=True,
            )

            decode_conv, decode_recurrent = zip(*(states() for _ in range(3)))
            decode_kv = torch.zeros_like(prefill_kv)
            decoded = []
            for token in range(4):
                output = torch.empty((1, 5120), dtype=torch.bfloat16, device="cuda")
                model.run_with_tensors(
                    inputs(
                        embeds[token : token + 1],
                        positions[:, token : token + 1],
                        decode_conv,
                        decode_recurrent,
                        decode_kv,
                        int(token > 0),
                        [token],
                    ),
                    {"hidden_states": output},
                    sync=True,
                )
                decoded.append(output.clone())
            decoded = torch.cat(decoded)
            torch.testing.assert_close(decoded, prefill, atol=0.08, rtol=0.04)
            self.assertTrue(torch.isfinite(prefill).all())


if __name__ == "__main__":
    unittest.main()
