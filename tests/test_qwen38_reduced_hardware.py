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


def runtime_inputs(embeds, positions, kv_cache, conv_states, recurrent_states, initial):
    tokens = embeds.shape[0]
    result = {
        "inputs_embeds": embeds,
        "position_ids": positions,
        "slot_mapping": positions,
        "query_start_locations": torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        "context_lengths": torch.tensor(
            [int(positions[-1]) + 1], dtype=torch.int32, device="cuda"
        ),
        "block_tables": torch.tensor([[0]], dtype=torch.int32, device="cuda"),
        "max_query_len": torch.empty((tokens, 0), dtype=torch.int32, device="cuda"),
        "max_seq_len": torch.empty(
            (int(positions[-1]) + 1, 0), dtype=torch.int32, device="cuda"
        ),
        "conv_state_line_stride": torch.tensor(
            [conv_states[0].stride(0)], dtype=torch.int64, device="cuda"
        ),
        "recurrent_state_line_stride": torch.tensor(
            [recurrent_states[0].stride(0)], dtype=torch.int64, device="cuda"
        ),
    }
    dummy = {
        torch.bfloat16: torch.zeros(1, dtype=torch.bfloat16, device="cuda"),
        torch.float32: torch.zeros(1, dtype=torch.float32, device="cuda"),
        torch.int32: torch.zeros(1, dtype=torch.int32, device="cuda"),
    }
    for index in range(4):
        if index == 3:
            result[f"kv_cache_{index}"] = kv_cache
            result[f"conv_state_dummy_{index}"] = dummy[torch.bfloat16]
            result[f"recurrent_state_dummy_{index}"] = dummy[torch.float32]
            result[f"state_indices_dummy_{index}"] = dummy[torch.int32]
            result[f"has_initial_state_dummy_{index}"] = dummy[torch.int32]
        else:
            result[f"kv_cache_dummy_{index}"] = dummy[torch.bfloat16]
            result[f"conv_state_{index}"] = conv_states[index]
            result[f"recurrent_state_{index}"] = recurrent_states[index]
            result[f"state_indices_{index}"] = torch.tensor(
                [0], dtype=torch.int32, device="cuda"
            )
            result[f"has_initial_state_{index}"] = torch.tensor(
                [initial], dtype=torch.int32, device="cuda"
            )
    return result


def exact_runtime_inputs(model, candidates):
    expected = set(model.get_input_name_to_index_map())
    missing = sorted(expected - set(candidates))
    if missing:
        raise ValueError("hardware harness missing compiled inputs: " + ", ".join(missing))
    return {name: value for name, value in candidates.items() if name in expected}


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_REDUCED") == "1",
    "set PAITON_RUN_QWEN38_REDUCED=1 with artifact/checkpoint paths",
)
class Qwen38ReducedHardwareTest(unittest.TestCase):
    def test_checkpoint_prefill_matches_stateful_decode(self):
        artifact = Path(os.environ["PAITON_QWEN38_REDUCED_ARTIFACT"])
        checkpoint = Path(os.environ["PAITON_QWEN38_CHECKPOINT"])
        self.assertEqual(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx1201")
        self.assertEqual(checkpoint.stat().st_size, 19_893_384_832)
        manifest = json.loads(artifact.with_suffix(".manifest.json").read_text())
        self.assertEqual(manifest["paiton_qwen38_contract"]["version"], 2)
        self.assertEqual(
            manifest["paiton_qwen38_contract"]["kv_cache_physical_layout"],
            "blocks_KV_tokens_heads_dim",
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
            self.assertEqual(unquantized.peak_source_bytes, 81_920)
            self.assertEqual(qronos.peak_pending_linears, 1)

            torch.manual_seed(38)
            embeds = torch.randn((4, 5120), dtype=torch.bfloat16, device="cuda") * 0.01

            def states():
                return (
                    torch.zeros((2, 30720), dtype=torch.bfloat16, device="cuda"),
                    torch.zeros((2, 48, 128, 128), dtype=torch.float32, device="cuda"),
                )

            prefill_conv, prefill_recurrent = zip(*(states() for _ in range(3)))
            prefill_kv = torch.zeros(
                (3, 2, 16, 4, 256), dtype=torch.bfloat16, device="cuda"
            )
            prefill_out = torch.empty((4, 5120), dtype=torch.bfloat16, device="cuda")
            model.run_with_tensors(
                exact_runtime_inputs(model, runtime_inputs(
                    embeds,
                    torch.arange(4, dtype=torch.int64, device="cuda"),
                    prefill_kv,
                    prefill_conv,
                    prefill_recurrent,
                    0,
                )),
                {"hidden_states": prefill_out},
                sync=True,
            )

            decode_conv, decode_recurrent = zip(*(states() for _ in range(3)))
            decode_kv = torch.zeros_like(prefill_kv)
            decoded = []
            for position in range(4):
                output = torch.empty((1, 5120), dtype=torch.bfloat16, device="cuda")
                model.run_with_tensors(
                    exact_runtime_inputs(model, runtime_inputs(
                        embeds[position : position + 1],
                        torch.tensor([position], dtype=torch.int64, device="cuda"),
                        decode_kv,
                        decode_conv,
                        decode_recurrent,
                        int(position > 0),
                    )),
                    {"hidden_states": output},
                    sync=True,
                )
                decoded.append(output.clone())
            decoded = torch.cat(decoded)
            torch.testing.assert_close(decoded, prefill_out, atol=0.08, rtol=0.04)
            for prefill, decode in zip(prefill_conv, decode_conv):
                torch.testing.assert_close(decode[0], prefill[0], atol=0.02, rtol=0.02)
            for prefill, decode in zip(prefill_recurrent, decode_recurrent):
                torch.testing.assert_close(decode[0], prefill[0], atol=3e-5, rtol=3e-5)
