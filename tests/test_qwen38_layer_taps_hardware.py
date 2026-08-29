"""Opt-in first-divergence diagnostic for the full Qwen3.8 backbone."""

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
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import (
    Qwen38UnquantizedLoader,
)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, float | int]:
    values = tensor.float()
    finite = torch.isfinite(values)
    finite_values = values[finite]
    return {
        "numel": values.numel(),
        "nan": int(torch.isnan(values).sum().item()),
        "posinf": int(torch.isposinf(values).sum().item()),
        "neginf": int(torch.isneginf(values).sum().item()),
        "max_abs_finite": (
            float(finite_values.abs().max().item()) if finite_values.numel() else 0.0
        ),
        "mean_abs_finite": (
            float(finite_values.abs().mean().item()) if finite_values.numel() else 0.0
        ),
    }


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_TAPS") == "1",
    "set PAITON_RUN_QWEN38_TAPS=1 with tap artifact/checkpoint/result paths",
)
class Qwen38LayerTapsHardwareTest(unittest.TestCase):
    def test_reports_selected_boundaries_without_repeated_full_runs(self) -> None:
        artifact = Path(os.environ["PAITON_QWEN38_TAPS_ARTIFACT"])
        checkpoint = Path(os.environ["PAITON_QWEN38_CHECKPOINT"])
        result_path = Path(os.environ["PAITON_QWEN38_RESULT_PATH"])
        manifest = json.loads(
            artifact.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        contract = manifest["paiton_qwen38_contract"]
        self.assertEqual(contract["num_hidden_layers"], 64)
        self.assertEqual(contract["qronos_linear_count"], 496)
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0],
            "gfx1201",
        )

        with Model(str(artifact)) as model, safe_open(
            str(checkpoint), framework="pt", device="cpu"
        ) as source:
            unquantized = Qwen38UnquantizedLoader(manifest)
            for name, tensor in unquantized.iter_from_random_access_source(source):
                model.set_constant_with_tensor(name, tensor.cuda().contiguous())
            qronos = QronosStreamingTransformer(
                qwen38_specs_from_manifest(manifest), max_pending_linears=1
            )
            for transformed in qronos.iter_from_random_access_source(source):
                for name, tensor in transformed.constants():
                    model.set_constant_with_tensor(name, tensor.cuda().contiguous())
            qronos.finish()

            prompt = [151644, 8948, 198, 151645]
            embedding = source.get_slice("model.language_model.embed_tokens.weight")
            embeds = torch.cat(
                [embedding[token_id : token_id + 1] for token_id in prompt]
            ).cuda()
            positions = torch.arange(4, dtype=torch.int64, device="cuda")
            inputs = {
                "inputs_embeds": embeds,
                "position_ids": positions,
                "slot_mapping": positions,
                "query_start_locations": torch.tensor(
                    [0, 4], dtype=torch.int32, device="cuda"
                ),
                "context_lengths": torch.tensor(
                    [4], dtype=torch.int32, device="cuda"
                ),
                "block_tables": torch.tensor(
                    [[0]], dtype=torch.int32, device="cuda"
                ),
                "max_query_len": torch.empty(
                    (4, 0), dtype=torch.int32, device="cuda"
                ),
                "max_seq_len": torch.empty(
                    (4, 0), dtype=torch.int32, device="cuda"
                ),
            }
            conv_states = []
            recurrent_states = []
            for index in range(64):
                if (index + 1) % 4 == 0:
                    inputs[f"kv_cache_{index}"] = torch.zeros(
                        (1, 2, 16, 4, 256),
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                else:
                    conv = torch.zeros(
                        (1, 30720), dtype=torch.bfloat16, device="cuda"
                    )
                    recurrent = torch.zeros(
                        (1, 48, 128, 128), dtype=torch.float32, device="cuda"
                    )
                    conv_states.append(conv)
                    recurrent_states.append(recurrent)
                    inputs[f"conv_state_{index}"] = conv
                    inputs[f"recurrent_state_{index}"] = recurrent
                    inputs[f"state_indices_{index}"] = torch.tensor(
                        [0], dtype=torch.int32, device="cuda"
                    )
                    inputs[f"has_initial_state_{index}"] = torch.tensor(
                        [0], dtype=torch.int32, device="cuda"
                    )
            inputs["conv_state_line_stride"] = torch.tensor(
                [conv_states[0].stride(0)], dtype=torch.int64, device="cuda"
            )
            inputs["recurrent_state_line_stride"] = torch.tensor(
                [recurrent_states[0].stride(0)], dtype=torch.int64, device="cuda"
            )
            expected_inputs = set(model.get_input_name_to_index_map())
            self.assertEqual(set(inputs), expected_inputs)

            output_specs = {
                tensor["name"]: tensor
                for tensor in manifest["interface"]["tensors"]
                if "output" in tensor["roles"]
            }
            output_names = set(model.get_output_name_to_index_map())
            self.assertEqual(output_names, set(output_specs))
            outputs = {
                name: torch.empty(
                    (4, output_specs[name]["shape_values"][-1][-1]),
                    dtype=torch.bfloat16,
                    device="cuda",
                )
                for name in output_names
            }
            model.run_with_tensors(inputs, outputs, sync=True)

            if len(output_names) == 65:
                ordered_names = [
                    f"layer_{index:02d}_hidden_states" for index in range(64)
                ]
                ordered_names.append("hidden_states")
            else:
                expected_suffixes = (
                    "layer_input",
                    "input_norm",
                    "qkv_pre_conv",
                    "qkv_post_conv",
                    "projection_a",
                    "projection_b",
                    "projection_z",
                    "gdn_core",
                    "gdn_out_proj",
                    "attention_residual",
                    "post_attention_norm",
                    "mlp_gate_pre_activation",
                    "mlp_gate",
                    "mlp_up",
                    "mlp_product",
                    "layer_output",
                )
                debug_layers = {
                    name.split("_", 2)[1]
                    for name in output_names
                    if name.startswith("layer_")
                }
                self.assertEqual(len(debug_layers), 1)
                debug_layer = debug_layers.pop()
                ordered_names = [
                    f"layer_{debug_layer}_{suffix}" for suffix in expected_suffixes
                ]
                ordered_names.append("hidden_states")
                self.assertEqual(set(ordered_names), output_names)
            result = {
                "prompt_token_ids": prompt,
                "inputs_embeds": _tensor_stats(embeds),
                "outputs": [
                    {"name": name, **_tensor_stats(outputs[name])}
                    for name in ordered_names
                ],
                "layer_62_conv_state": _tensor_stats(inputs["conv_state_62"]),
                "layer_62_recurrent_state": _tensor_stats(
                    inputs["recurrent_state_62"]
                ),
            }
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            first_nonfinite = next(
                (
                    record["name"]
                    for record in result["outputs"]
                    if record["nan"] or record["posinf"] or record["neginf"]
                ),
                None,
            )
            print("QWEN38_FIRST_NONFINITE=" + str(first_nonfinite))
            print("QWEN38_TAP_RESULT=" + json.dumps(result, sort_keys=True))
            if os.environ.get("PAITON_QWEN38_REQUIRE_FINITE") == "1":
                self.assertIsNone(first_nonfinite)
                self.assertEqual(result["layer_62_recurrent_state"]["nan"], 0)
                self.assertEqual(result["layer_62_recurrent_state"]["posinf"], 0)
                self.assertEqual(result["layer_62_recurrent_state"]["neginf"], 0)


if __name__ == "__main__":
    unittest.main()
