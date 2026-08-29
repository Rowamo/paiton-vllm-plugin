"""Opt-in full-backbone Qwen3.8 prefill/resume equivalence gate."""

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


def _stats(tensor: torch.Tensor) -> dict[str, float | int]:
    values = tensor.float()
    return {
        "nan": int(torch.isnan(values).sum().item()),
        "posinf": int(torch.isposinf(values).sum().item()),
        "neginf": int(torch.isneginf(values).sum().item()),
        "max_abs": float(values.abs().max().item()),
    }


@unittest.skipUnless(
    os.environ.get("PAITON_RUN_QWEN38_FULL_RESUME") == "1",
    "set PAITON_RUN_QWEN38_FULL_RESUME=1 with artifact/checkpoint/result paths",
)
class Qwen38FullResumeHardwareTest(unittest.TestCase):
    def test_five_token_prefill_matches_four_plus_one_decode(self) -> None:
        artifact = Path(os.environ["PAITON_QWEN38_FULL_ARTIFACT"])
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

            token_ids = [151644, 8948, 198, 151645, 198]
            embedding = source.get_slice("model.language_model.embed_tokens.weight")
            embeds = torch.cat(
                [embedding[token_id : token_id + 1] for token_id in token_ids]
            ).cuda()
            expected_inputs = set(model.get_input_name_to_index_map())

            def allocate_state():
                caches: dict[int, torch.Tensor] = {}
                conv: dict[int, torch.Tensor] = {}
                recurrent: dict[int, torch.Tensor] = {}
                for index in range(64):
                    if (index + 1) % 4 == 0:
                        caches[index] = torch.zeros(
                            (1, 2, 16, 4, 256),
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                    else:
                        conv[index] = torch.zeros(
                            (1, 30720), dtype=torch.bfloat16, device="cuda"
                        )
                        recurrent[index] = torch.zeros(
                            (1, 48, 128, 128), dtype=torch.float32, device="cuda"
                        )
                return caches, conv, recurrent

            def run(
                token_embeds: torch.Tensor,
                *,
                first_position: int,
                has_initial: int,
                state,
            ) -> torch.Tensor:
                caches, conv, recurrent = state
                count = token_embeds.shape[0]
                context_len = first_position + count
                inputs = {
                    "inputs_embeds": token_embeds,
                    "position_ids": torch.arange(
                        first_position,
                        context_len,
                        dtype=torch.int64,
                        device="cuda",
                    ),
                    "slot_mapping": torch.arange(
                        first_position,
                        context_len,
                        dtype=torch.int64,
                        device="cuda",
                    ),
                    "query_start_locations": torch.tensor(
                        [0, count], dtype=torch.int32, device="cuda"
                    ),
                    "context_lengths": torch.tensor(
                        [context_len], dtype=torch.int32, device="cuda"
                    ),
                    "block_tables": torch.tensor(
                        [[0]], dtype=torch.int32, device="cuda"
                    ),
                    "max_query_len": torch.empty(
                        (count, 0), dtype=torch.int32, device="cuda"
                    ),
                    "max_seq_len": torch.empty(
                        (context_len, 0), dtype=torch.int32, device="cuda"
                    ),
                }
                for index in range(64):
                    if index in caches:
                        inputs[f"kv_cache_{index}"] = caches[index]
                    else:
                        inputs[f"conv_state_{index}"] = conv[index]
                        inputs[f"recurrent_state_{index}"] = recurrent[index]
                        inputs[f"state_indices_{index}"] = torch.tensor(
                            [0], dtype=torch.int32, device="cuda"
                        )
                        inputs[f"has_initial_state_{index}"] = torch.tensor(
                            [has_initial], dtype=torch.int32, device="cuda"
                        )
                first_conv = next(iter(conv.values()))
                first_recurrent = next(iter(recurrent.values()))
                inputs["conv_state_line_stride"] = torch.tensor(
                    [first_conv.stride(0)], dtype=torch.int64, device="cuda"
                )
                inputs["recurrent_state_line_stride"] = torch.tensor(
                    [first_recurrent.stride(0)], dtype=torch.int64, device="cuda"
                )
                self.assertEqual(set(inputs), expected_inputs)
                output = torch.empty(
                    (count, 5120), dtype=torch.bfloat16, device="cuda"
                )
                model.run_with_tensors(
                    inputs, {"hidden_states": output}, sync=True
                )
                return output

            full_state = allocate_state()
            full = run(
                embeds, first_position=0, has_initial=0, state=full_state
            )
            resumed_state = allocate_state()
            run(
                embeds[:4], first_position=0, has_initial=0, state=resumed_state
            )
            resumed = run(
                embeds[4:], first_position=4, has_initial=1, state=resumed_state
            )
            delta = resumed.float() - full[4:].float()
            result = {
                "token_ids": token_ids,
                "full_last": _stats(full[4:]),
                "resumed": _stats(resumed),
                "max_abs_delta": float(delta.abs().max().item()),
                "mean_abs_delta": float(delta.abs().mean().item()),
            }
            result_path.write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            print("QWEN38_FULL_RESUME_RESULT=" + json.dumps(result, sort_keys=True))
            self.assertEqual(result["full_last"]["nan"], 0)
            self.assertEqual(result["resumed"]["nan"], 0)
            torch.testing.assert_close(
                resumed, full[4:], atol=0.08, rtol=0.04
            )


if __name__ == "__main__":
    unittest.main()
