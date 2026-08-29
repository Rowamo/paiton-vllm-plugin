import importlib
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class ProtocolA:
    pass


class ProtocolB:
    pass


class ProtocolC:
    pass


class FakeGDNBase(nn.Module):
    def __init__(self, config, vllm_config, prefix):
        super().__init__()
        self.tp_size = 1
        self.num_spec = 0


class FakeAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.kv_cache = ()


class FakeAttentionType:
    DECODER = "decoder"


class FakeLogitsProcessor:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def __call__(self, lm_head, hidden_states):
        return hidden_states


class FakeEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    def forward(self, input_ids):
        return self.weight[input_ids]

    def weight_loader(self, param, value):
        param.data.copy_(value)


class FakeShapeCalculator:
    @classmethod
    def gated_delta_net_state_shape(cls, *args):
        return (3, 10240), (48, 128, 128)


class FakeDtypeCalculator:
    @classmethod
    def gated_delta_net_state_dtype(cls, *args):
        return torch.bfloat16, torch.float32


class FakeCopyCalculator:
    @classmethod
    def gated_delta_net_state_copy_func(cls):
        return (lambda *args: None, lambda *args: None)


class FakeCompiledModel:
    def __init__(self):
        self.inputs = None
        self.noncontiguous_input_names = None

    def run_with_tensors(
        self, inputs, outputs, sync=False, noncontiguous_input_names=frozenset()
    ):
        self.inputs = inputs
        self.noncontiguous_input_names = noncontiguous_input_names
        outputs["hidden_states"].fill_(2)
        return outputs

    def get_input_name_to_index_map(self):
        return {}


def pinned_api_stubs(forward_context):
    names = [
        "vllm",
        "vllm.config",
        "vllm.distributed",
        "vllm.forward_context",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.logits_processor",
        "vllm.model_executor.layers.mamba",
        "vllm.model_executor.layers.mamba.gdn",
        "vllm.model_executor.layers.mamba.gdn.base",
        "vllm.model_executor.layers.mamba.mamba_utils",
        "vllm.model_executor.layers.vocab_parallel_embedding",
        "vllm.model_executor.models",
        "vllm.model_executor.models.interfaces",
        "vllm.model_executor.models.utils",
        "vllm.sequence",
        "vllm.attention",
    ]
    result = {name: module(name) for name in names}
    result["vllm.config"].VllmConfig = object
    result["vllm.distributed"].get_tensor_model_parallel_rank = lambda: 0
    result["vllm.distributed"].get_tensor_model_parallel_world_size = lambda: 1
    result["vllm.forward_context"].get_forward_context = lambda: forward_context
    result["vllm.model_executor.layers.logits_processor"].LogitsProcessor = (
        FakeLogitsProcessor
    )
    result["vllm.model_executor.layers.mamba.gdn.base"].GatedDeltaNetAttention = (
        FakeGDNBase
    )
    utils = result["vllm.model_executor.layers.mamba.mamba_utils"]
    utils.MambaStateCopyFunc = object
    utils.MambaStateCopyFuncCalculator = FakeCopyCalculator
    utils.MambaStateDtypeCalculator = FakeDtypeCalculator
    utils.MambaStateShapeCalculator = FakeShapeCalculator
    embeddings = result["vllm.model_executor.layers.vocab_parallel_embedding"]
    embeddings.ParallelLMHead = FakeEmbedding
    embeddings.VocabParallelEmbedding = FakeEmbedding
    interfaces = result["vllm.model_executor.models.interfaces"]
    interfaces.HasInnerState = ProtocolA
    interfaces.IsHybrid = ProtocolB
    interfaces.SupportsMRoPE = ProtocolC
    result["vllm.model_executor.models.utils"].make_empty_intermediate_tensors_factory = (
        lambda *args: lambda *inner_args: None
    )
    result["vllm.sequence"].IntermediateTensors = object
    result["vllm.attention"].Attention = FakeAttention
    result["vllm.attention"].AttentionType = FakeAttentionType
    return result


class Qwen38ModelContractTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("paiton_vllm_plugin.models.paiton_qwen38", None)
        sys.modules.pop("paiton_vllm_plugin.vllm_compat", None)

    def test_pinned_hybrid_surface_and_compiled_input_binding(self):
        full_meta = SimpleNamespace(
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
            seq_lens=torch.tensor([2], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
            max_query_len=2,
            max_seq_len=2,
        )
        gdn_metas = [
            SimpleNamespace(
                num_spec_decodes=0,
                non_spec_query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
                non_spec_state_indices_tensor=torch.tensor(
                    [index + 1], dtype=torch.int32
                ),
                has_initial_state=torch.tensor([False]),
            )
            for index in range(3)
        ]
        context = SimpleNamespace(attn_metadata={
            "model.layers.0.linear_attn": gdn_metas[0],
            "model.layers.1.linear_attn": gdn_metas[1],
            "model.layers.2.linear_attn": gdn_metas[2],
            "model.layers.3.self_attn": full_meta,
        })
        with patch.dict(sys.modules, pinned_api_stubs(context)):
            imported = importlib.import_module(
                "paiton_vllm_plugin.models.paiton_qwen38"
            )
            cls = imported.PaitonQwen38ForCausalLM
            self.assertTrue(cls.has_inner_state)
            self.assertTrue(cls.is_hybrid)
            instance = cls.__new__(cls)
            nn.Module.__init__(instance)
            instance.config = SimpleNamespace(
                hidden_size=8,
                num_key_value_heads=4,
                head_dim=256,
            )
            instance.contract = {"kv_cache_block_size": 16}
            instance.layer_types = (
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            )
            instance.cache_layers = nn.ModuleDict()
            for index in range(3):
                layer = nn.Module()
                # Padded page strides ensure runtime stride values are not guessed.
                page = torch.zeros((2, 40000), dtype=torch.bfloat16)
                conv = page[:, :30720].view(2, 3, 10240)
                recurrent = torch.zeros((2, 48, 128, 128), dtype=torch.float32)
                layer.kv_cache = (conv, recurrent)
                instance.cache_layers[str(index)] = layer
            full = nn.Module()
            full.kv_cache = torch.zeros(
                (7, 2, 16, 4, 256), dtype=torch.bfloat16
            )
            instance.cache_layers["3"] = full
            instance._dummy_inputs = {}
            instance.compiled_model = FakeCompiledModel()
            instance.compiled_input_names = {
                "inputs_embeds", "position_ids", "slot_mapping",
                "query_start_locations", "context_lengths", "block_tables",
                "max_query_len", "max_seq_len", "conv_state_line_stride",
                "recurrent_state_line_stride", "conv_state_0", "conv_state_1",
                "conv_state_2", "recurrent_state_0", "recurrent_state_1",
                "recurrent_state_2", "state_indices_0", "state_indices_1",
                "state_indices_2", "has_initial_state_0",
                "has_initial_state_1", "has_initial_state_2", "kv_cache_3",
            }

            positions = torch.arange(2).expand(3, -1)
            output = instance.forward(
                torch.tensor([1, 2]),
                positions,
                inputs_embeds=torch.zeros((2, 8), dtype=torch.bfloat16),
            )
            self.assertTrue(torch.all(output == 2))
            inputs = instance.compiled_model.inputs
            self.assertEqual(inputs["position_ids"].ndim, 1)
            self.assertEqual(inputs["conv_state_line_stride"].item(), 40000)
            self.assertEqual(
                inputs["recurrent_state_line_stride"].item(), 48 * 128 * 128
            )
            self.assertEqual(inputs["has_initial_state_0"].dtype, torch.int32)
            self.assertEqual(inputs["has_initial_state_0"].item(), 0)
            self.assertEqual(inputs["state_indices_0"].item(), 1)
            self.assertEqual(inputs["state_indices_1"].item(), 2)
            self.assertEqual(inputs["state_indices_2"].item(), 3)
            self.assertIn("kv_cache_3", inputs)
            self.assertNotIn("kv_cache_dummy_0", inputs)
            self.assertEqual(
                instance.compiled_model.noncontiguous_input_names,
                frozenset({
                    "conv_state_0", "recurrent_state_0",
                    "conv_state_1", "recurrent_state_1",
                    "conv_state_2", "recurrent_state_2",
                }),
            )

            context.attn_metadata = None
            profile_embeds = torch.randn((2, 8), dtype=torch.bfloat16)
            profile_output = instance.forward(
                torch.tensor([1, 2]),
                positions,
                inputs_embeds=profile_embeds,
            )
            self.assertIs(profile_output, profile_embeds)


if __name__ == "__main__":
    unittest.main()
