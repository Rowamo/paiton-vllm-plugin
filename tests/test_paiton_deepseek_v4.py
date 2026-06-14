import types
import unittest
from unittest import mock

import torch
from transformers import AutoConfig

from install_bundle import MODEL_TYPE_TO_ARCH
from paiton_vllm_plugin import register_paiton_models
from paiton_vllm_plugin.models.paiton_base import PaitonModelBase
from paiton_vllm_plugin.models.deepseek_v4_config import DeepseekV4Config
from paiton_vllm_plugin.models.paiton_deepseek_v4 import (
    PaitonDeepseekV4ForCausalLM,
)
from paiton_vllm_plugin.runtime.core import (
    runtime_uses_fnuz_fp8,
    torch_dtype_to_string,
)


def _make_model() -> PaitonDeepseekV4ForCausalLM:
    model = PaitonDeepseekV4ForCausalLM.__new__(PaitonDeepseekV4ForCausalLM)
    model.tp_size = 1
    model.tp_rank = 0
    model.parallel_config = types.SimpleNamespace(expert_placement_strategy="linear")
    model.config = types.SimpleNamespace(
        num_hidden_layers=1,
        n_routed_experts=2,
        num_experts_per_tok=1,
        num_hash_layers=1,
        vocab_size=4,
        head_dim=8,
        index_head_dim=3,
        index_topk=4,
        compress_ratios=[4],
    )
    model.dtype = torch.bfloat16
    return model


class PaitonDeepseekV4Tests(unittest.TestCase):
    def test_bundle_installer_knows_deepseek_v4_architecture(self) -> None:
        self.assertEqual(
            MODEL_TYPE_TO_ARCH["deepseek_v4"],
            "PaitonDeepseekV4ForCausalLM",
        )

    def test_registers_deepseek_v4_with_transformers_autoconfig(self) -> None:
        register_paiton_models()

        cfg = AutoConfig.for_model("deepseek_v4", vocab_size=1024)

        self.assertIsInstance(cfg, DeepseekV4Config)
        self.assertEqual(cfg.model_type, "deepseek_v4")
        self.assertEqual(cfg.vocab_size, 1024)

    def test_kv_cache_accessor_does_not_bool_check_tensors(self) -> None:
        cache = torch.zeros((1, 2, 3), dtype=torch.uint8)
        ctx = types.SimpleNamespace(kv_cache=cache)

        got = PaitonDeepseekV4ForCausalLM._get_kv_cache_tensor(ctx)

        self.assertEqual(got.data_ptr(), cache.data_ptr())
        self.assertEqual(tuple(got.shape), tuple(cache.shape))

        list_ctx = types.SimpleNamespace(kv_cache=[cache])
        self.assertIs(PaitonDeepseekV4ForCausalLM._get_kv_cache_tensor(list_ctx), cache)

    def test_runtime_keeps_legacy_fp8_dtype_alias_for_pdata(self) -> None:
        self.assertEqual(
            torch_dtype_to_string(torch.float8_e4m3fn),
            "float8_e4m3fnuz",
        )

    def test_runtime_uses_fnuz_fp8_on_gfx94x(self) -> None:
        props = types.SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ):
            self.assertTrue(runtime_uses_fnuz_fp8())

    def test_runtime_does_not_use_fnuz_fp8_on_gfx950(self) -> None:
        props = types.SimpleNamespace(gcnArchName="gfx950:sramecc+:xnack-")
        with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
            "torch.cuda.get_device_properties", return_value=props
        ):
            self.assertFalse(runtime_uses_fnuz_fp8())

    def test_base_fp8_weight_conversion_is_skipped_on_gfx950(self) -> None:
        weight = torch.tensor([1.0, -0.0], dtype=torch.float32).to(torch.float8_e4m3fn)

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_base.runtime_uses_fnuz_fp8",
            return_value=False,
        ), mock.patch.object(torch.Tensor, "cuda", lambda self: self):
            got = PaitonModelBase._convert_fp8_weights(object(), weight)

        self.assertEqual(got.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(got.view(torch.uint8), weight.view(torch.uint8)))

    def test_base_fp8_weight_conversion_reinterprets_fn_on_gfx94x(self) -> None:
        weight = torch.tensor([1.0, -0.0], dtype=torch.float32).to(torch.float8_e4m3fn)

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_base.runtime_uses_fnuz_fp8",
            return_value=True,
        ), mock.patch.object(torch.Tensor, "cuda", lambda self: self):
            got = PaitonModelBase._convert_fp8_weights(object(), weight)

        self.assertEqual(got.dtype, torch.float8_e4m3fnuz)
        # e4m3fn negative zero (0x80) becomes +0 in fnuz instead of NaN.
        self.assertEqual(got.view(torch.uint8)[1].item(), 0)

    def test_disables_vllm_sliding_window_on_attention_context(self) -> None:
        model = _make_model()
        model.config.sliding_window = 128
        model._deepseek_sliding_window = model.config.sliding_window
        impl = types.SimpleNamespace(sliding_window=128)
        attn_layer = types.SimpleNamespace(sliding_window=128, impl=impl)
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": attn_layer},
        )

        model._disable_vllm_sliding_window_check()

        self.assertIsNone(model.config.sliding_window)
        self.assertIsNone(attn_layer.sliding_window)
        self.assertIsNone(impl.sliding_window)
        self.assertEqual(model._deepseek_sliding_window, 128)

    def test_sparse_mla_missing_inputs_raise_by_default(self) -> None:
        model = _make_model()
        inputs = {}
        keepalive = []
        expected = {
            "input_ids",
            "sparse_mla_kv_2",
            "sparse_mla_indices_2",
            "sparse_mla_topk_length_2",
        }
        input_ids = torch.zeros((3,), dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "Placeholder sparse MLA inputs"):
            model._ensure_sparse_mla_inputs(expected, inputs, input_ids, keepalive)

        self.assertEqual(inputs, {})
        self.assertEqual(keepalive, [])

    def test_sparse_mla_defaults_fill_missing_expected_inputs_when_enabled(self) -> None:
        model = _make_model()
        inputs = {}
        keepalive = []
        expected = {
            "input_ids",
            "sparse_mla_kv_2",
            "sparse_mla_indices_2",
            "sparse_mla_topk_length_2",
        }
        input_ids = torch.zeros((3,), dtype=torch.int32)

        with mock.patch.dict("os.environ", {"PAITON_ALLOW_SPARSE_MLA_PLACEHOLDERS": "1"}):
            model._ensure_sparse_mla_inputs(expected, inputs, input_ids, keepalive)

        self.assertIn("sparse_mla_kv_2", inputs)
        self.assertIn("sparse_mla_indices_2", inputs)
        self.assertIn("sparse_mla_topk_length_2", inputs)
        self.assertEqual(inputs["sparse_mla_kv_2"].shape, [0, 1, 8])
        self.assertEqual(inputs["sparse_mla_indices_2"].shape, [3, 4])
        self.assertEqual(inputs["sparse_mla_topk_length_2"].shape, [3])
        self.assertEqual(len(keepalive), 3)

    def test_sparse_mla_defaults_fill_missing_indexer_kv_when_enabled(self) -> None:
        model = _make_model()
        inputs = {}
        keepalive = []
        expected = {
            "input_ids",
            "sparse_mla_indexer_kv_2",
        }
        input_ids = torch.zeros((3,), dtype=torch.int32)

        with mock.patch.dict("os.environ", {"PAITON_ALLOW_SPARSE_MLA_PLACEHOLDERS": "1"}):
            model._ensure_sparse_mla_inputs(expected, inputs, input_ids, keepalive)

        self.assertIn("sparse_mla_indexer_kv_2", inputs)
        self.assertEqual(inputs["sparse_mla_indexer_kv_2"].shape, [0, 1, 3])
        self.assertEqual(len(keepalive), 4)

    def test_sparse_mla_width_includes_compressed_topk_and_swa_window(self) -> None:
        model = _make_model()
        model.config.index_topk = 512
        model.config.compress_ratios = [0, 0, 4, 128]
        model._deepseek_sliding_window = 128
        inputs = {}
        keepalive = []
        expected = {
            "input_ids",
            "sparse_mla_indices_2",
        }
        input_ids = torch.zeros((3,), dtype=torch.int32)

        with mock.patch.dict("os.environ", {"PAITON_ALLOW_SPARSE_MLA_PLACEHOLDERS": "1"}):
            model._ensure_sparse_mla_inputs(expected, inputs, input_ids, keepalive)

        self.assertEqual(model._sparse_mla_index_width(), 640)
        self.assertEqual(inputs["sparse_mla_indices_2"].shape, [3, 640])

    def test_build_recent_sparse_mla_indices_uses_physical_slots(self) -> None:
        query_start_loc = torch.tensor([0, 3, 4], dtype=torch.int32)
        seq_lens = torch.tensor([3, 5], dtype=torch.int32)
        block_table = torch.tensor(
            [
                [7, 8, -1],
                [2, 4, -1],
            ],
            dtype=torch.int32,
        )

        indices, topk_length = (
            PaitonDeepseekV4ForCausalLM._build_recent_sparse_mla_indices(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                num_tokens=4,
                index_topk=4,
                block_size=2,
                device=torch.device("cpu"),
            )
        )

        self.assertEqual(topk_length.tolist(), [1, 2, 3, 3])
        self.assertEqual(
            indices.tolist(),
            [
                [14, -1, -1, -1],
                [14, 15, -1, -1],
                [14, 15, 16, -1],
                [5, 8, 9, -1],
            ],
        )

    def test_build_recent_sparse_mla_indices_honors_recent_window(self) -> None:
        query_start_loc = torch.tensor([0, 5], dtype=torch.int32)
        seq_lens = torch.tensor([5], dtype=torch.int32)
        block_table = torch.tensor([[3, 4, 5]], dtype=torch.int32)

        indices, topk_length = (
            PaitonDeepseekV4ForCausalLM._build_recent_sparse_mla_indices(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                num_tokens=5,
                index_topk=4,
                block_size=2,
                device=torch.device("cpu"),
                recent_window=2,
            )
        )

        self.assertEqual(topk_length.tolist(), [1, 2, 2, 2, 2])
        self.assertEqual(
            indices.tolist(),
            [
                [6, -1, -1, -1],
                [6, 7, -1, -1],
                [7, 8, -1, -1],
                [8, 9, -1, -1],
                [9, 10, -1, -1],
            ],
        )

    def test_sparse_mla_cache_capacity_uses_generated_indices(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8)
        slot_mapping = torch.tensor([1], dtype=torch.int64)
        sparse_indices = torch.tensor([[1, 6]], dtype=torch.int32)

        cache = model._get_sparse_mla_kv_cache(
            0,
            kv_cache,
            slot_mapping,
            sparse_indices,
        )

        self.assertEqual(tuple(cache.shape), (7, 1, 8))

    def test_sparse_mla_indexer_cache_uses_index_head_dim(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8)
        slot_mapping = torch.tensor([1], dtype=torch.int64)
        sparse_indices = torch.tensor([[1, 6]], dtype=torch.int32)

        cache = model._get_sparse_mla_indexer_kv_cache(
            0,
            kv_cache,
            slot_mapping,
            sparse_indices,
        )

        self.assertEqual(tuple(cache.shape), (7, 1, 3))
        self.assertEqual(cache.dtype, torch.bfloat16)

    def test_compressor_state_cache_shape_matches_compiler_contract(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 32, 4, 1, 8), dtype=torch.uint8)
        slot_mapping = torch.tensor([1, 9], dtype=torch.int64)
        block_tables = torch.tensor([[0, 5, -1]], dtype=torch.int32)

        compressor_cache = model._get_deepseek_v4_state_cache(
            0,
            kv_cache,
            slot_mapping,
            block_tables,
            indexer=False,
        )
        indexer_cache = model._get_deepseek_v4_state_cache(
            0,
            kv_cache,
            slot_mapping,
            block_tables,
            indexer=True,
        )

        self.assertEqual(tuple(compressor_cache.shape), (6, 4, 32))
        self.assertEqual(tuple(indexer_cache.shape), (6, 4, 12))
        self.assertEqual(compressor_cache.dtype, torch.float32)
        self.assertEqual(indexer_cache.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_forward_uses_sync_run(self) -> None:
        model = _make_model()
        model.num_layers = 0
        model.cache_dtype = torch.uint8
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {"input_ids": 0},
        )

        run_calls = []

        def _run(inputs, outputs, stream_ptr=None, sync=True):
            run_calls.append(sync)

        model.model.run = _run

        attn_metadata = types.SimpleNamespace(
            max_query_len=1,
            max_seq_len=1,
            slot_mapping=torch.zeros((1,), dtype=torch.int64),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
        )
        model.compilation_config = types.SimpleNamespace(static_forward_context={})

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_deepseek_v4.get_forward_context",
            return_value=types.SimpleNamespace(attn_metadata={"0": attn_metadata}),
        ):
            input_ids = torch.zeros((1,), dtype=torch.int64, device="cuda")
            positions = torch.zeros((1,), dtype=torch.int64, device="cuda")
            model.forward(input_ids, positions)

        self.assertEqual(run_calls, [True])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_maps_deepseek_v4_checkpoint_tensors(self) -> None:
        model = _make_model()
        pt_params = {
            "embed.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "head.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "norm.weight": torch.randn(4, dtype=torch.bfloat16),
            "hc_head_fn": torch.randn(1, 4),
            "hc_head_base": torch.randn(1),
            "hc_head_scale": torch.randn(1),
            "layers.0.attn.wq_a.weight": torch.randn(2, 4, dtype=torch.bfloat16),
            "layers.0.attn.wkv.weight": torch.randn(1, 4, dtype=torch.bfloat16),
            "layers.0.attn.wq_a.scale": torch.ones(1, 1),
            "layers.0.attn.wkv.scale": torch.full((1, 1), 2.0),
            "layers.0.attn.wq_b.weight": torch.randn(4, 2, dtype=torch.bfloat16),
            "layers.0.attn.wq_b.scale": torch.ones(1, 1),
            "layers.0.attn.wo_b.weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "layers.0.attn.wo_b.scale": torch.ones(1, 1),
            "layers.0.attn.q_norm.weight": torch.randn(2, dtype=torch.bfloat16),
            "layers.0.attn.kv_norm.weight": torch.randn(1, dtype=torch.bfloat16),
            "layers.0.attn.attn_sink": torch.randn(64, dtype=torch.float32),
            "layers.0.attn.compressor.wkv.weight": torch.randn(
                16, 4, dtype=torch.bfloat16),
            "layers.0.attn.compressor.wgate.weight": torch.randn(
                16, 4, dtype=torch.bfloat16),
            "layers.0.attn.compressor.ape": torch.randn(4, 16),
            "layers.0.attn.compressor.norm.weight": torch.randn(
                8, dtype=torch.bfloat16),
            "layers.0.attn.indexer.wq_b.weight": torch.randn(
                16, 2, dtype=torch.bfloat16),
            "layers.0.attn.indexer.wq_b.scale": torch.ones(1, 1),
            "layers.0.attn.indexer.weights_proj.weight": torch.randn(
                4, 4, dtype=torch.bfloat16),
            "layers.0.attn.indexer.compressor.wkv.weight": torch.randn(
                8, 4, dtype=torch.bfloat16),
            "layers.0.attn.indexer.compressor.wgate.weight": torch.randn(
                8, 4, dtype=torch.bfloat16),
            "layers.0.attn.indexer.compressor.ape": torch.randn(4, 8),
            "layers.0.attn.indexer.compressor.norm.weight": torch.randn(
                4, dtype=torch.bfloat16),
            "layers.0.attn_norm.weight": torch.randn(4, dtype=torch.bfloat16),
            "layers.0.ffn.shared_experts.w1.weight": torch.randn(
                2, 4, dtype=torch.bfloat16),
            "layers.0.ffn.shared_experts.w3.weight": torch.randn(
                2, 4, dtype=torch.bfloat16),
            "layers.0.ffn.shared_experts.w1.scale": torch.ones(1, 1),
            "layers.0.ffn.shared_experts.w3.scale": torch.ones(1, 1),
            "layers.0.ffn.shared_experts.w2.weight": torch.randn(
                4, 2, dtype=torch.bfloat16),
            "layers.0.ffn.shared_experts.w2.scale": torch.ones(1, 1),
            "layers.0.ffn.gate.weight": torch.randn(2, 4, dtype=torch.bfloat16),
            "layers.0.ffn.gate.tid2eid": torch.tensor(
                [[0], [1], [1], [0]],
                dtype=torch.int64,
            ),
            "layers.0.ffn_norm.weight": torch.randn(4, dtype=torch.bfloat16),
            "layers.0.hc_attn_fn": torch.randn(1, 4),
            "layers.0.hc_attn_base": torch.randn(1),
            "layers.0.hc_attn_scale": torch.randn(1),
            "layers.0.hc_ffn_fn": torch.randn(1, 4),
            "layers.0.hc_ffn_base": torch.randn(1),
            "layers.0.hc_ffn_scale": torch.randn(1),
        }
        for expert_id in range(2):
            prefix = f"layers.0.ffn.experts.{expert_id}"
            pt_params[f"{prefix}.w1.weight"] = torch.randint(
                0, 255, (2, 2), dtype=torch.uint8)
            pt_params[f"{prefix}.w3.weight"] = torch.randint(
                0, 255, (2, 2), dtype=torch.uint8)
            pt_params[f"{prefix}.w2.weight"] = torch.randint(
                0, 255, (4, 1), dtype=torch.uint8)
            pt_params[f"{prefix}.w1.scale"] = torch.randint(
                0, 255, (2, 1), dtype=torch.uint8)
            pt_params[f"{prefix}.w3.scale"] = torch.randint(
                0, 255, (2, 1), dtype=torch.uint8)
            pt_params[f"{prefix}.w2.scale"] = torch.randint(
                0, 255, (4, 1), dtype=torch.uint8)

        expected = {
            "embed_tokens_weight",
            "lm_head_weight",
            "norm_weight",
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "layers_0_attn_fused_wqa_wkv_weight",
            "layers_0_attn_fused_wqa_wkv_weight_scale_inv",
            "layers_0_attn_wq_b_weight",
            "layers_0_attn_wq_b_weight_scale_inv",
            "layers_0_attn_wo_b_weight",
            "layers_0_attn_wo_b_weight_scale_inv",
            "layers_0_attn_q_norm_weight",
            "layers_0_attn_kv_norm_weight",
            "layers_0_attn_attn_sink",
            "layers_0_attn_compressor_fused_wkv_wgate_weight",
            "layers_0_attn_compressor_ape",
            "layers_0_attn_compressor_norm_weight",
            "layers_0_attn_indexer_wq_b_weight",
            "layers_0_attn_indexer_wq_b_weight_scale_inv",
            "layers_0_attn_indexer_weights_proj_weight",
            "layers_0_attn_indexer_compressor_fused_wkv_wgate_weight",
            "layers_0_attn_indexer_compressor_ape",
            "layers_0_attn_indexer_compressor_norm_weight",
            "layers_0_attn_norm_weight",
            "layers_0_ffn_shared_experts_gate_up_proj_weight",
            "layers_0_ffn_shared_experts_gate_up_proj_weight_scale_inv",
            "layers_0_ffn_shared_experts_down_proj_weight",
            "layers_0_ffn_shared_experts_down_proj_weight_scale_inv",
            "layers_0_ffn_gate_weight",
            "layers_0_ffn_norm_weight",
            "layers_0_ffn_experts_w13_weight",
            "layers_0_ffn_experts_w13_weight_scale",
            "layers_0_ffn_experts_w2_weight",
            "layers_0_ffn_experts_w2_weight_scale",
            "layers_0_ffn_experts_hash_indices_table",
            "layers_0_hc_attn_fn",
            "layers_0_hc_attn_base",
            "layers_0_hc_attn_scale",
            "layers_0_hc_ffn_fn",
            "layers_0_hc_ffn_base",
            "layers_0_hc_ffn_scale",
        }

        mapped = model.map_pt_params(pt_params, expected_constant_names=expected)

        self.assertEqual(set(mapped), expected)
        self.assertEqual(mapped["layers_0_attn_fused_wqa_wkv_weight"].shape, (3, 4))
        self.assertEqual(mapped["layers_0_attn_attn_sink"].shape, (64,))
        self.assertEqual(mapped["layers_0_attn_attn_sink"].dtype, torch.float32)
        self.assertEqual(
            mapped["layers_0_attn_compressor_fused_wkv_wgate_weight"].shape,
            (32, 4),
        )
        torch.testing.assert_close(
            mapped["layers_0_attn_compressor_fused_wkv_wgate_weight"].cpu(),
            torch.cat(
                [
                    pt_params["layers.0.attn.compressor.wkv.weight"],
                    pt_params["layers.0.attn.compressor.wgate.weight"],
                ],
                dim=0,
            ),
        )
        self.assertEqual(
            mapped["layers_0_attn_indexer_wq_b_weight_scale_inv"].shape,
            (1, 1),
        )
        self.assertEqual(
            mapped["layers_0_attn_indexer_compressor_fused_wkv_wgate_weight"].shape,
            (16, 4),
        )
        torch.testing.assert_close(
            mapped["layers_0_attn_indexer_compressor_fused_wkv_wgate_weight"].cpu(),
            torch.cat(
                [
                    pt_params["layers.0.attn.indexer.compressor.wkv.weight"],
                    pt_params["layers.0.attn.indexer.compressor.wgate.weight"],
                ],
                dim=0,
            ),
        )
        self.assertEqual(mapped["layers_0_ffn_experts_w13_weight"].shape, (2, 4, 2))
        self.assertEqual(mapped["layers_0_ffn_experts_w13_weight"].dtype, torch.uint8)
        self.assertEqual(
            mapped["layers_0_ffn_experts_w13_weight_scale"].shape,
            (2, 4, 1),
        )
        self.assertEqual(
            mapped["layers_0_ffn_experts_hash_indices_table"].shape,
            (4, 1),
        )
        torch.testing.assert_close(
            mapped["layers_0_ffn_experts_hash_indices_table"].cpu(),
            pt_params["layers.0.ffn.gate.tid2eid"].to(torch.int32),
        )
        self.assertEqual(
            mapped["layers_0_ffn_experts_hash_indices_table"].dtype,
            torch.int32,
        )


if __name__ == "__main__":
    unittest.main()
