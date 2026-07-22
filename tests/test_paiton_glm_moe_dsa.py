# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Paiton GLM MoE DSA plugin weight mapping.

These tests exercise `PaitonGlmMoeDsaForCausalLM.map_pt_params` against a
synthetic checkpoint tensor stream (no vLLM, no compiled .so donation). They
verify:
- Plain MLA attention projections map 1:1 (no qkv fusion).
- Routed MXFP4 experts are packed: gate+up -> w13, down -> w2 (+UE8M0 scales).
- Shared experts are packed the same way into a 1-expert buffer.
- Dense-layer MLPs get a fused gate_up_proj_weight (+ independent down_proj).
- The router gate + e_score_correction_bias are emitted unchanged.
- Global embed/norm/lm_head map 1:1.
"""

import os
import types
import unittest
from unittest import mock

import torch

from paiton_vllm_plugin.models.paiton_glm_moe_dsa import (
    PaitonGlmMoeDsaForCausalLM,
    _resolve_compiled_moe_kernel,
)


def _make_model(n_layers: int = 4, first_k_dense: int = 1) -> "PaitonGlmMoeDsaForCausalLM":
    model = PaitonGlmMoeDsaForCausalLM.__new__(PaitonGlmMoeDsaForCausalLM)
    model.tp_size = 1
    model.tp_rank = 0
    model.num_layers = n_layers
    model.parallel_config = types.SimpleNamespace(
        expert_placement_strategy="linear",
        enable_expert_parallel=False,
    )
    model.config = types.SimpleNamespace(
        num_hidden_layers=n_layers,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=first_k_dense,
        moe_layer_freq=1,
        topk_method="noaux_tc",
        hidden_size=64,
        num_attention_heads=32,
        moe_intermediate_size=64,
        vocab_size=64,
        kv_lora_rank=512,
        qk_head_dim=192,
        qk_rope_head_dim=64,
        torch_dtype=torch.bfloat16,
    )
    model._n_routed_experts = 4
    model._topk = 2
    model._first_k_dense_replace = first_k_dense
    model._moe_layer_freq = 1
    model._has_correction_bias = True
    model.dtype = torch.bfloat16
    model.cache_dtype = torch.bfloat16
    return model


def _pt_attn(layer: int) -> dict:
    pt = {}
    for n in ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]:
        pt[f"model.layers.{layer}.self_attn.{n}.weight"] = torch.zeros([8, 8])
    pt[f"model.layers.{layer}.self_attn.q_a_layernorm.weight"] = torch.ones([8])
    pt[f"model.layers.{layer}.self_attn.kv_a_layernorm.weight"] = torch.ones([8])
    pt[f"model.layers.{layer}.input_layernorm.weight"] = torch.ones([64])
    pt[f"model.layers.{layer}.post_attention_layernorm.weight"] = torch.ones([64])
    return pt


def _pt_indexer(layer: int) -> dict:
    return {
        f"model.layers.{layer}.self_attn.indexer.wq_b.weight": torch.arange(
            32 * 16, dtype=torch.float32
        ).reshape(32, 16),
        f"model.layers.{layer}.self_attn.indexer.wk.weight": torch.arange(
            16 * 64, dtype=torch.float32
        ).reshape(16, 64),
        f"model.layers.{layer}.self_attn.indexer.weights_proj.weight": torch.arange(
            4 * 64, dtype=torch.float32
        ).reshape(4, 64),
        f"model.layers.{layer}.self_attn.indexer.k_norm.weight": torch.ones(16),
        f"model.layers.{layer}.self_attn.indexer.k_norm.bias": torch.ones(16),
    }


def _pt_dense_mlp(layer: int) -> dict:
    return {
        f"model.layers.{layer}.mlp.gate_proj.weight": torch.zeros([128, 64]),
        f"model.layers.{layer}.mlp.up_proj.weight": torch.zeros([128, 64]),
        f"model.layers.{layer}.mlp.down_proj.weight": torch.zeros([64, 128]),
    }


def _pt_sparse_mlp(layer: int, n_experts: int = 4) -> dict:
    pt = {
        f"model.layers.{layer}.mlp.gate_proj.weight": torch.zeros([n_experts, 64]),
        f"model.layers.{layer}.mlp.gate.e_score_correction_bias": torch.zeros([n_experts]),
    }
    for eid in range(n_experts):
        for proj, shp, scl in [
            ("gate_proj", [64, 32], [64, 2]),
            ("up_proj", [64, 32], [64, 2]),
            ("down_proj", [64, 32], [64, 2]),
        ]:
            pt[f"model.layers.{layer}.mlp.experts.{eid}.{proj}.weight"] = torch.zeros(
                shp, dtype=torch.uint8
            )
            pt[
                f"model.layers.{layer}.mlp.experts.{eid}.{proj}.weight_scale"
            ] = torch.zeros(scl, dtype=torch.uint8)
    for proj, shp, scl in [
        ("gate_proj", [64, 32], [64, 2]),
        ("up_proj", [64, 32], [64, 2]),
        ("down_proj", [64, 32], [64, 2]),
    ]:
        pt[
            f"model.layers.{layer}.mlp.shared_experts.{proj}.weight"
        ] = torch.zeros(shp, dtype=torch.uint8)
        pt[
            f"model.layers.{layer}.mlp.shared_experts.{proj}.weight_scale"
        ] = torch.zeros(scl, dtype=torch.uint8)
    return pt


class GlmMoeDsaWeightMappingTests(unittest.TestCase):
    def _run(
        self,
        model,
        pt_params,
        expected=None,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
    ):
        epgrp = mock.MagicMock()
        epgrp.rank_in_group = ep_rank
        epgrp.world_size = ep_size
        # The synthetic fixture uses hidden_size=64, below the minimum K block
        # accepted by either production CK preshuffle.  Keep generic-name
        # mapping tests on the unshuffled test backend; layout-specific tests
        # explicitly select FlatMM and mock its byte permutation below.
        uses_flatmm_names = bool(
            expected and any("_flatmm" in name for name in expected)
        )
        test_moe_kernel = (
            "ck_flatmm_fp4" if uses_flatmm_names else "paiton"
        )
        with mock.patch.dict(
            os.environ,
            {"PAITON_MOE_KERNEL": test_moe_kernel},
        ), mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa.get_tensor_model_parallel_world_size",
            return_value=tp_size,
        ), mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ), mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa.get_ep_group",
            return_value=epgrp,
        ):
            return model.map_pt_params(pt_params, expected_constant_names=expected)

    def test_flatmm_layout_is_inferred_from_compiled_constants(self):
        expected = {"layers_3_mlp_experts_w13_weight_flatmm"}
        self.assertEqual(
            _resolve_compiled_moe_kernel(None, expected),
            "ck_flatmm_fp4",
        )

    def test_fused_shared_flatmm_packs_shared_as_last_expert(self):
        model = _make_model(n_layers=1, first_k_dense=0)
        pt = _pt_attn(0)
        pt.update(_pt_sparse_mlp(0, n_experts=4))
        expected = {
            "layers_0_mlp_experts_w13_weight_flatmm_fused_shared",
            "layers_0_mlp_experts_w13_weight_scale_flatmm_fused_shared",
            "layers_0_mlp_experts_w2_weight_flatmm_fused_shared",
            "layers_0_mlp_experts_w2_weight_scale_flatmm_fused_shared",
        }
        # The production FlatMM layout requires K>=256; this lightweight
        # mapping fixture uses K=64, so mock only the byte permutation while
        # retaining the concatenation/shape/name checks under test.
        with mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa."
            "_preshuffle_flatmm_mxfp4_weight",
            side_effect=lambda value, *args, **kwargs: value,
        ), mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa."
            "_preshuffle_flatmm_mxfp4_scale",
            side_effect=lambda value, *args, **kwargs: value,
        ):
            mapped = self._run(model, pt, expected=expected)
        self.assertEqual(set(mapped), expected)
        self.assertEqual(
            tuple(
                mapped[
                    "layers_0_mlp_experts_w13_weight_flatmm_fused_shared"
                ].shape
            ),
            (5, 128, 32),
        )
        self.assertEqual(
            tuple(
                mapped[
                    "layers_0_mlp_experts_w2_weight_flatmm_fused_shared"
                ].shape
            ),
            (5, 64, 32),
        )

    def test_tp_group_does_not_shard_experts_when_ep_is_disabled(self):
        model = _make_model(n_layers=1, first_k_dense=0)
        pt = _pt_attn(0)
        pt.update(_pt_sparse_mlp(0, n_experts=4))
        expected = {
            "layers_0_mlp_experts_w13_weight_flatmm_fused_shared",
        }
        with mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa."
            "_preshuffle_flatmm_mxfp4_weight",
            side_effect=lambda value, *args, **kwargs: value,
        ), mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa."
            "_preshuffle_flatmm_mxfp4_scale",
            side_effect=lambda value, *args, **kwargs: value,
        ):
            mapped = self._run(
                model,
                pt,
                expected=expected,
                tp_size=2,
                ep_size=2,
                ep_rank=1,
            )
        self.assertEqual(
            tuple(
                mapped[
                    "layers_0_mlp_experts_w13_weight_flatmm_fused_shared"
                ].shape
            ),
            (5, 128, 32),
        )

    def test_fused_shared_ep_packs_local_weights_and_global_mask(self):
        model = _make_model(n_layers=1, first_k_dense=0)
        model.parallel_config.enable_expert_parallel = True
        pt = _pt_attn(0)
        pt.update(_pt_sparse_mlp(0, n_experts=4))
        weight_name = (
            "layers_0_mlp_experts_w13_weight_flatmm_fused_shared"
        )
        mask_name = "layers_0_mlp_experts_local_expert_mask"
        expected = {weight_name, mask_name}
        with mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa."
            "_preshuffle_flatmm_mxfp4_weight",
            side_effect=lambda value, *args, **kwargs: value,
        ), mock.patch(
            "paiton_vllm_plugin.models.paiton_glm_moe_dsa."
            "_preshuffle_flatmm_mxfp4_scale",
            side_effect=lambda value, *args, **kwargs: value,
        ):
            mapped = self._run(
                model,
                pt,
                expected=expected,
                tp_size=2,
                ep_size=2,
                ep_rank=0,
            )
        self.assertEqual(tuple(mapped[weight_name].shape), (3, 128, 32))
        self.assertEqual(mapped[mask_name].tolist(), [1, 1, 0, 0, 1])

    def test_flatmm_artifact_rejects_legacy_loader_layout(self):
        expected = {"layers_3_mlp_experts_w13_weight_flatmm"}
        with self.assertRaisesRegex(RuntimeError, "expects FlatMM"):
            _resolve_compiled_moe_kernel("ck_moe_fp4", expected)

    def test_legacy_flatmm_artifact_can_use_explicit_loader_layout(self):
        expected = {"layers_3_mlp_experts_w13_weight"}
        self.assertEqual(
            _resolve_compiled_moe_kernel("ck_flatmm_fp4", expected),
            "ck_flatmm_fp4",
        )

    def test_dense_and_sparse_layer_constant_names(self):
        model = _make_model(n_layers=4, first_k_dense=1)
        pt = {}
        pt.update(_pt_attn(0))  # dense
        pt.update(_pt_dense_mlp(0))
        for layer in (1, 2, 3):  # sparse
            pt.update(_pt_attn(layer))
            pt.update(_pt_sparse_mlp(layer))
        pt["model.embed_tokens.weight"] = torch.zeros([64, 64])
        pt["model.norm.weight"] = torch.ones([64])
        pt["lm_head.weight"] = torch.zeros([64, 64])

        mapped = self._run(model, pt)
        # Critical names exist.
        expected_names = {
            "layers_0_mlp_gate_up_proj_weight",
            "layers_0_mlp_down_proj_weight",
            "embed_tokens_weight",
            "lm_head_weight",
            "norm_weight",
        }
        for layer in (1, 2, 3):
            expected_names |= {
                f"layers_{layer}_mlp_gate_proj_weight",
                f"layers_{layer}_mlp_gate_e_score_correction_bias",
                f"layers_{layer}_mlp_experts_w13_weight",
                f"layers_{layer}_mlp_experts_w13_weight_scale",
                f"layers_{layer}_mlp_experts_w2_weight",
                f"layers_{layer}_mlp_experts_w2_weight_scale",
                f"layers_{layer}_mlp_shared_experts_w13_weight",
                f"layers_{layer}_mlp_shared_experts_w13_weight_scale",
                f"layers_{layer}_mlp_shared_experts_w2_weight",
                f"layers_{layer}_mlp_shared_experts_w2_weight_scale",
            }
        missing = expected_names - set(mapped.keys())
        self.assertFalse(missing, f"missing mapped constants: {sorted(missing)}")

    def test_expert_packing_shapes(self):
        model = _make_model(n_layers=2, first_k_dense=0)
        pt = _pt_attn(0)
        pt.update(_pt_sparse_mlp(0, n_experts=4))
        pt["model.embed_tokens.weight"] = torch.zeros([64, 64])
        pt["model.norm.weight"] = torch.ones([64])
        pt["lm_head.weight"] = torch.zeros([64, 64])
        mapped = self._run(model, pt)

        # 4 experts, gate+up stacked -> [4, 128, 32]
        self.assertEqual(
            tuple(mapped["layers_0_mlp_experts_w13_weight"].shape), (4, 128, 32)
        )
        self.assertEqual(mapped["layers_0_mlp_experts_w13_weight"].dtype, torch.uint8)
        # w2 = down_proj -> [4, 64, 32]
        self.assertEqual(
            tuple(mapped["layers_0_mlp_experts_w2_weight"].shape), (4, 64, 32)
        )
        # Scales: [4, 128, 2] and [4, 64, 2]
        self.assertEqual(
            tuple(mapped["layers_0_mlp_experts_w13_weight_scale"].shape), (4, 128, 2)
        )
        self.assertEqual(
            tuple(mapped["layers_0_mlp_experts_w2_weight_scale"].shape), (4, 64, 2)
        )
        # Shared experts: 1-expert buffer.
        self.assertEqual(
            tuple(mapped["layers_0_mlp_shared_experts_w13_weight"].shape), (1, 128, 32)
        )
        self.assertEqual(
            tuple(mapped["layers_0_mlp_shared_experts_w2_weight"].shape), (1, 64, 32)
        )

    def test_expected_filter_skips_unneeded_names(self):
        """If expected_constant_names is provided, names outside it are dropped."""
        model = _make_model(n_layers=2, first_k_dense=0)
        pt = _pt_attn(0)
        pt.update(_pt_sparse_mlp(0))
        pt["model.embed_tokens.weight"] = torch.zeros([64, 64])
        pt["model.norm.weight"] = torch.ones([64])
        pt["lm_head.weight"] = torch.zeros([64, 64])
        # Only ask for the routed w13 + gate; everything else should be skipped.
        expected = {
            "layers_0_mlp_experts_w13_weight",
            "layers_0_mlp_gate_proj_weight",
        }
        mapped = self._run(model, pt, expected=expected)
        self.assertEqual(set(mapped.keys()), expected)

    def test_indexer_weights_are_replicated_under_tp(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        pt = _pt_indexer(0)
        expected = {
            "layers_0_self_attn_indexer_wq_b_weight",
            "layers_0_self_attn_indexer_wk_weight",
            "layers_0_self_attn_indexer_weights_proj_weight",
            "layers_0_self_attn_indexer_k_norm_weight",
            "layers_0_self_attn_indexer_k_norm_bias",
        }
        mapped = self._run(model, pt, expected=expected, tp_size=2, tp_rank=1)

        self.assertEqual(
            tuple(mapped["layers_0_self_attn_indexer_wq_b_weight"].shape),
            (32, 16),
        )
        self.assertEqual(
            tuple(mapped["layers_0_self_attn_indexer_weights_proj_weight"].shape),
            (4, 64),
        )
        self.assertTrue(
            torch.equal(
                mapped["layers_0_self_attn_indexer_k_norm_bias"].cpu(),
                torch.ones(16),
            )
        )

    def test_glm_runtime_reuses_matching_single_head_latent_kv_cache(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        reference = torch.empty((2, 3, 16, 1, 576), dtype=torch.bfloat16)
        binding = types.SimpleNamespace(
            layer_idx=0,
            kv_cache=reference,
            ctx=types.SimpleNamespace(kv_cache=[reference]),
            kv_cache_data_ptr=reference.data_ptr(),
            kv_cache_pdata=None,
            kv_cache_view=reference,
        )

        kv_cache, pdata = model._refresh_deepseek_kv_binding(binding)

        self.assertIs(kv_cache, reference)
        self.assertEqual(pdata.shape, [2, 3, 16, 1, 576])

    def test_glm_runtime_allocates_private_cache_for_legacy_layout(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        reference = torch.empty((2, 3, 16, 32, 192), dtype=torch.bfloat16)
        binding = types.SimpleNamespace(
            layer_idx=0,
            kv_cache=reference,
            ctx=types.SimpleNamespace(kv_cache=[reference]),
            kv_cache_data_ptr=reference.data_ptr(),
            kv_cache_pdata=None,
            kv_cache_view=reference,
        )

        kv_cache, pdata = model._refresh_deepseek_kv_binding(binding)

        self.assertEqual(tuple(kv_cache.shape), (2, 3, 16, 1, 576))
        self.assertIsNot(kv_cache, reference)
        self.assertEqual(kv_cache.dtype, torch.bfloat16)
        self.assertEqual(pdata.shape, [2, 3, 16, 1, 576])

    def test_glm_runtime_normalizes_triton_kv_cache_layout(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        reference = torch.empty((4, 2, 16, 1, 576), dtype=torch.bfloat16)
        binding = types.SimpleNamespace(
            layer_idx=0,
            kv_cache=reference,
            ctx=types.SimpleNamespace(kv_cache=[reference]),
            kv_cache_data_ptr=reference.data_ptr(),
            kv_cache_pdata=None,
            kv_cache_view=reference,
        )

        kv_cache, pdata = model._refresh_deepseek_kv_binding(binding)

        self.assertEqual(tuple(kv_cache.shape), (2, 4, 16, 1, 576))
        self.assertIsNot(kv_cache, reference)
        self.assertEqual(kv_cache.dtype, torch.bfloat16)
        self.assertEqual(pdata.shape, [2, 4, 16, 1, 576])

    def test_glm_kv_cache_spec_mutates_registered_attention_layers(self):
        model = _make_model(n_layers=2, first_k_dense=1)
        ctx0_impl = types.SimpleNamespace(
            num_heads=32,
            head_size=192,
            head_size_v=192,
            num_kv_heads=32,
            scale=192 ** -0.5,
            num_queries_per_kv=1,
        )
        ctx1_impl = types.SimpleNamespace(
            num_heads=32,
            head_size=192,
            head_size_v=192,
            num_kv_heads=32,
            scale=192 ** -0.5,
            num_queries_per_kv=1,
        )
        ctx0 = types.SimpleNamespace(
            num_heads=32,
            head_size=192,
            head_size_v=192,
            num_kv_heads=32,
            scale=192 ** -0.5,
            impl=ctx0_impl,
        )
        ctx1 = types.SimpleNamespace(
            num_heads=32,
            head_size=192,
            head_size_v=192,
            num_kv_heads=32,
            scale=192 ** -0.5,
            impl=ctx1_impl,
        )
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": ctx0, "1": ctx1},
        )

        model._configure_glm_kv_cache_spec(types.SimpleNamespace())

        self.assertIs(model.compilation_config.static_forward_context["0"], ctx0)
        self.assertIs(model.compilation_config.static_forward_context["1"], ctx1)
        for ctx in (ctx0, ctx1, ctx0_impl, ctx1_impl):
            self.assertEqual(ctx.num_heads, 32)
            self.assertEqual(ctx.head_size, 576)
            self.assertEqual(ctx.head_size_v, 576)
            self.assertEqual(ctx.num_kv_heads, 1)
            self.assertAlmostEqual(ctx.scale, 192 ** -0.5)
        self.assertEqual(ctx0_impl.num_queries_per_kv, 32)
        self.assertEqual(ctx1_impl.num_queries_per_kv, 32)

    def test_glm_latent_kv_cache_uses_model_dtype_unless_fp8_enabled(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        model.cache_dtype = torch.float8_e4m3fnuz
        reference = torch.empty((2, 3, 16, 32, 192), dtype=torch.float8_e4m3fnuz)

        kv_cache = model._get_glm_latent_kv_cache(0, reference)
        self.assertEqual(kv_cache.dtype, torch.bfloat16)

        model.config.fp8_kv_cache = True
        kv_cache = model._get_glm_latent_kv_cache(1, reference)
        self.assertEqual(kv_cache.dtype, torch.float8_e4m3fnuz)

    def test_glm_sparse_extents_include_block_table_capacity(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        slot_mapping = torch.tensor([0], dtype=torch.int64)
        sparse_indices = torch.tensor([[0]], dtype=torch.int32)
        block_tables = torch.tensor([[0, 31]], dtype=torch.int32)

        required_slots, _, max_block, required_blocks = (
            model._compute_step_slot_extents(
                slot_mapping,
                sparse_indices,
                block_tables,
            )
        )

        self.assertEqual(max_block, 31)
        self.assertEqual(required_blocks, 32)
        self.assertEqual(required_slots, 32 * 16)

    def test_glm_sparse_extents_use_runtime_cache_block_size(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        model.config.kv_cache_block_size = 16
        reference = torch.empty((2, 4, 32, 1, 576), dtype=torch.bfloat16)
        model._deepseek_input_plan = types.SimpleNamespace(first_kv_cache=reference)

        required_slots, _, max_block, required_blocks = (
            model._compute_step_slot_extents(
                torch.tensor([0], dtype=torch.int64),
                torch.tensor([[0]], dtype=torch.int32),
                torch.tensor([[0, 2]], dtype=torch.int32),
            )
        )

        self.assertEqual(max_block, 2)
        self.assertEqual(required_blocks, 3)
        self.assertEqual(required_slots, 3 * 32)

    def test_glm_compressed_offset_input_is_zero(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        self.assertEqual(model._sparse_mla_compressed_offset_input_value(4096), 0)

    def test_glm_shared_sparse_indices_alias_previous_full_indexer(self):
        model = _make_model(n_layers=7, first_k_dense=1)
        model.config.indexer_types = [
            "full",
            "full",
            "full",
            "shared",
            "shared",
            "shared",
            "full",
        ]

        self.assertEqual(model._sparse_mla_runtime_alias_key(0), 0)
        self.assertEqual(model._sparse_mla_runtime_alias_key(3), 2)
        self.assertEqual(model._sparse_mla_runtime_alias_key(4), 2)
        self.assertEqual(model._sparse_mla_runtime_alias_key(5), 2)
        self.assertEqual(model._sparse_mla_runtime_alias_key(6), 6)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_glm_forward_aliases_shared_sparse_index_buffers(self):
        model = _make_model(n_layers=2, first_k_dense=1)
        model.num_layers = 2
        model.config.num_hidden_layers = 2
        model.config.indexer_types = ["full", "shared"]
        model.config.index_topk = 4
        model._index_topk = 4
        model._deepseek_sliding_window = 0
        model._paiton_graph_mode = False

        captured_inputs = {}

        def _run(inputs, outputs, stream_ptr=None, sync=True, graph_mode=False):
            del outputs, stream_ptr, sync, graph_mode
            captured_inputs.update(inputs)

        expected_inputs = {
            "input_ids",
            "position_ids",
            "slot_mapping",
            "query_start_locations",
            "context_lengths",
            "block_tables",
            "max_query_len",
            "max_seq_len",
            "kv_cache_0",
            "kv_cache_1",
            "sparse_mla_indices_0",
            "sparse_mla_indices_1",
            "sparse_mla_topk_length_0",
            "sparse_mla_topk_length_1",
        }
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: expected_inputs,
            run=_run,
        )

        kv_cache0 = torch.zeros((2, 2, 4, 1, 576), dtype=torch.bfloat16, device="cuda")
        kv_cache1 = torch.zeros((2, 2, 4, 1, 576), dtype=torch.bfloat16, device="cuda")
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={
                "0": types.SimpleNamespace(kv_cache=kv_cache0),
                "1": types.SimpleNamespace(kv_cache=kv_cache1),
            },
        )

        attn_metadata = types.SimpleNamespace(
            max_query_len=3,
            max_seq_len=3,
            slot_mapping=torch.tensor([0, 1, 2], dtype=torch.int64, device="cuda"),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([3], dtype=torch.int32, device="cuda"),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        )

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_deepseek_v4.get_forward_context",
            return_value=types.SimpleNamespace(attn_metadata={"0": attn_metadata}),
        ):
            input_ids = torch.zeros((3,), dtype=torch.int64, device="cuda")
            positions = torch.arange(3, dtype=torch.int64, device="cuda")
            model.forward(input_ids, positions)

        self.assertEqual(
            captured_inputs["sparse_mla_indices_0"].data_ptr,
            captured_inputs["sparse_mla_indices_1"].data_ptr,
        )
        self.assertEqual(
            captured_inputs["sparse_mla_topk_length_0"].data_ptr,
            captured_inputs["sparse_mla_topk_length_1"].data_ptr,
        )

    def test_sparse_runtime_capacity_guard_reports_undersized_cache(self):
        cache = torch.empty((7, 1, 64), dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "sparse_mla_kv_1 has 7 rows"):
            PaitonGlmMoeDsaForCausalLM._validate_sparse_runtime_capacity(
                "sparse_mla_kv_1",
                cache,
                8,
            )

    def test_glm_indexer_quant_starts_from_zero_rows(self):
        model = _make_model(n_layers=1, first_k_dense=1)
        cache = torch.empty((512, 1, 128), dtype=torch.bfloat16)
        self.assertEqual(model._indexer_num_existing_rows(0, cache), 0)

    def test_glm_router_aliases_map_to_compiled_gate_proj_names(self):
        """GLM sparse routers load as gate.weight, but artifacts expect gate_proj."""
        model = _make_model(n_layers=4, first_k_dense=3)
        pt = _pt_attn(3)
        pt.update(_pt_sparse_mlp(3))
        pt["model.layers.3.mlp.gate.weight"] = pt.pop(
            "model.layers.3.mlp.gate_proj.weight"
        )
        pt["model.layers.3.mlp.gate.expert_bias"] = pt.pop(
            "model.layers.3.mlp.gate.e_score_correction_bias"
        )

        expected = {
            "layers_3_mlp_gate_proj_weight",
            "layers_3_mlp_gate_e_score_correction_bias",
        }
        mapped = self._run(model, pt, expected=expected)

        self.assertEqual(set(mapped.keys()), expected)


if __name__ == "__main__":
    unittest.main()
