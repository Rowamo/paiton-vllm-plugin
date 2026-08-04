# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Paiton Kimi K3 weight-mapping / loader rules.

These tests exercise ``PaitonKimiK3ForCausalLM.map_pt_params`` against a
synthetic K3 checkpoint tensor stream (no vLLM scheduler, no compiled .so).
They verify the K3-specific loader corrections required by the K3
correctness plan:

- Conv1d depthwise weights are TP-sharded on the channel axis
  ([dim, 1, k] -> [local_proj_dim, k] per rank).
- A_log is TP-sliced per rank from the flattened checkpoint
  ([128] -> [num_local_heads] per rank, ignoring unused trailing values).
- o_norm is tiled from [head_dim] to [local_proj_dim].
- dt_bias is column-parallel sharded to [local_proj_dim].
- f_a_proj is replicated (not TP-sharded) -- the default fallthrough would
  wrongly split it.
- Shape validation rejects malformed special weights.
"""

import os
import types
import unittest
from unittest import mock

import torch

from paiton_vllm_plugin.models.paiton_kimi_k3 import (
    PaitonKimiK3ForCausalLM,
    _PaitonKimiK3StateLayer,
)
from paiton_vllm_plugin.paiton_attention_backend import (
    PaitonKimiK3AttentionBackend,
    PaitonTritonAttentionBackend,
)


def _cache_config(block_size=16, mamba_cache_mode="none"):
    return types.SimpleNamespace(
        block_size=block_size, mamba_cache_mode=mamba_cache_mode,
        mamba_cache_dtype="auto",
    )


def _contract(tp_size=1, block_size=16, dtype="bfloat16",
              mamba_modes=("none", "align")):
    return {
        "version": 5,
        "tp_size": tp_size,
        "ep_size": tp_size,
        "num_hidden_layers": 4,
        "max_batch_size": 1,
        "max_num_batched_tokens": 64,
        "kv_cache_block_size": block_size,
        "decode_partition_size": 512,
        "model_dtype": dtype,
        "cache_dtype": dtype,
        "fp8_kv_cache": False,
        "moe_kernel": "ck_flatmm_fp4",
        "flatmm_required": True,
        "speculative_metadata_supported": False,
        "mamba_cache_modes": list(mamba_modes),
        "kda_conv_state_dtype": dtype,
        "kda_recurrent_state_dtype": "float32",
        "kda_conv_state_packed": True,
        "kda_conv_state_layout": "SD",
        "kda_state_page_stride_aware": True,
        "kda_state_indices_per_layer": True,
        "kda_state_page_stride_runtime": True,
        "mla_cache_layout": "blocks_first",
        "mla_cache_block_stride_runtime": True,
    }


def _contract_model(tp_size=1, contract=None):
    model = PaitonKimiK3ForCausalLM.__new__(PaitonKimiK3ForCausalLM)
    model.tp_size = tp_size
    model.dtype = torch.bfloat16
    model.cache_dtype = torch.bfloat16
    model.config = types.SimpleNamespace(
        paiton_kimi_k3_contract=contract, num_hidden_layers=4
    )
    return model


# K3-like dimensions (small, but divisible by the tested TP sizes).
KDA_NUM_HEADS = 96
KDA_HEAD_DIM = 128
CONV_K = 4
HIDDEN = 7168


def _make_k3(tp_size: int = 1) -> "PaitonKimiK3ForCausalLM":
    model = PaitonKimiK3ForCausalLM.__new__(PaitonKimiK3ForCausalLM)
    model.tp_size = tp_size
    model.num_layers = 1
    model.parallel_config = types.SimpleNamespace(
        expert_placement_strategy="linear",
        enable_expert_parallel=False,
    )
    model.config = types.SimpleNamespace(
        num_hidden_layers=1,
        num_experts=896,
        num_experts_per_token=16,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        hidden_size=HIDDEN,
        num_attention_heads=96,
        moe_intermediate_size=3072,
        vocab_size=163840,
        torch_dtype=torch.bfloat16,
        ep_size=1,
        routed_expert_hidden_size=3584,
        linear_attn_config={
            "num_heads": KDA_NUM_HEADS,
            "head_dim": KDA_HEAD_DIM,
            "short_conv_kernel_size": CONV_K,
        },
    )
    model._n_routed_experts = 896
    model._topk = 16
    model._first_k_dense_replace = 1
    model._moe_layer_freq = 1
    model._has_correction_bias = True
    model._index_topk = 2048
    model._index_skip_topk_offset = 0
    model._index_topk_freq = 1
    model.dtype = torch.bfloat16
    model.cache_dtype = torch.bfloat16
    # KDA dims.
    model._kda_num_heads = KDA_NUM_HEADS
    model._kda_head_dim = KDA_HEAD_DIM
    model._kda_conv_kernel_size = CONV_K
    model._kda_local_proj_dim = (KDA_NUM_HEADS * KDA_HEAD_DIM) // tp_size
    model._kda_local_num_heads = KDA_NUM_HEADS // tp_size
    model._is_kda_layer = [True]
    return model


def _run(model, pt_params, expected=None, tp_size=1, tp_rank=0):
    epgrp = mock.MagicMock()
    epgrp.rank_in_group = 0
    epgrp.world_size = 1
    with mock.patch.dict(os.environ, {"PAITON_MOE_KERNEL": "ck_flatmm_fp4"}), mock.patch(
        "paiton_vllm_plugin.models.paiton_kimi_k3.get_tensor_model_parallel_world_size",
        return_value=tp_size,
    ), mock.patch(
        "paiton_vllm_plugin.models.paiton_kimi_k3.get_tensor_model_parallel_rank",
        return_value=tp_rank,
    ), mock.patch(
        "paiton_vllm_plugin.models.paiton_kimi_k3.get_ep_group",
        return_value=epgrp,
    ):
        return model.map_pt_params(pt_params, expected_constant_names=expected)


class KimiK3LoaderRulesTests(unittest.TestCase):
    proj_dim = KDA_NUM_HEADS * KDA_HEAD_DIM  # 12288

    def _conv_name(self, proj: str) -> str:
        return f"language_model.model.layers.0.self_attn.{proj}_conv1d.weight"

    def test_compact_logits_are_scattered_to_request_last_tokens(self) -> None:
        compact = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)

        output = PaitonKimiK3ForCausalLM._expand_compact_logits(
            compact, query_start_loc, num_tokens=5
        )

        torch.testing.assert_close(output[2], compact[0])
        torch.testing.assert_close(output[4], compact[1])

    def test_compact_logits_skip_zero_length_graph_padding(self) -> None:
        compact = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [90.0, 91.0], [92.0, 93.0]]
        )
        query_start_loc = torch.tensor([0, 2, 3, 3, 3], dtype=torch.int32)

        output = PaitonKimiK3ForCausalLM._expand_compact_logits(
            compact, query_start_loc, num_tokens=3
        )

        torch.testing.assert_close(output[1], compact[0])
        torch.testing.assert_close(output[2], compact[1])

    def test_pure_decode_always_consumes_existing_kda_state(self) -> None:
        metadata = types.SimpleNamespace(
            has_initial_state=None, num_prefills=0, num_decodes=2
        )
        state_indices = torch.tensor([7, 11, -1], dtype=torch.int32)

        result = PaitonKimiK3ForCausalLM._resolve_kda_has_initial_state(
            metadata,
            seq_lens=torch.tensor([8, 5, 0], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
            state_indices=state_indices,
        )

        torch.testing.assert_close(
            result, torch.tensor([1, 1, 0], dtype=torch.int32)
        )

    def test_prefill_preserves_scheduler_has_initial_state_mask(self) -> None:
        scheduler_mask = torch.tensor([False, True])
        metadata = types.SimpleNamespace(
            has_initial_state=scheduler_mask, num_prefills=2, num_decodes=0
        )

        result = PaitonKimiK3ForCausalLM._resolve_kda_has_initial_state(
            metadata,
            seq_lens=torch.tensor([3, 6], dtype=torch.int32),
            query_start_loc=torch.tensor([0, 3, 6], dtype=torch.int32),
            state_indices=torch.tensor([2, 4], dtype=torch.int32),
        )

        torch.testing.assert_close(
            result, torch.tensor([0, 1], dtype=torch.int32)
        )

    def test_kda_metadata_is_selected_per_layer(self) -> None:
        layer0 = types.SimpleNamespace(
            non_spec_state_indices_tensor=torch.tensor([1])
        )
        layer1 = types.SimpleNamespace(
            non_spec_state_indices_tensor=torch.tensor([2])
        )
        metadata = {"0": layer0, "1": layer1}

        self.assertIs(
            PaitonKimiK3ForCausalLM._get_kda_layer_metadata(metadata, 1),
            layer1,
        )
        with self.assertRaisesRegex(RuntimeError, "layer-specific"):
            PaitonKimiK3ForCausalLM._get_kda_layer_metadata(metadata, 2)

    def test_conv1d_tp_sharded_on_channel_axis(self) -> None:
        tp_size = 2
        for tp_rank in range(tp_size):
            model = _make_k3(tp_size=tp_size)
            w = torch.randn(self.proj_dim, 1, CONV_K, dtype=torch.float32)
            local = self.proj_dim // tp_size
            expected = {"layers_0_self_attn_q_conv1d"}
            mapped = _run(
                model, {self._conv_name("q"): w},
                expected=expected, tp_size=tp_size, tp_rank=tp_rank,
            )
            out = mapped[f"layers_0_self_attn_q_conv1d"]
            self.assertEqual(out.shape, (local, CONV_K))
            # Rank r owns channels [r*local:(r+1)*local].
            ref = w.squeeze(1)[tp_rank * local:(tp_rank + 1) * local]
            torch.testing.assert_close(out.cpu(), ref)

    def test_conv1d_rejects_non_divisible_dim(self) -> None:
        model = _make_k3(tp_size=2)
        # dim not divisible by tp_size=2 -> the loader must raise.
        bad = torch.randn(self.proj_dim + 1, 1, CONV_K, dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "divisible by"):
            _run(
                model, {self._conv_name("q"): bad},
                expected={"layers_0_self_attn_q_conv1d"},
                tp_size=2, tp_rank=0,
            )

    def test_a_log_tp_sliced_per_rank(self) -> None:
        tp_size = 8
        local_heads = KDA_NUM_HEADS // tp_size  # 12
        # Flattened checkpoint has 128 values; only first 96 are meaningful.
        a_log = torch.randn(128, dtype=torch.float32)
        for tp_rank in range(tp_size):
            model = _make_k3(tp_size=tp_size)
            mapped = _run(
                model,
                {"language_model.model.layers.0.self_attn.A_log": a_log},
                expected={"layers_0_self_attn_A_log"},
                tp_size=tp_size, tp_rank=tp_rank,
            )
            out = mapped["layers_0_self_attn_A_log"]
            self.assertEqual(out.shape, (local_heads,))
            ref = a_log.reshape(-1)[tp_rank * local_heads:(tp_rank + 1) * local_heads]
            torch.testing.assert_close(out.cpu(), ref)

    def test_a_log_legacy_4d_form_is_flattened(self) -> None:
        # A legacy checkpoint may store A_log as a 4-D tensor; only the
        # flattened first num_heads values must be used.
        tp_size = 1
        local_heads = KDA_NUM_HEADS
        a_log = torch.randn(2, 4, 4, 4, dtype=torch.float32)  # 128 values
        model = _make_k3(tp_size=tp_size)
        mapped = _run(
            model,
            {"language_model.model.layers.0.self_attn.A_log": a_log},
            expected={"layers_0_self_attn_A_log"},
            tp_size=tp_size, tp_rank=0,
        )
        out = mapped["layers_0_self_attn_A_log"]
        self.assertEqual(out.shape, (local_heads,))
        torch.testing.assert_close(out.cpu(), a_log.reshape(-1)[:local_heads])

    def test_o_norm_tiled_to_local_proj_dim(self) -> None:
        tp_size = 4
        local_proj = self.proj_dim // tp_size
        model = _make_k3(tp_size=tp_size)
        w = torch.randn(KDA_HEAD_DIM, dtype=torch.float32)
        mapped = _run(
            model,
            {"language_model.model.layers.0.self_attn.o_norm.weight": w},
            expected={"layers_0_self_attn_o_norm"},
            tp_size=tp_size, tp_rank=0,
        )
        out = mapped["layers_0_self_attn_o_norm"]
        self.assertEqual(out.shape, (local_proj,))
        num_local_heads = local_proj // KDA_HEAD_DIM
        torch.testing.assert_close(out.cpu(), w.repeat(num_local_heads))

    def test_dt_bias_column_parallel_sharded(self) -> None:
        tp_size = 2
        local_proj = self.proj_dim // tp_size
        model = _make_k3(tp_size=tp_size)
        w = torch.randn(self.proj_dim, dtype=torch.float32)
        mapped = _run(
            model,
            {"language_model.model.layers.0.self_attn.dt_bias": w},
            expected={"layers_0_self_attn_dt_bias"},
            tp_size=tp_size, tp_rank=1,
        )
        out = mapped["layers_0_self_attn_dt_bias"]
        self.assertEqual(out.shape, (local_proj,))
        torch.testing.assert_close(out.cpu(), w[local_proj:2 * local_proj])

    def test_f_a_proj_is_replicated_under_tp(self) -> None:
        tp_size = 4
        model = _make_k3(tp_size=tp_size)
        w = torch.randn(KDA_HEAD_DIM, HIDDEN, dtype=torch.bfloat16)
        for tp_rank in range(tp_size):
            mapped = _run(
                model,
                {"language_model.model.layers.0.self_attn.f_a_proj.weight": w},
                expected={"layers_0_self_attn_f_a_proj_weight"},
                tp_size=tp_size, tp_rank=tp_rank,
            )
            out = mapped["layers_0_self_attn_f_a_proj_weight"]
            # Replicated: every rank gets the full [head_dim, hidden] weight.
            self.assertEqual(out.shape, (KDA_HEAD_DIM, HIDDEN))
            torch.testing.assert_close(out.cpu(), w)

    def test_latent_moe_down_and_up_projections_are_replicated(self) -> None:
        tp_size = 8
        # Use reduced dimensions while preserving K3's exact projection
        # orientation. The real BF16 down projection is [3584, 7168] and must
        # bind 51,380,224 bytes on every rank, not one eighth of that.
        hidden, latent = 16, 8
        down = torch.randn(latent, hidden, dtype=torch.bfloat16)
        up = torch.randn(hidden, latent, dtype=torch.bfloat16)
        down_name = (
            "language_model.model.layers.0.block_sparse_moe."
            "routed_expert_down_proj.weight"
        )
        up_name = (
            "language_model.model.layers.0.block_sparse_moe."
            "routed_expert_up_proj.weight"
        )
        expected_names = {
            "layers_0_mlp_routed_expert_down_proj_weight",
            "layers_0_mlp_routed_expert_up_proj_weight",
        }
        for tp_rank in range(tp_size):
            model = _make_k3(tp_size=tp_size)
            model.config.hidden_size = hidden
            model.config.routed_expert_hidden_size = latent
            mapped = _run(
                model,
                {down_name: down, up_name: up},
                expected=expected_names,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            torch.testing.assert_close(
                mapped["layers_0_mlp_routed_expert_down_proj_weight"].cpu(),
                down,
            )
            torch.testing.assert_close(
                mapped["layers_0_mlp_routed_expert_up_proj_weight"].cpu(),
                up,
            )

    def test_latent_moe_projection_shape_is_validated(self) -> None:
        model = _make_k3(tp_size=8)
        name = (
            "language_model.model.layers.0.block_sparse_moe."
            "routed_expert_down_proj.weight"
        )
        bad = torch.empty(8, 15, dtype=torch.bfloat16)
        model.config.hidden_size = 16
        model.config.routed_expert_hidden_size = 8
        with self.assertRaisesRegex(RuntimeError, "expected replicated"):
            _run(
                model,
                {name: bad},
                expected={"layers_0_mlp_routed_expert_down_proj_weight"},
                tp_size=8,
                tp_rank=0,
            )


class KimiK3ContractValidationTests(unittest.TestCase):
    def _vllm_config(self, block_size=16, mamba_cache_mode="none"):
        return types.SimpleNamespace(
            cache_config=_cache_config(block_size, mamba_cache_mode),
            speculative_config=None,
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        )

    def test_legacy_artifact_without_contract_is_rejected(self) -> None:
        model = _contract_model(tp_size=1, contract=None)
        with self.assertRaisesRegex(RuntimeError, "Recompile"):
            model._validate_k3_contract(self._vllm_config())

    def test_tp_mismatch_is_rejected(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract(tp_size=8))
        with self.assertRaisesRegex(RuntimeError, "TP size mismatch"):
            model._validate_k3_contract(self._vllm_config())

    def test_block_size_mismatch_is_rejected(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract(block_size=16))
        with self.assertRaisesRegex(RuntimeError, "block_size mismatch"):
            model._validate_k3_contract(self._vllm_config(block_size=32))

    def test_mamba_cache_mode_all_is_rejected(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract())
        with self.assertRaisesRegex(RuntimeError, "mamba_cache_mode"):
            model._validate_k3_contract(
                self._vllm_config(mamba_cache_mode="all")
            )

    def test_mamba_cache_mode_none_and_align_accepted(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract())
        for mode in ("none", "align"):
            contract = model._validate_k3_contract(
                self._vllm_config(mamba_cache_mode=mode)
            )
            self.assertEqual(contract["tp_size"], 1)
            self.assertIn(mode, contract["mamba_cache_modes"])

    def test_legacy_per_projection_state_contract_is_rejected(self) -> None:
        legacy = _contract()
        legacy["version"] = 1
        legacy.pop("kda_conv_state_packed")
        model = _contract_model(tp_size=1, contract=legacy)
        with self.assertRaisesRegex(RuntimeError, "legacy per-projection"):
            model._validate_k3_contract(self._vllm_config())

    def test_legacy_kv_first_mla_contract_is_rejected(self) -> None:
        legacy = _contract()
        legacy["version"] = 2
        legacy.pop("mla_cache_layout")
        model = _contract_model(tp_size=1, contract=legacy)
        with self.assertRaisesRegex(RuntimeError, "K/V-first MLA cache"):
            model._validate_k3_contract(self._vllm_config())

    def test_artifact_without_runtime_mla_stride_is_rejected(self) -> None:
        legacy = _contract()
        legacy["mla_cache_block_stride_runtime"] = False
        model = _contract_model(tp_size=1, contract=legacy)
        with self.assertRaisesRegex(RuntimeError, "runtime MLA cache block stride"):
            model._validate_k3_contract(self._vllm_config())

    def test_artifact_without_kda_page_stride_is_rejected(self) -> None:
        legacy = _contract()
        legacy["version"] = 3
        legacy.pop("kda_state_page_stride_aware")
        model = _contract_model(tp_size=1, contract=legacy)
        with self.assertRaisesRegex(RuntimeError, "contiguous KDA state rows"):
            model._validate_k3_contract(self._vllm_config())

    def test_artifact_without_per_layer_kda_metadata_is_rejected(self) -> None:
        legacy = _contract()
        legacy["version"] = 4
        legacy.pop("kda_state_indices_per_layer")
        legacy.pop("kda_state_page_stride_runtime")
        model = _contract_model(tp_size=1, contract=legacy)
        with self.assertRaisesRegex(RuntimeError, "one KDA state index/stride"):
            model._validate_k3_contract(self._vllm_config())

    def test_conv_state_layout_mismatch_is_rejected(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract())
        with mock.patch(
            "paiton_vllm_plugin.models.paiton_kimi_k3.get_conv_state_layout",
            return_value="DS",
        ), self.assertRaisesRegex(RuntimeError, "layout mismatch"):
            model._validate_k3_contract(self._vllm_config())

    def test_speculative_runtime_is_rejected_when_contract_is_false(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract())
        config = self._vllm_config()
        config.speculative_config = types.SimpleNamespace(num_speculative_tokens=1)
        with self.assertRaisesRegex(RuntimeError, "does not support speculative"):
            model._validate_k3_contract(config)

    def test_scheduler_capacity_above_artifact_is_rejected(self) -> None:
        model = _contract_model(tp_size=1, contract=_contract())
        config = self._vllm_config()
        config.scheduler_config = types.SimpleNamespace(
            max_num_batched_tokens=65, max_num_seqs=1
        )
        with self.assertRaisesRegex(RuntimeError, "token capacity"):
            model._validate_k3_contract(config)

    def test_mamba_state_dtype_mismatch_is_rejected(self) -> None:
        contract = _contract()
        contract["kda_conv_state_dtype"] = "float16"
        model = _contract_model(tp_size=1, contract=contract)
        with self.assertRaisesRegex(RuntimeError, "state dtype mismatch"):
            model._validate_k3_contract(self._vllm_config())


class KimiK3StateDescriptorTests(unittest.TestCase):
    def test_mamba_state_shape_uses_kda_defaults_for_partial_config(self) -> None:
        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(num_attention_heads=80),
            ),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=2),
            speculative_config=None,
        )
        with mock.patch(
            "paiton_vllm_plugin.models.paiton_kimi_k3."
            "MambaStateShapeCalculator.kda_state_shape",
            return_value="shape",
        ) as state_shape:
            self.assertEqual(
                PaitonKimiK3ForCausalLM.get_mamba_state_shape_from_config(
                    vllm_config
                ),
                "shape",
            )
        state_shape.assert_called_once_with(
            2, 80, 128, conv_kernel_size=4, num_spec=0
        )

    def test_attn_residual_scratch_bank_is_reused_and_cleared(self) -> None:
        model = _make_k3(tp_size=1)
        model._num_attn_res_blocks = 2
        model._attn_res_block_residual_capacity = 4
        model._attn_res_block_residual_bank = None

        first = model._attn_res_block_residual_for_forward(3, torch.device("cpu"))
        first.fill_(1)
        second = model._attn_res_block_residual_for_forward(2, torch.device("cpu"))

        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertTrue(torch.count_nonzero(second).eq(0))

    def test_paiton_backends_expose_rank_matching_stride_orders(self) -> None:
        self.assertEqual(
            PaitonTritonAttentionBackend.get_kv_cache_shape(4, 16, 1, 576),
            (2, 4, 16, 1, 576),
        )
        self.assertEqual(
            PaitonTritonAttentionBackend.get_kv_cache_stride_order(),
            (0, 1, 2, 3, 4),
        )
        self.assertEqual(
            PaitonKimiK3AttentionBackend.get_kv_cache_shape(4, 16, 1, 576),
            (4, 2, 16, 1, 576),
        )
        self.assertEqual(
            PaitonKimiK3AttentionBackend.get_kv_cache_stride_order(),
            (0, 1, 2, 3, 4),
        )
        self.assertEqual(
            PaitonKimiK3AttentionBackend.get_kv_cache_stride_order(True),
            (1, 0, 2, 3, 4, 5),
        )
        self.assertTrue(PaitonKimiK3AttentionBackend.indexes_kv_by_block_stride())

    def test_latent_cache_requires_blocks_first_layout(self) -> None:
        model = _make_k3(tp_size=1)
        cache = torch.empty(4, 2, 16, 1, 576, dtype=torch.bfloat16)
        self.assertIs(model._get_glm_latent_kv_cache(1, cache), cache)
        legacy = torch.empty(2, 4, 16, 1, 576, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "latent MLA cache layout"):
            model._get_glm_latent_kv_cache(1, legacy)

    def test_latent_cache_reports_padded_runtime_block_stride(self) -> None:
        model = _make_k3(tp_size=1)
        logical_page = 2 * 16 * 576
        physical_stride = logical_page + 128
        backing = torch.empty(4 * physical_stride, dtype=torch.bfloat16)
        cache = torch.as_strided(
            backing,
            size=(4, 2, 16, 1, 576),
            stride=(physical_stride, 16 * 576, 576, 576, 1),
        )
        self.assertIs(model._get_glm_latent_kv_cache(1, cache), cache)
        self.assertEqual(
            model._get_mla_cache_block_stride(1, cache), physical_stride
        )

    def test_latent_cache_rejects_promoted_logical_block_size(self) -> None:
        model = _make_k3(tp_size=1)
        model.config.kv_cache_block_size = 16
        incompatible = torch.empty(4, 2, 720, 1, 576, dtype=torch.bfloat16)

        with self.assertRaisesRegex(RuntimeError, "latent MLA cache layout"):
            model._get_glm_latent_kv_cache(1, incompatible)

    def test_bound_mla_cache_supports_current_and_legacy_vllm_binding(self) -> None:
        model = _make_k3(tp_size=1)
        cache = torch.empty(4, 2, 16, 1, 576, dtype=torch.bfloat16)
        ctx = types.SimpleNamespace(kv_cache=cache)
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"1": ctx}
        )

        # Current vLLM binds the tensor directly. It must not be indexed at
        # [0], which would drop the blocks dimension and produce a rank-4 view.
        self.assertIs(model._get_bound_mla_cache(1), cache)

        # Keep compatibility with vLLM versions that bind [tensor].
        ctx.kv_cache = [cache]
        self.assertIs(model._get_bound_mla_cache(1), cache)

        ctx.kv_cache = []
        with self.assertRaisesRegex(RuntimeError, "no vLLM-bound KV cache"):
            model._get_bound_mla_cache(1)

    def test_runtime_input_selection_drops_pruned_nope_inputs(self) -> None:
        candidates = {
            "input_ids": object(),
            "position_ids": object(),
            "max_query_len": object(),
            "max_seq_len": object(),
            "state_indices": object(),
        }
        expected = {"input_ids", "state_indices"}

        selected = PaitonKimiK3ForCausalLM._select_expected_inputs(
            candidates, expected
        )

        self.assertEqual(set(selected), expected)
        PaitonKimiK3ForCausalLM._validate_runtime_input_names(selected, expected)

        with self.assertRaisesRegex(
            RuntimeError, r"missing=\['state_indices'\].*unexpected=\['extra'\]"
        ):
            PaitonKimiK3ForCausalLM._validate_runtime_input_names(
                {"input_ids": object(), "extra": object()}, expected
            )

    def test_descriptor_matches_vllm_packed_kda_state(self) -> None:
        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
            cache_config=_cache_config(),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=8),
        )
        layer = _PaitonKimiK3StateLayer(
            vllm_config,
            num_heads=KDA_NUM_HEADS,
            head_dim=KDA_HEAD_DIM,
            conv_size=CONV_K,
        )
        conv_shape, recurrent_shape = layer.get_state_shape()
        packed_dim = 3 * KDA_NUM_HEADS * KDA_HEAD_DIM // 8
        self.assertIn(
            conv_shape,
            ((CONV_K - 1, packed_dim), (packed_dim, CONV_K - 1)),
        )
        self.assertEqual(recurrent_shape, (KDA_NUM_HEADS // 8, KDA_HEAD_DIM, KDA_HEAD_DIM))
        self.assertEqual(layer.get_state_dtype(), (torch.bfloat16, torch.float32))

    def test_register_replaces_only_kda_layers(self) -> None:
        model = _make_k3(tp_size=1)
        model.num_layers = 2
        model._is_kda_layer = [True, False]
        mla_sentinel = types.SimpleNamespace(attn_backend=None)
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": object(), "1": mla_sentinel}
        )
        vllm_config = types.SimpleNamespace(
            model_config=types.SimpleNamespace(dtype=torch.bfloat16),
            cache_config=_cache_config(),
            parallel_config=types.SimpleNamespace(tensor_parallel_size=1),
        )
        model._register_kda_state_layers(vllm_config)
        self.assertIsInstance(
            model.compilation_config.static_forward_context["0"],
            _PaitonKimiK3StateLayer,
        )
        self.assertIs(model.compilation_config.static_forward_context["1"], mla_sentinel)
        self.assertIs(mla_sentinel.attn_backend, PaitonKimiK3AttentionBackend)


if __name__ == "__main__":
    unittest.main()
