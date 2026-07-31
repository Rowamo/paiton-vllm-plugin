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


def _cache_config(block_size=16, mamba_cache_mode="none"):
    return types.SimpleNamespace(
        block_size=block_size, mamba_cache_mode=mamba_cache_mode,
        mamba_cache_dtype="auto",
    )


def _contract(tp_size=1, block_size=16, dtype="bfloat16",
              mamba_modes=("none", "align")):
    return {
        "version": 2,
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

    def test_conv1d_tp_sharded_on_channel_axis(self) -> None:
        tp_size = 2
        for tp_rank in range(tp_size):
            model = _make_k3(tp_size=tp_size)
            w = torch.randn(self.proj_dim, 1, CONV_K, dtype=torch.float32)
            local = self.proj_dim // tp_size
            expected = {f"layers_0_self_attn_q_conv1d"}
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
        mla_sentinel = object()
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


if __name__ == "__main__":
    unittest.main()
