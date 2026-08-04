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
    model._paiton_graph_mode = False
    return model


def _find_backing_tensor(
    backings: list[torch.Tensor],
    data_ptr: int,
) -> torch.Tensor:
    for tensor in backings:
        if tensor.data_ptr() == data_ptr:
            return tensor
    raise AssertionError(f"missing backing tensor for ptr={data_ptr}")


class _CountingKvContext:
    def __init__(self, kv_cache: torch.Tensor) -> None:
        self._kv_cache = kv_cache
        self.kv_cache_reads = 0

    @property
    def kv_cache(self) -> torch.Tensor:
        self.kv_cache_reads += 1
        return self._kv_cache

    @kv_cache.setter
    def kv_cache(self, value: torch.Tensor) -> None:
        self._kv_cache = value


def _shuffle_fp8_weight(weight: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    in_rows, in_cols = layout
    block_cols = in_cols * 2
    elems_per_16b = 16 // weight.element_size()
    weight_view = weight.view(
        -1,
        weight.shape[-2] // in_rows,
        in_rows,
        weight.shape[-1] // block_cols,
        block_cols // elems_per_16b,
        elems_per_16b,
    )
    weight_view = weight_view.permute(0, 1, 3, 4, 2, 5).contiguous()
    return weight_view.view(*weight.shape)


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

    def test_deepseek_input_plan_caches_kv_cache_pdata(self) -> None:
        model = _make_model()
        model.num_layers = 2
        model.cache_dtype = torch.uint8
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {
                "kv_cache_0": 0,
                "kv_cache_1": 1,
                "input_ids": 2,
            },
        )
        ctx0 = _CountingKvContext(torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8))
        ctx1 = _CountingKvContext(torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8))
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": ctx0, "1": ctx1},
        )

        plan0 = model._get_deepseek_input_plan()
        plan1 = model._get_deepseek_input_plan()

        self.assertIs(plan0, plan1)
        self.assertEqual(ctx0.kv_cache_reads, 1)
        self.assertEqual(ctx1.kv_cache_reads, 1)
        self.assertEqual(
            plan0.ordered_input_names,
            ("kv_cache_0", "kv_cache_1", "input_ids"),
        )
        self.assertIs(
            plan0.layer_bindings[0].kv_cache_pdata,
            plan1.layer_bindings[0].kv_cache_pdata,
        )

    def test_deepseek_input_plan_rebuilds_when_static_context_changes(self) -> None:
        model = _make_model()
        model.num_layers = 1
        model.cache_dtype = torch.uint8
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {
                "kv_cache_0": 0,
                "input_ids": 1,
            },
        )
        ctx0 = _CountingKvContext(torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8))
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": ctx0},
        )
        plan0 = model._get_deepseek_input_plan()

        ctx1 = _CountingKvContext(torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8))
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": ctx1},
        )
        plan1 = model._get_deepseek_input_plan()

        self.assertIsNot(plan0, plan1)
        self.assertIs(plan1.layer_bindings[0].kv_cache, ctx1._kv_cache)

    def test_deepseek_kv_binding_refresh_can_validate_context_replacement(self) -> None:
        model = _make_model()
        model.num_layers = 1
        model.cache_dtype = torch.uint8
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {
                "kv_cache_0": 0,
                "input_ids": 1,
            },
        )
        ctx = _CountingKvContext(torch.empty((2, 2, 4, 1, 8), dtype=torch.uint8))
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": ctx},
        )
        binding = model._get_deepseek_input_plan().layer_bindings[0]

        replacement = torch.empty((2, 3, 4, 1, 8), dtype=torch.uint8)
        ctx.kv_cache = replacement
        kv_cache, pdata = model._refresh_deepseek_kv_binding(
            binding,
            validate_context=True,
        )

        self.assertIs(kv_cache, replacement)
        self.assertEqual(pdata.data_ptr, replacement.data_ptr())
        self.assertEqual(pdata.shape, list(replacement.shape))

    def test_forward_bound_run_updates_only_pointers_for_stable_shapes(self) -> None:
        model = _make_model()
        model.num_layers = 0
        model.cache_dtype = torch.uint8

        bind_calls = []
        update_calls = []
        run_bound_calls = []

        class _Runtime:
            _bound_run_available = True

            def get_input_name_to_index_map(self):
                return {
                    "input_ids": 0,
                    "position_ids": 1,
                    "slot_mapping": 2,
                    "query_start_locations": 3,
                    "context_lengths": 4,
                    "block_tables": 5,
                    "max_query_len": 6,
                    "max_seq_len": 7,
                }

            def bind_inputs(self, ordered_inputs):
                bind_calls.append([pd.data_ptr for pd in ordered_inputs])

            def update_input_pointers(self, ptrs):
                update_calls.append(list(ptrs))

            def run_bound(self, outputs, stream_ptr=None, sync=True,
                          graph_mode=False):
                del outputs, graph_mode
                run_bound_calls.append((stream_ptr, sync))

        model.model = _Runtime()
        model.compilation_config = types.SimpleNamespace(static_forward_context={})
        attn_metadata = types.SimpleNamespace(
            max_query_len=1,
            max_seq_len=1,
            slot_mapping=torch.zeros((1,), dtype=torch.int64),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([1], dtype=torch.int32),
            block_table=torch.zeros((1, 1), dtype=torch.int32),
        )

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_deepseek_v4.get_forward_context",
            return_value=types.SimpleNamespace(attn_metadata={"0": attn_metadata}),
        ), mock.patch(
            "torch.cuda.current_stream",
            return_value=types.SimpleNamespace(cuda_stream=123),
        ):
            input_ids = torch.zeros((1,), dtype=torch.int64)
            positions = torch.zeros((1,), dtype=torch.int64)
            model.forward(input_ids, positions)
            model.forward(input_ids, positions)

        self.assertEqual(len(bind_calls), 1)
        self.assertEqual(len(update_calls), 1)
        self.assertEqual(run_bound_calls, [(123, False), (123, False)])
        self.assertEqual(len(update_calls[0]), 8)

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

    def test_base_non_fp8_weight_conversion_keeps_cpu_tensor(self) -> None:
        weight = torch.tensor([1.0], dtype=torch.float32)

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_base.runtime_uses_fnuz_fp8",
            return_value=False,
        ):
            got = PaitonModelBase._convert_fp8_weights(object(), weight)

        self.assertIs(got, weight)
        self.assertFalse(got.is_cuda)

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

    def test_recent_window_indices_overlap_c4_compressor_write_slots(self) -> None:
        """Diagnostic: short prompts already alias raw-SWA and compressed slots.

        For compress_ratio=4, the compressor writes compressed keys at token
        slots 3, 7, 11, ... . The runtime's recent-window sparse indices for a
        short prompt include those same physical slots, so a single
        ``sparse_mla_kv`` cache cannot preserve both the raw recent-window keys
        and the compressed keys at once.
        """
        query_start_loc = torch.tensor([0, 8], dtype=torch.int32)
        seq_lens = torch.tensor([8], dtype=torch.int32)
        block_table = torch.tensor([[0, 1]], dtype=torch.int32)

        indices, topk_length = (
            PaitonDeepseekV4ForCausalLM._build_recent_sparse_mla_indices(
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                num_tokens=8,
                index_topk=8,
                block_size=4,
                device=torch.device("cpu"),
                recent_window=8,
            )
        )

        self.assertEqual(topk_length[-1].item(), 8)
        recent_slots = {slot for slot in indices[-1].tolist() if slot >= 0}
        compressor_write_slots = {3, 7}
        self.assertEqual(recent_slots, set(range(8)))
        self.assertEqual(recent_slots & compressor_write_slots, compressor_write_slots)

    def test_generic_sparse_metadata_lookup_misses_c128a_fields(self) -> None:
        attn_metadata = types.SimpleNamespace(
            c128a_global_decode_topk_indices=torch.tensor([[11, 15]], dtype=torch.int32),
            c128a_decode_topk_lens=torch.tensor([2], dtype=torch.int32),
            c128a_prefill_topk_indices=torch.tensor([[0, 1]], dtype=torch.int32),
        )

        sparse_indices = PaitonDeepseekV4ForCausalLM._first_tensor_attr(
            attn_metadata,
            (
                "sparse_mla_indices",
                "sparse_mla_topk_indices",
                "topk_indices",
                "topk_indices_buffer",
            ),
        )
        sparse_lens = PaitonDeepseekV4ForCausalLM._first_tensor_attr(
            attn_metadata,
            (
                "sparse_mla_topk_length",
                "sparse_mla_topk_lengths",
                "topk_length",
                "topk_lengths",
            ),
        )

        self.assertIsNone(sparse_indices)
        self.assertIsNone(sparse_lens)

    def test_synthesized_c128a_metadata_covers_prefill_rows(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        model._deepseek_sliding_window = 128

        synth = model._synthesize_c128a_metadata(
            positions=torch.tensor([127, 255], dtype=torch.int64),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            compress_ratio=128,
            slot_mapping=torch.arange(2, dtype=torch.int64),
        )

        self.assertNotIn("c128a_global_decode_topk_indices", synth)
        self.assertNotIn("c128a_decode_topk_lens", synth)
        self.assertEqual(tuple(synth["c128a_prefill_topk_indices"].shape), (2, 128))
        self.assertEqual(
            synth["c128a_prefill_topk_indices"][0, :2].tolist(),
            [0, -1],
        )
        self.assertEqual(
            synth["c128a_prefill_topk_indices"][1, :3].tolist(),
            [0, 1, -1],
        )

    def test_build_c128_sparse_inputs_synthesizes_missing_metadata(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        model.config.index_topk = 4
        model._deepseek_sliding_window = 2

        recent_indices = torch.tensor(
            [[20, 21, -1, -1], [30, 31, -1, -1]],
            dtype=torch.int32,
        )
        recent_topk_length = torch.tensor([2, 2], dtype=torch.int32)

        sparse_indices, sparse_topk_length = model._build_c128a_sparse_inputs(
            types.SimpleNamespace(
                block_table=torch.tensor([[5, 6, -1]], dtype=torch.int32),
                block_size=256,
            ),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            compressed_slot_offset=100,
            recent_indices=recent_indices,
            recent_topk_length=recent_topk_length,
            layer_idx=0,
            positions=torch.tensor([127, 255], dtype=torch.int64),
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        )

        self.assertEqual(tuple(sparse_indices.shape), (2, 128))
        self.assertEqual(sparse_indices[0, :3].tolist(), [1507, 20, 21])
        self.assertEqual(sparse_indices[1, :4].tolist(), [1507, 1635, 30, 31])
        self.assertEqual(sparse_topk_length.tolist(), [3, 4])

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

        self.assertGreaterEqual(cache.shape[0], 7)
        self.assertEqual(tuple(cache.shape[1:]), (1, 8))

    def test_sparse_mla_compressed_offset_uses_raw_cache_capacity(self) -> None:
        slot_mapping = torch.tensor([1], dtype=torch.int64)
        sparse_indices = torch.tensor([[1, 6]], dtype=torch.int32)

        offset = PaitonDeepseekV4ForCausalLM._compressed_sparse_slot_offset(
            slot_mapping,
            sparse_indices,
        )

        self.assertEqual(offset, 7)

    def test_sparse_mla_cache_doubles_capacity_for_compressed_namespace(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 5, 4, 1, 8), dtype=torch.uint8)
        slot_mapping = torch.tensor([1], dtype=torch.int64)
        sparse_indices = torch.tensor([[1, 6]], dtype=torch.int32)

        cache = model._get_sparse_mla_kv_cache(
            0,
            kv_cache,
            slot_mapping,
            sparse_indices,
            compressed_slot_offset=7,
        )

        self.assertGreaterEqual(cache.shape[0], 14)
        self.assertEqual(tuple(cache.shape[1:]), (1, 8))

    def test_sparse_mla_cache_rehomes_compressed_rows_when_offset_grows(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 5, 4, 1, 8), dtype=torch.uint8)

        cache = model._get_sparse_mla_kv_cache(
            0,
            kv_cache,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int32),
            compressed_slot_offset=2,
        )
        cache[0, 0, 0] = 11
        cache[2, 0, 0] = 22

        grown = model._get_sparse_mla_kv_cache(
            0,
            kv_cache,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[1, 6]], dtype=torch.int32),
            compressed_slot_offset=7,
        )

        self.assertEqual(grown[0, 0, 0].item(), 11)
        self.assertEqual(grown[7, 0, 0].item(), 22)

    def test_sparse_mla_cache_reuses_capacity_with_stable_rounded_offset(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 128, 4, 1, 8), dtype=torch.uint8)

        first = model._get_sparse_mla_kv_cache(
            0,
            kv_cache,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[1, 120]], dtype=torch.int32),
            compressed_slot_offset=256,
            required_slots=121,
        )
        second = model._get_sparse_mla_kv_cache(
            0,
            kv_cache,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[1, 180]], dtype=torch.int32),
            compressed_slot_offset=256,
            required_slots=181,
        )

        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertGreaterEqual(second.shape[0], 512)

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

        self.assertGreaterEqual(cache.shape[0], 7)
        self.assertEqual(tuple(cache.shape[1:]), (1, 3))
        self.assertEqual(cache.dtype, torch.bfloat16)

    def test_sparse_mla_indexer_cache_doubles_capacity_for_compressed_namespace(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 5, 4, 1, 8), dtype=torch.uint8)
        slot_mapping = torch.tensor([1], dtype=torch.int64)
        sparse_indices = torch.tensor([[1, 6]], dtype=torch.int32)

        cache = model._get_sparse_mla_indexer_kv_cache(
            0,
            kv_cache,
            slot_mapping,
            sparse_indices,
            compressed_slot_offset=7,
        )

        self.assertGreaterEqual(cache.shape[0], 14)
        self.assertEqual(tuple(cache.shape[1:]), (1, 3))

    def test_sparse_mla_indexer_cache_rehomes_compressed_rows_when_offset_grows(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 5, 4, 1, 8), dtype=torch.uint8)

        cache = model._get_sparse_mla_indexer_kv_cache(
            0,
            kv_cache,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[1]], dtype=torch.int32),
            compressed_slot_offset=2,
        )
        cache[0, 0, 0] = 33
        cache[2, 0, 0] = 44

        grown = model._get_sparse_mla_indexer_kv_cache(
            0,
            kv_cache,
            torch.tensor([1], dtype=torch.int64),
            torch.tensor([[1, 6]], dtype=torch.int32),
            compressed_slot_offset=7,
        )

        self.assertEqual(grown[0, 0, 0].item(), 33)
        self.assertEqual(grown[7, 0, 0].item(), 44)

    def test_c128_prefill_local_indices_map_to_compressed_slots(self) -> None:
        mapped = (
            PaitonDeepseekV4ForCausalLM._map_c128a_prefill_local_indices_to_slots(
                prefill_local_indices=torch.tensor(
                    [[0, 1, -1], [0, 1, 2]],
                    dtype=torch.int32,
                ),
                block_table=torch.tensor([[5, 6, -1]], dtype=torch.int32),
                query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
                compress_ratio=128,
                compressed_slot_offset=100,
                block_size=256,
            )
        )

        self.assertEqual(
            mapped.tolist(),
            [
                [1507, 1635, -1],
                [1507, 1635, 1763],
            ],
        )

    def test_c128_dense_decode_slots_map_to_compiler_slots(self) -> None:
        mapped = PaitonDeepseekV4ForCausalLM._map_c128_dense_slots_to_compiler_slots(
            torch.tensor(
                [[0, 1, -1], [2, 3, -1]],
                dtype=torch.int32,
            ),
            compressed_slot_offset=100,
            compress_ratio=128,
            block_size=256,
        )

        self.assertEqual(
            mapped.tolist(),
            [
                [227, 355, -1],
                [483, 611, -1],
            ],
        )

    def test_c128_dense_decode_slots_request_aware_no_aliasing(self) -> None:
        """Regression test: batched decode must not alias compressed KV slots.

        With 2 decode requests whose synthesized dense indices both start at 0,
        the request-aware mapping must produce distinct physical slots per
        request (via per-request block_table lookup). The legacy non-request-
        aware mapping would map d=0 for both requests to the same slot.
        """
        # 2 decode tokens, both with dense indices [0, 1]
        dense_indices = torch.tensor(
            [[0, 1], [0, 1]],
            dtype=torch.int32,
        )
        # 2 requests with different block tables
        block_table = torch.tensor(
            [[5, 6, -1], [10, 11, -1]],
            dtype=torch.int32,
        )
        query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)

        mapped = PaitonDeepseekV4ForCausalLM._map_c128_dense_slots_to_compiler_slots(
            dense_indices,
            compressed_slot_offset=100,
            compress_ratio=128,
            block_size=256,
            block_table=block_table,
            query_start_loc=query_start_loc,
        )

        # Request 0: block_table[0]=[5,6,-1]
        #   d=0: block_id=5, raw_offset=0*128+127=127 -> 100+5*256+127 = 1507
        #   d=1: block_id=5, raw_offset=1*128+127=255 -> 100+5*256+255 = 1635
        # Request 1: block_table[1]=[10,11,-1]
        #   d=0: block_id=10, raw_offset=0*128+127=127 -> 100+10*256+127 = 2787
        #   d=1: block_id=10, raw_offset=1*128+127=255 -> 100+10*256+255 = 2915
        self.assertEqual(mapped[0].tolist(), [1507, 1635])
        self.assertEqual(mapped[1].tolist(), [2787, 2915])

        # Critical: no slot overlap between the two requests
        slots_0 = set(mapped[0].tolist())
        slots_1 = set(mapped[1].tolist())
        overlap = slots_0 & slots_1
        self.assertEqual(
            overlap, set(),
            f"Cross-request slot aliasing detected: {overlap}",
        )

    def test_c128_sparse_inputs_preserve_recent_window_when_no_compressed_tokens(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        model._deepseek_sliding_window = 4

        recent_indices = torch.tensor(
            [
                [7, -1, -1, -1],
                [7, 8, -1, -1],
                [7, 8, 9, -1],
            ],
            dtype=torch.int32,
        )
        recent_topk_length = torch.tensor([1, 2, 3], dtype=torch.int32)
        layer_attn_metadata = types.SimpleNamespace(
            c128a_prefill_topk_indices=torch.full((3, 4), -1, dtype=torch.int32),
            block_table=torch.tensor([[3, 4]], dtype=torch.int32),
            block_size=256,
        )

        sparse_indices, sparse_topk_length = model._build_c128a_sparse_inputs(
            layer_attn_metadata,
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
            compressed_slot_offset=16,
            recent_indices=recent_indices,
            recent_topk_length=recent_topk_length,
            layer_idx=0,
        )

        self.assertEqual(tuple(sparse_indices.shape), (3, 128))
        self.assertTrue(torch.equal(sparse_indices[:, :4], recent_indices))
        self.assertTrue(torch.equal(sparse_indices[:, 4:], torch.full((3, 124), -1, dtype=torch.int32)))
        self.assertEqual(sparse_topk_length.tolist(), recent_topk_length.tolist())

    def test_c128_sparse_inputs_fall_back_to_recent_window_when_metadata_is_absent(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        model._deepseek_sliding_window = 4

        recent_indices = torch.tensor(
            [
                [7, -1, -1, -1],
                [7, 8, -1, -1],
            ],
            dtype=torch.int32,
        )
        recent_topk_length = torch.tensor([1, 2], dtype=torch.int32)

        sparse_indices, sparse_topk_length = model._build_c128a_sparse_inputs(
            types.SimpleNamespace(block_table=torch.tensor([[3, 4]], dtype=torch.int32)),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            compressed_slot_offset=16,
            recent_indices=recent_indices,
            recent_topk_length=recent_topk_length,
            layer_idx=0,
        )

        self.assertEqual(tuple(sparse_indices.shape), (2, 128))
        self.assertTrue(torch.equal(sparse_indices[:, :4], recent_indices))
        self.assertEqual(sparse_topk_length.tolist(), recent_topk_length.tolist())

    def test_c128_sparse_inputs_prefix_mapped_compressed_slots_before_recent_window(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        model.config.index_topk = 4
        model._deepseek_sliding_window = 2

        layer_attn_metadata = types.SimpleNamespace(
            c128a_prefill_topk_indices=torch.tensor(
                [[0, 1, -1, -1], [0, 1, 2, -1]],
                dtype=torch.int32,
            ),
            block_table=torch.tensor([[5, 6, -1]], dtype=torch.int32),
            block_size=256,
        )
        recent_indices = torch.tensor(
            [[20, 21, -1, -1], [30, 31, -1, -1]],
            dtype=torch.int32,
        )
        recent_topk_length = torch.tensor([2, 2], dtype=torch.int32)

        sparse_indices, sparse_topk_length = model._build_c128a_sparse_inputs(
            layer_attn_metadata,
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
            compressed_slot_offset=100,
            recent_indices=recent_indices,
            recent_topk_length=recent_topk_length,
            layer_idx=0,
        )

        self.assertEqual(tuple(sparse_indices.shape), (2, 128))
        self.assertEqual(sparse_indices[0, :4].tolist(), [1507, 1635, 20, 21])
        self.assertEqual(sparse_indices[1, :5].tolist(), [1507, 1635, 1763, 30, 31])
        self.assertTrue(torch.equal(sparse_indices[:, 5:], torch.full((2, 123), -1, dtype=torch.int32)))
        self.assertEqual(sparse_topk_length.tolist(), [4, 5])

    def test_c128_sparse_inputs_remap_dense_decode_slots(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        model.config.index_topk = 4
        model._deepseek_sliding_window = 2

        layer_attn_metadata = types.SimpleNamespace(
            c128a_global_decode_topk_indices=torch.tensor(
                [[[0, 1, -1, -1]]],
                dtype=torch.int32,
            ),
            c128a_decode_topk_lens=torch.tensor([2], dtype=torch.int32),
            block_table=torch.tensor([[5, 6, -1]], dtype=torch.int32),
            block_size=256,
        )

        sparse_indices, sparse_topk_length = model._build_c128a_sparse_inputs(
            layer_attn_metadata,
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            compressed_slot_offset=100,
            recent_indices=None,
            recent_topk_length=None,
            layer_idx=0,
        )

        # With block_table[0]=[5,6,-1], block_size=256, compress_ratio=128,
        # compressed_block_size=2:
        #   d=0: block_table[0, 0]=5, offset=0*128+127=127 -> 100+5*256+127=1507
        #   d=1: block_table[0, 0]=5, offset=1*128+127=255 -> 100+5*256+255=1635
        # The request-aware path uses the actual block ID from block_table
        # instead of the raw physical_blocks index, so different requests'
        # compressed KV lands at distinct physical slots.
        self.assertEqual(tuple(sparse_indices.shape), (1, 128))
        self.assertEqual(sparse_indices[0, :2].tolist(), [1507, 1635])
        self.assertTrue(
            torch.equal(
                sparse_indices[:, 2:],
                torch.full((1, 126), -1, dtype=torch.int32),
            )
        )
        self.assertEqual(sparse_topk_length.tolist(), [2])

    def test_compressor_state_cache_shape_matches_compiler_contract(self) -> None:
        model = _make_model()
        kv_cache = torch.empty((2, 32, 16, 1, 8), dtype=torch.uint8)
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

        self.assertGreaterEqual(compressor_cache.shape[0], 6)
        self.assertGreaterEqual(indexer_cache.shape[0], 6)
        self.assertEqual(tuple(compressor_cache.shape[1:]), (16, 32))
        self.assertEqual(tuple(indexer_cache.shape[1:]), (16, 12))
        self.assertEqual(compressor_cache.dtype, torch.float32)
        self.assertEqual(indexer_cache.dtype, torch.float32)

    def test_state_cache_uses_compiled_artifact_block_size_when_available(self) -> None:
        model = _make_model()
        model.model = types.SimpleNamespace(
            get_input_maximum_shape=lambda name: (
                [32, 4, 32]
                if name == "compressor_state_cache_0"
                else [32, 4, 12]
            )
        )
        kv_cache = torch.empty((2, 32, 16, 1, 8), dtype=torch.uint8)
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

        # slot 9 with a 4-token state-cache page writes block 2, so this must
        # allocate 3 pages. The broken version allocated 1 page from kv block 16.
        self.assertGreaterEqual(compressor_cache.shape[0], 6)
        self.assertGreaterEqual(indexer_cache.shape[0], 6)
        self.assertEqual(tuple(compressor_cache.shape[1:]), (4, 32))
        self.assertEqual(tuple(indexer_cache.shape[1:]), (4, 12))

    def test_c128_compressor_state_cache_falls_back_to_kv_block_size(self) -> None:
        model = _make_model()
        model.config.compress_ratios = [128]
        kv_cache = torch.empty((2, 32, 16, 1, 8), dtype=torch.uint8)
        slot_mapping = torch.tensor([31], dtype=torch.int64)
        block_tables = torch.tensor([[0, 5, -1]], dtype=torch.int32)

        compressor_cache = model._get_deepseek_v4_state_cache(
            0,
            kv_cache,
            slot_mapping,
            block_tables,
            indexer=False,
        )

        self.assertGreaterEqual(compressor_cache.shape[0], 6)
        self.assertEqual(tuple(compressor_cache.shape[1:]), (16, 16))
        self.assertEqual(compressor_cache.dtype, torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_forward_uses_async_run_for_decode(self) -> None:
        model = _make_model()
        model.num_layers = 0
        model.cache_dtype = torch.uint8
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {"input_ids": 0},
        )
        # The bound-run path is disabled for the mock (no bind_inputs method).
        model.model._bound_run_available = False

        run_calls = []

        def _run(inputs, outputs, stream_ptr=None, sync=True, graph_mode=False):
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

        self.assertEqual(run_calls, [False])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_graph_decode_inputs_survive_intervening_prefill(self) -> None:
        model = _make_model()
        model.num_layers = 0
        model.cache_dtype = torch.uint8
        model._paiton_graph_mode = True
        model._paiton_graph_max_seq_len = 8960
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {
                "input_ids": 0,
                "max_query_len": 1,
                "max_seq_len": 2,
            },
            _bound_run_available=False,
        )

        run_calls = []

        def _run(inputs, outputs, stream_ptr=None, sync=True, graph_mode=False):
            del outputs, stream_ptr, sync
            run_calls.append(
                (
                    graph_mode,
                    inputs["input_ids"].data_ptr,
                    tuple(inputs["max_query_len"].shape),
                    tuple(inputs["max_seq_len"].shape),
                )
            )

        model.model.run = _run
        model.compilation_config = types.SimpleNamespace(static_forward_context={})

        def _metadata(num_tokens: int, max_query_len: int):
            return types.SimpleNamespace(
                max_query_len=max_query_len,
                max_seq_len=num_tokens,
                slot_mapping=torch.arange(
                    num_tokens, dtype=torch.int64, device="cuda"
                ),
                query_start_loc=torch.tensor(
                    [0, num_tokens], dtype=torch.int32, device="cuda"
                ),
                seq_lens=torch.tensor(
                    [num_tokens], dtype=torch.int32, device="cuda"
                ),
                block_table=torch.zeros(
                    (1, 1), dtype=torch.int32, device="cuda"
                ),
            )

        for num_tokens, max_query_len in ((1, 1), (2, 2), (1, 1)):
            metadata = _metadata(num_tokens, max_query_len)
            with mock.patch(
                "paiton_vllm_plugin.models.paiton_deepseek_v4.get_forward_context",
                return_value=types.SimpleNamespace(attn_metadata={"0": metadata}),
            ):
                model.forward(
                    torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
                    torch.arange(num_tokens, dtype=torch.int64, device="cuda"),
                )

        graph_calls = [call for call in run_calls if call[0]]
        self.assertEqual(len(graph_calls), 2)
        self.assertEqual(graph_calls[0][1], graph_calls[1][1])
        self.assertEqual(graph_calls[0][2], (1, 0))
        self.assertEqual(graph_calls[1][2], (1, 0))
        self.assertEqual(graph_calls[0][3], (8960, 0))
        self.assertEqual(graph_calls[1][3], (8960, 0))
        self.assertIn(1, model._paiton_graph_scratch_caches)
        self.assertEqual(model._paiton_graph_scratch_caches[1]["nt"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_forward_expands_compact_logits_back_to_token_rows(self) -> None:
        model = _make_model()
        model.num_layers = 0
        model.cache_dtype = torch.uint8

        captured_outputs = {}

        run_calls = []

        def _run(inputs, outputs, stream_ptr=None, sync=True, graph_mode=False):
            del inputs, stream_ptr
            run_calls.append(sync)
            captured_outputs.update(outputs)

        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: {"input_ids": 0},
            run=_run,
        )

        attn_metadata = types.SimpleNamespace(
            max_query_len=3,
            max_seq_len=3,
            slot_mapping=torch.zeros((3,), dtype=torch.int64, device="cuda"),
            query_start_loc=torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([3], dtype=torch.int32, device="cuda"),
            block_table=torch.zeros((1, 1), dtype=torch.int32, device="cuda"),
        )
        model.compilation_config = types.SimpleNamespace(static_forward_context={})

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_deepseek_v4.get_forward_context",
            return_value=types.SimpleNamespace(attn_metadata={"0": attn_metadata}),
        ):
            input_ids = torch.zeros((3,), dtype=torch.int64, device="cuda")
            positions = torch.arange(3, dtype=torch.int64, device="cuda")
            output = model.forward(input_ids, positions)

        self.assertEqual(tuple(output.shape), (3, model.config.vocab_size))
        self.assertEqual(captured_outputs["logits"].shape, [1, model.config.vocab_size])
        self.assertEqual(run_calls, [True])

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_forward_allocates_distinct_indexer_aux_buffers(self) -> None:
        model = _make_model()
        model.num_layers = 1
        model.cache_dtype = torch.uint8
        model._deepseek_sliding_window = 0

        captured_inputs = {}

        def _run(inputs, outputs, stream_ptr=None, sync=True, graph_mode=False):
            del outputs, stream_ptr, sync
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
            "indexer_logits_workspace",
            "kv_cache_0",
            "sparse_mla_indexer_kv_0",
            "indexer_q_fp8_0",
            "indexer_weights_0",
            "indexer_k_fp8_0",
            "indexer_k_scale_0",
            "cu_seqlen_ks_0",
            "cu_seqlen_ke_0",
            "sparse_mla_indices_0",
            "sparse_mla_topk_length_0",
        }
        model.model = types.SimpleNamespace(
            get_input_name_to_index_map=lambda: expected_inputs,
            run=_run,
        )

        kv_cache = torch.zeros((2, 2, 4, 1, 8), dtype=torch.uint8, device="cuda")
        model.compilation_config = types.SimpleNamespace(
            static_forward_context={"0": types.SimpleNamespace(kv_cache=kv_cache)},
        )

        attn_metadata = types.SimpleNamespace(
            max_query_len=2,
            max_seq_len=8,
            slot_mapping=torch.tensor([0, 1], dtype=torch.int64, device="cuda"),
            query_start_loc=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([2], dtype=torch.int32, device="cuda"),
            block_table=torch.tensor([[0, 1]], dtype=torch.int32, device="cuda"),
        )

        with mock.patch(
            "paiton_vllm_plugin.models.paiton_deepseek_v4.get_forward_context",
            return_value=types.SimpleNamespace(attn_metadata={"0": attn_metadata}),
        ):
            input_ids = torch.zeros((2,), dtype=torch.int64, device="cuda")
            positions = torch.arange(2, dtype=torch.int64, device="cuda")
            model.forward(input_ids, positions)

        self.assertIn("indexer_q_fp8_0", captured_inputs)
        self.assertIn("indexer_weights_0", captured_inputs)
        self.assertIn("indexer_k_fp8_0", captured_inputs)
        self.assertIn("indexer_k_scale_0", captured_inputs)
        self.assertIn("cu_seqlen_ks_0", captured_inputs)
        self.assertIn("cu_seqlen_ke_0", captured_inputs)
        self.assertIn("indexer_logits_workspace", captured_inputs)
        self.assertEqual(captured_inputs["indexer_logits_workspace"].shape, [2, 8])
        logits_workspace = _find_backing_tensor(
            model._run_input_backings,
            captured_inputs["indexer_logits_workspace"].data_ptr,
        )
        self.assertGreaterEqual(logits_workspace.numel(), 16)
        self.assertNotEqual(
            captured_inputs["cu_seqlen_ks_0"].data_ptr,
            captured_inputs["cu_seqlen_ke_0"].data_ptr,
        )
        self.assertEqual(
            captured_inputs["indexer_k_fp8_0"].shape[0],
            captured_inputs["sparse_mla_indexer_kv_0"].shape[0],
        )
        self.assertEqual(
            captured_inputs["indexer_k_scale_0"].shape[0],
            captured_inputs["sparse_mla_indexer_kv_0"].shape[0],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_forward_clones_sparse_indices_per_layer(self) -> None:
        model = _make_model()
        model.num_layers = 2
        model.cache_dtype = torch.uint8
        model._deepseek_sliding_window = 2
        model.config.num_hidden_layers = 2
        model.config.compress_ratios = [4, 4]

        captured_inputs = {}

        def _run(inputs, outputs, stream_ptr=None, sync=True, graph_mode=False):
            del outputs, stream_ptr, sync
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

        kv_cache0 = torch.zeros((2, 2, 4, 1, 8), dtype=torch.uint8, device="cuda")
        kv_cache1 = torch.zeros((2, 2, 4, 1, 8), dtype=torch.uint8, device="cuda")
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

        self.assertIn("sparse_mla_indices_0", captured_inputs)
        self.assertIn("sparse_mla_indices_1", captured_inputs)
        self.assertIn("sparse_mla_topk_length_0", captured_inputs)
        self.assertIn("sparse_mla_topk_length_1", captured_inputs)

        self.assertNotEqual(
            captured_inputs["sparse_mla_indices_0"].data_ptr,
            captured_inputs["sparse_mla_indices_1"].data_ptr,
        )
        self.assertNotEqual(
            captured_inputs["sparse_mla_topk_length_0"].data_ptr,
            captured_inputs["sparse_mla_topk_length_1"].data_ptr,
        )

        layer0_indices = _find_backing_tensor(
            model._run_input_backings,
            captured_inputs["sparse_mla_indices_0"].data_ptr,
        )
        layer1_indices = _find_backing_tensor(
            model._run_input_backings,
            captured_inputs["sparse_mla_indices_1"].data_ptr,
        )
        layer0_lengths = _find_backing_tensor(
            model._run_input_backings,
            captured_inputs["sparse_mla_topk_length_0"].data_ptr,
        )
        layer1_lengths = _find_backing_tensor(
            model._run_input_backings,
            captured_inputs["sparse_mla_topk_length_1"].data_ptr,
        )

        torch.testing.assert_close(layer0_indices, layer1_indices)
        torch.testing.assert_close(layer0_lengths, layer1_lengths)

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

    def test_map_pt_params_preshuffles_dynamic_fp8_linear_weights(self) -> None:
        model = _make_model()
        model.quantized = True
        model.dynamic_quant = True
        model.config.q_lora_rank = 32

        fp8 = torch.float8_e4m3fn
        def make_fp8(shape: tuple[int, int], start: float, end: float) -> torch.Tensor:
            return torch.linspace(start, end, shape[0] * shape[1]).reshape(shape).to(fp8)

        pt_params = {
            "layers.0.attn.wq_a.weight": make_fp8((16, 32), -4.0, 4.0),
            "layers.0.attn.wkv.weight": make_fp8((16, 32), 4.0, -4.0),
            "layers.0.attn.wo_b.weight": make_fp8((32, 32), -3.5, 3.5),
            "layers.0.attn.indexer.wq_b.weight": make_fp8((32, 32), -2.5, 2.5),
            "layers.0.ffn.shared_experts.w1.weight": make_fp8((16, 32), -1.5, 1.5),
            "layers.0.ffn.shared_experts.w3.weight": make_fp8((16, 32), 1.5, -1.5),
            "layers.0.ffn.shared_experts.w2.weight": make_fp8((32, 32), -0.75, 0.75),
        }
        expected = {
            "layers_0_attn_fused_wqa_wkv_weight",
            "layers_0_attn_wo_b_weight",
            "layers_0_attn_indexer_wq_b_weight",
            "layers_0_ffn_shared_experts_gate_up_proj_weight",
            "layers_0_ffn_shared_experts_down_proj_weight",
        }

        mapped = model.map_pt_params(pt_params, expected_constant_names=expected)

        torch.testing.assert_close(
            mapped["layers_0_attn_fused_wqa_wkv_weight"].cpu(),
            _shuffle_fp8_weight(
                torch.cat(
                    [
                        pt_params["layers.0.attn.wq_a.weight"],
                        pt_params["layers.0.attn.wkv.weight"],
                    ],
                    dim=0,
                )
            ),
        )
        torch.testing.assert_close(
            mapped["layers_0_attn_wo_b_weight"].cpu(),
            _shuffle_fp8_weight(pt_params["layers.0.attn.wo_b.weight"]),
        )
        torch.testing.assert_close(
            mapped["layers_0_attn_indexer_wq_b_weight"].cpu(),
            _shuffle_fp8_weight(pt_params["layers.0.attn.indexer.wq_b.weight"]),
        )
        torch.testing.assert_close(
            mapped["layers_0_ffn_shared_experts_gate_up_proj_weight"].cpu(),
            _shuffle_fp8_weight(
                torch.cat(
                    [
                        pt_params["layers.0.ffn.shared_experts.w1.weight"],
                        pt_params["layers.0.ffn.shared_experts.w3.weight"],
                    ],
                    dim=0,
                )
            ),
        )
        torch.testing.assert_close(
            mapped["layers_0_ffn_shared_experts_down_proj_weight"].cpu(),
            _shuffle_fp8_weight(pt_params["layers.0.ffn.shared_experts.w2.weight"]),
        )


if __name__ == "__main__":
    unittest.main()
