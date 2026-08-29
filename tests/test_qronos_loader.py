import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from paiton_vllm_plugin.runtime.core.utils.qronos_loader import (
    QronosLinearSpec,
    QronosParallelism,
    QronosStreamingTransformer,
    qwen38_specs_from_manifest,
)
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import (
    Qwen38UnquantizedLoader,
    configure_qwen38_cache_contract,
    qwen38_unquantized_specs_from_manifest,
    resolve_qwen38_safetensors,
)


def tensors(k, n, offset=0):
    weight = torch.full((k, n // 8), offset, dtype=torch.int32)
    scale = torch.arange(k // 128 * n, dtype=torch.float32).reshape(
        k // 128, n
    ) + offset
    zero = torch.zeros((k // 128, n // 8), dtype=torch.int32)
    return weight, scale, zero


def consume_triple(transformer, spec, values, order=("weight", "scale", "zero")):
    suffix = {
        "weight": ".weight",
        "scale": ".weight_scale",
        "zero": ".weight_zero_point",
    }
    mapping = dict(zip(("weight", "scale", "zero"), values))
    result = None
    for component in order:
        value = transformer.consume(
            spec.source_prefix + suffix[component], mapping[component]
        )
        if value is not None:
            result = value
    return result


class TestQronosStreamingTransformer(unittest.TestCase):
    def test_rejects_ambiguous_target_constant_names(self):
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            QronosStreamingTransformer((
                QronosLinearSpec("layer.q", "same", "same", 128, 8),
            ))
        with self.assertRaisesRegex(ValueError, "duplicate Qronos target"):
            QronosStreamingTransformer((
                QronosLinearSpec("layer.q", "shared", "q_scale", 128, 8),
                QronosLinearSpec("layer.k", "k_weight", "shared", 128, 8),
            ))

    def test_finalizes_each_linear_without_model_dictionary(self):
        specs = (
            QronosLinearSpec("layer.0.q", "l0_q_weight", "l0_q_scale", 256, 64),
            QronosLinearSpec("layer.0.k", "l0_k_weight", "l0_k_scale", 256, 32),
        )
        loader = QronosStreamingTransformer(specs, max_pending_linears=1)
        first = consume_triple(loader, specs[0], tensors(256, 64))
        self.assertIsNotNone(first)
        self.assertEqual(loader.pending_bytes, 0)
        self.assertEqual([name for name, _ in first.constants()], [
            "l0_q_weight", "l0_q_scale"
        ])
        second = consume_triple(
            loader, specs[1], tensors(256, 32), order=("zero", "weight", "scale")
        )
        self.assertIsNotNone(second)
        loader.finish()
        self.assertEqual(loader.peak_pending_linears, 1)

    def test_rejects_non_streaming_tensor_order(self):
        specs = tuple(
            QronosLinearSpec(f"layer.{i}.q", f"w{i}", f"s{i}", 128, 8)
            for i in range(3)
        )
        loader = QronosStreamingTransformer(specs, max_pending_linears=2)
        for spec in specs[:2]:
            self.assertIsNone(
                loader.consume(spec.source_prefix + ".weight", tensors(128, 8)[0])
            )
        with self.assertRaisesRegex(MemoryError, "streaming window"):
            loader.consume(specs[2].source_prefix + ".weight", tensors(128, 8)[0])

    def test_random_access_ignores_physical_order_and_fetches_one_triple(self):
        specs = (
            QronosLinearSpec("layer.0.q", "w0", "s0", 128, 16),
            QronosLinearSpec("layer.1.q", "w1", "s1", 128, 16),
        )
        values = {}
        for index, spec in enumerate(specs):
            weight, scale, zero = tensors(128, 16, offset=index)
            values[spec.source_prefix + ".weight_scale"] = scale
            values[spec.source_prefix + ".weight"] = weight
            values[spec.source_prefix + ".weight_zero_point"] = zero

        class Source:
            def __init__(self):
                self.accesses = []

            def keys(self):
                # Deliberately grouped like the pinned checkpoint data order.
                return [
                    *(spec.source_prefix + ".weight_scale" for spec in specs),
                    *(spec.source_prefix + ".weight" for spec in specs),
                    *(spec.source_prefix + ".weight_zero_point" for spec in specs),
                ]

            def get_tensor(self, name):
                self.accesses.append(name)
                return values[name]

        source = Source()
        loader = QronosStreamingTransformer(specs, max_pending_linears=1)
        results = list(loader.iter_from_random_access_source(source))
        loader.finish()
        self.assertEqual(len(results), 2)
        self.assertEqual(
            source.accesses,
            [
                "layer.0.q.weight",
                "layer.0.q.weight_scale",
                "layer.0.q.weight_zero_point",
                "layer.1.q.weight",
                "layer.1.q.weight_scale",
                "layer.1.q.weight_zero_point",
            ],
        )
        self.assertEqual(loader.peak_pending_linears, 1)

    def test_random_access_rejects_undeclared_quantized_linear(self):
        spec = QronosLinearSpec("layer.q", "w", "s", 128, 8)
        weight, scale, zero = tensors(128, 8)
        source = {
            "layer.q.weight": weight,
            "layer.q.weight_scale": scale,
            "layer.q.weight_zero_point": zero,
            "layer.extra.weight_scale": scale,
        }
        loader = QronosStreamingTransformer((spec,))
        with self.assertRaisesRegex(ValueError, "undeclared quantized"):
            list(loader.iter_from_random_access_source(source))

    def test_reduced_contract_allows_only_later_source_layers(self):
        spec = QronosLinearSpec(
            "model.language_model.layers.0.q", "w", "s", 128, 8
        )
        weight, scale, zero = tensors(128, 8)
        source = {
            spec.source_prefix + ".weight": weight,
            spec.source_prefix + ".weight_scale": scale,
            spec.source_prefix + ".weight_zero_point": zero,
            "model.language_model.layers.4.q.weight_scale": scale,
            "model.language_model.layers.63.q.weight_zero_point": zero,
        }
        loader = QronosStreamingTransformer(
            (spec,), allowed_extra_layer_range=(4, 64)
        )
        self.assertEqual(len(list(loader.iter_from_random_access_source(source))), 1)
        loader.finish()
        source["model.language_model.layers.3.q.weight_scale"] = scale
        with self.assertRaisesRegex(ValueError, "undeclared quantized"):
            list(QronosStreamingTransformer(
                (spec,), allowed_extra_layer_range=(4, 64)
            ).iter_from_random_access_source(source))

    def test_safetensors_random_access_path(self):
        spec = QronosLinearSpec("layer.q", "w", "s", 128, 16)
        weight, scale, zero = tensors(128, 16)
        with tempfile.TemporaryDirectory(prefix="paiton_qronos_source_") as root:
            path = Path(root) / "model.safetensors"
            save_file(
                {
                    "layer.q.weight_scale": scale,
                    "layer.q.weight": weight,
                    "layer.q.weight_zero_point": zero,
                },
                path,
            )
            loader = QronosStreamingTransformer((spec,), max_pending_linears=1)
            results = list(loader.iter_safetensors(path))
            loader.finish()
        self.assertEqual(len(results), 1)
        self.assertEqual(tuple(results[0].weights.packed_weight.shape), (16, 16))
        self.assertEqual(loader.peak_pending_linears, 1)

    def test_column_parallel_shards_packed_n_before_transform(self):
        spec = QronosLinearSpec(
            "layer.q", "q_weight", "q_scale", 256, 64, QronosParallelism.COLUMN
        )
        source = tensors(256, 64)
        rank1 = QronosStreamingTransformer((spec,), tp_rank=1, tp_size=2)
        result = consume_triple(rank1, spec, source)
        rank1.finish()
        self.assertEqual(tuple(result.weights.packed_weight.shape), (32, 32))
        self.assertEqual(tuple(result.weights.scales.shape), (32, 2))
        torch.testing.assert_close(result.weights.scales, source[1][:, 32:].t())

    def test_row_parallel_shards_group_aligned_k(self):
        spec = QronosLinearSpec(
            "layer.o", "o_weight", "o_scale", 256, 16, QronosParallelism.ROW
        )
        source = tensors(256, 16)
        rank1 = QronosStreamingTransformer((spec,), tp_rank=1, tp_size=2)
        result = consume_triple(rank1, spec, source)
        rank1.finish()
        self.assertEqual(tuple(result.weights.packed_weight.shape), (16, 16))
        self.assertEqual(tuple(result.weights.scales.shape), (16, 1))
        torch.testing.assert_close(result.weights.scales[:, 0], source[1][1])

    def test_exact_shape_dtype_duplicate_and_missing_checks(self):
        spec = QronosLinearSpec("layer.q", "q_weight", "q_scale", 256, 64)
        loader = QronosStreamingTransformer((spec,))
        with self.assertRaisesRegex(ValueError, "shape must be exactly"):
            loader.consume(
                "layer.q.weight",
                torch.zeros((32, 256), dtype=torch.int32),
            )
        with self.assertRaisesRegex(ValueError, "dtype must be"):
            loader.consume(
                "layer.q.weight",
                torch.zeros((256, 8), dtype=torch.int64),
            )
        loader.consume("layer.q.weight", tensors(256, 64)[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            loader.consume("layer.q.weight", tensors(256, 64)[0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            loader.finish()

    def test_pending_byte_limit_is_enforced(self):
        spec = QronosLinearSpec("layer.q", "q_weight", "q_scale", 256, 64)
        loader = QronosStreamingTransformer((spec,), max_pending_bytes=1024)
        with self.assertRaisesRegex(MemoryError, "memory contract"):
            loader.consume("layer.q.weight", tensors(256, 64)[0])


def qwen38_manifest_fixture():
    layouts = [
        {
            "source_prefix": "model.language_model.layers.0.linear_attn.in_proj_a",
            "target_weight_name": "layers_0_linear_attn_in_proj_a_weight",
            "target_scale_name": "layers_0_linear_attn_in_proj_a_weight_scale",
            "input_size": 5120,
            "output_size": 48,
            "parallelism": "replicated",
            "padded_output_size": None,
        }
    ]
    layout_json = json.dumps(layouts, sort_keys=True, separators=(",", ":"))
    contract = {
        "version": 1,
        "product_model_type": "qwen3_8",
        "compatibility_api_model_type": "qwen3_5",
        "scope": "text-only",
        "multimodal": False,
        "mtp_speculative": False,
        "tp_size": 1,
        "source_num_hidden_layers": 64,
        "num_hidden_layers": 1,
        "num_gdn_layers": 1,
        "num_full_attention_layers": 0,
        "max_num_batched_tokens": 8192,
        "max_context_length": 8192,
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "gdn_conv_state_dtype": "bfloat16",
        "gdn_recurrent_state_dtype": "float32",
        "gdn_conv_state_layout": "SD",
        "rotary_dim": 64,
        "rope_theta": 10_000_000,
        "mrope_section": [11, 11, 10],
        "rotary_inv_freq_binding": "compiler_owned",
        "qronos_group_size": 128,
        "qronos_checkpoint_layout": "Kx(N/8)_packed_i32",
        "qronos_kernel_layout": "paiton_w4a16_g128_v1",
        "qronos_transform_version": "quark_qronos_reorder_signed_v1",
        "qronos_linear_count": 1,
        "qronos_layout_sha256": hashlib.sha256(layout_json.encode()).hexdigest(),
        "zero_centered_norm_transform": "gamma=1+checkpoint_bf16",
    }
    return {
        "target": {"arch": "gfx1201", "family": "rdna4", "wave_size": 32},
        "paiton_qwen38_contract": contract,
        "qronos_linears": layouts,
        "interface": {
            "tensors": [
                {
                    "name": "layers_0_linear_attn_in_proj_a_weight",
                    "dtype": "int32",
                    "roles": ["param"],
                    "binding": "unbound",
                    "shape_values": [[48], [640]],
                },
                {
                    "name": "layers_0_linear_attn_in_proj_a_weight_scale",
                    "dtype": "float32",
                    "roles": ["param"],
                    "binding": "unbound",
                    "shape_values": [[48], [40]],
                },
            ]
        },
    }


class TestQwen38ManifestSpecs(unittest.TestCase):
    def test_derives_exact_loader_spec(self):
        specs = qwen38_specs_from_manifest(qwen38_manifest_fixture())
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0].input_size, 5120)
        self.assertEqual(specs[0].output_size, 48)
        self.assertIs(specs[0].parallelism, QronosParallelism.REPLICATED)

    def test_rejects_contract_hash_target_and_same_byte_transpose(self):
        mutations = ("hash", "arch", "shape")
        for mutation in mutations:
            manifest = qwen38_manifest_fixture()
            if mutation == "hash":
                manifest["paiton_qwen38_contract"]["qronos_layout_sha256"] = "0" * 64
            elif mutation == "arch":
                manifest["target"]["arch"] = "gfx942"
            else:
                manifest["interface"]["tensors"][0]["shape_values"] = [[640], [48]]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qwen38_specs_from_manifest(manifest)

    def test_rejects_undeclared_packed_constant(self):
        manifest = qwen38_manifest_fixture()
        manifest["interface"]["tensors"].append(
            {
                "name": "unexpected_weight_scale",
                "dtype": "float32",
                "roles": ["param"],
                "shape_values": [[1]],
            }
        )
        with self.assertRaisesRegex(ValueError, "undeclared packed"):
            qwen38_specs_from_manifest(manifest)


def qwen38_backbone_manifest_fixture():
    manifest = qwen38_manifest_fixture()
    records = manifest["interface"]["tensors"]
    shapes = {
        "layers_0_input_layernorm_weight": (5120,),
        "layers_0_post_attention_layernorm_weight": (5120,),
        "layers_0_linear_attn_A_log": (48,),
        "layers_0_linear_attn_conv1d": (10240, 1, 4),
        "layers_0_linear_attn_dt_bias": (48,),
        "layers_0_linear_attn_norm_weight": (128,),
        "norm_weight": (5120,),
    }
    records.extend(
        {
            "name": name,
            "dtype": "bfloat16",
            "roles": ["param"],
            "binding": "unbound",
            "shape_values": [[dim] for dim in shape],
        }
        for name, shape in shapes.items()
    )
    records.append({
        "name": "rotary_emb_inv_freq",
        "dtype": "float32",
        "roles": ["param"],
        "binding": "compiler_owned",
        "shape_values": [[32]],
    })
    return manifest


class TestQwen38UnquantizedLoader(unittest.TestCase):
    def test_cache_contract_resolves_only_fp32_recurrence_auto(self):
        cache = type("Cache", (), {
            "cache_dtype": "auto",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "auto",
            "mamba_cache_mode": "align",
        })()
        configure_qwen38_cache_contract(cache, resolve_auto=True)
        self.assertEqual(cache.mamba_ssm_cache_dtype, "float32")
        cache.mamba_cache_dtype = "float32"
        with self.assertRaisesRegex(ValueError, "BF16 convolution"):
            configure_qwen38_cache_contract(cache, resolve_auto=False)

    def test_derives_and_loads_every_backbone_constant_boundedly(self):
        manifest = qwen38_backbone_manifest_fixture()
        specs = qwen38_unquantized_specs_from_manifest(manifest)
        self.assertEqual(len(specs), 7)
        source = {
            spec.source_name: torch.zeros(spec.shape, dtype=torch.bfloat16)
            for spec in specs
        }
        loaded = dict(Qwen38UnquantizedLoader(manifest).iter_from_random_access_source(source))
        self.assertEqual(
            set(loaded),
            {spec.target_name for spec in specs},
        )
        self.assertTrue(torch.all(loaded["layers_0_input_layernorm_weight"] == 1))
        self.assertTrue(torch.all(loaded["layers_0_linear_attn_norm_weight"] == 0))

    def test_mlp_qronos_entries_do_not_change_attention_layer_kind(self):
        manifest = qwen38_backbone_manifest_fixture()
        mlp = dict(manifest["qronos_linears"][0])
        mlp["source_prefix"] = "model.language_model.layers.0.mlp.gate_proj"
        mlp["target_weight_name"] = "layers_0_mlp_gate_proj_weight"
        mlp["target_scale_name"] = "layers_0_mlp_gate_proj_weight_scale"
        layouts = manifest["qronos_linears"]
        layouts.append(mlp)
        canonical = json.dumps(layouts, sort_keys=True, separators=(",", ":"))
        contract = manifest["paiton_qwen38_contract"]
        contract["qronos_linear_count"] = 2
        contract["qronos_layout_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        manifest["interface"]["tensors"].extend((
            {
                "name": mlp["target_weight_name"],
                "dtype": "int32",
                "roles": ["param"],
                "binding": "unbound",
                "shape_values": [[48], [640]],
            },
            {
                "name": mlp["target_scale_name"],
                "dtype": "float32",
                "roles": ["param"],
                "binding": "unbound",
                "shape_values": [[48], [40]],
            },
        ))
        self.assertEqual(len(qwen38_unquantized_specs_from_manifest(manifest)), 7)

    def test_rejects_missing_wrong_dtype_shape_and_extra_abi_constant(self):
        manifest = qwen38_backbone_manifest_fixture()
        specs = qwen38_unquantized_specs_from_manifest(manifest)
        source = {
            spec.source_name: torch.zeros(spec.shape, dtype=torch.bfloat16)
            for spec in specs
        }
        missing = dict(source)
        missing.pop(specs[0].source_name)
        with self.assertRaisesRegex(ValueError, "missing backbone"):
            list(Qwen38UnquantizedLoader(manifest).iter_from_random_access_source(missing))
        wrong_dtype = dict(source)
        wrong_dtype[specs[0].source_name] = wrong_dtype[specs[0].source_name].float()
        with self.assertRaisesRegex(ValueError, "dtype must be"):
            list(Qwen38UnquantizedLoader(manifest).iter_from_random_access_source(wrong_dtype))
        wrong_shape = dict(source)
        wrong_shape[specs[0].source_name] = torch.zeros((1, 5120), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "shape must be exactly"):
            list(Qwen38UnquantizedLoader(manifest).iter_from_random_access_source(wrong_shape))
        manifest["interface"]["tensors"].append({
            "name": "unexpected_bf16",
            "dtype": "bfloat16",
            "roles": ["param"],
            "binding": "unbound",
            "shape_values": [[1]],
        })
        with self.assertRaisesRegex(ValueError, "undeclared Qwen3.8 backbone"):
            qwen38_unquantized_specs_from_manifest(manifest)

    def test_resolves_only_the_single_file_checkpoint_contract(self):
        with tempfile.TemporaryDirectory(prefix="paiton_qwen38_checkpoint_") as root:
            root = Path(root)
            checkpoint = root / "model.safetensors"
            checkpoint.touch()
            self.assertEqual(resolve_qwen38_safetensors(str(root)), checkpoint)
            (root / "model.safetensors.index.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "one model.safetensors"):
                resolve_qwen38_safetensors(str(root))


if __name__ == "__main__":
    unittest.main()
