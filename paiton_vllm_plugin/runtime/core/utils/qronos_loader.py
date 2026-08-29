"""Bounded-memory, shape-strict Qronos linear checkpoint transformation."""

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Dict, Optional, Tuple

import torch

from .qronos_w4a16 import QronosW4A16Weights, transform_qronos_w4a16


class QronosParallelism(str, Enum):
    REPLICATED = "replicated"
    COLUMN = "column"
    ROW = "row"


@dataclass(frozen=True)
class QronosLinearSpec:
    source_prefix: str
    target_weight_name: str
    target_scale_name: str
    input_size: int
    output_size: int
    parallelism: QronosParallelism = QronosParallelism.REPLICATED
    padded_output_size: Optional[int] = None

    def __post_init__(self):
        if not self.source_prefix or self.source_prefix.endswith("."):
            raise ValueError("source_prefix must be a non-empty module name")
        if not self.target_weight_name or not self.target_scale_name:
            raise ValueError("target constant names must be non-empty")
        if self.input_size <= 0 or self.input_size % 128:
            raise ValueError("input_size must be positive and group-128 aligned")
        if self.output_size <= 0 or self.output_size % 8:
            raise ValueError("output_size must be positive and pack-8 aligned")
        object.__setattr__(self, "parallelism", QronosParallelism(self.parallelism))


@dataclass(frozen=True)
class TransformedQronosLinear:
    spec: QronosLinearSpec
    weights: QronosW4A16Weights

    def constants(self) -> Tuple[Tuple[str, torch.Tensor], Tuple[str, torch.Tensor]]:
        return (
            (self.spec.target_weight_name, self.weights.packed_weight),
            (self.spec.target_scale_name, self.weights.scales),
        )


class QronosStreamingTransformer:
    """Consume checkpoint tensors and finalize each packed linear immediately.

    Only incomplete triples are retained. A small pending-linear limit rejects
    checkpoint iteration orders that would otherwise recreate a whole-model
    dictionary in memory.
    """

    _SUFFIXES = {
        ".weight": "weight",
        ".weight_scale": "scale",
        ".weight_zero_point": "zero",
    }

    def __init__(
        self,
        specs,
        *,
        tp_rank: int = 0,
        tp_size: int = 1,
        max_pending_linears: int = 2,
        max_pending_bytes: int = 512 * 1024 * 1024,
        allowed_extra_layer_range: Optional[Tuple[int, int]] = None,
    ):
        specs = tuple(specs)
        if not specs:
            raise ValueError("Qronos streaming transformer requires at least one spec")
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError(f"invalid TP rank/size {tp_rank}/{tp_size}")
        if max_pending_linears <= 0 or max_pending_bytes <= 0:
            raise ValueError("pending limits must be positive")
        self.specs: Dict[str, QronosLinearSpec] = {}
        self._source_names: Dict[str, Tuple[str, str]] = {}
        target_names = set()
        for spec in specs:
            if spec.source_prefix in self.specs:
                raise ValueError(f"duplicate Qronos source prefix {spec.source_prefix}")
            if spec.target_weight_name == spec.target_scale_name:
                raise ValueError("weight and scale target names must be distinct")
            if spec.target_weight_name in target_names or spec.target_scale_name in target_names:
                raise ValueError("duplicate Qronos target constant name")
            target_names.update((spec.target_weight_name, spec.target_scale_name))
            self.specs[spec.source_prefix] = spec
            for suffix, component in self._SUFFIXES.items():
                self._source_names[spec.source_prefix + suffix] = (
                    spec.source_prefix,
                    component,
                )
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.max_pending_linears = max_pending_linears
        self.max_pending_bytes = max_pending_bytes
        if allowed_extra_layer_range is not None:
            start, stop = allowed_extra_layer_range
            if not 0 <= start <= stop:
                raise ValueError("invalid allowed extra Qronos layer range")
        self.allowed_extra_layer_range = allowed_extra_layer_range
        self._pending: Dict[str, Dict[str, torch.Tensor]] = {}
        self._completed = set()
        self.pending_bytes = 0
        self.peak_pending_bytes = 0
        self.peak_pending_linears = 0

    def handles(self, name: str) -> bool:
        return name in self._source_names

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _expected_shapes(self, spec: QronosLinearSpec):
        return {
            "weight": (spec.input_size, spec.output_size // 8),
            "scale": (spec.input_size // 128, spec.output_size),
            "zero": (spec.input_size // 128, spec.output_size // 8),
        }

    def _prepare_component(
        self,
        spec: QronosLinearSpec,
        component: str,
        name: str,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        expected_shape = self._expected_shapes(spec)[component]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"{name} shape must be exactly {expected_shape}, got {tuple(tensor.shape)}"
            )
        expected_dtype = {
            "weight": torch.int32,
            "scale": torch.float32,
            "zero": torch.int32,
        }[component]
        if tensor.dtype is not expected_dtype:
            raise ValueError(
                f"{name} dtype must be {expected_dtype}, got {tensor.dtype}"
            )
        return tensor.detach().to(device="cpu").contiguous()

    def _local_sizes(self, spec: QronosLinearSpec) -> Tuple[int, int]:
        local_k, local_n = spec.input_size, spec.output_size
        if spec.parallelism is QronosParallelism.COLUMN:
            if local_n % self.tp_size:
                raise ValueError(
                    f"{spec.source_prefix} N={local_n} is not divisible by TP={self.tp_size}"
                )
            local_n //= self.tp_size
            if local_n % 8:
                raise ValueError(
                    f"{spec.source_prefix} TP-local N={local_n} is not pack-8 aligned"
                )
        elif spec.parallelism is QronosParallelism.ROW:
            if local_k % self.tp_size:
                raise ValueError(
                    f"{spec.source_prefix} K={local_k} is not divisible by TP={self.tp_size}"
                )
            local_k //= self.tp_size
            if local_k % 128:
                raise ValueError(
                    f"{spec.source_prefix} TP-local K={local_k} is not group-128 aligned"
                )
        return local_k, local_n

    def _shard(
        self,
        spec: QronosLinearSpec,
        parts: Dict[str, torch.Tensor],
    ):
        local_k, local_n = self._local_sizes(spec)
        weight, scale, zero = parts["weight"], parts["scale"], parts["zero"]
        if spec.parallelism is QronosParallelism.COLUMN:
            start_n = self.tp_rank * local_n
            end_n = start_n + local_n
            weight = weight[:, start_n // 8 : end_n // 8].contiguous()
            scale = scale[:, start_n:end_n].contiguous()
            zero = zero[:, start_n // 8 : end_n // 8].contiguous()
        elif spec.parallelism is QronosParallelism.ROW:
            start_k = self.tp_rank * local_k
            end_k = start_k + local_k
            start_group = start_k // 128
            end_group = end_k // 128
            weight = weight[start_k:end_k].contiguous()
            scale = scale[start_group:end_group].contiguous()
            zero = zero[start_group:end_group].contiguous()
        return weight, scale, zero, local_k, local_n

    def consume(
        self, name: str, tensor: torch.Tensor
    ) -> Optional[TransformedQronosLinear]:
        source = self._source_names.get(name)
        if source is None:
            return None
        prefix, component = source
        if prefix in self._completed:
            raise ValueError(f"duplicate Qronos tensor after completion: {name}")
        pending = self._pending.setdefault(prefix, {})
        if component in pending:
            raise ValueError(f"duplicate Qronos tensor: {name}")

        spec = self.specs[prefix]
        cpu_tensor = self._prepare_component(
            spec, component, name, tensor
        )
        pending[component] = cpu_tensor
        self.pending_bytes += self._tensor_bytes(cpu_tensor)
        self.peak_pending_bytes = max(self.peak_pending_bytes, self.pending_bytes)
        self.peak_pending_linears = max(
            self.peak_pending_linears, len(self._pending)
        )
        if len(self._pending) > self.max_pending_linears:
            raise MemoryError(
                "Qronos checkpoint tensor order exceeds bounded streaming window: "
                f"{len(self._pending)} pending linears > {self.max_pending_linears}"
            )
        if self.pending_bytes > self.max_pending_bytes:
            raise MemoryError(
                "Qronos pending checkpoint tensors exceed memory contract: "
                f"{self.pending_bytes} > {self.max_pending_bytes} bytes"
            )
        if len(pending) != 3:
            return None

        source_bytes = sum(self._tensor_bytes(value) for value in pending.values())
        weight, scale, zero, _, local_n = self._shard(spec, pending)
        padded_output_size = spec.padded_output_size
        if padded_output_size is not None and spec.parallelism is QronosParallelism.COLUMN:
            if padded_output_size % self.tp_size:
                raise ValueError(
                    f"{spec.source_prefix} padded N is not divisible by TP size"
                )
            padded_output_size //= self.tp_size
        if padded_output_size is None:
            padded_output_size = (local_n + 7) // 8 * 8
        transformed = transform_qronos_w4a16(
            weight,
            scale,
            zero,
            padded_output_size=padded_output_size,
        )
        del self._pending[prefix]
        self.pending_bytes -= source_bytes
        self._completed.add(prefix)
        return TransformedQronosLinear(spec=spec, weights=transformed)

    @staticmethod
    def _source_keys(source):
        keys = source.keys()
        return set(keys)

    @staticmethod
    def _source_tensor(source, name):
        if hasattr(source, "get_tensor"):
            return source.get_tensor(name)
        return source[name]

    def iter_from_random_access_source(self, source):
        """Fetch one declared tensor triple at a time from an mmap-like source."""
        if self._pending or self._completed:
            raise RuntimeError(
                "random-access loading requires a fresh QronosStreamingTransformer"
            )
        available = self._source_keys(source)
        required = set(self._source_names)
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                "random-access Qronos source is missing tensors: "
                + ", ".join(missing[:20])
            )
        # Scale and zero-point names unambiguously identify quantized linears.
        # Reject any such checkpoint tensors not declared by the compiled graph.
        extra_quant_metadata = sorted(
            name
            for name in available - required
            if name.endswith(".weight_scale")
            or name.endswith(".weight_zero_point")
        )
        rejected_extra = []
        for name in extra_quant_metadata:
            match = re.match(r"^model\.language_model\.layers\.(\d+)\.", name)
            allowed = self.allowed_extra_layer_range
            if (
                match is None
                or allowed is None
                or not allowed[0] <= int(match.group(1)) < allowed[1]
            ):
                rejected_extra.append(name)
        if rejected_extra:
            raise ValueError(
                "random-access Qronos source has undeclared quantized linears: "
                + ", ".join(rejected_extra[:20])
            )

        for prefix, spec in self.specs.items():
            parts = {}
            for suffix, component in self._SUFFIXES.items():
                name = prefix + suffix
                parts[component] = self._prepare_component(
                    spec,
                    component,
                    name,
                    self._source_tensor(source, name),
                )
            source_bytes = sum(
                self._tensor_bytes(value) for value in parts.values()
            )
            if source_bytes > self.max_pending_bytes:
                raise MemoryError(
                    f"{prefix} tensor triple exceeds memory contract: "
                    f"{source_bytes} > {self.max_pending_bytes} bytes"
                )
            self.peak_pending_bytes = max(
                self.peak_pending_bytes, source_bytes
            )
            self.peak_pending_linears = max(self.peak_pending_linears, 1)
            weight, scale, zero, _, local_n = self._shard(spec, parts)
            padded_output_size = spec.padded_output_size
            if (
                padded_output_size is not None
                and spec.parallelism is QronosParallelism.COLUMN
            ):
                if padded_output_size % self.tp_size:
                    raise ValueError(
                        f"{spec.source_prefix} padded N is not divisible by TP size"
                    )
                padded_output_size //= self.tp_size
            if padded_output_size is None:
                padded_output_size = (local_n + 7) // 8 * 8
            transformed = transform_qronos_w4a16(
                weight,
                scale,
                zero,
                padded_output_size=padded_output_size,
            )
            self._completed.add(prefix)
            yield TransformedQronosLinear(spec=spec, weights=transformed)

    def iter_safetensors(self, path):
        """Random-access one safetensors file without following its data order."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as source:
            yield from self.iter_from_random_access_source(source)

    def finish(self) -> None:
        if self._pending:
            details = ", ".join(
                f"{prefix}: {sorted(parts)}"
                for prefix, parts in sorted(self._pending.items())
            )
            raise ValueError(f"incomplete Qronos tensor triples: {details}")
        missing = sorted(set(self.specs) - self._completed)
        if missing:
            raise ValueError(
                "missing Qronos linears: " + ", ".join(missing[:20])
            )


def qwen38_specs_from_manifest(manifest) -> Tuple[QronosLinearSpec, ...]:
    """Validate contract v3 and derive strict loader specs from the artifact.

    The manifest is the authority for compiled constant names and physical
    shapes. A same-byte transposition is rejected here before the C++ runtime's
    byte-count-only binding check can accept it.
    """

    if not isinstance(manifest, dict):
        raise ValueError("Qwen3.8 artifact manifest must be an object")
    target = manifest.get("target")
    if not isinstance(target, dict):
        raise ValueError("Qwen3.8 manifest is missing target metadata")
    if target.get("arch") not in ("gfx1200", "gfx1201"):
        raise ValueError("Qwen3.8 contract v3 requires gfx1200/gfx1201")
    if target.get("family") != "rdna4" or target.get("wave_size") != 32:
        raise ValueError("Qwen3.8 contract v3 requires RDNA4 wave32")

    contract = manifest.get("paiton_qwen38_contract")
    if not isinstance(contract, dict):
        raise ValueError("manifest is missing paiton_qwen38_contract")
    required_contract = {
        "version": 3,
        "product_model_type": "qwen3_8",
        "compatibility_api_model_type": "qwen3_5",
        "scope": "text-only",
        "multimodal": False,
        "mtp_speculative": False,
        "tp_size": 1,
        "source_num_hidden_layers": 64,
        "activation_dtype": "bfloat16",
        "kv_cache_dtype": "bfloat16",
        "kv_cache_physical_layout": "blocks_KV_tokens_heads_dim",
        "num_key_value_heads": 4,
        "head_dim": 256,
        "runtime_shell_parameters": [
            {
                "name": "model.embed_tokens.weight",
                "dtype": "bfloat16",
                "shape": [248320, 5120],
            },
            {
                "name": "lm_head.weight",
                "dtype": "bfloat16",
                "shape": [248320, 5120],
            },
        ],
        "gdn_conv_state_dtype": "bfloat16",
        "gdn_recurrent_state_dtype": "float32",
        "gdn_conv_state_layout": "SD",
        "gdn_conv_state_shape": [3, 10240],
        "gdn_recurrent_state_shape": [48, 128, 128],
        "rotary_dim": 64,
        "rope_theta": 10_000_000,
        "mrope_section": [11, 11, 10],
        "rotary_inv_freq_binding": "compiler_owned",
        "qronos_group_size": 128,
        "qronos_checkpoint_layout": "Kx(N/8)_packed_i32",
        "qronos_kernel_layout": "paiton_w4a16_g128_v1",
        "qronos_transform_version": "quark_qronos_reorder_signed_v1",
        "zero_centered_norm_transform": "gamma=1+checkpoint_bf16",
    }
    for key, expected in required_contract.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"unsupported Qwen3.8 contract {key}={contract.get(key)!r}; "
                f"expected {expected!r}"
            )
    if not 1 <= int(contract.get("max_num_batched_tokens", 0)) <= 8192:
        raise ValueError("Qwen3.8 contract max_num_batched_tokens must be in [1,8192]")
    if not 1 <= int(contract.get("max_context_length", 0)) <= 8192:
        raise ValueError("Qwen3.8 contract max_context_length must be in [1,8192]")
    if not 1 <= int(contract.get("max_batch_size", 0)) <= 256:
        raise ValueError("Qwen3.8 contract max_batch_size must be in [1,256]")

    raw_layouts = manifest.get("qronos_linears")
    if not isinstance(raw_layouts, list) or not raw_layouts:
        raise ValueError("manifest must declare non-empty qronos_linears")
    if contract.get("qronos_linear_count") != len(raw_layouts):
        raise ValueError("Qwen3.8 qronos_linear_count does not match manifest layouts")
    canonical = json.dumps(raw_layouts, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode()).hexdigest() != contract.get(
        "qronos_layout_sha256"
    ):
        raise ValueError("Qwen3.8 Qronos layout hash mismatch")

    parallelism = {
        "replicated": QronosParallelism.REPLICATED,
        "column": QronosParallelism.COLUMN,
        "row": QronosParallelism.ROW,
    }
    specs = []
    for index, layout in enumerate(raw_layouts):
        if not isinstance(layout, dict):
            raise ValueError(f"qronos_linears[{index}] must be an object")
        try:
            mode = parallelism[layout["parallelism"]]
            spec = QronosLinearSpec(
                source_prefix=layout["source_prefix"],
                target_weight_name=layout["target_weight_name"],
                target_scale_name=layout["target_scale_name"],
                input_size=int(layout["input_size"]),
                output_size=int(layout["output_size"]),
                parallelism=mode,
                padded_output_size=(
                    int(layout["padded_output_size"])
                    if layout.get("padded_output_size") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid qronos_linears[{index}]: {error}") from error
        specs.append(spec)

    # Reuse constructor duplicate checks before inspecting ABI tensor records.
    QronosStreamingTransformer(specs)
    interface = manifest.get("interface")
    tensors = interface.get("tensors") if isinstance(interface, dict) else None
    if not isinstance(tensors, list):
        raise ValueError("Qwen3.8 manifest interface.tensors must be a list")
    by_name = {}
    for tensor in tensors:
        if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
            raise ValueError("invalid Qwen3.8 interface tensor record")
        if tensor["name"] in by_name:
            raise ValueError(f"duplicate interface tensor {tensor['name']}")
        by_name[tensor["name"]] = tensor

    declared_targets = set()
    for spec in specs:
        padded_n = spec.padded_output_size or ((spec.output_size + 7) // 8 * 8)
        expected = {
            spec.target_weight_name: ("int32", [padded_n, spec.input_size // 8]),
            spec.target_scale_name: ("float32", [padded_n, spec.input_size // 128]),
        }
        for name, (dtype, shape) in expected.items():
            declared_targets.add(name)
            tensor = by_name.get(name)
            if tensor is None:
                raise ValueError(f"manifest interface is missing Qronos constant {name}")
            if "param" not in tensor.get("roles", []):
                raise ValueError(f"Qronos constant {name} must have param role")
            if tensor.get("binding") != "unbound":
                raise ValueError(f"Qronos constant {name} must be unbound")
            if tensor.get("dtype") != dtype:
                raise ValueError(
                    f"Qronos constant {name} dtype must be {dtype}, got {tensor.get('dtype')}"
                )
            shape_values = tensor.get("shape_values")
            exact_shape = [values[0] for values in shape_values] if isinstance(shape_values, list) else None
            if (
                exact_shape is None
                or any(not isinstance(values, list) or len(values) != 1 for values in shape_values)
                or exact_shape != shape
            ):
                raise ValueError(
                    f"Qronos constant {name} shape must be exactly {shape}, got {shape_values}"
                )
    extra_packed = sorted(
        name
        for name, tensor in by_name.items()
        if name not in declared_targets
        and "param" in tensor.get("roles", [])
        and (
            (name.endswith("_weight") and tensor.get("dtype") == "int32")
            or name.endswith("_weight_scale")
        )
    )
    if extra_packed:
        raise ValueError(
            "manifest has undeclared packed Qronos constants: "
            + ", ".join(extra_packed[:20])
        )
    return tuple(specs)
