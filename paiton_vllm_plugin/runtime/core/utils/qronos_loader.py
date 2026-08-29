"""Bounded-memory, shape-strict Qronos linear checkpoint transformation."""

from dataclasses import dataclass
from enum import Enum
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
        if extra_quant_metadata:
            raise ValueError(
                "random-access Qronos source has undeclared quantized linears: "
                + ", ".join(extra_quant_metadata[:20])
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
