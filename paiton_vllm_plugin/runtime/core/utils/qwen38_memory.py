"""Manifest-driven VRAM preflight for Qwen3.8 artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import math
from typing import Any, Mapping

from .qronos_loader import qwen38_specs_from_manifest


_DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
}
_MINIMUM_HEADROOM_BYTES = 2 * 1024**3
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Qwen38MemoryEstimate:
    compiled_unbound_constants_bytes: int
    compiler_owned_constants_bytes: int
    runtime_shell_parameters_bytes: int
    kv_cache_bytes: int
    gdn_state_bytes: int
    hybrid_cache_bytes: int
    activation_blob_bytes: int
    workspace_bytes: int
    allocator_headroom_bytes: int
    required_bytes: int
    available_bytes: int | None
    device_total_bytes: int | None
    fits: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype_bytes(dtype: object) -> int:
    try:
        return _DTYPE_BYTES[str(dtype)]
    except KeyError as error:
        raise ValueError(f"unsupported Qwen3.8 memory-estimator dtype {dtype!r}") from error


def _shape_numel(shape: object, *, label: str) -> int:
    if not isinstance(shape, list) or not shape or any(
        not isinstance(dim, int) or dim <= 0 for dim in shape
    ):
        raise ValueError(f"{label} must have a positive static shape")
    return math.prod(shape)


def _interface_numel(record: Mapping[str, Any]) -> int:
    shape_values = record.get("shape_values")
    if not isinstance(shape_values, list) or any(
        not isinstance(domain, list)
        or len(domain) != 1
        or not isinstance(domain[0], int)
        or domain[0] <= 0
        for domain in shape_values
    ):
        raise ValueError(
            f"manifest parameter {record.get('name')!r} lacks a positive static shape"
        )
    return math.prod(domain[0] for domain in shape_values)


def estimate_qwen38_memory(
    manifest: Mapping[str, Any],
    *,
    available_bytes: int | None = None,
    device_total_bytes: int | None = None,
    headroom_fraction: float = 0.10,
    hybrid_cache_reservation_bytes: int | None = None,
) -> Qwen38MemoryEstimate:
    """Estimate the complete minimum VRAM needed by one artifact contract."""

    qwen38_specs_from_manifest(manifest)
    if not 0.0 <= headroom_fraction <= 1.0:
        raise ValueError("headroom_fraction must be in [0,1]")
    if available_bytes is not None and available_bytes <= 0:
        raise ValueError("available_bytes must be positive")
    if (
        hybrid_cache_reservation_bytes is not None
        and hybrid_cache_reservation_bytes <= 0
    ):
        raise ValueError("hybrid_cache_reservation_bytes must be positive")

    contract = manifest["paiton_qwen38_contract"]
    interface = manifest.get("interface")
    records = interface.get("tensors") if isinstance(interface, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("Qwen3.8 manifest is missing its tensor interface")

    compiled_unbound = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Qwen3.8 tensor interface entries must be objects")
        if "param" not in record.get("roles", ()) or record.get("binding") != "unbound":
            continue
        compiled_unbound += _interface_numel(record) * _dtype_bytes(
            record.get("dtype")
        )

    shell_bytes = 0
    for record in contract["runtime_shell_parameters"]:
        shell_bytes += _shape_numel(
            record.get("shape"), label=f"runtime shell parameter {record.get('name')!r}"
        ) * _dtype_bytes(record.get("dtype"))

    planning = manifest.get("memory_planning")
    required_planning_fields = (
        "activation_blob_bytes",
        "compiler_owned_constant_bytes",
        "shared_workspace_bytes",
        "unique_workspace_bytes",
    )
    if not isinstance(planning, Mapping) or any(
        not isinstance(planning.get(field), int) or planning[field] < 0
        for field in required_planning_fields
    ):
        raise ValueError("Qwen3.8 manifest lacks exact memory-planning metadata")

    activation_bytes = planning["activation_blob_bytes"]
    compiler_owned_bytes = planning["compiler_owned_constant_bytes"]
    workspace_bytes = (
        planning["shared_workspace_bytes"] + planning["unique_workspace_bytes"]
    )

    dtype_bytes = _dtype_bytes(contract["kv_cache_dtype"])
    kv_cache_bytes = (
        int(contract["max_context_length"])
        * int(contract["num_full_attention_layers"])
        * 2
        * int(contract["num_key_value_heads"])
        * int(contract["head_dim"])
        * dtype_bytes
    )

    conv_numel = _shape_numel(
        contract["gdn_conv_state_shape"], label="GDN convolution state"
    )
    recurrent_numel = _shape_numel(
        contract["gdn_recurrent_state_shape"], label="GDN recurrent state"
    )
    per_gdn_state = (
        conv_numel * _dtype_bytes(contract["gdn_conv_state_dtype"])
        + recurrent_numel * _dtype_bytes(contract["gdn_recurrent_state_dtype"])
    )
    gdn_state_bytes = (
        per_gdn_state
        * int(contract["num_gdn_layers"])
        * int(contract["max_batch_size"])
    )
    hybrid_cache_bytes = max(
        kv_cache_bytes + gdn_state_bytes,
        hybrid_cache_reservation_bytes or 0,
    )

    subtotal = (
        compiled_unbound
        + compiler_owned_bytes
        + shell_bytes
        + hybrid_cache_bytes
        + activation_bytes
        + workspace_bytes
    )
    headroom_bytes = max(
        _MINIMUM_HEADROOM_BYTES,
        math.ceil(subtotal * headroom_fraction),
    )
    required_bytes = subtotal + headroom_bytes
    fits = None if available_bytes is None else required_bytes <= available_bytes
    return Qwen38MemoryEstimate(
        compiled_unbound_constants_bytes=compiled_unbound,
        compiler_owned_constants_bytes=compiler_owned_bytes,
        runtime_shell_parameters_bytes=shell_bytes,
        kv_cache_bytes=kv_cache_bytes,
        gdn_state_bytes=gdn_state_bytes,
        hybrid_cache_bytes=hybrid_cache_bytes,
        activation_blob_bytes=activation_bytes,
        workspace_bytes=workspace_bytes,
        allocator_headroom_bytes=headroom_bytes,
        required_bytes=required_bytes,
        available_bytes=available_bytes,
        device_total_bytes=device_total_bytes,
        fits=fits,
    )


def preflight_qwen38_memory(
    manifest: Mapping[str, Any],
    *,
    device: int = 0,
    hybrid_cache_reservation_bytes: int | None = None,
) -> Qwen38MemoryEstimate:
    """Compare the manifest estimate with current selected-device free VRAM."""

    import torch

    with torch.cuda.device(device):
        available_bytes, total_bytes = torch.cuda.mem_get_info()
    estimate = estimate_qwen38_memory(
        manifest,
        available_bytes=int(available_bytes),
        device_total_bytes=int(total_bytes),
        hybrid_cache_reservation_bytes=hybrid_cache_reservation_bytes,
    )
    _LOGGER.info(
        "Qwen3.8 VRAM preflight requires %.2f GiB including %.2f GiB "
        "headroom; %.2f GiB is free on device %d",
        estimate.required_bytes / 1024**3,
        estimate.allocator_headroom_bytes / 1024**3,
        estimate.available_bytes / 1024**3,
        device,
    )
    if not estimate.fits:
        raise MemoryError(
            "Qwen3.8 artifact memory preflight failed: "
            f"requires {estimate.required_bytes} bytes including headroom, "
            f"but only {estimate.available_bytes} bytes are free"
        )
    return estimate
