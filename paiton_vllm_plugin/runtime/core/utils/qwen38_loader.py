"""Strict bounded-memory loading for non-Qronos Qwen3.8 backbone constants."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch

from .qronos_loader import qwen38_specs_from_manifest


def configure_qwen38_cache_contract(cache_config, *, resolve_auto: bool) -> None:
    """Enforce BF16 KV/conv and FP32 recurrence for contract v2."""

    if cache_config.cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Paiton Qwen3.8 requires BF16 full-attention KV cache")
    if cache_config.mamba_cache_dtype not in ("auto", "bfloat16"):
        raise ValueError("Paiton Qwen3.8 requires BF16 convolution state")
    if cache_config.mamba_ssm_cache_dtype == "auto" and resolve_auto:
        cache_config.mamba_ssm_cache_dtype = "float32"
    if cache_config.mamba_ssm_cache_dtype != "float32":
        raise ValueError("Paiton Qwen3.8 requires FP32 recurrent state")
    if getattr(cache_config, "enable_prefix_caching", False):
        raise ValueError(
            "Paiton Qwen3.8 contract v2 does not yet support aligned prefix caching"
        )
    if cache_config.mamba_cache_mode != "none":
        raise ValueError(
            "Paiton Qwen3.8 contract v2 requires mamba_cache_mode=none"
        )


@dataclass(frozen=True)
class Qwen38TensorSpec:
    source_name: str
    target_name: str
    shape: Tuple[int, ...]
    add_one: bool = False


def _exact_shape(record):
    values = record.get("shape_values") if isinstance(record, dict) else None
    if not isinstance(values, list) or any(
        not isinstance(dim, list) or len(dim) != 1 or not isinstance(dim[0], int)
        for dim in values
    ):
        return None
    return tuple(dim[0] for dim in values)


def qwen38_unquantized_specs_from_manifest(
    manifest,
) -> Tuple[Qwen38TensorSpec, ...]:
    """Derive the exact checkpoint-to-ABI mapping from contract v2."""

    qronos_specs = qwen38_specs_from_manifest(manifest)
    contract = manifest["paiton_qwen38_contract"]
    num_layers = int(contract.get("num_hidden_layers", 0))
    source_layers = int(contract.get("source_num_hidden_layers", 0))
    if not 1 <= num_layers <= source_layers == 64:
        raise ValueError("invalid Qwen3.8 compiled/source layer counts")
    if int(contract.get("rotary_dim", 0)) != 64:
        raise ValueError("Qwen3.8 contract v2 requires rotary_dim=64")
    if int(contract.get("rope_theta", 0)) != 10_000_000:
        raise ValueError("Qwen3.8 contract v2 requires rope_theta=10000000")
    if contract.get("mrope_section") != [11, 11, 10]:
        raise ValueError("Qwen3.8 contract v2 requires mrope_section=[11,11,10]")

    interface = manifest["interface"]["tensors"]
    by_name = {record["name"]: record for record in interface}
    layer_kinds = {}
    for spec in qronos_specs:
        parts = spec.source_prefix.split(".")
        try:
            layer_index = int(parts[3])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"invalid Qwen3.8 checkpoint layer prefix {spec.source_prefix}"
            ) from error
        if ".linear_attn." in spec.source_prefix:
            kind = "gdn"
        elif ".self_attn." in spec.source_prefix:
            kind = "full"
        else:
            continue
        previous = layer_kinds.setdefault(layer_index, kind)
        if previous != kind:
            raise ValueError(f"mixed Qwen3.8 layer kind at index {layer_index}")
    if set(layer_kinds) != set(range(num_layers)):
        raise ValueError("Qwen3.8 Qronos layouts do not cover every compiled layer")
    if sum(kind == "gdn" for kind in layer_kinds.values()) != int(
        contract.get("num_gdn_layers", -1)
    ):
        raise ValueError("Qwen3.8 GDN layer count does not match layouts")
    if sum(kind == "full" for kind in layer_kinds.values()) != int(
        contract.get("num_full_attention_layers", -1)
    ):
        raise ValueError("Qwen3.8 full-attention layer count does not match layouts")

    specs = []
    for index in range(num_layers):
        source = f"model.language_model.layers.{index}"
        target = f"layers_{index}"
        specs.extend((
            Qwen38TensorSpec(
                f"{source}.input_layernorm.weight",
                f"{target}_input_layernorm_weight",
                (5120,),
                True,
            ),
            Qwen38TensorSpec(
                f"{source}.post_attention_layernorm.weight",
                f"{target}_post_attention_layernorm_weight",
                (5120,),
                True,
            ),
        ))
        if layer_kinds[index] == "gdn":
            specs.extend((
                Qwen38TensorSpec(
                    f"{source}.linear_attn.A_log",
                    f"{target}_linear_attn_A_log",
                    (48,),
                ),
                Qwen38TensorSpec(
                    f"{source}.linear_attn.conv1d.weight",
                    f"{target}_linear_attn_conv1d",
                    (10240, 1, 4),
                ),
                Qwen38TensorSpec(
                    f"{source}.linear_attn.dt_bias",
                    f"{target}_linear_attn_dt_bias",
                    (48,),
                ),
                Qwen38TensorSpec(
                    f"{source}.linear_attn.norm.weight",
                    f"{target}_linear_attn_norm_weight",
                    (128,),
                ),
            ))
        else:
            specs.extend((
                Qwen38TensorSpec(
                    f"{source}.self_attn.q_norm.weight",
                    f"{target}_self_attn_q_norm_weight",
                    (256,),
                    True,
                ),
                Qwen38TensorSpec(
                    f"{source}.self_attn.k_norm.weight",
                    f"{target}_self_attn_k_norm_weight",
                    (256,),
                    True,
                ),
            ))
    specs.append(Qwen38TensorSpec(
        "model.language_model.norm.weight", "norm_weight", (5120,), True
    ))

    target_names = set()
    for spec in specs:
        if spec.target_name in target_names:
            raise ValueError(f"duplicate Qwen3.8 target constant {spec.target_name}")
        target_names.add(spec.target_name)
        record = by_name.get(spec.target_name)
        if record is None:
            raise ValueError(f"manifest is missing Qwen3.8 constant {spec.target_name}")
        if "param" not in record.get("roles", []):
            raise ValueError(f"Qwen3.8 constant {spec.target_name} must have param role")
        if record.get("binding") != "unbound":
            raise ValueError(f"Qwen3.8 constant {spec.target_name} must be unbound")
        if record.get("dtype") != "bfloat16":
            raise ValueError(f"Qwen3.8 constant {spec.target_name} must be bfloat16")
        if _exact_shape(record) != spec.shape:
            raise ValueError(
                f"Qwen3.8 constant {spec.target_name} shape must be exactly "
                f"{spec.shape}, got {record.get('shape_values')}"
            )

    inv_freq = by_name.get("rotary_emb_inv_freq")
    expected_inv_shape = (int(contract["rotary_dim"]) // 2,)
    if (
        inv_freq is None
        or "param" not in inv_freq.get("roles", [])
        or inv_freq.get("dtype") != "float32"
        or inv_freq.get("binding") != "compiler_owned"
        or _exact_shape(inv_freq) != expected_inv_shape
    ):
        raise ValueError("manifest has an invalid rotary_emb_inv_freq constant")

    declared = target_names | {"rotary_emb_inv_freq"}
    declared.update(
        name
        for spec in qronos_specs
        for name in (spec.target_weight_name, spec.target_scale_name)
    )
    extra = sorted(
        record["name"]
        for record in interface
        if "param" in record.get("roles", []) and record["name"] not in declared
    )
    if extra:
        raise ValueError(
            "manifest has undeclared Qwen3.8 backbone constants: "
            + ", ".join(extra[:20])
        )
    return tuple(specs)


class Qwen38UnquantizedLoader:
    """Fetch and transform one small BF16 backbone constant at a time."""

    def __init__(self, manifest):
        self.manifest = manifest
        self.specs = qwen38_unquantized_specs_from_manifest(manifest)
        self.peak_source_bytes = 0

    @staticmethod
    def _source_tensor(source, name):
        if hasattr(source, "get_tensor"):
            return source.get_tensor(name)
        return source[name]

    def iter_from_random_access_source(self, source):
        available = set(source.keys())
        required = {spec.source_name for spec in self.specs}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                "Qwen3.8 checkpoint is missing backbone tensors: "
                + ", ".join(missing[:20])
            )
        for spec in self.specs:
            tensor = self._source_tensor(source, spec.source_name)
            if tensor.dtype is not torch.bfloat16:
                raise ValueError(
                    f"{spec.source_name} dtype must be torch.bfloat16, got {tensor.dtype}"
                )
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(
                    f"{spec.source_name} shape must be exactly {spec.shape}, "
                    f"got {tuple(tensor.shape)}"
                )
            self.peak_source_bytes = max(
                self.peak_source_bytes, tensor.numel() * tensor.element_size()
            )
            value = tensor.detach().to(device="cpu").contiguous()
            if spec.add_one:
                value = value.add(torch.ones_like(value))
            yield spec.target_name, value

    def iter_safetensors(self, path: Path | str):
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as source:
            yield from self.iter_from_random_access_source(source)


def resolve_qwen38_safetensors(
    model_ref: str,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
    download_dir: str | None = None,
) -> Path:
    """Resolve the pinned single-file checkpoint without copying it."""

    local = Path(model_ref)
    if local.exists():
        root = local if local.is_dir() else local.parent
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(
            repo_id=model_ref,
            repo_type="model",
            revision=revision,
            token=token,
            cache_dir=download_dir,
            allow_patterns=["model.safetensors", "model.safetensors.index.json"],
        ))
    index = root / "model.safetensors.index.json"
    if index.exists():
        raise ValueError("Qwen3.8 contract v2 requires one model.safetensors file")
    checkpoint = root / "model.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Qwen3.8 checkpoint not found at {checkpoint}")
    return checkpoint
