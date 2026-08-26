# SPDX-License-Identifier: Apache-2.0
"""Text-only Paiton adapter for the pinned Qwen3.8-27B Quark checkpoint."""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.models.qwen38_contract import (
    ADMITTED_TOKEN_COUNTS,
    Qwen38ArtifactFamily,
    Qwen38ContractError,
    TensorRecord,
    load_qwen38_artifact_family,
)
from paiton_vllm_plugin.models.qwen38_state import (
    LINEAR_ATTENTION_LAYERS,
    Qwen38SchedulerStateBridge,
)
from paiton_vllm_plugin.models.qwen38_schedule import (
    MAX_ACTIVE_SEQUENCES,
    MAX_BLOCKS_PER_SEQUENCE,
    KV_BLOCK_SIZE,
    Qwen38ScheduledMetadataBridge,
    Qwen38ScheduledStep,
    REQUIRED_PHYSICAL_KV_BLOCKS,
)
from paiton_vllm_plugin.models.qwen38_scheduled_contract import (
    SCHEDULED_MANIFEST_ENV,
    load_qwen38_scheduled_artifact_family,
)
from paiton_vllm_plugin.models.qwen38_greedy import Qwen38GreedyOutputProvider
from paiton_vllm_plugin.models.qwen38_trace import (
    Qwen38RuntimeTrace,
    install_qwen38_worker_cprofile,
)
from paiton_vllm_plugin.models.runtime_compat import current_device_stream_ptr
from paiton_vllm_plugin.runtime.core import Model


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu()
    return hashlib.sha256(memoryview(value.numpy())).hexdigest()


def _torch_dtype_for_record(record: TensorRecord) -> torch.dtype:
    mapping = {"BF16": torch.bfloat16, "U8": torch.uint8}
    try:
        return mapping[record.dtype]
    except KeyError as error:
        raise Qwen38ContractError(
            f"unsupported frozen checkpoint dtype {record.dtype!r} for "
            f"{record.checkpoint_name}"
        ) from error


def _materialize_artifact_constant_maps(
    family: Qwen38ArtifactFamily,
    logical_weights: Mapping[str, torch.Tensor],
    device: torch.device,
    *,
    artifacts: Mapping[object, object] | None = None,
    device_cache: dict[str, torch.Tensor] | None = None,
) -> tuple[dict[object, dict[str, torch.Tensor]], list[dict[str, object]]]:
    """Apply only hash-bound concatenation recipes from the family manifest."""

    required = family.required_records
    if set(logical_weights) != set(required):
        raise Qwen38ContractError(
            "logical checkpoint weights differ from the required text contract"
        )
    if artifacts is None:
        artifacts = family.artifacts
    if device_cache is None:
        device_cache = {}
    runtime_maps: dict[object, dict[str, torch.Tensor]] = {}
    plans: list[dict[str, object]] = []
    for artifact_key, artifact in sorted(
        artifacts.items(),
        key=lambda item: (
            int(getattr(item[1], "tokens")),
            str(getattr(item[1], "provider_route", "")),
        ),
    ):
        tokens = int(getattr(artifact, "tokens"))
        consumed = {
            source
            for derived in artifact.derived_constants
            for source in derived.source_names
        }
        runtime_weights: dict[str, torch.Tensor] = {}
        for source, record in sorted(required.items()):
            if source in consumed:
                continue
            assert record.runtime_name is not None
            value = device_cache.get(source)
            if value is None:
                value = logical_weights[source].detach().contiguous().to(device=device)
                device_cache[source] = value
            runtime_weights[record.runtime_name] = value

        derived_entries: list[dict[str, object]] = []
        for derived in artifact.derived_constants:
            source_tensors = []
            for source in derived.source_names:
                value = device_cache.get(source)
                if value is None:
                    value = (
                        logical_weights[source]
                        .detach()
                        .contiguous()
                        .to(device=device)
                    )
                    device_cache[source] = value
                source_tensors.append(value)
            value = torch.cat(
                source_tensors, dim=derived.concatenate_dimension
            ).contiguous()
            expected_dtype = {"U8": torch.uint8, "BF16": torch.bfloat16}[
                derived.dtype
            ]
            if value.dtype != expected_dtype or tuple(value.shape) != derived.shape:
                raise Qwen38ContractError(
                    f"materialized constant differs for M={tokens} "
                    f"{derived.runtime_name}: dtype={value.dtype} "
                    f"shape={tuple(value.shape)}"
                )
            runtime_weights[derived.runtime_name] = value
            derived_entries.append(
                {
                    "runtime_name": derived.runtime_name,
                    "source_names": list(derived.source_names),
                    "concatenate_dimension": derived.concatenate_dimension,
                    "dtype": derived.dtype,
                    "shape": list(derived.shape),
                    "provider_id": derived.provider_id,
                }
            )
        if set(runtime_weights) != set(artifact.physical_runtime_names):
            raise Qwen38ContractError(
                f"materialized runtime constants differ from M={tokens} physical ABI"
            )
        runtime_maps[artifact_key] = runtime_weights
        plans.append(
            {
                "tokens": tokens,
                "provider_route": getattr(artifact, "provider_route", None),
                "binding_plan_sha256": artifact.binding_plan_sha256,
                "required_runtime_constant_count": len(runtime_weights),
                "direct_runtime_constant_count": (
                    len(runtime_weights) - len(artifact.derived_constants)
                ),
                "derived_runtime_constant_count": len(artifact.derived_constants),
                "logical_checkpoint_source_count": len(required),
                "logical_source_coverage": "exactly-once",
                "derived_constants": derived_entries,
            }
        )
    return runtime_maps, plans


class PaitonQwen38ForCausalLM(nn.Module):
    """A narrow vLLM model for one accepted text-only serving envelope.

    The compiled decoder returns one final-token logit row.  ``forward``
    exposes that row through a zero-stride token dimension so vLLM can select
    its ordinary final-token index without allocating all-token logits.  Prompt
    logprobs are therefore outside this adapter's admitted contract.
    """

    is_hybrid = True
    has_inner_state = True
    supports_multimodal = False
    supports_mrope = True
    supports_prompt_logprobs = False
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        install_qwen38_worker_cprofile()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_text_config
        self._scheduled_enabled = bool(os.environ.get(SCHEDULED_MANIFEST_ENV))
        self._validate_serving_config(
            vllm_config,
            scheduled=self._scheduled_enabled,
        )
        self.family = load_qwen38_artifact_family()
        self.scheduled_family = (
            load_qwen38_scheduled_artifact_family(self.family)
            if self._scheduled_enabled
            else None
        )
        self._validate_model_binding(vllm_config, self.family)
        self.greedy_output_provider = (
            Qwen38GreedyOutputProvider(self.family.greedy_output_provider)
            if (
                not self._scheduled_enabled
                and self.family.greedy_output_provider is not None
            )
            else None
        )
        triton_key_start = time.perf_counter()
        from triton.runtime.cache import triton_key

        triton_key()
        self._triton_key_warmup_ms = (time.perf_counter() - triton_key_start) * 1000.0
        self.models = {
            tokens: Model(str(record.path))
            for tokens, record in self.family.artifacts.items()
        }
        self.scheduled_models = (
            {
                artifact_key: Model(str(record.path))
                for artifact_key, record in self.scheduled_family.artifacts.items()
            }
            if self.scheduled_family is not None
            else {}
        )
        self._validate_runtime_abis()
        self.state_bridge = Qwen38SchedulerStateBridge(vllm_config)
        self.scheduled_bridge = (
            Qwen38ScheduledMetadataBridge(
                self.state_bridge.gdn_layers,
                self.state_bridge.attention_layers,
            )
            if self._scheduled_enabled
            else None
        )
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            self.config.vocab_size,
            logits_as_input=True,
        )
        self.dtype = torch.bfloat16
        self.unpadded_vocab_size = self.config.vocab_size
        self._weights_loaded = False
        graph_mode = os.environ.get("PAITON_QWEN38_WHOLE_MODEL_GRAPH", "0")
        if graph_mode not in ("0", "1"):
            raise Qwen38ContractError(
                "PAITON_QWEN38_WHOLE_MODEL_GRAPH must be exactly 0 or 1"
            )
        self._whole_model_graph = graph_mode == "1"
        if self._scheduled_enabled and self._whole_model_graph:
            raise Qwen38ContractError(
                "scheduled multi-sequence execution does not require or admit "
                "whole-model graph mode"
            )
        self._profile_inputs: dict[int, dict[str, torch.Tensor]] = {}
        self._profile_outputs: dict[int, dict[str, torch.Tensor]] = {}
        self._sessions: dict[int, object] = {}
        self._scheduled_inputs: dict[
            tuple[int, str], dict[str, torch.Tensor]
        ] = {}
        self._scheduled_outputs: dict[
            tuple[int, str], dict[str, torch.Tensor]
        ] = {}
        self._scheduled_sessions: dict[tuple[int, str], object] = {}
        self._scheduled_storage_identity: dict[
            tuple[int, str], dict[str, tuple[int, int]]
        ] = {}
        self._scheduled_route_selection_counts: dict[str, int] = {}
        self._scheduled_pending_logits: torch.Tensor | None = None
        self._scheduled_allocation_only_logits_pending = False
        self._rotary_cos: torch.Tensor | None = None
        self._rotary_sin: torch.Tensor | None = None
        self._position_ids: torch.Tensor | None = None
        self.runtime_trace = Qwen38RuntimeTrace()
        self.weight_load_result: dict[str, object] | None = None
        self.activation_proof = {
            "model_class": f"{type(self).__module__}:{type(self).__name__}",
            "manifest": str(self.family.manifest_path),
            "manifest_sha256": self.family.manifest_sha256,
            "plugin_commit": self.family.plugin_commit,
            "plugin_tree": self.family.plugin_tree,
            "artifacts": {
                str(tokens): {
                    "path": str(record.path),
                    "sha256": record.sha256,
                }
                for tokens, record in self.family.artifacts.items()
            },
            "stock_fallback": False,
            "triton_key_warmup_ms": self._triton_key_warmup_ms,
            "whole_model_graph": self._whole_model_graph,
            "scheduled_multisequence": self._scheduled_enabled,
            "scheduled_manifest": (
                str(self.scheduled_family.manifest_path)
                if self.scheduled_family is not None
                else None
            ),
            "scheduled_manifest_sha256": (
                self.scheduled_family.manifest_sha256
                if self.scheduled_family is not None
                else None
            ),
            "scheduled_artifact_routes": (
                {
                    f"m{record.tokens}:{record.provider_route}": {
                        "path": str(record.path),
                        "sha256": record.sha256,
                        "provider_id": record.provider_id,
                        "predicate_sha256": (
                            record.provider_route_predicate_sha256
                        ),
                    }
                    for record in self.scheduled_family.artifacts.values()
                }
                if self.scheduled_family is not None
                else None
            ),
            "greedy_output_provider": (
                self.greedy_output_provider.activation
                if self.greedy_output_provider is not None
                else None
            ),
        }

    @staticmethod
    def _validate_serving_config(
        vllm_config: VllmConfig,
        *,
        scheduled: bool = False,
    ) -> None:
        parallel = vllm_config.parallel_config
        scheduler = vllm_config.scheduler_config
        cache = vllm_config.cache_config
        failures: list[str] = []
        if parallel.tensor_parallel_size != 1:
            failures.append("tensor_parallel_size must be 1")
        if parallel.pipeline_parallel_size != 1:
            failures.append("pipeline_parallel_size must be 1")
        if parallel.data_parallel_size != 1:
            failures.append("data_parallel_size must be 1")
        if getattr(parallel, "prefill_context_parallel_size", 1) != 1:
            failures.append("prefill_context_parallel_size must be 1")
        if getattr(parallel, "decode_context_parallel_size", 1) != 1:
            failures.append("decode_context_parallel_size must be 1")
        if getattr(parallel, "enable_dbo", False):
            failures.append("DBO must be disabled")
        expected_sequences = MAX_ACTIVE_SEQUENCES if scheduled else 1
        if scheduler.max_num_seqs != expected_sequences:
            failures.append(f"max_num_seqs must be {expected_sequences}")
        required_tokens = 4096 if scheduled else 2048
        if scheduled and scheduler.max_num_batched_tokens != required_tokens:
            failures.append(
                f"max_num_batched_tokens must be exactly {required_tokens}"
            )
        elif not scheduled and scheduler.max_num_batched_tokens < required_tokens:
            failures.append(
                f"max_num_batched_tokens must admit M={required_tokens}"
            )
        if vllm_config.model_config.max_model_len != 4096:
            failures.append("max_model_len must be exactly 4096")
        chunked_prefill = getattr(scheduler, "enable_chunked_prefill", False)
        if scheduled and not chunked_prefill:
            failures.append("chunked prefill must be enabled for scheduled artifacts")
        if not scheduled and chunked_prefill:
            failures.append("chunked prefill must be disabled for fixed-M artifacts")
        if vllm_config.speculative_config is not None:
            failures.append("MTP/speculative decoding is not supported")
        if vllm_config.lora_config is not None:
            failures.append("LoRA is not supported")
        if scheduled and cache.block_size != KV_BLOCK_SIZE:
            failures.append(
                f"scheduler KV page size must be exactly {KV_BLOCK_SIZE}"
            )
        elif not scheduled and (cache.block_size <= 0 or cache.block_size % 16):
            failures.append("scheduler KV page size must be a positive multiple of 16")
        if scheduled and getattr(cache, "num_gpu_blocks_override", None) != (
            REQUIRED_PHYSICAL_KV_BLOCKS
        ):
            failures.append(
                "num_gpu_blocks_override must be exactly "
                f"{REQUIRED_PHYSICAL_KV_BLOCKS}"
            )
        if cache.cache_dtype not in ("auto", "bfloat16"):
            failures.append("KV cache dtype must resolve to BF16")
        if cache.mamba_cache_mode != "none":
            failures.append("mamba_cache_mode must be 'none'")
        if cache.enable_prefix_caching:
            failures.append("prefix caching is outside the fixed-M serving envelope")
        if cache.mamba_block_size != 4096:
            failures.append("mamba_block_size must equal max_model_len (4096)")
        if failures:
            raise Qwen38ContractError(
                "unsupported Qwen3.8 serving configuration: " + "; ".join(failures)
            )

    @staticmethod
    def _validate_model_binding(
        vllm_config: VllmConfig, family: Qwen38ArtifactFamily
    ) -> None:
        model_ref = Path(vllm_config.model_config.model).resolve()
        if model_ref != family.checkpoint_path.parent:
            raise Qwen38ContractError(
                f"model directory {model_ref} is not the pinned checkpoint directory "
                f"{family.checkpoint_path.parent}"
            )
        revision = vllm_config.model_config.revision
        if revision is not None:
            # A local byte-identified checkout has no meaningful HF revision
            # field.  If supplied, it must still be the immutable target.
            from paiton_vllm_plugin.models.qwen38_contract import TARGET_REVISION

            if revision != TARGET_REVISION:
                raise Qwen38ContractError(
                    f"checkpoint revision must be {TARGET_REVISION}, got {revision}"
                )
        if self_arch := getattr(vllm_config.model_config, "architecture", None):
            if self_arch != "PaitonQwen38ForCausalLM":
                raise Qwen38ContractError(
                    f"Paiton architecture override is missing: {self_arch!r}"
                )

    def _validate_runtime_abis(self) -> None:
        expected_states = {str(item["name"]) for item in self.family.state_records}
        expected_inputs = {"input_ids", "positions", "cos", "sin", *expected_states}
        for tokens, model in self.models.items():
            actual_constants = set(model.get_constant_names())
            expected_physical = set(self.family.artifacts[tokens].physical_runtime_names)
            if actual_constants != expected_physical:
                raise Qwen38ContractError(
                    f"artifact M={tokens} constant ABI mismatch: "
                    f"missing={sorted(expected_physical - actual_constants)}, "
                    f"unexpected={sorted(actual_constants - expected_physical)}"
                )
            actual_inputs = set(model.get_input_name_to_index_map())
            if actual_inputs != expected_inputs:
                raise Qwen38ContractError(
                    f"artifact M={tokens} input ABI mismatch: "
                    f"missing={sorted(expected_inputs - actual_inputs)}, "
                    f"unexpected={sorted(actual_inputs - expected_inputs)}"
                )
            if model.get_output_name_to_index_map() != {"logits": 0}:
                raise Qwen38ContractError(f"artifact M={tokens} output name ABI differs")
            if model.get_output_maximum_shape("logits") != [1, 248320]:
                raise Qwen38ContractError(f"artifact M={tokens} output shape ABI differs")
        if self.scheduled_family is None:
            return
        metadata_inputs = {
            "actual_token_count",
            "actual_sequence_count",
            "query_start_locations",
            "token_to_sequence",
            "gdn_has_initial_state",
            "slot_mapping",
            "block_tables",
            "logit_row_indices",
        }
        metadata_inputs.update(
            f"gdn_state_indices_{layer}" for layer in LINEAR_ATTENTION_LAYERS
        )
        scheduled_inputs = {"input_ids", "positions", "cos", "sin", *expected_states}
        scheduled_inputs.update(metadata_inputs)
        for artifact_key, model in self.scheduled_models.items():
            record = self.scheduled_family.artifacts[artifact_key]
            tokens = record.tokens
            actual_constants = set(model.get_constant_names())
            expected_physical = set(record.physical_runtime_names)
            if actual_constants != expected_physical:
                raise Qwen38ContractError(
                    "scheduled artifact "
                    f"M={tokens} route={record.provider_route} constant ABI mismatch: "
                    f"missing={sorted(expected_physical - actual_constants)}, "
                    f"unexpected={sorted(actual_constants - expected_physical)}"
                )
            actual_inputs = set(model.get_input_name_to_index_map())
            if actual_inputs != scheduled_inputs:
                raise Qwen38ContractError(
                    f"scheduled artifact M={tokens} input ABI mismatch: "
                    f"missing={sorted(scheduled_inputs - actual_inputs)}, "
                    f"unexpected={sorted(actual_inputs - scheduled_inputs)}"
                )
            if model.get_output_name_to_index_map() != {"logits": 0}:
                raise Qwen38ContractError(
                    f"scheduled artifact M={tokens} output name ABI differs"
                )
            expected_output = [min(tokens, MAX_ACTIVE_SEQUENCES), 248320]
            if model.get_output_maximum_shape("logits") != expected_output:
                raise Qwen38ContractError(
                    f"scheduled artifact M={tokens} output shape ABI differs"
                )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return Qwen38SchedulerStateBridge.mamba_state_shape()

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return Qwen38SchedulerStateBridge.mamba_state_dtype()

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return Qwen38SchedulerStateBridge.mamba_state_copy_func()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        raise Qwen38ContractError(
            "external embeddings and multimodal embedding merge are not supported"
        )

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[object],
    ) -> tuple[torch.Tensor, int]:
        if mm_features:
            raise Qwen38ContractError(
                "vision features are not admitted by the text-only M-RoPE contract"
            )
        positions = torch.arange(len(input_tokens), dtype=torch.int64)
        return positions.unsqueeze(0).expand(3, -1).clone(), 0

    @staticmethod
    def _text_positions(positions: torch.Tensor, tokens: int) -> torch.Tensor:
        if positions.ndim == 1 and tuple(positions.shape) == (tokens,):
            return positions.to(dtype=torch.int64).contiguous()
        if positions.ndim == 2 and tuple(positions.shape) == (3, tokens):
            return positions[0].to(dtype=torch.int64).contiguous()
        raise Qwen38ContractError(
            f"text positions must have shape ({tokens},) or (3, {tokens}), "
            f"got {tuple(positions.shape)}"
        )

    @staticmethod
    def _assert_text_positions(
        positions: torch.Tensor, expected: torch.Tensor
    ) -> None:
        # One reduction validates both the text-only M-RoPE planes and their
        # exact scheduler position sequence.  Keeping this as an async device
        # assertion preserves fail-closed behavior without three launches.
        torch._assert_async(torch.all(positions == expected))

    @staticmethod
    def _rotary_inputs(
        positions: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = positions.to(device=device, dtype=torch.float32)
        frequency = 1.0 / (
            10_000_000.0
            ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64.0)
        )
        phase = torch.outer(values, frequency)
        phase = torch.cat((phase, phase), dim=-1)
        return phase.cos(), phase.sin()

    @staticmethod
    def _is_allocation_only_metadata(metadata: object) -> bool:
        return metadata is None or (
            isinstance(metadata, dict)
            and (not metadata or not any(value is not None for value in metadata.values()))
        )

    def _initialize_rotary_tables(self, device: torch.device) -> None:
        if self._rotary_cos is not None or self._rotary_sin is not None:
            raise Qwen38ContractError("rotary tables may be initialized exactly once")
        self._position_ids = torch.arange(4096, dtype=torch.int64, device=device)
        positions = torch.arange(4096, dtype=torch.float32, device=device)
        frequency = 1.0 / (
            10_000_000.0
            ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64.0)
        )
        phase = torch.outer(positions, frequency)
        phase = torch.cat((phase, phase), dim=-1)
        self._rotary_cos = phase.cos()
        self._rotary_sin = phase.sin()

    @staticmethod
    def _scheduled_state_inputs(
        step: Qwen38ScheduledStep,
    ) -> dict[str, torch.Tensor]:
        state_inputs: dict[str, torch.Tensor] = {}
        for layer, (conv, recurrent) in step.gdn_storage.items():
            state_inputs[f"gdn_conv_state_{layer}"] = conv
            state_inputs[f"gdn_recurrent_state_{layer}"] = recurrent
        for layer, cache in step.attention_storage.items():
            state_inputs[f"kv_cache_{layer}"] = cache
        return state_inputs

    @staticmethod
    def _tensor_binding_identity(tensor: torch.Tensor) -> tuple[int, int]:
        return tensor.data_ptr(), tensor.numel() * tensor.element_size()

    def _ensure_scheduled_session(
        self,
        step: Qwen38ScheduledStep,
        device: torch.device,
    ) -> None:
        bucket = step.token_bucket
        artifact_key = (bucket, step.provider_route)
        state_inputs = self._scheduled_state_inputs(step)
        state_identity = {
            name: self._tensor_binding_identity(tensor)
            for name, tensor in state_inputs.items()
        }
        if artifact_key in self._scheduled_sessions:
            if state_identity != self._scheduled_storage_identity[artifact_key]:
                raise Qwen38ContractError(
                    "scheduler-owned state addresses changed after scheduled "
                    f"M={bucket} session binding"
                )
            return
        model = self.scheduled_models.get(artifact_key)
        if model is None:
            raise Qwen38ContractError(
                "scheduled model family has no artifact for "
                f"M={bucket} route={step.provider_route}"
            )
        output_rows = min(bucket, MAX_ACTIVE_SEQUENCES)
        inputs = {
            "input_ids": torch.zeros(bucket, dtype=torch.int64, device=device),
            "positions": torch.zeros(bucket, dtype=torch.int64, device=device),
            "cos": torch.zeros((bucket, 64), dtype=torch.float32, device=device),
            "sin": torch.zeros((bucket, 64), dtype=torch.float32, device=device),
            "actual_token_count": torch.zeros(1, dtype=torch.int32, device=device),
            "actual_sequence_count": torch.zeros(1, dtype=torch.int32, device=device),
            "query_start_locations": torch.zeros(
                MAX_ACTIVE_SEQUENCES + 1, dtype=torch.int32, device=device
            ),
            "token_to_sequence": torch.zeros(
                bucket, dtype=torch.int32, device=device
            ),
            "gdn_has_initial_state": torch.zeros(
                MAX_ACTIVE_SEQUENCES, dtype=torch.int32, device=device
            ),
            "slot_mapping": torch.zeros(bucket, dtype=torch.int64, device=device),
            "block_tables": torch.zeros(
                (MAX_ACTIVE_SEQUENCES, MAX_BLOCKS_PER_SEQUENCE),
                dtype=torch.int32,
                device=device,
            ),
            "logit_row_indices": torch.zeros(
                output_rows, dtype=torch.int32, device=device
            ),
            **state_inputs,
        }
        inputs.update(
            {
                f"gdn_state_indices_{layer}": torch.zeros(
                    MAX_ACTIVE_SEQUENCES,
                    dtype=torch.int32,
                    device=device,
                )
                for layer in step.gdn_state_indices
            }
        )
        outputs = {
            "logits": torch.empty(
                (output_rows, self.config.vocab_size),
                dtype=torch.bfloat16,
                device=device,
            )
        }
        self._scheduled_inputs[artifact_key] = inputs
        self._scheduled_outputs[artifact_key] = outputs
        self._scheduled_sessions[artifact_key] = model.create_persistent_tensor_session(
            inputs,
            outputs,
            static_output_shapes={
                "logits": (output_rows, self.config.vocab_size)
            },
        )
        self._scheduled_storage_identity[artifact_key] = state_identity
        route_label = f"m{bucket}:{step.provider_route}"
        self.activation_proof.setdefault("scheduled_sessions", {})[route_label] = {
            "state_tensor_count": len(state_inputs),
            "logical_bound_state_span_bytes": sum(
                size for _, size in state_identity.values()
            ),
            "output_rows": output_rows,
            "graph_mode": False,
            "provider_route": step.provider_route,
            "provider_route_predicate_sha256": (
                step.provider_route_predicate_sha256
            ),
        }

    def _populate_scheduled_inputs(
        self,
        step: Qwen38ScheduledStep,
        input_ids: torch.Tensor,
        text_positions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        inputs = self._scheduled_inputs[(step.token_bucket, step.provider_route)]
        tokens = step.token_count
        sequences = step.sequence_count
        if input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_contiguous():
            raise Qwen38ContractError("input_ids must be contiguous int32 or int64")
        self._async_position_guards(text_positions)
        assert self._rotary_cos is not None and self._rotary_sin is not None
        inputs["input_ids"][:tokens].copy_(input_ids)
        inputs["positions"][:tokens].copy_(text_positions)
        inputs["cos"][:tokens].copy_(self._rotary_cos[text_positions])
        inputs["sin"][:tokens].copy_(self._rotary_sin[text_positions])
        if tokens < step.token_bucket:
            inputs["input_ids"][tokens:].zero_()
            inputs["positions"][tokens:].zero_()
            inputs["cos"][tokens:].zero_()
            inputs["sin"][tokens:].zero_()
        inputs["actual_token_count"].fill_(tokens)
        inputs["actual_sequence_count"].fill_(sequences)
        inputs["query_start_locations"][: sequences + 1].copy_(
            step.query_start_locations
        )
        inputs["token_to_sequence"][:tokens].copy_(step.token_to_sequence)
        for layer, indices in step.gdn_state_indices.items():
            name = f"gdn_state_indices_{layer}"
            inputs[name][:sequences].copy_(indices)
        inputs["gdn_has_initial_state"][:sequences].copy_(
            step.gdn_has_initial_state
        )
        inputs["slot_mapping"][:tokens].copy_(step.slot_mapping)
        inputs["block_tables"][:sequences].copy_(step.block_tables)
        inputs["logit_row_indices"][:sequences].copy_(step.logit_row_indices)
        return inputs

    @staticmethod
    def _async_position_guards(text_positions: torch.Tensor) -> None:
        torch._assert_async(
            torch.all((text_positions >= 0) & (text_positions < 4096))
        )

    def _forward_scheduled(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        text_positions: torch.Tensor,
        metadata: Mapping[str, object],
        device: torch.device,
    ) -> torch.Tensor:
        if self.scheduled_bridge is None:
            raise Qwen38ContractError("scheduled metadata bridge is not initialized")
        if positions.ndim == 2:
            torch._assert_async(torch.all(positions == text_positions.unsqueeze(0)))
        step = self.scheduled_bridge.prepare(
            metadata,
            int(input_ids.shape[0]),
            text_positions,
        )
        self.runtime_trace.mark(
            "scheduled_metadata_ready",
            sequences=step.sequence_count,
            token_bucket=step.token_bucket,
            provider_route=step.provider_route,
            provider_route_predicate_sha256=(
                step.provider_route_predicate_sha256
            ),
        )
        self._ensure_scheduled_session(step, device)
        inputs = self._populate_scheduled_inputs(step, input_ids, text_positions)
        self.runtime_trace.mark("descriptors_ready", input_count=len(inputs))
        if self._scheduled_pending_logits is not None:
            raise Qwen38ContractError("previous scheduled logits were not consumed")
        route_key = (step.token_bucket, step.provider_route)
        self._scheduled_route_selection_counts[step.provider_route] = (
            self._scheduled_route_selection_counts.get(step.provider_route, 0) + 1
        )
        result = self._scheduled_sessions[route_key].run(
            stream_ptr=current_device_stream_ptr(device),
            sync=False,
            graph_mode=False,
        )
        self.runtime_trace.mark("model_enqueued")
        logits = result["logits"][: step.sequence_count]
        self._scheduled_pending_logits = logits
        self.activation_proof["last_scheduled_step"] = {
            "actual_tokens": step.token_count,
            "actual_sequences": step.sequence_count,
            "token_bucket": step.token_bucket,
            "provider_route": step.provider_route,
            "provider_route_predicate_sha256": (
                step.provider_route_predicate_sha256
            ),
            "provider_route_selection_counts": dict(
                sorted(self._scheduled_route_selection_counts.items())
            ),
            "paiton_model_calls": 1,
            "stock_fallback_calls": 0,
        }
        self.runtime_trace.mark("wrapper_handoff")
        return logits[0:1].expand(step.token_count, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise Qwen38ContractError("pipeline parallel execution is not supported")
        if inputs_embeds is not None:
            raise Qwen38ContractError("vision and external embeddings are not supported")
        forbidden = {
            key
            for key, value in kwargs.items()
            if value is not None
            and key
            in {
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "multimodal_embeddings",
            }
        }
        if forbidden:
            raise Qwen38ContractError(
                f"vision inputs are explicitly rejected: {sorted(forbidden)}"
            )
        if not self._weights_loaded:
            raise Qwen38ContractError("Paiton constants have not been strictly loaded")
        if input_ids.ndim != 1:
            raise Qwen38ContractError(
                f"flattened input_ids must have rank 1, got {tuple(input_ids.shape)}"
            )
        tokens = int(input_ids.shape[0])
        self.runtime_trace.begin(tokens=tokens)
        device = input_ids.device
        if device.type != "cuda":
            raise Qwen38ContractError(f"Paiton inputs must be on HIP, got {device}")
        text_positions = self._text_positions(positions, tokens)
        self.runtime_trace.mark("positions_ready")
        forward_context: ForwardContext = get_forward_context()
        metadata = forward_context.attn_metadata
        allocation_only = self._is_allocation_only_metadata(metadata)
        self.runtime_trace.classify_allocation_only(allocation_only)
        if allocation_only:
            # vLLM performs allocation-only dummy forwards before scheduler KV
            # state is bound.  They are not requests and must not consume or
            # create request lifecycle state.  Constants and runtime containers
            # are already resident, so a compact zero row is sufficient for
            # the runner's memory accounting pass.
            result = torch.zeros(
                (1, self.config.vocab_size), dtype=torch.bfloat16, device=device
            ).expand(tokens, -1)
            if self._scheduled_enabled:
                if (
                    self._scheduled_pending_logits is not None
                    or self._scheduled_allocation_only_logits_pending
                ):
                    raise Qwen38ContractError(
                        "previous scheduled logits were not consumed"
                    )
                # vLLM's startup memory profile performs this allocation-only
                # forward and then independently profiles the sampler through
                # compute_logits().  Record that exact handoff so a missing
                # real scheduled result still fails closed.
                self._scheduled_allocation_only_logits_pending = True
            self.runtime_trace.mark("allocation_only_handoff")
            self.runtime_trace.finish()
            return result
        if not isinstance(metadata, dict):
            raise Qwen38ContractError("microbatch/DBO attention metadata is not admitted")
        if self._scheduled_enabled:
            return self._forward_scheduled(
                input_ids,
                positions,
                text_positions,
                metadata,
                device,
            )
        if tokens not in ADMITTED_TOKEN_COUNTS:
            self.family.artifact_for_tokens(tokens)  # raises the canonical error
        if self._whole_model_graph and tokens != 1:
            raise Qwen38ContractError(
                "whole-model graph mode is initialized only for the M=1 profile"
            )
        prepared = self.state_bridge.prepare(
            metadata, tokens, device, trace=self.runtime_trace
        )
        if not prepared.persistent:
            raise Qwen38ContractError("persistent state arena is not initialized")
        if input_ids.dtype not in (torch.int32, torch.int64) or not input_ids.is_contiguous():
            raise Qwen38ContractError("input_ids must be contiguous int32 or int64")
        profile_inputs = self._profile_inputs[tokens]
        profile_inputs["input_ids"].copy_(input_ids)
        start = prepared.max_seq_len - tokens
        assert self._rotary_cos is not None and self._rotary_sin is not None
        assert self._position_ids is not None
        expected_positions = self._position_ids[start : prepared.max_seq_len]
        self._assert_text_positions(
            positions,
            expected_positions,
        )
        if tokens == 1:
            profile_inputs["positions"].copy_(text_positions)
            profile_inputs["cos"].copy_(
                self._rotary_cos[start : prepared.max_seq_len]
            )
            profile_inputs["sin"].copy_(
                self._rotary_sin[start : prepared.max_seq_len]
            )
        elif not prepared.fresh_prefill or start != 0:
            raise Qwen38ContractError(
                "M>1 profiles require a fresh zero-origin prefill"
            )
        self.runtime_trace.mark("rotary_ready")
        inputs = profile_inputs
        self.runtime_trace.mark("descriptors_ready", input_count=len(inputs))
        result = self._sessions[tokens].run(
            stream_ptr=current_device_stream_ptr(device),
            sync=False,
            graph_mode=self._whole_model_graph,
        )
        self.runtime_trace.mark("model_enqueued")
        self.state_bridge.commit(prepared)
        self.runtime_trace.mark("state_committed")
        # Compact-logits ABI: vLLM samples only the last index in this admitted
        # envelope.  All token rows alias the single compiled final-token row.
        output = result["logits"].expand(tokens, -1)
        self.runtime_trace.mark("wrapper_handoff")
        return output

    def _initialize_persistent_runtime(self, device: torch.device) -> None:
        initialization_start = time.perf_counter()
        if self.greedy_output_provider is not None:
            self.greedy_output_provider.initialize(device)
        state_inputs = self.state_bridge.initialize_persistent_arena(device)
        self._initialize_rotary_tables(device)
        for tokens, model in self.models.items():
            inputs = {
                "input_ids": torch.empty(tokens, dtype=torch.int64, device=device),
                "positions": self._position_ids[:tokens].clone(),
                "cos": self._rotary_cos[:tokens].clone(),
                "sin": self._rotary_sin[:tokens].clone(),
                **state_inputs,
            }
            outputs = {
                "logits": torch.empty(
                    (1, self.config.vocab_size), dtype=torch.bfloat16, device=device
                )
            }
            self._profile_inputs[tokens] = inputs
            self._profile_outputs[tokens] = outputs
            self._sessions[tokens] = model.create_persistent_tensor_session(
                inputs,
                outputs,
                static_output_shapes={"logits": (1, self.config.vocab_size)},
            )
        stream_ptr = current_device_stream_ptr(device)
        paiton_warmup_start = time.perf_counter()
        warmed_profiles = (1,)
        for tokens in warmed_profiles:
            inputs = self._profile_inputs[tokens]
            inputs["input_ids"].zero_()
            self._sessions[tokens].run(
                stream_ptr=stream_ptr,
                sync=False,
                graph_mode=self._whole_model_graph,
            )
        torch.cuda.synchronize(device)
        paiton_warmup_ms = (time.perf_counter() - paiton_warmup_start) * 1000.0
        self.state_bridge.reset_persistent_arena(force=True)

        logprob_warmup_start = time.perf_counter()
        from vllm.v1.sample.ops.logprobs import batched_count_greater_than

        logprobs = torch.zeros(
            (1, self.config.vocab_size), dtype=torch.float32, device=device
        )
        selected = torch.zeros((1, 1), dtype=torch.float32, device=device)
        batched_count_greater_than(logprobs, selected)
        torch.cuda.synchronize(device)
        logprob_warmup_ms = (time.perf_counter() - logprob_warmup_start) * 1000.0
        self.activation_proof["persistent_runtime"] = {
            "state_tensor_count": len(state_inputs),
            "state_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in state_inputs.values()
            ),
            "profile_sessions": sorted(self._sessions),
            "eager_warmed_profiles": list(warmed_profiles),
            "graph_capture_profiles": (
                list(warmed_profiles) if self._whole_model_graph else []
            ),
            "paiton_eager_warmup_ms": paiton_warmup_ms,
            "logprob_rank_warmup_ms": logprob_warmup_ms,
            "total_initialization_ms": (
                time.perf_counter() - initialization_start
            )
            * 1000.0,
        }

    def _initialize_scheduled_runtime(self, device: torch.device) -> None:
        initialization_start = time.perf_counter()
        self._initialize_rotary_tables(device)
        logprob_warmup_start = time.perf_counter()
        from vllm.v1.sample.ops.logprobs import batched_count_greater_than

        logprobs = torch.zeros(
            (1, self.config.vocab_size), dtype=torch.float32, device=device
        )
        selected = torch.zeros((1, 1), dtype=torch.float32, device=device)
        batched_count_greater_than(logprobs, selected)
        torch.cuda.synchronize(device)
        self.activation_proof["scheduled_runtime"] = {
            "scheduler_owned_state": True,
            "private_state_arena": False,
            "profile_sessions": [],
            "graph_capture_profiles": [],
            "logprob_rank_warmup_ms": (
                time.perf_counter() - logprob_warmup_start
            )
            * 1000.0,
            "total_initialization_ms": (
                time.perf_counter() - initialization_start
            )
            * 1000.0,
        }

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.runtime_trace.mark("logits_handoff_entry")
        if self._scheduled_enabled:
            if self._scheduled_pending_logits is None:
                if not self._scheduled_allocation_only_logits_pending:
                    raise Qwen38ContractError(
                        "scheduled forward produced no pending logits"
                    )
                if (
                    hidden_states.ndim != 2
                    or int(hidden_states.shape[-1]) != self.config.vocab_size
                ):
                    raise Qwen38ContractError(
                        "allocation-only sampler input must already contain logits"
                    )
                self._scheduled_allocation_only_logits_pending = False
            else:
                if self._scheduled_allocation_only_logits_pending:
                    raise Qwen38ContractError(
                        "scheduled and allocation-only logits cannot both be pending"
                    )
                hidden_states = self._scheduled_pending_logits
                self._scheduled_pending_logits = None
        result = self.logits_processor(None, hidden_states)
        if self.greedy_output_provider is not None and tuple(result.shape) == (
            1,
            self.config.vocab_size,
        ):
            self.greedy_output_provider.register_logits(result)
        self.runtime_trace.mark("logits_handoff_exit")
        self.runtime_trace.finish()
        return result

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        if self._weights_loaded:
            raise Qwen38ContractError("weights may be loaded exactly once")
        records = self.family.tensor_records
        seen: set[str] = set()
        logical_weights: dict[str, torch.Tensor] = {}
        category_counts: dict[str, int] = {}
        category_digests: dict[str, hashlib._Hash] = {}
        for name, tensor in weights:
            if name in seen:
                raise Qwen38ContractError(f"duplicate checkpoint tensor: {name}")
            record = records.get(name)
            if record is None:
                raise Qwen38ContractError(f"unknown checkpoint tensor: {name}")
            seen.add(name)
            if tuple(tensor.shape) != record.shape:
                raise Qwen38ContractError(
                    f"shape mismatch for {name}: {tuple(tensor.shape)} != {record.shape}"
                )
            expected_dtype = _torch_dtype_for_record(record)
            if tensor.dtype != expected_dtype:
                raise Qwen38ContractError(
                    f"dtype mismatch for {name}: {tensor.dtype} != {expected_dtype}"
                )
            actual_sha = _tensor_sha256(tensor)
            if actual_sha != record.sha256:
                raise Qwen38ContractError(
                    f"byte identity mismatch for {name}: {actual_sha} != {record.sha256}"
                )
            category_counts[record.category] = category_counts.get(record.category, 0) + 1
            digest = category_digests.setdefault(record.category, hashlib.sha256())
            digest.update(name.encode())
            digest.update(b"\0")
            digest.update(actual_sha.encode())
            digest.update(b"\n")
            if record.disposition == "required":
                assert record.runtime_name is not None
                # Physical fusion is limited to byte-preserving concatenation
                # recipes validated from the hash-bound artifact manifest.
                logical_weights[name] = tensor.detach().contiguous()
            elif record.disposition != "deferred":
                raise Qwen38ContractError(
                    f"invalid disposition {record.disposition!r} for {name}"
                )

        missing = set(records) - seen
        if missing:
            raise Qwen38ContractError(
                f"checkpoint iterator omitted {len(missing)} tensors; "
                f"first missing={sorted(missing)[:8]}"
            )
        if len(seen) != 1695 or len(logical_weights) != 1347:
            raise Qwen38ContractError(
                f"strict weight count failure: seen={len(seen)}, "
                f"logical={len(logical_weights)}"
            )
        trace_device = torch.device("cuda", torch.cuda.current_device())
        device_cache: dict[str, torch.Tensor] = {}
        runtime_maps, transforms = _materialize_artifact_constant_maps(
            self.family,
            logical_weights,
            trace_device,
            device_cache=device_cache,
        )
        for tokens, model in self.models.items():
            model.set_many_constants_with_tensors(runtime_maps[tokens])
        if self.scheduled_family is not None:
            scheduled_maps, scheduled_transforms = (
                _materialize_artifact_constant_maps(
                    self.family,
                    logical_weights,
                    trace_device,
                    artifacts=self.scheduled_family.artifacts,
                    device_cache=device_cache,
                )
            )
            for artifact_key, model in self.scheduled_models.items():
                model.set_many_constants_with_tensors(
                    scheduled_maps[artifact_key]
                )
            transforms.extend(
                {"artifact_family": "scheduled", **plan}
                for plan in scheduled_transforms
            )
            self._initialize_scheduled_runtime(trace_device)
        else:
            self._initialize_persistent_runtime(trace_device)
        self.runtime_trace.initialize(device=trace_device, identity=self.activation_proof)
        self._weights_loaded = True
        self.weight_load_result = {
            "checkpoint_tensor_count": len(seen),
            "required_text_loaded_once": len(logical_weights),
            "vision_deferred": category_counts.get("vision", 0),
            "mtp_deferred": category_counts.get("mtp", 0),
            "category_counts": dict(sorted(category_counts.items())),
            "category_named_digests": {
                name: digest.hexdigest()
                for name, digest in sorted(category_digests.items())
            },
            "transforms": transforms,
            "complete_checkpoint_bf16_dequantization": False,
            "passed": True,
        }
        return seen


__all__ = ["PaitonQwen38ForCausalLM"]
