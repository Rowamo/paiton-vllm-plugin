# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned state bridge for the admitted Qwen3.8 serving envelope."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from paiton_vllm_plugin.models.qwen38_contract import Qwen38ContractError
from paiton_vllm_plugin.paiton_attention_backend import (
    PaitonTritonAttentionBackend,
)
from paiton_vllm_plugin.vllm_compat import Attention, AttentionType


GDN_CONV_STORAGE_SHAPE = (3, 10240)
GDN_CONV_ARTIFACT_SHAPE = (10240, 3)
GDN_RECURRENT_SHAPE = (48, 128, 128)
KV_ARTIFACT_SHAPE = (2, 256, 16, 4, 256)
KV_BLOCK_SIZE = 16
MAX_SEQUENCE = 4096
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
LINEAR_ATTENTION_LAYERS = tuple(
    layer for layer in range(64) if layer not in FULL_ATTENTION_LAYERS
)


class PaitonQwen38GDNStateLayer(MambaBase):
    """A state-only vLLM layer; Paiton executes the corresponding math."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache: tuple[torch.Tensor, torch.Tensor] = (
            torch.tensor([]),
            torch.tensor([]),
        )
        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise Qwen38ContractError(f"duplicate scheduler state layer: {prefix}")
        context[prefix] = self

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return torch.bfloat16, torch.float32

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        # vLLM stores convolution state contiguous along feature dimension.
        # The bridge transposes a request-owned block into the artifact ABI.
        return GDN_CONV_STORAGE_SHAPE, GDN_RECURRENT_SHAPE


def make_attention_state_layer(vllm_config: VllmConfig, prefix: str) -> Attention:
    config = vllm_config.model_config.hf_text_config
    return Attention(
        num_heads=config.num_attention_heads,
        head_size=config.head_dim,
        scale=config.head_dim**-0.5,
        num_kv_heads=config.num_key_value_heads,
        cache_config=vllm_config.cache_config,
        quant_config=None,
        prefix=prefix,
        attn_type=AttentionType.DECODER,
        attn_backend=PaitonTritonAttentionBackend,
    )


@dataclass
class PreparedQwen38State:
    inputs: dict[str, torch.Tensor]
    state_indices: dict[int, torch.Tensor]
    block_ids: dict[int, torch.Tensor]
    physical_block_count: int
    physical_block_size: int
    max_seq_len: int
    fresh_prefill: bool
    persistent: bool = False


class Qwen38SchedulerStateBridge(nn.Module):
    """Map scheduler-owned state pages to and from the fixed artifact ABI.

    The compatibility path maps scheduler pages to the artifact ABI.  The
    serving path instead owns one persistent batch-1 arena whose addresses are
    bound to the generated runtimes once.  Scheduler metadata is still fully
    validated, but state does not make a redundant round trip through vLLM's
    pages on every token.
    """

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__()
        self.gdn_layers: dict[int, PaitonQwen38GDNStateLayer] = {}
        self.attention_layers: dict[int, Attention] = {}
        for layer in range(64):
            prefix = f"model.language_model.layers.{layer}"
            if layer in LINEAR_ATTENTION_LAYERS:
                self.gdn_layers[layer] = PaitonQwen38GDNStateLayer(
                    vllm_config, f"{prefix}.linear_attn"
                )
            else:
                self.attention_layers[layer] = make_attention_state_layer(
                    vllm_config, f"{prefix}.self_attn"
                )
        self._active = False
        self._decode_steps = 0
        self._sequence_length = 0
        self._persistent_inputs: dict[str, torch.Tensor] | None = None
        self._persistent_reset_groups: tuple[tuple[torch.Tensor, ...], ...] = ()
        self._persistent_dirty = False

    def initialize_persistent_arena(
        self, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Allocate the single admitted request slot exactly once."""

        if self._persistent_inputs is not None:
            raise Qwen38ContractError("persistent state arena may be initialized once")
        inputs: dict[str, torch.Tensor] = {}
        for layer in self.gdn_layers:
            inputs[f"gdn_conv_state_{layer}"] = torch.zeros(
                GDN_CONV_ARTIFACT_SHAPE, dtype=torch.bfloat16, device=device
            )
            inputs[f"gdn_recurrent_state_{layer}"] = torch.zeros(
                GDN_RECURRENT_SHAPE, dtype=torch.float32, device=device
            )
        for layer in self.attention_layers:
            inputs[f"kv_cache_{layer}"] = torch.zeros(
                KV_ARTIFACT_SHAPE, dtype=torch.bfloat16, device=device
            )
        self._persistent_inputs = inputs
        reset_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for tensor in inputs.values():
            reset_by_dtype.setdefault(tensor.dtype, []).append(tensor)
        self._persistent_reset_groups = tuple(
            tuple(group) for group in reset_by_dtype.values()
        )
        self._persistent_dirty = False
        return inputs

    @property
    def persistent_ready(self) -> bool:
        return self._persistent_inputs is not None

    def reset_persistent_arena(self, *, force: bool = False) -> None:
        if self._persistent_inputs is None:
            raise Qwen38ContractError("persistent state arena is not initialized")
        if self._persistent_dirty or force:
            for group in self._persistent_reset_groups:
                torch._foreach_zero_(group)
            self._persistent_dirty = False

    @staticmethod
    def mamba_state_shape() -> tuple[tuple[int, ...], tuple[int, ...]]:
        return GDN_CONV_STORAGE_SHAPE, GDN_RECURRENT_SHAPE

    @staticmethod
    def mamba_state_dtype() -> tuple[torch.dtype, torch.dtype]:
        return torch.bfloat16, torch.float32

    @staticmethod
    def mamba_state_copy_func() -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def _phase_and_indices(
        self,
        metadata: Mapping[str, Any],
        tokens: int,
        *,
        materialize_indices: bool = True,
    ) -> tuple[bool, dict[int, torch.Tensor]]:
        first_key = f"model.language_model.layers.{LINEAR_ATTENTION_LAYERS[0]}.linear_attn"
        gdn_metadata = metadata.get(first_key)
        if not isinstance(gdn_metadata, GDNAttentionMetadata):
            raise Qwen38ContractError(
                "missing stock GDN scheduler metadata for the Paiton state bridge"
            )
        first_attention = next(iter(self.attention_layers))
        first_attention_key = (
            f"model.language_model.layers.{first_attention}.self_attn"
        )
        first_attention_metadata = metadata.get(first_attention_key)
        single_token_fresh = (
            tokens == 1
            and gdn_metadata.num_prefills == 0
            and gdn_metadata.num_decodes == 1
            and getattr(first_attention_metadata, "max_seq_len", None) == 1
        )
        if gdn_metadata.num_prefills == 1 and gdn_metadata.num_decodes == 0:
            fresh = True
        elif (
            gdn_metadata.num_prefills == 0
            and gdn_metadata.num_decodes == 1
            and tokens == 1
        ):
            if single_token_fresh:
                fresh = True
            elif not self._active:
                raise Qwen38ContractError("decode has no active scheduler-owned request")
            else:
                fresh = False
        else:
            raise Qwen38ContractError(
                "only one fresh prefill or one non-speculative decode is admitted; "
                f"got prefills={gdn_metadata.num_prefills}, "
                f"decodes={gdn_metadata.num_decodes}, M={tokens}"
            )

        expected_phase = (
            gdn_metadata.num_prefills,
            gdn_metadata.num_prefill_tokens,
            gdn_metadata.num_decodes,
            gdn_metadata.num_decode_tokens,
            gdn_metadata.num_spec_decodes,
            gdn_metadata.num_spec_decode_tokens,
            gdn_metadata.num_actual_tokens,
        )
        state_indices: dict[int, torch.Tensor] = {}
        for layer in self.gdn_layers:
            key = f"model.language_model.layers.{layer}.linear_attn"
            layer_metadata = metadata.get(key)
            if not isinstance(layer_metadata, GDNAttentionMetadata):
                raise Qwen38ContractError(
                    f"missing stock GDN scheduler metadata for layer {layer}"
                )
            actual_phase = (
                layer_metadata.num_prefills,
                layer_metadata.num_prefill_tokens,
                layer_metadata.num_decodes,
                layer_metadata.num_decode_tokens,
                layer_metadata.num_spec_decodes,
                layer_metadata.num_spec_decode_tokens,
                layer_metadata.num_actual_tokens,
            )
            if actual_phase != expected_phase:
                raise Qwen38ContractError(
                    f"GDN scheduler phase differs for layer {layer}: "
                    f"{actual_phase} != {expected_phase}"
                )
            if layer_metadata.num_actual_tokens != tokens:
                raise Qwen38ContractError(
                    f"scheduler token count {layer_metadata.num_actual_tokens} "
                    f"for GDN layer {layer} != input M={tokens}"
                )
            if layer_metadata.spec_sequence_masks is not None:
                raise Qwen38ContractError(
                    f"MTP/speculative state metadata is not admitted for GDN layer {layer}"
                )
            state_index = layer_metadata.non_spec_state_indices_tensor
            if state_index is None or tuple(state_index.shape) != (1,):
                raise Qwen38ContractError(
                    f"scheduler must provide exactly one request-owned GDN state "
                    f"index for layer {layer}"
                )
            if materialize_indices:
                state_indices[layer] = state_index.to(dtype=torch.int64).contiguous()
        return fresh, state_indices

    def prepare(
        self,
        metadata: Mapping[str, Any],
        tokens: int,
        device: torch.device,
        trace: Any | None = None,
    ) -> PreparedQwen38State:
        persistent = self._persistent_inputs is not None
        fresh, state_indices = self._phase_and_indices(
            metadata, tokens, materialize_indices=not persistent
        )
        if trace is not None:
            trace.mark("state_phase_ready", fresh=fresh)
        inputs: dict[str, torch.Tensor] = {}
        for layer, holder in self.gdn_layers.items():
            if len(holder.kv_cache) != 2:
                raise Qwen38ContractError(f"GDN layer {layer} has no bound state")
            conv_storage, recurrent_storage = holder.kv_cache
            if persistent:
                if (
                    not isinstance(conv_storage, torch.Tensor)
                    or not isinstance(recurrent_storage, torch.Tensor)
                    or tuple(conv_storage.shape[1:]) != GDN_CONV_STORAGE_SHAPE
                    or tuple(recurrent_storage.shape[1:]) != GDN_RECURRENT_SHAPE
                ):
                    raise Qwen38ContractError(
                        f"GDN layer {layer} has unsupported scheduler state storage"
                    )
                continue
            if fresh:
                conv = torch.zeros(
                    GDN_CONV_ARTIFACT_SHAPE,
                    dtype=torch.bfloat16,
                    device=device,
                )
                recurrent = torch.zeros(
                    GDN_RECURRENT_SHAPE,
                    dtype=torch.float32,
                    device=device,
                )
            else:
                state_index = state_indices[layer]
                conv = (
                    torch.index_select(conv_storage, 0, state_index)
                    .squeeze(0)
                    .transpose(0, 1)
                    .contiguous()
                )
                recurrent = (
                    torch.index_select(recurrent_storage, 0, state_index)
                    .squeeze(0)
                    .contiguous()
                )
            inputs[f"gdn_conv_state_{layer}"] = conv
            inputs[f"gdn_recurrent_state_{layer}"] = recurrent
        if trace is not None:
            trace.mark("gdn_adapter_ready")

        first_attention = next(iter(self.attention_layers))
        first_physical = self.attention_layers[first_attention].kv_cache
        if not isinstance(first_physical, torch.Tensor) or first_physical.ndim != 5:
            raise Qwen38ContractError("scheduler has not bound the first paged KV cache")
        physical_block_size = int(first_physical.shape[2])
        if physical_block_size <= 0 or physical_block_size % KV_BLOCK_SIZE:
            raise Qwen38ContractError(
                f"scheduler KV page size {physical_block_size} is not a positive "
                f"multiple of the artifact block size {KV_BLOCK_SIZE}"
            )
        max_seq_len: int | None = None
        block_ids: dict[int, torch.Tensor] = {}
        physical_block_count: int | None = None
        for layer in self.attention_layers:
            key = f"model.language_model.layers.{layer}.self_attn"
            attention_metadata = metadata.get(key)
            if attention_metadata is None:
                raise Qwen38ContractError(
                    f"missing scheduler full-attention metadata for layer {layer}"
                )
            if int(getattr(attention_metadata, "num_actual_tokens", -1)) != tokens:
                raise Qwen38ContractError(
                    f"full-attention scheduler token count differs for layer {layer}"
                )
            block_table = getattr(attention_metadata, "block_table", None)
            layer_max_seq_len = getattr(attention_metadata, "max_seq_len", None)
            if not isinstance(block_table, torch.Tensor) or block_table.ndim != 2:
                raise Qwen38ContractError(
                    f"scheduler block table is unavailable for attention layer {layer}"
                )
            if block_table.shape[0] != 1 or not isinstance(layer_max_seq_len, int):
                raise Qwen38ContractError(
                    f"only a batch-1 scheduler block table is admitted for layer {layer}"
                )
            if layer_max_seq_len <= 0 or layer_max_seq_len > MAX_SEQUENCE:
                raise Qwen38ContractError(
                    f"scheduler max sequence {layer_max_seq_len} for attention layer "
                    f"{layer} is outside [1, {MAX_SEQUENCE}]"
                )
            if max_seq_len is None:
                max_seq_len = layer_max_seq_len
                physical_block_count = ceil(max_seq_len / physical_block_size)
            elif layer_max_seq_len != max_seq_len:
                raise Qwen38ContractError(
                    f"scheduler max sequence differs for attention layer {layer}: "
                    f"{layer_max_seq_len} != {max_seq_len}"
                )
            assert physical_block_count is not None
            if block_table.shape[1] < physical_block_count:
                raise Qwen38ContractError(
                    f"scheduler block table is shorter than the context for "
                    f"attention layer {layer}"
                )
            if not persistent:
                block_ids[layer] = block_table[0, :physical_block_count].to(
                    dtype=torch.int64
                ).contiguous()
        assert max_seq_len is not None
        assert physical_block_count is not None
        prior_token_count = 0 if fresh else max_seq_len - tokens
        if prior_token_count < 0:
            raise Qwen38ContractError("scheduler context is shorter than the query")
        if fresh and max_seq_len != tokens:
            raise Qwen38ContractError(
                f"fresh request context {max_seq_len} != query M={tokens}"
            )
        if not fresh and max_seq_len != self._sequence_length + tokens:
            raise Qwen38ContractError(
                f"decode context {max_seq_len} does not continue persistent "
                f"sequence length {self._sequence_length} by M={tokens}"
            )
        if trace is not None:
            trace.mark(
                "kv_metadata_ready",
                max_seq_len=max_seq_len,
                physical_block_count=physical_block_count,
                prior_token_count=prior_token_count,
            )

        for layer, holder in self.attention_layers.items():
            physical = holder.kv_cache
            if not isinstance(physical, torch.Tensor) or physical.ndim != 5:
                raise Qwen38ContractError(
                    f"full-attention layer {layer} has no bound paged KV cache"
                )
            if (
                physical.shape[0] != 2
                or physical.shape[2] != physical_block_size
                or tuple(physical.shape[3:]) != (4, 256)
            ):
                raise Qwen38ContractError(
                    f"full-attention layer {layer} has unsupported KV shape "
                    f"{tuple(physical.shape)}"
                )
            if persistent:
                continue
            logical = torch.zeros(
                KV_ARTIFACT_SHAPE,
                dtype=torch.bfloat16,
                device=device,
            )
            if prior_token_count:
                physical_pages = torch.index_select(physical, 1, block_ids[layer])
                logical.reshape(2, MAX_SEQUENCE, 4, 256)[:, :prior_token_count].copy_(
                    physical_pages.reshape(2, -1, 4, 256)[:, :prior_token_count]
                )
            inputs[f"kv_cache_{layer}"] = logical
        if persistent and fresh:
            self.reset_persistent_arena()
        if trace is not None:
            trace.mark("kv_adapter_ready")

        return PreparedQwen38State(
            inputs=self._persistent_inputs if persistent else inputs,
            state_indices=state_indices,
            block_ids=block_ids,
            physical_block_count=physical_block_count,
            physical_block_size=physical_block_size,
            max_seq_len=max_seq_len,
            fresh_prefill=fresh,
            persistent=persistent,
        )

    def commit(self, prepared: PreparedQwen38State) -> None:
        if prepared.persistent:
            if prepared.fresh_prefill:
                self._active = True
                self._decode_steps = 0
            else:
                self._decode_steps += 1
            self._sequence_length = prepared.max_seq_len
            self._persistent_dirty = True
            return
        for layer, holder in self.gdn_layers.items():
            conv = prepared.inputs[f"gdn_conv_state_{layer}"]
            recurrent = prepared.inputs[f"gdn_recurrent_state_{layer}"]
            holder.kv_cache[0].index_copy_(
                0, prepared.state_indices[layer], conv.transpose(0, 1).unsqueeze(0)
            )
            holder.kv_cache[1].index_copy_(
                0, prepared.state_indices[layer], recurrent.unsqueeze(0)
            )
        for layer, holder in self.attention_layers.items():
            logical = prepared.inputs[f"kv_cache_{layer}"]
            physical_pages = torch.index_select(
                holder.kv_cache, 1, prepared.block_ids[layer]
            ).contiguous()
            physical_pages.reshape(2, -1, 4, 256)[:, : prepared.max_seq_len].copy_(
                logical.reshape(2, MAX_SEQUENCE, 4, 256)[:, : prepared.max_seq_len]
            )
            holder.kv_cache.index_copy_(
                1,
                prepared.block_ids[layer],
                physical_pages,
            )
        if prepared.fresh_prefill:
            self._active = True
            self._decode_steps = 0
        else:
            self._decode_steps += 1
        self._sequence_length = prepared.max_seq_len


__all__ = [
    "FULL_ATTENTION_LAYERS",
    "GDN_CONV_ARTIFACT_SHAPE",
    "GDN_CONV_STORAGE_SHAPE",
    "GDN_RECURRENT_SHAPE",
    "KV_ARTIFACT_SHAPE",
    "LINEAR_ATTENTION_LAYERS",
    "PaitonQwen38GDNStateLayer",
    "PreparedQwen38State",
    "Qwen38SchedulerStateBridge",
]
