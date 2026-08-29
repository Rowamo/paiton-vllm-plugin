"""vLLM wrapper for the product-facing Paiton Qwen3.8 text backbone."""

from collections.abc import Iterable
import json
from pathlib import Path
from typing import ClassVar, Literal

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid, SupportsMRoPE
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

from paiton_vllm_plugin.artifact_manifest import manifest_path_for
from paiton_vllm_plugin.models.artifact_resolver import resolve_artifact_dir
from paiton_vllm_plugin.models.model_path import resolve_model_so_path
from paiton_vllm_plugin.runtime.core import Model
from paiton_vllm_plugin.runtime.core.utils.qronos_loader import (
    QronosStreamingTransformer,
    qwen38_specs_from_manifest,
)
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import (
    Qwen38UnquantizedLoader,
    configure_qwen38_cache_contract,
    resolve_qwen38_safetensors,
)
from paiton_vllm_plugin.runtime.core.utils.qwen38_memory import (
    preflight_qwen38_memory,
)
from paiton_vllm_plugin.vllm_compat import Attention, AttentionType


class PaitonQwen38GDNCacheLayer(GatedDeltaNetAttention):
    """Expose only Qwen3.8 GDN cache/state semantics to vLLM."""

    def __init__(self, config, vllm_config: VllmConfig, prefix: str):
        super().__init__(config, vllm_config, prefix)
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

    def get_state_shape(self):
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
        )

    def forward(self, *args, **kwargs):
        raise RuntimeError("Paiton executes GDN inside the compiled backbone")


class PaitonQwen38ForCausalLM(nn.Module, HasInnerState, IsHybrid, SupportsMRoPE):
    """Qwen3.8 embedding/logits shell around a compiled hybrid text backbone."""

    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        if prefix:
            raise ValueError("Paiton Qwen3.8 does not support pipeline prefixes")
        if get_tensor_model_parallel_world_size() != 1:
            raise ValueError("Paiton Qwen3.8 contract v3 requires TP=1")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Paiton Qwen3.8 contract v3 requires PP=1")
        if vllm_config.speculative_config is not None:
            raise ValueError("Paiton Qwen3.8 contract v3 does not support speculative decode")
        configure_qwen38_cache_contract(
            vllm_config.cache_config, resolve_auto=False
        )

        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_text_config
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.dtype = vllm_config.model_config.dtype
        if self.dtype is not torch.bfloat16:
            raise ValueError("Paiton Qwen3.8 contract v3 requires BF16 model dtype")

        model_ref = vllm_config.model_config.model
        self.model_path = resolve_artifact_dir(
            model_ref,
            revision=vllm_config.model_config.revision,
            token=vllm_config.model_config.hf_token,
            download_dir=vllm_config.load_config.download_dir,
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.model_so_path = resolve_model_so_path(
            self.model_path,
            Path(model_ref).name,
            self.tp_size,
            max_input_tokens=max_tokens,
            decode_partition_size=getattr(self.config, "decode_partition_size", None),
        )
        with manifest_path_for(self.model_so_path).open(encoding="utf-8") as source:
            self.manifest = json.load(source)
        self.qronos_specs = qwen38_specs_from_manifest(self.manifest)
        self.contract = self.manifest["paiton_qwen38_contract"]
        self.memory_estimate = preflight_qwen38_memory(
            self.manifest,
            hybrid_cache_reservation_bytes=(
                vllm_config.cache_config.kv_cache_memory_bytes
            ),
        )
        self.num_layers = int(self.contract["num_hidden_layers"])
        if self.num_layers != self.config.num_hidden_layers:
            raise ValueError("Qwen3.8 artifact/config layer-count mismatch")
        if max_tokens > int(self.contract["max_num_batched_tokens"]):
            raise ValueError("vLLM token budget exceeds the Qwen3.8 artifact contract")
        if vllm_config.model_config.max_model_len > int(
            self.contract["max_context_length"]
        ):
            raise ValueError("vLLM max model length exceeds the Qwen3.8 artifact contract")
        if vllm_config.cache_config.block_size != int(
            self.contract["kv_cache_block_size"]
        ):
            raise ValueError("vLLM block size does not match the Qwen3.8 artifact")

        self.compiled_model = Model(str(self.model_so_path))
        self.compiled_input_names = set(
            self.compiled_model.get_input_name_to_index_map()
        )
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            prefix="model.embed_tokens",
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            params_dtype=torch.bfloat16,
            prefix="lm_head",
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

        self.layer_types = tuple(self.config.layer_types[: self.num_layers])
        static_context = {}
        self.cache_layers = nn.ModuleDict()
        num_q_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        for index, layer_type in enumerate(self.layer_types):
            if layer_type == "linear_attention":
                key = f"model.layers.{index}.linear_attn"
                layer = PaitonQwen38GDNCacheLayer(
                    self.config, vllm_config, prefix=key
                )
            elif layer_type == "full_attention":
                key = f"model.layers.{index}.self_attn"
                layer = Attention(
                    num_heads=num_q_heads,
                    head_size=self.config.head_dim,
                    scale=self.config.head_dim**-0.5,
                    num_kv_heads=num_kv_heads,
                    cache_config=vllm_config.cache_config,
                    quant_config=None,
                    prefix=key,
                    attn_type=AttentionType.DECODER,
                )
            else:
                raise ValueError(f"unsupported Qwen3.8 layer type {layer_type!r}")
            self.cache_layers[str(index)] = layer
            static_context[key] = layer
        vllm_config.compilation_config.static_forward_context = static_context
        self._dummy_inputs = {}

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def get_mrope_input_positions(
        self, input_tokens: list[int], mm_features: list[object]
    ) -> tuple[torch.Tensor, int]:
        if mm_features:
            raise ValueError("Paiton Qwen3.8 contract v3 is text-only")
        positions = torch.arange(len(input_tokens), dtype=torch.long)
        return positions.unsqueeze(0).expand(3, -1), 0

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: VllmConfig
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def _dummy(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        key = (dtype, device)
        value = self._dummy_inputs.get(key)
        if value is None:
            value = torch.zeros(1, dtype=dtype, device=device)
            self._dummy_inputs[key] = value
        return value

    def _metadata(self, index: int):
        layer_type = self.layer_types[index]
        suffix = "linear_attn" if layer_type == "linear_attention" else "self_attn"
        key = f"model.layers.{index}.{suffix}"
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict) or key not in metadata:
            raise RuntimeError(f"vLLM did not provide Qwen3.8 metadata for {key}")
        return metadata[key]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("Paiton Qwen3.8 contract v3 requires PP=1")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if positions.ndim == 2:
            positions = positions[0]
        if positions.ndim != 1:
            raise ValueError("Qwen3.8 text positions must be one-dimensional or 3-axis")

        # vLLM deliberately omits attention metadata during its eager memory
        # profile because KV caches do not exist yet. The compiled backbone
        # cannot execute without those cache bindings; its persistent runtime
        # and transformed constants have already been allocated during model
        # loading, so forwarding the correctly shaped embeddings lets vLLM
        # profile the runtime-owned logits/sampler path without inventing cache
        # addresses. A real scheduled forward always carries per-layer metadata.
        if get_forward_context().attn_metadata is None:
            return inputs_embeds

        full_index = next(
            index for index, kind in enumerate(self.layer_types) if kind == "full_attention"
        )
        gdn_index = next(
            index for index, kind in enumerate(self.layer_types) if kind == "linear_attention"
        )
        full_meta = self._metadata(full_index)
        gdn_meta = self._metadata(gdn_index)
        if getattr(gdn_meta, "num_spec_decodes", 0):
            raise ValueError("Paiton Qwen3.8 does not support speculative GDN metadata")
        query_starts = gdn_meta.non_spec_query_start_loc
        state_indices = gdn_meta.non_spec_state_indices_tensor
        if query_starts is None or state_indices is None:
            raise RuntimeError("vLLM did not provide non-speculative GDN state metadata")
        has_initial = gdn_meta.has_initial_state
        if has_initial is None:
            has_initial = torch.ones_like(state_indices, dtype=torch.int32)
        else:
            has_initial = has_initial.to(dtype=torch.int32)

        device = inputs_embeds.device
        inputs = {
            "inputs_embeds": inputs_embeds.contiguous(),
            "position_ids": positions.to(dtype=torch.int64).contiguous(),
            "slot_mapping": full_meta.slot_mapping.to(dtype=torch.int64).contiguous(),
            "query_start_locations": query_starts.to(dtype=torch.int32).contiguous(),
            "context_lengths": full_meta.seq_lens.to(dtype=torch.int32).contiguous(),
            "block_tables": full_meta.block_table.to(dtype=torch.int32).contiguous(),
            "max_query_len": torch.empty(
                (int(full_meta.max_query_len), 0), dtype=torch.int32, device=device
            ),
            "max_seq_len": torch.empty(
                (int(full_meta.max_seq_len), 0), dtype=torch.int32, device=device
            ),
        }
        conv_stride = recurrent_stride = None
        for index, layer_type in enumerate(self.layer_types):
            layer = self.cache_layers[str(index)]
            if layer_type == "linear_attention":
                conv_state, recurrent_state = layer.kv_cache
                conv_state = conv_state.view(conv_state.shape[0], -1)
                inputs[f"kv_cache_dummy_{index}"] = self._dummy(torch.bfloat16, device)
                inputs[f"conv_state_{index}"] = conv_state
                inputs[f"recurrent_state_{index}"] = recurrent_state
                inputs[f"state_indices_{index}"] = state_indices
                inputs[f"has_initial_state_{index}"] = has_initial
                conv_stride = conv_state.stride(0)
                recurrent_stride = recurrent_state.stride(0)
            else:
                kv_cache = layer.kv_cache
                if not isinstance(kv_cache, torch.Tensor) or kv_cache.ndim != 5:
                    raise RuntimeError(
                        "pinned vLLM did not bind the expected five-dimensional "
                        f"Qwen3.8 KV cache for layer {index}"
                    )
                expected_tail = (
                    2,
                    int(self.contract["kv_cache_block_size"]),
                    int(self.config.num_key_value_heads),
                    int(self.config.head_dim),
                )
                if tuple(kv_cache.shape[1:]) != expected_tail:
                    raise RuntimeError(
                        "pinned vLLM bound an incompatible Qwen3.8 page-first "
                        f"KV cache for layer {index}: got {tuple(kv_cache.shape)}, "
                        f"expected [blocks,{','.join(map(str, expected_tail))}]"
                    )
                inputs[f"kv_cache_{index}"] = kv_cache.view(torch.bfloat16)
                inputs[f"conv_state_dummy_{index}"] = self._dummy(torch.bfloat16, device)
                inputs[f"recurrent_state_dummy_{index}"] = self._dummy(torch.float32, device)
                inputs[f"state_indices_dummy_{index}"] = self._dummy(torch.int32, device)
                inputs[f"has_initial_state_dummy_{index}"] = self._dummy(torch.int32, device)
        if conv_stride is None or recurrent_stride is None:
            raise RuntimeError("Qwen3.8 artifact has no GDN state layers")
        inputs["conv_state_line_stride"] = torch.tensor(
            [conv_stride], dtype=torch.int64, device=device
        )
        inputs["recurrent_state_line_stride"] = torch.tensor(
            [recurrent_stride], dtype=torch.int64, device=device
        )
        output = torch.empty(
            (inputs_embeds.shape[0], self.config.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        missing = sorted(self.compiled_input_names - set(inputs))
        if missing:
            raise RuntimeError(
                "Qwen3.8 wrapper did not provide compiled inputs: "
                + ", ".join(missing)
            )
        exact_inputs = {
            name: value for name, value in inputs.items()
            if name in self.compiled_input_names
        }
        strided_state_names = frozenset(
            f"{state}_{index}"
            for index, layer_type in enumerate(self.layer_types)
            if layer_type == "linear_attention"
            for state in ("conv_state", "recurrent_state")
        )
        return self.compiled_model.run_with_tensors(
            exact_inputs,
            {"hidden_states": output},
            sync=False,
            noncontiguous_input_names=strided_state_names,
        )["hidden_states"]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        model_config = self.vllm_config.model_config
        checkpoint = resolve_qwen38_safetensors(
            model_config.model,
            revision=model_config.revision,
            token=model_config.hf_token,
            download_dir=self.vllm_config.load_config.download_dir,
        )
        from safetensors import safe_open

        loaded = set()
        with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
            embedding_name = "model.language_model.embed_tokens.weight"
            lm_head_name = "lm_head.weight"
            available = set(source.keys())
            for name, module in (
                (embedding_name, self.embed_tokens),
                (lm_head_name, self.lm_head),
            ):
                if name not in available:
                    raise ValueError(f"Qwen3.8 checkpoint is missing {name}")
                module.weight_loader(module.weight, source.get_tensor(name))
                loaded.add(name)

            unquantized = Qwen38UnquantizedLoader(self.manifest)
            for name, tensor in unquantized.iter_from_random_access_source(source):
                self.compiled_model.set_constant_with_tensor(
                    name, tensor.to(device="cuda").contiguous()
                )
                loaded.add(name)

            source_layers = int(self.contract["source_num_hidden_layers"])
            compiled_layers = int(self.contract["num_hidden_layers"])
            qronos = QronosStreamingTransformer(
                self.qronos_specs,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                max_pending_linears=1,
                allowed_extra_layer_range=(compiled_layers, source_layers),
            )
            for transformed in qronos.iter_from_random_access_source(source):
                for name, tensor in transformed.constants():
                    self.compiled_model.set_constant_with_tensor(
                        name, tensor.to(device="cuda").contiguous()
                    )
                    loaded.add(name)
            qronos.finish()
        return loaded
