"""Multimodal Qwen3.8 shell around the compiled Paiton text backbone."""

from collections.abc import Iterable

import torch
from torch import nn

from safetensors import safe_open
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.registry import cached_tokenizer_from_config

from paiton_vllm_plugin.models.paiton_qwen38 import PaitonQwen38ForCausalLM
from paiton_vllm_plugin.runtime.core.utils.qwen38_loader import (
    resolve_qwen38_safetensors,
)


_VISION_PREFIX = "model.visual."
_VISION_TENSOR_COUNT = 333
_VISION_PARAMETER_BYTES = 921_460_192
_DTYPE_BYTES = {"BF16": 2, "F32": 4, "I32": 4}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3_5ProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class PaitonQwen38ForConditionalGeneration(
    Qwen3VLForConditionalGeneration,
    IsHybrid,
):
    """Upstream BF16 vision/processor path with Paiton's Qwen3.8 LM."""

    supports_multimodal_pruning = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        if multimodal_config is None:
            raise ValueError("Paiton Qwen3.8 conditional generation requires multimodal config")
        if vllm_config.quant_config is not None:
            raise ValueError(
                "Paiton owns Qronos quantization; the vLLM vision shell must be unquantized"
            )

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)

        self.use_deepstack = hasattr(
            config.vision_config, "deepstack_visual_indexes"
        )
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = PaitonQwen38ForCausalLM(
                vllm_config=vllm_config,
                prefix="",
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return PaitonQwen38ForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        return PaitonQwen38ForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return PaitonQwen38ForCausalLM.get_mamba_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The incoming iterator would traverse the 19.9 GB checkpoint in file
        # order. Use the same random-access checkpoint as the strict text
        # loader so only one vision tensor or one Qronos triple is live at once.
        del weights
        model_config = self.language_model.vllm_config.model_config
        checkpoint = resolve_qwen38_safetensors(
            model_config.model,
            revision=model_config.revision,
            token=model_config.hf_token,
            download_dir=self.language_model.vllm_config.load_config.download_dir,
        )
        with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
            vision_names = tuple(
                name for name in source.keys() if name.startswith(_VISION_PREFIX)
            )
            vision_bytes = 0
            for name in vision_names:
                tensor_slice = source.get_slice(name)
                elements = 1
                for extent in tensor_slice.get_shape():
                    elements *= extent
                try:
                    vision_bytes += elements * _DTYPE_BYTES[tensor_slice.get_dtype()]
                except KeyError as error:
                    raise ValueError(
                        f"unsupported Qwen3.8 vision tensor dtype for {name}"
                    ) from error
            if len(vision_names) != _VISION_TENSOR_COUNT:
                raise ValueError(
                    f"Qwen3.8 vision tensor count mismatch: {len(vision_names)}"
                )
            if vision_bytes != _VISION_PARAMETER_BYTES:
                raise ValueError(
                    f"Qwen3.8 vision parameter bytes mismatch: {vision_bytes}"
                )
            loader = AutoWeightsLoader(self)
            loaded = loader.load_weights(
                ((name, source.get_tensor(name)) for name in vision_names),
                mapper=self.hf_to_vllm_mapper,
            )
        loaded.update(self.language_model.load_weights(()))
        return loaded
