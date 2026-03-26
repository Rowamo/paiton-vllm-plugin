from pathlib import Path
import torch
from torch import nn
from typing import Dict, Iterable, Set, Tuple, List, Optional

from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.forward_context import ForwardContext, get_forward_context

from paiton_vllm_plugin.models.artifact_resolver import resolve_artifact_dir
from paiton_vllm_plugin.models.model_path import resolve_model_so_path
from paiton_vllm_plugin.runtime.core import Model
from paiton_vllm_plugin.vllm_compat import Attention, AttentionType

class PaitonQwen3ForCausalLM(nn.Module):
    packed_modules_mapping: Dict[str, List[str]] = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed input IDs. For Paiton models, embeddings are handled internally
        by the compiled model, so this is a no-op that returns a dummy tensor
        to satisfy the VllmModelForTextGeneration protocol.
        """
        hidden_size = getattr(getattr(self, "config", None), "hidden_size", 4096)
        return torch.zeros(
            (*input_ids.shape, hidden_size),
            dtype=getattr(self, "dtype", torch.float32),
            device=input_ids.device,
        )

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        self.config = vllm_config.model_config.hf_config
        model_ref = vllm_config.model_config.model
        self.model_path = resolve_artifact_dir(
            model_ref,
            revision=vllm_config.model_config.revision,
            token=vllm_config.model_config.hf_token,
            download_dir=vllm_config.load_config.download_dir,
        )
        max_input_tokens = getattr(
            getattr(vllm_config, "scheduler_config", None),
            "max_num_batched_tokens",
            None,
        )
        decode_partition_size = getattr(
            self.config,
            "decode_partition_size",
            None,
        )
        model_so_path = resolve_model_so_path(
            self.model_path,
            Path(model_ref).name,
            self.tp_size,
            max_input_tokens=max_input_tokens,
            decode_partition_size=decode_partition_size,
        )
        self.model = Model(model_so_path)
        self.dtype = self.config.torch_dtype
        self.cache_dtype = self.dtype if vllm_config.cache_config.cache_dtype == "auto" else torch.float8_e4m3fnuz
        self.quantized = vllm_config.quant_config is not None
        self.static_quant = self.quantized and vllm_config.quant_config.activation_scheme == "static"
        self.dynamic_quant = self.quantized and vllm_config.quant_config.activation_scheme == "dynamic"
        self.unpadded_vocab_size = self.config.vocab_size

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            self.config.vocab_size,
            logit_scale,
            logits_as_input=True, # output of run_with_tensors is logits
        )

        # vLLM 0.8 changes some things, now we need to set up a
        # so-called static forward context to put kv-caches into.
        self.num_layers = self.config.num_hidden_layers
        self.compilation_config = vllm_config.compilation_config
        num_q_heads = self.config.num_attention_heads // self.tp_size
        num_kv_heads = max(1, self.config.num_key_value_heads // self.tp_size)
        head_size = self.config.head_dim
        scale = head_size ** -0.5

        self.compilation_config.static_forward_context = {
            str(i) : Attention(
                num_heads=num_q_heads,
                head_size=head_size,
                scale=scale,
                num_kv_heads=num_kv_heads,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                prefix=str(i),
                attn_type=AttentionType.DECODER,
            ) for i in range(self.num_layers)
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        output = torch.empty([input_ids.shape[0], self.config.vocab_size], dtype=torch.float32, device="cuda")
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not attn_metadata:
            return output

        attn_metadata = attn_metadata['0'] # It's the same for all layers, so just use the first one
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len
        inputs = {
            "input_ids": input_ids,
            "position_ids": positions,
            "slot_mapping": attn_metadata.slot_mapping,
            "query_start_locations": attn_metadata.query_start_loc,
            "context_lengths": attn_metadata.seq_lens,
            "block_tables": attn_metadata.block_table,
            "max_query_len": torch.empty([max_query_len, 0], dtype=torch.int32, device="cuda"),
            "max_seq_len": torch.empty([max_seq_len, 0], dtype=torch.int32, device="cuda")
        }
        for i in range(self.num_layers):
            idx = f"kv_cache_{i}"
            inputs[idx] = self.compilation_config.static_forward_context[str(i)].kv_cache[0].view(self.cache_dtype)

        outputs = {
            "logits": output,
        }
        model_output = self.model.run_with_tensors(inputs, outputs, sync=False)
        # self.model.profile_with_tensors(inputs, outputs, num_iters=10, filename=f"vllm_profile_{attn_metadata.max_decode_seq_len}.json")
        return model_output["logits"]

    def get_rank_weight(self, weight: torch.Tensor, dim: int) -> torch.Tensor:
        if weight.dim() == 0:
            weight = weight.reshape([1])
        local_weight = torch.split(weight, weight.shape[dim] // self.tp_size, dim)[self.tp_rank]
        return local_weight

    def get_rank_bias(self, bias: torch.Tensor) -> torch.Tensor:
        if self.tp_rank == 0:
            return bias
        else:
            return torch.zeros_like(bias)

    def map_pt_params(self, pt_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Check if module is fused, return the fused module name,
        # unfused module name, and full fused parameter name if so
        def find_fusion(name: str) -> Optional[Tuple[str, str, str]]:
            for fused_module, modules in self.packed_modules_mapping.items():
                for module in modules:
                    if module in name:
                        return fused_module, module, name.replace(module, fused_module)
            return None, None, None

        # In Paiton, we use '_' instead of '.':
        def convert_name(name: str) -> str:
            return name.replace("model.", "").replace(".", "_")

        # CK GEMM & MoE require pre-shuffling the weights for faster GEMM.
        def shuffle_weight(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
            IN, IK = layout
            BK = IK * 2
            K = 16 // x.element_size()
            BN = IN
            assert x.shape[-2] % BN == 0
            assert x.shape[-1] % BK == 0

            x_ = x
            x_ = x_.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
            x_ = x_.permute(0, 1, 3, 4, 2, 5)
            x_ = x_.contiguous()
            x_ = x_.view(*x.shape)
            return x_

        params_paiton: Dict[str, torch.Tensor] = {}

        for name, param in pt_params.items():
            fused_module, unfused_module, full_fused_name = find_fusion(name)
            if fused_module is not None:
                if convert_name(full_fused_name) in params_paiton:
                    continue

                modules_to_fuse = self.packed_modules_mapping[fused_module]
                params_names = [name.replace(unfused_module, module) for module in modules_to_fuse]
                params = [pt_params[param_name] for param_name in params_names]
                name = full_fused_name

                if name.endswith(".weight"):
                    # gate_up and qkv are column parallel, split across dim=0
                    params = [self.get_rank_weight(param, dim=0).cuda() for param in params]

                    if self.static_quant:
                        # Due to fusion of weights with different scales
                        # we have to de-quantize, apply the uniform scale and quantize again
                        # First find all the weight scales
                        scales_names = [name.replace(".weight", ".weight_scale") for name in params_names]
                        scales = [pt_params[name] for name in scales_names]
                        # Then de-quantize all the weights
                        dequant_weights = [w.to(torch.float32) * s for (w, s) in zip(params, scales)]
                        # Find the maximum scale
                        max_weight_scale = max([scale.item() for scale in scales])
                        # Re-quantize all the weights with the maximum scale
                        params = [(w / max_weight_scale).to(params[0].dtype) for w in dequant_weights]

                    # Fuse the weights
                    paiton_param = torch.cat(params, dim=0)
                    if self.dynamic_quant:
                        paiton_param = shuffle_weight(paiton_param)
                elif name.endswith(".bias"):
                    bias = torch.cat(params, dim=0)
                    paiton_param = self.get_rank_bias(bias)
                elif name.endswith(".input_scale"):
                    for param in params[1:]:
                        assert param == params[0], f"input_scale must be equal for all of {params_names}"
                    paiton_param = (params[0] * 2).float()
                elif name.endswith(".weight_scale"):
                    # We always use the maximum weight scale
                    max_weight_scale = max([scale.item() for scale in params])
                    paiton_param = torch.FloatTensor([max_weight_scale * 2])
                elif name.endswith(".weight_scale_inv"):
                    # gate_up and qkv are column parallel, split across dim=0
                    params = [self.get_rank_weight(param, dim=0).cuda() for param in params]
                    # Fuse the scales
                    paiton_param = (torch.cat(params, dim=0) * 2).float()
            elif name.endswith("down_proj.weight") or name.endswith("o_proj.weight"):
                paiton_param = self.get_rank_weight(param, dim=1) # row parallel, split across dim=1
                if self.dynamic_quant:
                    paiton_param = shuffle_weight(paiton_param)
            elif name.endswith(".bias"):
                paiton_param = self.get_rank_bias(param)
            elif name.endswith("norm.weight"):
                paiton_param = param # we do not split normalization weights across ranks
            elif name.endswith(".input_scale"):
                paiton_param = (param * 2).float()
            elif name.endswith(".weight_scale") or name.endswith(".weight_scale_inv"):
                paiton_param = (param * 2).float()
            elif name.endswith(".kv_scale"):
                # this is an "old" way of doing things now, so k and v scales must be separate
                k_scale_name = name.replace("kv_scale", "k_scale")
                v_scale_name = name.replace("kv_scale", "v_scale")
                params_paiton[convert_name(k_scale_name)] = (param * 2).cuda()
                params_paiton[convert_name(v_scale_name)] = (param * 2).cuda()
                continue
            else:
                paiton_param = self.get_rank_weight(param, dim=0)

            if paiton_param.dtype == torch.float8_e4m3fn:
                weight_as_int8 = paiton_param.view(torch.int8).cuda()
                weight_as_int8[weight_as_int8 == -128] = 0 # e4m3fn `-0` is `NaN` in e4m3fnuz, set it to `0`
                paiton_param = weight_as_int8.view(torch.float8_e4m3fnuz)

            params_paiton[convert_name(name)] = paiton_param.cuda()

        return params_paiton

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        if self.tp_rank == 0:
            processed_logits = self.logits_processor(None, hidden_states)
            return processed_logits
        return None

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # Avoid keeping the entire checkpoint alive on GPU via `dict(weights)`.
        # Materialize on CPU first, then upload only final constants to GPU.
        pt_params = {name: tensor.detach().cpu() for name, tensor in weights}
        self.model.set_many_constants_with_tensors(self.map_pt_params(pt_params))
        return set()
