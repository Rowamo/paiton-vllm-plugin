"""Checkpoint mapping for Paiton DeepSeek V4 artifacts."""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from torch import Tensor
from vllm.distributed.parallel_state import get_ep_group
from vllm.platforms import current_platform

from paiton_vllm_plugin.runtime.core import runtime_uses_fnuz_fp8


class DeepseekV4WeightsMixin:
    def map_pt_params(
        self,
        pt_params: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        def convert_name(name: str) -> str:
            return name.replace("model.", "").replace(".", "_")

        def fix_fp8(w: torch.Tensor) -> torch.Tensor:
            if runtime_uses_fnuz_fp8() and w.dtype == torch.float8_e4m3fn:
                w_int8 = w.view(torch.int8).cuda()
                w_int8[w_int8 == -128] = 0
                return w_int8.view(torch.float8_e4m3fnuz)
            return w.cuda()

        def shuffle_weight(weight: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
            """Pre-shuffle dynamic-FP8 weights for CK blockscale GEMMs."""
            in_rows, in_cols = layout
            block_cols = in_cols * 2
            elems_per_16b = 16 // weight.element_size()
            assert weight.shape[-2] % in_rows == 0
            assert weight.shape[-1] % block_cols == 0

            weight_view = weight.view(
                -1,
                weight.shape[-2] // in_rows,
                in_rows,
                weight.shape[-1] // block_cols,
                block_cols // elems_per_16b,
                elems_per_16b,
            )
            weight_view = weight_view.permute(0, 1, 3, 4, 2, 5).contiguous()
            return weight_view.view(*weight.shape)

        def scale_to_float(scale: torch.Tensor) -> torch.Tensor:
            return scale.float()

        def scale_to_uint8(scale: torch.Tensor) -> torch.Tensor:
            if scale.dtype == torch.uint8:
                return scale
            if scale.element_size() == 1:
                return scale.view(torch.uint8)
            return scale.to(torch.uint8)

        def packed_to_uint8(weight: torch.Tensor) -> torch.Tensor:
            if weight.dtype == torch.uint8:
                return weight
            if weight.element_size() == 1:
                return weight.view(torch.uint8)
            return weight.to(torch.uint8)

        def fuse_wkv_wgate(name: str, param: torch.Tensor) -> tuple[str, torch.Tensor]:
            wgate_name = name.replace(".wkv.", ".wgate.")
            out_name = convert_name(name.replace(".wkv.", ".fused_wkv_wgate."))
            value = torch.cat([param, pt_params[wgate_name]], dim=0)
            return out_name, value

        def fuse_wkv_wgate_scale(name: str,
                                 param: torch.Tensor) -> tuple[str, torch.Tensor]:
            wgate_name = name.replace(".wkv.", ".wgate.")
            out_name = convert_name(
                name.replace(".wkv.scale", ".fused_wkv_wgate.weight_scale_inv"))
            value = torch.cat(
                [scale_to_float(param),
                 scale_to_float(pt_params[wgate_name])],
                dim=0,
            ) * fp8_scale_factor
            return out_name, value

        fp8_scale_factor = 2.0 if current_platform.is_fp8_fnuz() else 1.0
        params_paiton: Dict[str, torch.Tensor] = {}

        def add_expected_constant(name: str, value: torch.Tensor) -> None:
            if expected_constant_names is None or name in expected_constant_names:
                params_paiton[name] = value.cuda()

        try:
            ep_group = get_ep_group()
            ep_rank = ep_group.rank_in_group
            ep_size = ep_group.world_size
        except Exception:
            ep_rank = 0
            ep_size = 1

        num_experts = getattr(self.config, "n_routed_experts",
                              getattr(self.config, "num_experts", 0))
        if num_experts and num_experts % ep_size != 0:
            raise ValueError(
                f"EP world_size must divide n_routed_experts "
                f"(ep_size={ep_size}, n_routed_experts={num_experts})")
        num_local_experts = num_experts // ep_size if num_experts else 0
        placement = getattr(self.parallel_config, "expert_placement_strategy",
                            "linear")
        if placement == "round_robin":
            local_expert_ids = list(range(ep_rank, num_experts, ep_size))
        else:
            start = ep_rank * num_local_experts
            local_expert_ids = list(range(start, start + num_local_experts))

        expert_regex = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(.+)")
        layers_experts = [
            [{} for _ in range(num_experts)]
            for _ in range(getattr(self.config, "num_hidden_layers", 0))
        ]

        for name, param in pt_params.items():
            expert_match = expert_regex.match(name)
            if expert_match:
                layer_id = int(expert_match[1])
                expert_id = int(expert_match[2])
                layers_experts[layer_id][expert_id][expert_match[3]] = param
                continue

            if name == "embed.weight":
                out_name = "embed_tokens_weight"
                value = self.get_rank_weight(param, dim=0)
            elif name == "head.weight":
                out_name = "lm_head_weight"
                value = self.get_rank_weight(param, dim=0)
            elif name == "norm.weight" or name.startswith("hc_head"):
                out_name = convert_name(name)
                value = param
            elif ".attn.wq_a.weight" in name:
                wq_a = param
                wkv = pt_params[name.replace(".wq_a.weight", ".wkv.weight")]
                out_name = convert_name(name.replace(
                    ".wq_a.weight", ".fused_wqa_wkv.weight"))
                value = self.get_rank_weight(torch.cat([wq_a, wkv], dim=0), dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif ".attn.wq_a.scale" in name:
                wq_a = scale_to_float(param)
                wkv = scale_to_float(pt_params[name.replace(".wq_a.scale",
                                                            ".wkv.scale")])
                out_name = convert_name(name.replace(
                    ".wq_a.scale", ".fused_wqa_wkv.weight_scale_inv"))
                value = self.get_rank_weight(
                    torch.cat([wq_a, wkv], dim=0), dim=0) * fp8_scale_factor
            elif ".attn.wkv." in name:
                continue
            elif ".attn.wq_b.weight" in name:
                out_name = convert_name(name)
                value = self.get_rank_weight(param, dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif ".attn.wo_a.weight" in name:
                out_name = convert_name(name)
                # wo_a is consumed by the fused grouped blockscale kernel, not
                # CK, so keep it in normal row-major [N, K] layout.
                value = self.get_rank_weight(param, dim=0)
            elif ".attn.wq_b.scale" in name or ".attn.wo_a.scale" in name:
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = self.get_rank_weight(
                    scale_to_float(param), dim=0) * fp8_scale_factor
            elif ".attn.wo_b.weight" in name:
                out_name = convert_name(name)
                value = self.get_rank_weight(param, dim=1)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif ".attn.wo_b.scale" in name:
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = self.get_rank_weight(
                    scale_to_float(param), dim=1) * fp8_scale_factor
            elif name.endswith(".attn.q_norm.weight") or name.endswith(
                    ".attn.kv_norm.weight") or name.endswith(".attn_norm.weight"):
                out_name = convert_name(name)
                value = param
            elif name.endswith(".attn.attn_sink"):
                out_name = convert_name(name)
                local_sink = self.get_rank_weight(param.float(), dim=0)
                padded_heads = max(int(local_sink.numel()), 64)
                value = torch.full(
                    (padded_heads,),
                    -float("inf"),
                    dtype=torch.float32,
                    device=local_sink.device,
                )
                value[: local_sink.numel()].copy_(local_sink)
            elif ".attn.compressor.wkv.weight" in name:
                out_name, value = fuse_wkv_wgate(name, param)
            elif ".attn.compressor.wkv.scale" in name:
                out_name, value = fuse_wkv_wgate_scale(name, param)
            elif ".attn.compressor.wgate." in name:
                continue
            elif (
                name.endswith(".attn.compressor.ape")
                or name.endswith(".attn.compressor.norm.weight")
                or name.endswith(".attn.compressor.fused_wkv_wgate.weight")
                or name.endswith(".attn.compressor.fused_wkv_wgate.scale")
            ):
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = scale_to_float(param) * fp8_scale_factor if name.endswith(
                    ".scale") else param
            elif ".attn.indexer.compressor.wkv.weight" in name:
                out_name, value = fuse_wkv_wgate(name, param)
            elif ".attn.indexer.compressor.wkv.scale" in name:
                out_name, value = fuse_wkv_wgate_scale(name, param)
            elif ".attn.indexer.compressor.wgate." in name:
                continue
            elif (
                name.endswith(".attn.indexer.compressor.ape")
                or name.endswith(".attn.indexer.compressor.norm.weight")
                or name.endswith(".attn.indexer.compressor.fused_wkv_wgate.weight")
                or name.endswith(".attn.indexer.compressor.fused_wkv_wgate.scale")
            ):
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = scale_to_float(param) * fp8_scale_factor if name.endswith(
                    ".scale") else param
            elif ".attn.indexer.wq_b.weight" in name or (
                    ".attn.indexer.weights_proj.weight" in name):
                out_name = convert_name(name)
                value = param
                if getattr(self, "dynamic_quant", False) and ".attn.indexer.wq_b.weight" in name:
                    value = shuffle_weight(value)
            elif ".attn.indexer.wq_b.scale" in name or (
                    ".attn.indexer.weights_proj.scale" in name):
                out_name = convert_name(name.replace(".scale",
                                                     ".weight_scale_inv"))
                value = scale_to_float(param) * fp8_scale_factor
            elif name.endswith(".ffn.shared_experts.w1.weight"):
                w1 = param
                w3 = pt_params[name.replace(".w1.weight", ".w3.weight")]
                out_name = convert_name(name.replace(
                    ".w1.weight", ".gate_up_proj.weight"))
                value = self.get_rank_weight(torch.cat([w1, w3], dim=0), dim=0)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif name.endswith(".ffn.shared_experts.w1.scale"):
                w1 = scale_to_float(param)
                w3 = scale_to_float(pt_params[name.replace(".w1.scale",
                                                           ".w3.scale")])
                out_name = convert_name(name.replace(
                    ".w1.scale", ".gate_up_proj.weight_scale_inv"))
                value = self.get_rank_weight(
                    torch.cat([w1, w3], dim=0), dim=0) * fp8_scale_factor
            elif name.endswith(".ffn.shared_experts.w3.weight") or name.endswith(
                    ".ffn.shared_experts.w3.scale"):
                continue
            elif name.endswith(".ffn.shared_experts.w2.weight"):
                out_name = convert_name(name.replace(".w2.weight",
                                                     ".down_proj.weight"))
                value = self.get_rank_weight(param, dim=1)
                if getattr(self, "dynamic_quant", False):
                    value = shuffle_weight(value)
            elif name.endswith(".ffn.shared_experts.w2.scale"):
                out_name = convert_name(name.replace(
                    ".w2.scale", ".down_proj.weight_scale_inv"))
                value = self.get_rank_weight(
                    scale_to_float(param), dim=1) * fp8_scale_factor
            elif name.endswith(".ffn.gate.tid2eid"):
                out_name = convert_name(
                    name.replace(".gate.tid2eid",
                                 ".experts.hash_indices_table"))
                value = param.to(dtype=torch.int32)
            elif name.endswith(".ffn.gate.weight"):
                out_name = convert_name(name)
                value = param
            elif name.endswith(".ffn.gate.bias"):
                out_name = convert_name(name.replace(
                    ".gate.bias", ".experts.e_score_correction_bias"))
                value = param.float()
            elif name.endswith(".ffn_norm.weight"):
                out_name = convert_name(name)
                value = param
            elif name.endswith("_fn") or name.endswith("_base") or name.endswith(
                    "_scale"):
                out_name = convert_name(name)
                value = param
            else:
                continue

            if out_name.endswith("_weight_scale_inv"):
                value = value.float()
            add_expected_constant(
                out_name,
                fix_fp8(value) if value.dtype == torch.float8_e4m3fn else value,
            )

        for layer_id, experts in enumerate(layers_experts):
            if not any(experts):
                continue
            local_experts = [experts[i] for i in local_expert_ids]

            w13 = torch.stack(
                [
                    torch.cat(
                        [
                            packed_to_uint8(expert["w1.weight"]),
                            packed_to_uint8(expert["w3.weight"]),
                        ],
                        dim=0,
                    )
                    for expert in local_experts
                ],
                dim=0,
            )
            add_expected_constant(
                convert_name(f"layers.{layer_id}.ffn.experts.w13_weight"),
                w13,
            )

            w13_scale = torch.stack(
                [
                    torch.cat(
                        [
                            scale_to_uint8(expert["w1.scale"]),
                            scale_to_uint8(expert["w3.scale"]),
                        ],
                        dim=0,
                    )
                    for expert in local_experts
                ],
                dim=0,
            )
            add_expected_constant(
                convert_name(f"layers.{layer_id}.ffn.experts.w13_weight_scale"),
                w13_scale,
            )

            w2 = torch.stack(
                [packed_to_uint8(expert["w2.weight"]) for expert in local_experts],
                dim=0,
            )
            add_expected_constant(
                convert_name(f"layers.{layer_id}.ffn.experts.w2_weight"),
                w2,
            )

            w2_scale = torch.stack(
                [scale_to_uint8(expert["w2.scale"]) for expert in local_experts],
                dim=0,
            )
            add_expected_constant(
                convert_name(f"layers.{layer_id}.ffn.experts.w2_weight_scale"),
                w2_scale,
            )

            mask_name = convert_name(
                f"layers.{layer_id}.ffn.experts.local_expert_mask")
            if expected_constant_names is None or mask_name in expected_constant_names:
                local_mask = torch.zeros((num_experts,), dtype=torch.int32)
                local_mask[local_expert_ids] = 1
                params_paiton[mask_name] = local_mask.cuda()

        self._add_deepseek_router_defaults(params_paiton, expected_constant_names)
        self._add_deepseek_attention_defaults(params_paiton, expected_constant_names,
                                             fp8_scale_factor)
        return params_paiton

    def _add_deepseek_attention_defaults(
        self,
        params_paiton: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]],
        fp8_scale_factor: float,
    ) -> None:
        if expected_constant_names is None:
            return

        for name in expected_constant_names:
            if name in params_paiton:
                continue
            if name.endswith("_attn_k_scale") or name.endswith("_attn_v_scale"):
                params_paiton[name] = torch.tensor(
                    [fp8_scale_factor],
                    dtype=torch.float32,
                    device="cuda",
                )

    def _add_deepseek_router_defaults(
        self,
        params_paiton: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]],
    ) -> None:
        if expected_constant_names is None:
            return

        num_hash_layers = getattr(self.config, "num_hash_layers", 0)
        num_experts = getattr(self.config, "n_routed_experts",
                              getattr(self.config, "num_experts", 0))
        topk = getattr(self.config, "num_experts_per_tok", 0)
        vocab_size = getattr(self.config, "vocab_size", 0)

        for layer_id in range(num_hash_layers):
            name = f"layers_{layer_id}_ffn_experts_hash_indices_table"
            if name not in expected_constant_names or name in params_paiton:
                continue
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(os.getenv("PAITON_DEEPSEEK_V4_HASH_SEED", "0")) +
                            layer_id)
            table = torch.stack(
                [
                    torch.randperm(num_experts, generator=gen)[:topk]
                    for _ in range(vocab_size)
                ],
                dim=0,
            ).to(dtype=torch.int32)
            params_paiton[name] = table.cuda()

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            torch.cuda.set_device(int(local_rank_env))

        expected_all = set(self.model.get_constant_names(unbound_constants_only=False))
        loaded_names: Set[str] = set()
        global_params: Dict[str, Tensor] = {}
        layer_params: Dict[str, Tensor] = {}
        current_layer: Optional[int] = None
        layer_regex = re.compile(r"layers\.(\d+)\.")

        def set_mapped(mapped: Dict[str, Tensor]) -> None:
            if not mapped:
                return
            self.model.set_many_constants_with_tensors(mapped)
            loaded_names.update(mapped)

        def flush_layer() -> None:
            nonlocal layer_params
            if current_layer is None or not layer_params:
                return
            layer_prefix = f"layers_{current_layer}_"
            expected_layer = {
                name for name in expected_all if name.startswith(layer_prefix)
            }
            set_mapped(self.map_pt_params(
                layer_params,
                expected_constant_names=expected_layer,
            ))
            layer_params = {}

        for name, tensor in weights:
            if name.startswith("mtp."):
                continue
            param = tensor.detach().cpu()
            match = layer_regex.match(name)
            if match:
                layer_id = int(match[1])
                if current_layer is None:
                    current_layer = layer_id
                elif layer_id != current_layer:
                    if layer_id < current_layer:
                        raise RuntimeError(
                            "DeepSeek V4 weight stream is not layer-contiguous; "
                            f"saw layer {layer_id} after layer {current_layer}.")
                    flush_layer()
                    current_layer = layer_id
                layer_params[name] = param
            else:
                global_params[name] = param

        flush_layer()

        expected_global = {
            name for name in expected_all if not name.startswith("layers_")
        }
        set_mapped(self.map_pt_params(
            global_params,
            expected_constant_names=expected_global,
        ))

        missing = sorted(expected_all - loaded_names)
        if missing:
            raise RuntimeError(
                "Paiton DeepSeek V4 constants mismatch: missing expected "
                f"constants during load_weights() (mapped={len(loaded_names)}, "
                f"expected={len(expected_all)}). First 50 missing:\n- "
                + "\n- ".join(missing[:50]))
        return set()

