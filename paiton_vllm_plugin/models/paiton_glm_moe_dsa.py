# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Paiton-compiled GLM MoE DSA models.

GLM-5.2-MXFP4 is a sparse MLA + MoE model whose experts (routed and shared)
are stored in the same packed MXFP4 layout (E2M1 weights + UE8M0 per-32
scales) as DeepSeek V4. Its GLM DSA indexer is computed in the compiled graph
and routes with a sigmoid + e_score_correction_bias gate (folded in-graph by
the compiler).

The base Qwen3-MoE wrapper provides the vLLM scheduler/runtime integration
and the MLA-attention constant mapping; this class overrides the MoE
weight-packing path to emit the MXFP4 fused expert tensors and the
shared-expert tensors the compiled artifact expects.
"""

from __future__ import annotations

import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch import Tensor

from vllm.distributed.parallel_state import get_ep_group
from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

from paiton_vllm_plugin.models.paiton_deepseek_v4 import PaitonDeepseekV4ForCausalLM
from paiton_vllm_plugin.runtime.core import runtime_uses_fnuz_fp8, torch_to_paiton_data


# Per-expert checkpoint tensor suffixes that make up a packed MXFP4 expert.
# amd/GLM-5.2-MXFP4 stores routed/shared experts as separate per-projection
# tensors; the loader packs gate_proj + up_proj -> w13, down_proj -> w2.
_EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")


def _packed_to_uint8(weight: torch.Tensor) -> torch.Tensor:
    """Identity for U8 MXFP4 payloads; reinterpret single-byte dtypes as U8."""
    if weight.dtype == torch.uint8:
        return weight
    if weight.element_size() == 1:
        return weight.view(torch.uint8)
    return weight.to(torch.uint8)


def _scale_to_uint8(scale: torch.Tensor) -> torch.Tensor:
    """Identity for U8 UE8M0 scale payloads; reinterpret single-byte as U8."""
    if scale.dtype == torch.uint8:
        return scale
    if scale.element_size() == 1:
        return scale.view(torch.uint8)
    return scale.to(torch.uint8)


def _fix_fp8(w: torch.Tensor) -> torch.Tensor:
    """Convert FP8 payloads only on devices that require FNUZ."""
    if runtime_uses_fnuz_fp8() and w.dtype == torch.float8_e4m3fn:
        w_int8 = w.view(torch.int8).cuda()
        w_int8[w_int8 == -128] = 0
        return w_int8.view(torch.float8_e4m3fnuz)
    return w.cuda()


def _preshuffle_mxfp4_weight(src: torch.Tensor, K: int, NXdl: int = 16) -> torch.Tensor:
    """Pre-shuffle MXFP4 weight for CK DeviceMoeGemmMXBPreShuffle.

    The scatter in the CK example is equivalent to a reshape + permute:
      src [N0, NLane, K0, KLane, KPack]
      -> permute(0, 2, 3, 1, 4) -> contiguous -> reshape(N, K_pk)
    This avoids materializing the O(N * K) index tensor that the naive
    2D scatter needs (~25 GB for GLM routed w13).
    """
    KPack = 16
    NLane = NXdl
    KLane = 64 // NLane  # 4
    K_pk = K // 2
    K0 = K_pk // (KLane * KPack)
    N_total = src.shape[0]
    assert src.shape[1] == K_pk, f"weight K_pk mismatch: {src.shape[1]} vs {K_pk}"
    assert N_total % NLane == 0, f"N_total {N_total} not divisible by NLane {NLane}"
    return (
        src.reshape(N_total // NLane, NLane, K0, KLane, KPack)
        .permute(0, 2, 3, 1, 4)
        .contiguous()
        .reshape(N_total, K_pk)
    )


def _preshuffle_mxfp4_scale(src: torch.Tensor, K: int, KLast: bool = True) -> torch.Tensor:
    """Pre-shuffle MXFP4 e8m0 scale for CK DeviceMoeGemmMXBPreShuffle.

    The scatter is equivalent to a reshape + permute:
      src [MN0, MNXdlPack, XdlMNThread, K0, KXdlPack, XdlKThread]
      -> permute(0, 3, 5, 2, 4, 1) -> contiguous -> reshape(MN, K)
    For KLast=False (Col layout) the source is transposed first.
    """
    MNXdlPack = 2
    KXdlPack = 2
    XdlMNThread = 16
    XdlKThread = 4
    K0 = K // KXdlPack // XdlKThread
    MN = src.shape[0]
    assert src.shape[1] == K, f"scale K mismatch: {src.shape[1]} vs {K}"
    assert MN % (MNXdlPack * XdlMNThread) == 0
    MN0 = MN // (MNXdlPack * XdlMNThread)
    if not KLast:
        src = src.t().contiguous()
    return (
        src.reshape(MN0, MNXdlPack, XdlMNThread, K0, KXdlPack, XdlKThread)
        .permute(0, 3, 5, 2, 4, 1)
        .contiguous()
        .reshape(MN, K)
    )


def _preshuffle_flatmm_mxfp4_weight(
    src: torch.Tensor,
    K: int,
    *,
    gate_up: bool,
) -> torch.Tensor:
    """Pre-shuffle packed MXFP4 weights for CK Tile A16W4 MoE FlatMM."""
    experts, n_dim, packed_k = src.shape
    KPack = 16
    NLane = 16
    KLane = 64 // NLane
    K_pk = K // 2
    K0 = K_pk // (KLane * KPack)
    assert packed_k == K_pk

    if gate_up:
        half_n = n_dim // 2
        assert n_dim % (2 * NLane) == 0
        return (
            src.reshape(
                experts, 2, half_n // NLane, NLane, K0, KLane, KPack
            )
            .permute(0, 2, 1, 4, 5, 3, 6)
            .contiguous()
            .reshape_as(src)
        )

    assert n_dim % NLane == 0
    return (
        src.reshape(experts, n_dim // NLane, NLane, K0, KLane, KPack)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .reshape_as(src)
    )


def _preshuffle_flatmm_mxfp4_scale(
    src: torch.Tensor,
    *,
    gate_up: bool,
) -> torch.Tensor:
    """Pre-shuffle UE8M0 scales for CK Tile A16W4 MoE FlatMM."""
    experts, n_dim, k_blocks = src.shape
    KPack = 2
    NPack = 2
    NLane = 16
    KLane = 64 // NLane
    assert k_blocks % (KPack * KLane) == 0
    assert n_dim % (NLane * NPack) == 0
    K0 = k_blocks // (KPack * KLane)
    scale_kn = src.permute(0, 2, 1).contiguous()

    if gate_up:
        shuffled = (
            scale_kn.reshape(
                experts,
                K0,
                KPack,
                KLane,
                NPack,
                n_dim // (NLane * NPack),
                NLane,
            )
            .permute(0, 5, 1, 3, 6, 2, 4)
            .contiguous()
        )
    else:
        shuffled = (
            scale_kn.reshape(
                experts,
                K0,
                KPack,
                KLane,
                n_dim // (NLane * NPack),
                NPack,
                NLane,
            )
            .permute(0, 4, 1, 3, 6, 2, 5)
            .contiguous()
        )
    return shuffled.reshape_as(src)


def _artifact_has_flatmm_constant_names(
    expected_constant_names: Optional[Set[str]],
) -> bool:
    return bool(
        expected_constant_names
        and any(
            name.endswith("_mlp_experts_w13_weight_flatmm")
            or name.endswith(
                "_mlp_experts_w13_weight_flatmm_fused_shared"
            )
            for name in expected_constant_names
        )
    )


def _artifact_has_fused_shared_flatmm_constants(
    expected_constant_names: Optional[Set[str]],
) -> bool:
    return bool(
        expected_constant_names
        and any(
            name.endswith(
                "_mlp_experts_w13_weight_flatmm_fused_shared"
            )
            for name in expected_constant_names
        )
    )


def _artifact_has_fused_shared_ck_constants(
    expected_constant_names: Optional[Set[str]],
) -> bool:
    return bool(
        expected_constant_names
        and any(
            name.endswith("_mlp_experts_w13_weight_ck_fused_shared")
            for name in expected_constant_names
        )
    )


def _resolve_compiled_moe_kernel(
    requested_kernel: Optional[str],
    expected_constant_names: Optional[Set[str]],
) -> str:
    """Resolve the loader layout from layout-specific artifact constants."""
    artifact_uses_ck_fused = _artifact_has_fused_shared_ck_constants(
        expected_constant_names
    )
    if artifact_uses_ck_fused:
        if requested_kernel not in (None, "ck_moe_fp4_fused"):
            raise RuntimeError(
                "The compiled GLM artifact expects fused-shared CK "
                "BPreShuffle MoE weights, but PAITON_MOE_KERNEL="
                f"{requested_kernel!r}. Remove the variable or set "
                "PAITON_MOE_KERNEL=ck_moe_fp4_fused."
            )
        return "ck_moe_fp4_fused"

    artifact_uses_flatmm = _artifact_has_flatmm_constant_names(
        expected_constant_names
    )
    if artifact_uses_flatmm:
        if requested_kernel not in (None, "ck_flatmm_fp4"):
            raise RuntimeError(
                "The compiled GLM artifact expects FlatMM-preshuffled MoE "
                "weights, but PAITON_MOE_KERNEL="
                f"{requested_kernel!r}. Remove the variable or set "
                "PAITON_MOE_KERNEL=ck_flatmm_fp4."
            )
        return "ck_flatmm_fp4"

    # Artifacts compiled before the layout-specific naming fix use the generic
    # constant names for both layouts. Preserve an explicit FlatMM request as
    # the compatibility path for those existing .so files. Newly compiled
    # FlatMM artifacts always take the self-identifying branch above.
    kernel = requested_kernel or "ck_moe_fp4"
    if kernel not in (
        "paiton", "ck_moe_fp4", "ck_flatmm_fp4", "ck_moe_fp4_fused"
    ):
        raise ValueError(f"Unsupported PAITON_MOE_KERNEL={kernel!r}")
    return kernel


class PaitonGlmMoeDsaForCausalLM(PaitonDeepseekV4ForCausalLM):
    """Runtime wrapper for Paiton-compiled GLM MoE DSA artifacts."""

    # MLA decoupled projections: do NOT fuse q/k/v. gate/up are packed into the
    # MXFP4 expert tensors, so we also do not fuse them into a gate_up_proj.
    packed_modules_mapping: Dict[str, list] = {}

    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__(vllm_config, prefix=prefix)
        cfg = self.config
        self._n_routed_experts = int(
            getattr(cfg, "n_routed_experts", getattr(cfg, "num_experts", 0))
        )
        self._topk = int(getattr(cfg, "num_experts_per_tok", 0))
        self._first_k_dense_replace = int(getattr(cfg, "first_k_dense_replace", 0))
        self._moe_layer_freq = int(getattr(cfg, "moe_layer_freq", 1))
        self._has_correction_bias = (
            getattr(cfg, "topk_method", None) == "noaux_tc"
        )
        self._index_topk = int(getattr(cfg, "index_topk", 2048))
        self._index_topk_freq = int(getattr(cfg, "index_topk_freq", 4))
        self._index_skip_topk_offset = int(getattr(cfg, "index_skip_topk_offset", 3))
        self._configure_glm_kv_cache_spec(vllm_config)
        # Alias-key cache for the sparse indexer groups. Populated lazily so
        # that test harnesses using __new__ (bypassing __init__) still work.
        # The MLA latent cache needs no plugin-side state: it is a flat view
        # of the vLLM-allocated single latent plane (see
        # _get_glm_latent_kv_cache), never a private allocation.
        self._glm_alias_key_cache: Optional[Tuple] = None

    def _configure_glm_kv_cache_spec(self, vllm_config) -> None:
        """Ask vLLM to allocate the latent MLA cache shape GLM consumes."""
        num_q_heads = int(self.config.num_attention_heads) // int(self.tp_size)
        head_size = self._mla_head_dim()
        scale = float(getattr(self.config, "qk_head_dim", head_size)) ** -0.5

        static_context = getattr(
            self.compilation_config,
            "static_forward_context",
            {},
        )
        for i in range(self.num_layers):
            ctx = static_context[str(i)]
            ctx.num_heads = num_q_heads
            ctx.head_size = head_size
            # MLA latent semantics: one 576-wide latent per token, no
            # separate V plane. head_size_v=0 makes vLLM allocate a single
            # [num_blocks, 1, block_size, mla_head_dim] latent plane per
            # layer, which _get_glm_latent_kv_cache flattens for the sparse
            # MLA kernels. (head_size_v=head_size would allocate a second
            # 576-wide plane that nothing reads.)
            ctx.head_size_v = 0
            ctx.num_kv_heads = 1
            if hasattr(ctx, "scale"):
                ctx.scale = scale

            # vLLM's printable/debug state and some backend paths mirror these
            # fields on the implementation object. Keep it in sync without
            # creating a new Attention module, which would re-register prefix i.
            impl = getattr(ctx, "impl", None)
            if impl is not None:
                impl.num_heads = num_q_heads
                impl.head_size = head_size
                if hasattr(impl, "head_size_v"):
                    impl.head_size_v = 0
                impl.num_kv_heads = 1
                if hasattr(impl, "scale"):
                    impl.scale = scale
                if hasattr(impl, "num_queries_per_kv"):
                    impl.num_queries_per_kv = num_q_heads

    # ------------------------------------------------------------------ #
    # Layer classification helpers.
    # ------------------------------------------------------------------ #
    def _is_sparse_layer(self, layer_id: int) -> bool:
        return (
            layer_id >= self._first_k_dense_replace
            and layer_id % self._moe_layer_freq == 0
        )

    def _indexer_type(self, layer_id: int) -> str:
        indexer_types = getattr(self.config, "indexer_types", None)
        if indexer_types is not None and layer_id < len(indexer_types):
            return str(indexer_types[layer_id])
        layer_offset = max(layer_id - self._index_skip_topk_offset + 1, 0)
        return "full" if layer_offset % self._index_topk_freq == 0 else "shared"

    def _sparse_mla_index_width(self) -> int:
        return self._index_topk

    def _layer_compress_ratio(self, layer_idx: int) -> int:
        return 1

    def _index_head_dim(self) -> int:
        return int(getattr(self.config, "index_head_dim", 128))

    def _index_n_heads(self) -> int:
        return int(getattr(self.config, "index_n_heads", 32))

    def _mla_head_dim(self) -> int:
        return int(getattr(self.config, "kv_lora_rank", 512)) + int(
            getattr(self.config, "qk_rope_head_dim", 64)
        )

    def _get_glm_latent_kv_cache(
        self,
        layer_idx: int,
        reference_kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Return the flat sparse-MLA latent view of vLLM's MLA cache plane.

        vLLM allocates one latent plane per attention layer (head_size =
        mla_head_dim, head_size_v = 0): ``[num_blocks, 1, block_size,
        mla_head_dim]`` in the cache dtype (raw uint8 for fp8 pools).
        Flatten it to the ``[num_blocks * block_size, 1, mla_head_dim]``
        layout the compiled sparse-MLA kernels index. The result aliases
        the vLLM pool pages directly: no private latent cache is
        allocated, and prefix-cache page copy/reuse operates on the same
        physical pages the kernels read and write.
        """
        del layer_idx
        head_dim = self._mla_head_dim()
        cache_dtype = (
            self.cache_dtype
            if bool(getattr(self.config, "fp8_kv_cache", False))
            else self.dtype
        )
        # vLLM allocates the fp8 KV pool as raw bytes (torch.uint8) while the
        # compiled kernels treat the cache as an fp8 element type. Both are
        # 1-byte elements, so re-view the pool instead of shadow-copying it;
        # the actual bit semantics live entirely inside the Paiton kernels
        # (HIP fnuz encode on write, HIP fnuz decode on read).
        if (
            reference_kv_cache.dtype == torch.uint8
            and cache_dtype in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
            )
        ):
            reference_kv_cache = reference_kv_cache.view(cache_dtype)

        if reference_kv_cache.dim() != 4:
            raise RuntimeError(
                "GLM sparse MLA expects vLLM's single-plane MLA cache "
                "(num_blocks, 1, block_size, mla_head_dim); legacy packed "
                "K+V / Paiton-layout caches are no longer allocated, got "
                f"shape={tuple(reference_kv_cache.shape)}"
            )
        num_blocks = int(reference_kv_cache.shape[0])
        num_kv_heads = int(reference_kv_cache.shape[1])
        block_size = int(reference_kv_cache.shape[2])
        content = int(reference_kv_cache.shape[3])
        if num_kv_heads != 1 or block_size <= 0 or content != head_dim:
            raise RuntimeError(
                f"GLM sparse MLA expected the vLLM MLA latent plane "
                f"(num_blocks, 1, block_size, {head_dim}), got shape="
                f"{tuple(reference_kv_cache.shape)}"
            )
        if not reference_kv_cache.is_contiguous():
            # A block-compact KV layout (B outermost) strides each layer's
            # pages across other layers' blocks, breaking the flat
            # slot-major view the sparse-MLA kernels index. Layer-compact
            # layouts (the default) keep every layer's plane contiguous.
            raise RuntimeError(
                "GLM sparse MLA requires a contiguous per-layer latent "
                "plane; the vLLM KV cache layout produced a strided view "
                f"(shape={tuple(reference_kv_cache.shape)})."
            )
        return reference_kv_cache.view(num_blocks * block_size, 1, head_dim)

    def _refresh_deepseek_kv_binding(
        self,
        binding,
        *,
        validate_context: bool = False,
    ):
        # Fast path: replay only when the cached binding already holds the
        # flat rank-3 latent view. The input plan pre-populates
        # kv_cache_pdata from the raw rank-4 vLLM pool plane, so pdata
        # alone is not evidence that the latent view was ever derived —
        # the first bind must fall through and translate below.
        if (
            not validate_context
            and binding.kv_cache_pdata is not None
            and binding.kv_cache is not None
            and binding.kv_cache.dim() == 3
        ):
            return binding.kv_cache, binding.kv_cache_pdata

        reference_kv_cache = binding.kv_cache
        if validate_context:
            current = self._get_kv_cache_tensor(binding.ctx)
            if current is not None:
                reference_kv_cache = current

        latent_kv_cache = self._get_glm_latent_kv_cache(
            binding.layer_idx,
            reference_kv_cache,
        )
        current_ptr = latent_kv_cache.data_ptr()
        if (
            current_ptr == binding.kv_cache_data_ptr
            and latent_kv_cache is binding.kv_cache
            and binding.kv_cache_pdata is not None
        ):
            return binding.kv_cache, binding.kv_cache_pdata

        binding.kv_cache = latent_kv_cache
        binding.kv_cache_view = latent_kv_cache
        binding.kv_cache_pdata = torch_to_paiton_data(latent_kv_cache)
        binding.kv_cache_data_ptr = current_ptr
        return binding.kv_cache, binding.kv_cache_pdata

    def _runtime_kv_cache_block_size(self) -> int:
        cached = getattr(self, "_paiton_kv_cache_block_size", None)
        if cached is not None:
            return cached

        input_plan = getattr(self, "_deepseek_input_plan", None)
        first_kv_cache = getattr(input_plan, "first_kv_cache", None)
        if first_kv_cache is not None and first_kv_cache.dim() >= 3:
            block_size = first_kv_cache.shape[2]
            self._paiton_kv_cache_block_size = block_size
            return block_size

        compilation_config = getattr(self, "compilation_config", None)
        static_context = getattr(
            compilation_config,
            "static_forward_context",
            {},
        )
        for ctx in static_context.values():
            kv_cache = self._get_kv_cache_tensor(ctx)
            if kv_cache is not None and kv_cache.dim() >= 3:
                block_size = kv_cache.shape[2]
                self._paiton_kv_cache_block_size = block_size
                return block_size

        block_size = int(getattr(self.config, "kv_cache_block_size", 16))
        self._paiton_kv_cache_block_size = block_size
        return block_size

    def _compute_step_slot_extents(
        self,
        slot_mapping: Optional[torch.Tensor],
        sparse_indices: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
    ) -> tuple[int, int, int, int]:
        del slot_mapping, sparse_indices
        # GLM stores its sparse/indexer caches in the same physical namespace
        # as the vLLM block table. The highest block therefore covers both the
        # current slot and every index the compiled indexer can emit.
        #
        # PaitonGPUWorker tracks the physical block-ID high-water mark from
        # SchedulerOutput's CPU lists. This avoids a per-token GPU .item()
        # synchronization without confusing block_tables.shape[1] (logical
        # blocks per request) with the physical server block pool.
        #
        # Direct/unit-test callers and custom workers may not install that
        # watermark. Keep the exact GPU maximum as a correctness-first
        # fallback; it is slower, but cannot under-allocate or corrupt caches.
        block_size = self._runtime_kv_cache_block_size()
        required_blocks_bt = int(
            getattr(self, "_paiton_physical_block_high_water", 0) or 0
        )
        if required_blocks_bt <= 0:
            max_block = -1
            if block_tables is not None:
                valid_blocks = block_tables[block_tables >= 0]
                if valid_blocks.numel() > 0:
                    max_block = int(valid_blocks.max().item())
            required_blocks_bt = max_block + 1 if max_block >= 0 else 1
        else:
            max_block = required_blocks_bt - 1
        required_slots = required_blocks_bt * block_size
        return required_slots, -1, max_block, required_blocks_bt

    def _compiled_indexer_overwrites_sparse_inputs(self) -> bool:
        """GLM full indexers produce sparse indices without a window seed.

        The compiler fixes the GLM sparse-indexer ``window_size`` to zero, so
        its output does not depend on the input contents. Shared layers alias
        the most recent full indexer's output. This lets the runtime bind
        scratch buffers directly and skip recent-index construction.
        """
        return True

    def _sparse_mla_compressed_offset_input_value(
        self,
        step_sparse_slot_offset: int,
    ) -> int:
        # GLM stores sparse MLA/indexer rows in the same flat slot namespace as
        # the vLLM block table. DeepSeek appends compressed rows after the dense
        # cache and therefore binds a non-zero runtime offset; GLM must always
        # bind zero when the graph asks for this tensor.
        return 0

    def _sparse_mla_compressed_offset_is_static(self) -> bool:
        return True

    def _preallocate_sparse_mla_cache_capacity(self) -> bool:
        # GLM's flat indexer caches use vLLM physical slot IDs directly.
        # Their maximum size is therefore known from the paged cache.
        # Allocate that size once instead of geometrically growing the 21
        # indexer caches; the retired growth allocations otherwise remain
        # reserved by PyTorch and eventually starve ROCm kernel launches.
        # The MLA latent cache itself is no longer allocated here at all:
        # it is the vLLM latent plane, bound via _get_sparse_mla_kv_cache.
        return True

    def _get_sparse_mla_kv_cache(
        self,
        layer_idx: int,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        sparse_indices: Optional[torch.Tensor] = None,
        *,
        compressed_slot_offset: Optional[int] = None,
        required_slots: Optional[int] = None,
    ) -> torch.Tensor:
        """Bind the flat vLLM MLA latent plane; allocate no scratch latent.

        ``kv_cache`` here is already the refreshed binding tensor from
        _refresh_deepseek_kv_binding: the flat
        ``[num_blocks * block_size, 1, mla_head_dim]`` view of the vLLM
        single-plane MLA allocation. The compiled sparse-MLA kernels share
        those physical pages with vLLM's prefix-cache block copy/reuse, so
        no plugin-side latent cache exists (unlike the DeepSeek compressed
        namespace the base mixin serves).
        """
        del (
            layer_idx,
            slot_mapping,
            sparse_indices,
            compressed_slot_offset,
            required_slots,
        )
        if kv_cache is None or kv_cache.dim() != 3:
            raise RuntimeError(
                "GLM sparse MLA latent binding expected the flat vLLM MLA "
                "plane view [slots, 1, mla_head_dim], got shape="
                f"{tuple(kv_cache.shape) if kv_cache is not None else None}"
            )
        return kv_cache

    def _build_glm_alias_keys(self) -> List[int]:
        """Precompute per-layer sparse-MLA alias keys from config.indexer_types.

        Matches the walk-back in the previous _sparse_mla_runtime_alias_key but
        runs once for all layers instead of per layer per step.
        """
        n = self.num_layers
        indexer_types = getattr(self.config, "indexer_types", None)
        types: List[str] = []
        for layer_id in range(n):
            if indexer_types is not None and layer_id < len(indexer_types):
                types.append(str(indexer_types[layer_id]))
            else:
                layer_offset = max(
                    layer_id - self._index_skip_topk_offset + 1, 0
                )
                types.append(
                    "full" if layer_offset % self._index_topk_freq == 0 else "shared"
                )
        keys: List[int] = []
        for layer_id in range(n):
            if types[layer_id] != "shared":
                keys.append(layer_id)
                continue
            src = layer_id
            for s in range(layer_id - 1, -1, -1):
                if types[s] != "shared":
                    src = s
                    break
            keys.append(src)
        return keys

    def _sparse_mla_runtime_alias_key(self, layer_idx: int) -> int:
        # Cache keyed by the identity of config.indexer_types so that a
        # reassignment (e.g. in tests) invalidates the cache, while the
        # steady-state hot path is an O(1) list index.
        indexer_types = getattr(self.config, "indexer_types", None)
        cache = getattr(self, "_glm_alias_key_cache", None)
        if cache is None or cache[0] is not indexer_types:
            keys = self._build_glm_alias_keys()
            cache = (indexer_types, keys)
            self._glm_alias_key_cache = cache
        return cache[1][layer_idx]

    def _indexer_num_existing_rows(
        self,
        layer_idx: int,
        sparse_mla_indexer_kv: Optional[torch.Tensor],
    ) -> int:
        # GLM writes current indexer K rows directly into the flat cache inside
        # the compiled graph, then immediately quantizes them. Always quantize
        # from row 0; the DeepSeek high-water-mark optimization would skip the
        # freshly written rows because GLM does not append into a compressed
        # namespace.
        return 0

    def _indexer_num_existing_rows_is_static(self) -> bool:
        return True

    # ------------------------------------------------------------------ #
    # Weight mapping.
    # ------------------------------------------------------------------ #
    def map_pt_params(
        self,
        pt_params: Dict[str, Tensor],
        expected_constant_names: Optional[Set[str]] = None,
    ) -> Dict[str, Tensor]:
        def convert_name(name: str) -> str:
            return name.replace("model.", "").replace(".", "_")

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        def get_rank_weight(weight: Tensor, dim: int) -> Tensor:
            if weight.dim() == 0:
                weight = weight.reshape([1])
            return torch.split(weight, weight.shape[dim] // tp_size, dim)[tp_rank]

        def get_rank_bias(bias: Tensor) -> Tensor:
            return bias if tp_rank == 0 else torch.zeros_like(bias)

        params_paiton: Dict[str, Tensor] = {}

        # vLLM constructs an EP process group over TP ranks even when EP is
        # disabled. Only shard experts when the configuration explicitly
        # enables expert parallelism; a TP-only artifact replicates all routed
        # experts on every rank.
        enable_ep = bool(self.parallel_config.enable_expert_parallel)
        ep_group = get_ep_group()
        ep_rank = ep_group.rank_in_group if enable_ep else 0
        ep_size = ep_group.world_size if enable_ep else 1
        self._validate_compiled_ep_size(ep_size)
        if tp_size % ep_size != 0:
            raise RuntimeError(
                f"EP size must divide TP size (ep_size={ep_size}, "
                f"tp_size={tp_size})"
            )
        moe_tp_size = tp_size // ep_size
        compiled_moe_tp_size = int(
            getattr(self.config, "moe_tp_size", moe_tp_size)
        )
        if compiled_moe_tp_size != moe_tp_size:
            raise RuntimeError(
                "Compiled/runtime MoE tensor-parallel size mismatch: "
                f"compiled={compiled_moe_tp_size}, runtime={moe_tp_size}"
            )
        moe_tp_rank = tp_rank % moe_tp_size
        assert self._n_routed_experts % ep_size == 0, (
            f"EP world_size must divide num_experts (ep_size={ep_size}, "
            f"num_experts={self._n_routed_experts})")
        num_local_experts = self._n_routed_experts // ep_size
        placement = getattr(
            self.parallel_config, "expert_placement_strategy", "linear"
        )
        if placement == "round_robin":
            local_expert_ids = list(
                range(ep_rank, self._n_routed_experts, ep_size)
            )
        else:
            expert_start = ep_rank * num_local_experts
            local_expert_ids = list(
                range(expert_start, expert_start + num_local_experts)
            )

        # Per-layer expert buffer (routed). shared_experts handled inline below.
        layers_routed_experts = [
            [{} for _ in range(self._n_routed_experts)]
            for _ in range(self.config.num_hidden_layers)
        ]
        layers_shared_experts = [{} for _ in range(self.config.num_hidden_layers)]
        routed_re = re.compile(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)"
        )
        shared_re = re.compile(r"model\.layers\.(\d+)\.mlp\.shared_experts\.(.+)")

        def maybe_emit(out_name: str, value: Tensor) -> None:
            if (
                expected_constant_names is None
                or out_name in expected_constant_names
            ) and out_name not in params_paiton:
                params_paiton[out_name] = value

        num_hidden_layers = self.config.num_hidden_layers
        for name, param in pt_params.items():
            # ---- Skip the MTP/predictor layer (model.layers.N where N >=
            # num_hidden_layers). amd/GLM-5.2-MXFP4 stores these as
            # `model.layers.{N}.` (not `mtp.`), and the compiled Paiton
            # artifact only covers the main decoder layers -- vLLM runs MTP
            # itself. Drop these weights so they don't trigger out-of-range
            # buffer indexing or bogus constant names.
            m_layer = re.match(r"model\.layers\.(\d+)\.", name)
            if m_layer is not None and int(m_layer[1]) >= num_hidden_layers:
                continue

            # ---- GLM router aliases. ------------------------------------- #
            # HF/vLLM GLM MoE names the router `mlp.gate.weight`, while the
            # compiled artifact keeps the DeepSeek-style constant name
            # `mlp_gate_proj_weight`.
            if name.endswith("mlp.gate.weight"):
                out_name = convert_name(
                    name.replace(".mlp.gate.weight", ".mlp.gate_proj.weight")
                )
                maybe_emit(out_name, param.cuda())
                continue

            if name.endswith("mlp.gate.expert_bias"):
                out_name = convert_name(
                    name.replace(
                        ".mlp.gate.expert_bias",
                        ".mlp.gate.e_score_correction_bias",
                    )
                )
                maybe_emit(out_name, param.cuda())
                continue

            # ---- Routed MoE experts: buffer for MXFP4 packing below. ------- #
            m = routed_re.match(name)
            if m is not None:
                layer_id, expert_id, weight_name = int(m[1]), int(m[2]), m[3]
                layers_routed_experts[layer_id][expert_id][weight_name] = param
                continue

            # ---- Shared experts (sparse layers): buffer for MXFP4 packing. - #
            m = shared_re.match(name)
            if m is not None:
                layer_id, weight_name = int(m[1]), m[2]
                layers_shared_experts[layer_id][weight_name] = param
                continue

            # ---- All other (attention, dense MLP, router, norms, head). --- #
            out_name = convert_name(name)

            if name.endswith("down_proj.weight") or name.endswith("o_proj.weight"):
                # Row parallel: split across dim=1.
                value = get_rank_weight(param, dim=1)
                maybe_emit(out_name, _fix_fp8(value) if value.dtype == torch.float8_e4m3fn else value.cuda())
                continue

            if name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
                layer_match = re.match(r"model\.layers\.(\d+)\.", name)
                layer_id = int(layer_match[1]) if layer_match is not None else None
                if layer_id is not None and not self._is_sparse_layer(layer_id):
                    # Dense MLP projections are emitted only by the fused
                    # gate_up_proj path below.
                    continue
                # MoE router gate is replicated across TP ranks.
                maybe_emit(out_name, param.cuda())
                continue

            if name.endswith("mlp.gate.e_score_correction_bias"):
                # Router correction bias (noaux_tc). Replicated across TP ranks.
                # The checkpoint stores it in fp32 -- the compiled artifact
                # also declares it in fp32, so no cast is needed.
                maybe_emit(out_name, param.cuda())
                continue

            if name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                # Norm weights are replicated across TP ranks.
                maybe_emit(out_name, param.cuda())
                continue

            if (
                ".self_attn.indexer.wq_b.weight" in name
                or ".self_attn.indexer.wk.weight" in name
                or ".self_attn.indexer.weights_proj.weight" in name
                or ".self_attn.indexer.k_norm.weight" in name
                or ".self_attn.indexer.k_norm.bias" in name
            ):
                # GLM DSA indexer is replicated across TP ranks. The sparse
                # top-k kernel combines its own index_n_heads; sharding these
                # weights would make the compiled graph's expected shapes miss.
                maybe_emit(out_name, param.cuda())
                continue

            # MLA low-rank down-projections (hidden -> lora_rank) are replicated
            # across TP ranks (q_lora / kv_lora are shared). The head
            # projections (q_b_proj, kv_b_proj) are column-parallel (sharded
            # along dim=0 across heads), and o_proj is row-parallel (already
            # handled above). VocabParallelEmbedding (embed_tokens) and
            # ParallelLMHead (lm_head) are also column-parallel under TP.

            if (
                name.endswith("q_a_proj.weight")
                or name.endswith("kv_a_proj_with_mqa.weight")
            ):
                maybe_emit(out_name, param.cuda())
                continue

            # Fused q_a + kv_a projection: the compiler packs q_a_proj.weight
            # [q_lora_rank, hidden] and kv_a_proj_with_mqa.weight
            # [kv_lora_rank + rope, hidden] into a single
            # q_kv_a_proj_fused_weight [q_lora + kv_lora + rope, hidden].
            # Both source weights are replicated (not TP-sharded).

            # Skip the fused weight here; it's packed in the loop below.
            if name.endswith("q_kv_a_proj_fused.weight"):
                continue

            if name.endswith(".bias"):
                maybe_emit(out_name, get_rank_bias(param).cuda())
                continue

            # q_b_proj / kv_b_proj / embed_tokens / lm_head: column-parallel.
            value = get_rank_weight(param, dim=0)
            maybe_emit(out_name, _fix_fp8(value) if value.dtype == torch.float8_e4m3fn else value.cuda())

        # ---- Dense-layer MLP fusion (gate_proj + up_proj -> gate_up_proj). - #
        # Dense layers (0..first_k_dense_replace-1) emit fused gate_up_proj_weight
        # (column parallel, split across dim=0) and the down_proj already handled
        # above via the row-parallel branch.
        for layer_id in range(self.config.num_hidden_layers):
            if self._is_sparse_layer(layer_id):
                continue
            gp_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
            up_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
            if gp_name in pt_params and up_name in pt_params:
                fused = torch.cat(
                    [
                        get_rank_weight(pt_params[gp_name], dim=0),
                        get_rank_weight(pt_params[up_name], dim=0),
                    ],
                    dim=0,
                )
                maybe_emit(
                    convert_name(f"model.layers.{layer_id}.mlp.gate_up_proj.weight"),
                    fused.cuda(),
                )

        # ---- Fused q_a + kv_a projection packing. ------------------------- #
        # The compiler fuses q_a_proj and kv_a_proj_with_mqa into a single
        # q_kv_a_proj_fused weight. Pack the checkpoint's separate weights
        # into the fused layout: [q_lora, hidden] ++ [kv_lora + rope, hidden].
        _q_lora = int(getattr(self.config, "q_lora_rank", 0) or 0)
        _kv_lora = int(getattr(self.config, "kv_lora_rank", 512))
        _rope_dim = int(getattr(self.config, "qk_rope_head_dim", 64))
        for layer_id in range(self.config.num_hidden_layers):
            qa_name = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
            kv_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
            fused_name = convert_name(
                f"model.layers.{layer_id}.self_attn.q_kv_a_proj_fused.weight"
            )
            if (
                expected_constant_names is not None
                and fused_name not in expected_constant_names
            ):
                continue
            if qa_name in pt_params and kv_name in pt_params:
                qa_w = pt_params[qa_name].cuda()
                kv_w = pt_params[kv_name].cuda()
                fused_w = torch.cat([qa_w, kv_w], dim=0)
                maybe_emit(fused_name, fused_w)

        # ---- Fused wk + weights_proj indexer projection packing. ---------- #
        # When PAITON_FUSE_GLM_WK_WEIGHTS_PROJ=1 the compiler fuses the
        # full-indexer indexer.wk [head_dim, hidden] and indexer.weights_proj
        # [n_heads, hidden] into a single wk_weights_proj weight
        # [head_dim + n_heads, hidden]. Pack the checkpoint's separate weights
        # into the fused layout in checkpoint order (wk rows first, then
        # weights_proj rows). The fused name is layout-specific, so a merged
        # artifact cannot accidentally share a constant blob with a
        # separate-layout artifact. Both source projections are replicated
        # (the indexer is non-TP), so the fused projection is also replicated.
        # The merged GEMM runs BF16 (Contract A); cast the gate rows to the wk
        # weight dtype so the constant blob matches the fused Linear's dtype.
        _idx_hd = int(getattr(self.config, "index_head_dim", 0) or 0)
        _idx_nh = int(getattr(self.config, "index_n_heads", 0) or 0)
        for layer_id in range(self.config.num_hidden_layers):
            wk_name = (
                f"model.layers.{layer_id}.self_attn.indexer.wk.weight"
            )
            wp_name = (
                f"model.layers.{layer_id}.self_attn.indexer.weights_proj.weight"
            )
            fused_name = convert_name(
                f"model.layers.{layer_id}.self_attn.indexer."
                f"wk_weights_proj.weight"
            )
            if (
                expected_constant_names is not None
                and fused_name not in expected_constant_names
            ):
                # Artifact does not use the merged layout for this layer.
                continue
            # Only pack when the artifact actually expects the fused constant.
            # When expected_constant_names is None (test path), mirror the
            # q_kv_a loop and pack only if both sources are present.
            # Note: duplicate source-tensor detection happens upstream in
            # load_weights() (before the checkpoint stream is collapsed into
            # the pt_params dict); a dict cannot represent a repeated name, so
            # by this point each source weight is guaranteed unique.
            if (
                expected_constant_names is not None
                and fused_name in expected_constant_names
            ):
                missing = [
                    n for n in (wk_name, wp_name) if n not in pt_params
                ]
                if missing:
                    raise ValueError(
                        f"Cannot pack fused wk_weights_proj for layer "
                        f"{layer_id}: the compiled artifact expects the "
                        f"merged layout but the checkpoint is missing "
                        f"source weight(s) {missing}."
                    )
            if wk_name not in pt_params or wp_name not in pt_params:
                continue
            wk_w = pt_params[wk_name]
            wp_w = pt_params[wp_name]
            if wk_w.shape[0] != _idx_hd or wp_w.shape[0] != _idx_nh:
                raise ValueError(
                    f"wk_weights_proj shape mismatch for layer {layer_id}: "
                    f"wk.weight rows={wk_w.shape[0]} (expected {_idx_hd}), "
                    f"weights_proj.weight rows={wp_w.shape[0]} "
                    f"(expected {_idx_nh})."
                )
            if wk_w.shape[1] != wp_w.shape[1]:
                raise ValueError(
                    f"wk_weights_proj hidden-size mismatch for layer "
                    f"{layer_id}: wk.weight hidden={wk_w.shape[1]}, "
                    f"weights_proj.weight hidden={wp_w.shape[1]}."
                )
            fused_w = torch.cat(
                [wk_w.cuda(), wp_w.cuda().to(wk_w.dtype)], dim=0
            )
            assert fused_w.shape[0] == _idx_hd + _idx_nh
            maybe_emit(fused_name, fused_w)

        # ---- Pack routed MXFP4 experts into fused w13/w2 (+UE8M0 scales). - #
        # CK kernels require B weights/scales to be pre-shuffled at load time.
        # DeviceMoeGemmMXBPreShuffle and A16W4 FlatMM use different layouts.
        _moe_kernel = _resolve_compiled_moe_kernel(
            os.environ.get("PAITON_MOE_KERNEL"),
            expected_constant_names,
        )
        _use_ck_moe = _moe_kernel in ("ck_moe_fp4", "ck_moe_fp4_fused")
        _use_flatmm_moe = _moe_kernel == "ck_flatmm_fp4"
        _fused_shared_flatmm = _artifact_has_fused_shared_flatmm_constants(
            expected_constant_names
        )
        _fused_shared_ck = _artifact_has_fused_shared_ck_constants(
            expected_constant_names
        )
        _shared_use_ck_moe = _moe_kernel in ("ck_moe_fp4", "ck_flatmm_fp4")
        _routed_layout_suffix = (
            "_flatmm_fused_shared"
            if _fused_shared_flatmm
            else (
                "_ck_fused_shared"
                if _fused_shared_ck
                else (
                    "_flatmm"
                    if _artifact_has_flatmm_constant_names(expected_constant_names)
                    else ""
                )
            )
        )
        _hidden = int(self.config.hidden_size)
        _inter = int(getattr(self.config, "moe_intermediate_size", 0))
        if _inter % moe_tp_size != 0:
            raise RuntimeError(
                "MoE tensor-parallel size must divide moe_intermediate_size "
                f"(moe_tp_size={moe_tp_size}, intermediate_size={_inter})"
            )
        _local_inter = _inter // moe_tp_size
        _n_shared = int(getattr(self.config, "n_shared_experts", 1))
        _shared_inter = _inter * _n_shared

        def shard_moe_tensor(value: Tensor, dim: int) -> Tensor:
            if moe_tp_size == 1:
                return value
            if value.shape[dim] % moe_tp_size != 0:
                raise RuntimeError(
                    "Cannot tensor-parallel shard MoE tensor with shape "
                    f"{tuple(value.shape)} on dim {dim} across "
                    f"{moe_tp_size} ranks"
                )
            return value.chunk(moe_tp_size, dim=dim)[moe_tp_rank].contiguous()

        for layer_id, experts in enumerate(layers_routed_experts):
            if not any(experts):
                continue
            local_experts = [experts[i] for i in local_expert_ids]
            shared = layers_shared_experts[layer_id]
            if _fused_shared_flatmm and not shared:
                raise RuntimeError(
                    f"Layer {layer_id} is missing the shared expert required "
                    "by the fused-shared FlatMM artifact"
                )
            if _fused_shared_ck and not shared:
                raise RuntimeError(
                    f"Layer {layer_id} is missing the shared expert required "
                    "by the fused-shared CK artifact"
                )

            w13_parts = [
                torch.cat(
                    [
                        shard_moe_tensor(
                            _packed_to_uint8(e["gate_proj.weight"]), 0
                        ),
                        shard_moe_tensor(
                            _packed_to_uint8(e["up_proj.weight"]), 0
                        ),
                    ],
                    dim=0,
                )
                for e in local_experts
            ]
            if _fused_shared_flatmm or _fused_shared_ck:
                w13_parts.append(
                    torch.cat(
                        [
                            shard_moe_tensor(
                                _packed_to_uint8(shared["gate_proj.weight"]), 0
                            ),
                            shard_moe_tensor(
                                _packed_to_uint8(shared["up_proj.weight"]), 0
                            ),
                        ],
                        dim=0,
                    )
                )
            w13 = torch.stack(w13_parts, dim=0)
            if _use_flatmm_moe:
                w13 = _preshuffle_flatmm_mxfp4_weight(
                    w13.cuda(), _hidden, gate_up=True
                )
            elif _use_ck_moe:
                w13 = _preshuffle_mxfp4_weight(w13.cuda().reshape(-1, _hidden // 2), _hidden).reshape(w13.shape)
            else:
                w13 = w13.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.experts."
                    f"w13_weight{_routed_layout_suffix}"
                ),
                w13,
            )

            w13_scale_parts = [
                torch.cat(
                    [
                        shard_moe_tensor(
                            _scale_to_uint8(e["gate_proj.weight_scale"]), 0
                        ),
                        shard_moe_tensor(
                            _scale_to_uint8(e["up_proj.weight_scale"]), 0
                        ),
                    ],
                    dim=0,
                )
                for e in local_experts
            ]
            if _fused_shared_flatmm or _fused_shared_ck:
                w13_scale_parts.append(
                    torch.cat(
                        [
                            shard_moe_tensor(
                                _scale_to_uint8(shared["gate_proj.weight_scale"]), 0
                            ),
                            shard_moe_tensor(
                                _scale_to_uint8(shared["up_proj.weight_scale"]), 0
                            ),
                        ],
                        dim=0,
                    )
                )
            w13_scale = torch.stack(w13_scale_parts, dim=0)
            if _use_flatmm_moe:
                w13_scale = _preshuffle_flatmm_mxfp4_scale(
                    w13_scale.cuda(), gate_up=True
                )
            elif _use_ck_moe:
                w13_scale = _preshuffle_mxfp4_scale(w13_scale.cuda().reshape(-1, _hidden // 32), _hidden // 32).reshape(
                    w13_scale.shape)
            else:
                w13_scale = w13_scale.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.experts."
                    f"w13_weight_scale{_routed_layout_suffix}"
                ),
                w13_scale,
            )

            w2_parts = [
                shard_moe_tensor(_packed_to_uint8(e["down_proj.weight"]), 1)
                for e in local_experts
            ]
            if _fused_shared_flatmm or _fused_shared_ck:
                w2_parts.append(
                    shard_moe_tensor(
                        _packed_to_uint8(shared["down_proj.weight"]), 1
                    )
                )
            w2 = torch.stack(w2_parts, dim=0)
            if _use_flatmm_moe:
                w2 = _preshuffle_flatmm_mxfp4_weight(
                    w2.cuda(), _local_inter, gate_up=False
                )
            elif _use_ck_moe:
                w2 = _preshuffle_mxfp4_weight(
                    w2.cuda().reshape(-1, _local_inter // 2), _local_inter
                ).reshape(w2.shape)
            else:
                w2 = w2.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.experts."
                    f"w2_weight{_routed_layout_suffix}"
                ),
                w2,
            )

            w2_scale_parts = [
                shard_moe_tensor(
                    _scale_to_uint8(e["down_proj.weight_scale"]), 1
                )
                for e in local_experts
            ]
            if _fused_shared_flatmm or _fused_shared_ck:
                w2_scale_parts.append(
                    shard_moe_tensor(
                        _scale_to_uint8(shared["down_proj.weight_scale"]), 1
                    )
                )
            w2_scale = torch.stack(w2_scale_parts, dim=0)
            if _use_flatmm_moe:
                w2_scale = _preshuffle_flatmm_mxfp4_scale(
                    w2_scale.cuda(), gate_up=False
                )
            elif _use_ck_moe:
                w2_scale = _preshuffle_mxfp4_scale(w2_scale.cuda().reshape(-1, _local_inter // 32), _local_inter // 32).reshape(
                    w2_scale.shape)
            else:
                w2_scale = w2_scale.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.experts."
                    f"w2_weight_scale{_routed_layout_suffix}"
                ),
                w2_scale,
            )

            # Local expert mask: used by moe_sorting to filter/remap experts in
            # EP mode. A fused shared route is appended after all global routed
            # expert IDs. Enable it on one EP rank only so the compiled EP
            # all-reduce does not duplicate the shared-expert contribution.
            mask_name = convert_name(
                f"model.layers.{layer_id}.mlp.experts.local_expert_mask"
            )
            if expected_constant_names is None or mask_name in expected_constant_names:
                mask_experts = self._n_routed_experts + int(
                    _fused_shared_flatmm or _fused_shared_ck
                )
                local_mask = torch.zeros(
                    (mask_experts,), dtype=torch.int32
                )
                local_mask[local_expert_ids] = 1
                if (_fused_shared_flatmm or _fused_shared_ck) and ep_rank == 0:
                    local_mask[self._n_routed_experts] = 1
                params_paiton[mask_name] = local_mask.cuda()

        # ---- Pack shared MXFP4 experts (1-expert FusedMxfp4MoE). ---------- #
        # Sparse-layer shared experts share the same MXFP4 layout as routed
        # experts; the compiled model instantiates them as a 1-expert
        # FusedMxfp4MoE (num_experts=1, topk=1) and the loader binds a single
        # leading "expert 0" slice to that buffer.
        for layer_id, shared in enumerate(layers_shared_experts):
            if not shared:
                continue
            if _fused_shared_flatmm:
                continue
            w13 = torch.cat(
                [
                    shard_moe_tensor(
                        _packed_to_uint8(shared["gate_proj.weight"]), 0
                    ),
                    shard_moe_tensor(
                        _packed_to_uint8(shared["up_proj.weight"]), 0
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            if _shared_use_ck_moe:
                w13 = _preshuffle_mxfp4_weight(w13.cuda().reshape(-1, _hidden // 2), _hidden).reshape(w13.shape)
            else:
                w13 = w13.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.shared_experts.w13_weight"
                ),
                w13,
            )
            w13_scale = torch.cat(
                [
                    shard_moe_tensor(
                        _scale_to_uint8(shared["gate_proj.weight_scale"]), 0
                    ),
                    shard_moe_tensor(
                        _scale_to_uint8(shared["up_proj.weight_scale"]), 0
                    ),
                ],
                dim=0,
            ).unsqueeze(0)
            if _shared_use_ck_moe:
                w13_scale = _preshuffle_mxfp4_scale(w13_scale.cuda().reshape(-1, _hidden // 32), _hidden // 32).reshape(
                    w13_scale.shape)
            else:
                w13_scale = w13_scale.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.shared_experts.w13_weight_scale"
                ),
                w13_scale,
            )
            w2 = shard_moe_tensor(
                _packed_to_uint8(shared["down_proj.weight"]), 1
            ).unsqueeze(0)
            if _shared_use_ck_moe:
                local_shared_inter = _shared_inter // moe_tp_size
                w2 = _preshuffle_mxfp4_weight(
                    w2.cuda().reshape(-1, local_shared_inter // 2),
                    local_shared_inter,
                ).reshape(w2.shape)
            else:
                w2 = w2.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.shared_experts.w2_weight"
                ),
                w2,
            )
            w2_scale = shard_moe_tensor(
                _scale_to_uint8(shared["down_proj.weight_scale"]), 1
            ).unsqueeze(0)
            if _shared_use_ck_moe:
                local_shared_inter = _shared_inter // moe_tp_size
                w2_scale = _preshuffle_mxfp4_scale(
                    w2_scale.cuda().reshape(-1, local_shared_inter // 32),
                    local_shared_inter // 32,
                ).reshape(w2_scale.shape)
            else:
                w2_scale = w2_scale.cuda()
            maybe_emit(
                convert_name(
                    f"model.layers.{layer_id}.mlp.shared_experts.w2_weight_scale"
                ),
                w2_scale,
            )

        # FP8 E4M3 latent caches (`--kv-cache-dtype fp8_e4m3`): the compiled
        # artifact expects per-layer k_scale/v_scale constants that have no
        # checkpoint source. The fused RoPE write path applies them with the
        # implicit scale-1.0 convention (bounded RMSNorm/rope outputs), so
        # synthesize ones for any expected scale constant.
        if expected_constant_names is not None:
            scale_re = re.compile(r"layers_(\d+)_self_attn_([kv])_scale$")
            for name in sorted(expected_constant_names):
                if name in params_paiton:
                    continue
                if scale_re.match(name):
                    params_paiton[name] = torch.ones(
                        1, dtype=torch.float32, device="cuda"
                    )

        return params_paiton

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        import os

        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            torch.cuda.set_device(int(local_rank_env))

        # Ingest the checkpoint stream while tracking duplicate tensor names.
        # The plan requires the loader to reject duplicate source tensors: the
        # wk_weights_proj packing in map_pt_params consumes wk.weight and
        # weights_proj.weight and relies on each appearing at most once. A plain
        # dict comprehension would silently keep the last value for a repeated
        # name, hiding the corruption before the missing/shape validation runs,
        # so detect duplicates here before they are overwritten.
        pt_params: Dict[str, Tensor] = {}
        duplicate_names: Set[str] = set()
        for name, tensor in weights:
            if name in pt_params:
                duplicate_names.add(name)
            pt_params[name] = tensor.detach().cpu()
        if duplicate_names:
            raise ValueError(
                "Duplicate checkpoint tensor name(s) supplied to "
                "load_weights(): "
                + ", ".join(sorted(duplicate_names))
                + ". Each tensor must appear at most once in the checkpoint "
                "stream; the wk_weights_proj packing cannot disambiguate a "
                "repeated source weight."
            )

        expected_all = set(
            self.model.get_constant_names(unbound_constants_only=False)
        )
        mapped = self.map_pt_params(
            pt_params,
            expected_constant_names=expected_all,
        )

        missing = sorted(expected_all - set(mapped.keys()))
        extra = sorted(set(mapped.keys()) - expected_all)
        if missing:
            raise RuntimeError(
                "Paiton GLM constants mismatch: did not provide values for some "
                "expected constants during load_weights(). "
                f"(mapped={len(mapped)}, expected={len(expected_all)}). "
                "First 50 missing:\n- " + "\n- ".join(missing[:50])
            )
        if extra:
            raise RuntimeError(
                "Paiton GLM constants mismatch: produced constant names that the "
                "compiled artifact does not expect during load_weights(). "
                f"(mapped={len(mapped)}, expected={len(expected_all)}). "
                "First 50 extra:\n- " + "\n- ".join(extra[:50])
            )

        self.model.set_many_constants_with_tensors(mapped)
        return set()
