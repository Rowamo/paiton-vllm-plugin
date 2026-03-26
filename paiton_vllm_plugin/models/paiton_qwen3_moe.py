import re
import os
from pathlib import Path
import torch
from torch import nn, Tensor
from typing import Dict, Iterable, Set, Tuple, List, Optional

from vllm.config import VllmConfig, CUDAGraphMode
from vllm.sequence import IntermediateTensors
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.distributed import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import get_tp_group
from vllm.distributed.parallel_state import get_ep_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.platforms import current_platform

from paiton_vllm_plugin.models.artifact_resolver import resolve_artifact_dir
from paiton_vllm_plugin.models.model_path import resolve_model_so_path
from paiton_vllm_plugin.runtime.core import (
    Model,
    PData,
    torch_dtype_to_string,
    torch_to_paiton_data,
)
from paiton_vllm_plugin.vllm_compat import Attention, AttentionType

class PaitonQwen3MoeForCausalLM(nn.Module):
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

        # Ensure this worker uses the correct device before we allocate any CUDA
        # tensors (constants, masks, scales, etc.). If constants are allocated on
        # the wrong device, the runtime may treat them as unset and later crash.
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            torch.cuda.set_device(int(local_rank_env))

        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.parallel_config = vllm_config.parallel_config

        # Expert-parallel (EP) size is used by the compiled graph to enable EP
        # collectives inside MoE blocks (e.g., the EP all-reduce of expert
        # contributions). vLLM tracks EP via its own process groups; make sure
        # the compiled runtime sees a consistent EP_SIZE.
        try:
            ep_group = get_ep_group()
            os.environ.setdefault("EP_SIZE", str(ep_group.world_size))
            os.environ.setdefault("EP_RANK", str(ep_group.rank_in_group))
            # Provide the *global* torch.distributed rank to the compiled runtime.
            #
            # The generated runtime derives:
            #   comm_rank     = PAITON_RANK % tp_size
            #   comm_group_id = PAITON_RANK / tp_size
            #
            # Using the real global rank makes those computations consistent with
            # how vLLM launches workers:
            # - If global_world_size == tp_size (common for EP=TP setups),
            #   comm_group_id is 0 for all ranks, so everyone agrees on the same
            #   mqueue paths.
            # - If global_world_size > tp_size, ranks naturally partition into
            #   groups of size tp_size and get distinct comm_group_id values.
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                os.environ["PAITON_RANK"] = str(dist.get_rank())
            else:
                os.environ["PAITON_RANK"] = str(self.tp_rank)
        except Exception:
            # Best-effort only; EP may be disabled or group may be unavailable.
            os.environ["PAITON_RANK"] = str(self.tp_rank)

        self.config = vllm_config.model_config.hf_config
        model_ref = vllm_config.model_config.model
        self.model_path = resolve_artifact_dir(
            model_ref,
            revision=vllm_config.model_config.revision,
            token=vllm_config.model_config.hf_token,
            download_dir=vllm_config.load_config.download_dir,
        )
        max_input_tokens = getattr(getattr(vllm_config, "scheduler_config", None),
                                   "max_num_batched_tokens", None)
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

        # ---- Ensure a clean mqueue namespace for the compiled runtime ------
        # The compiled .so uses POSIX message queues to exchange NCCL unique
        # IDs between ranks.  Stale queues left by a previous crashed run
        # create a race condition: a non-root rank may open the stale queue
        # (which already exists) before the root rank can unlink and recreate
        # it, causing the two ranks to communicate through different queue
        # objects — an unrecoverable deadlock.
        #
        # Fix: rank 0 removes all paiton mqueues *from Python* before any
        # rank loads the .so, and a barrier ensures the cleanup is visible.
        import torch.distributed as _dist
        if _dist.is_available() and _dist.is_initialized() and _dist.get_world_size() > 1:
            if _dist.get_rank() == 0:
                import pathlib as _pathlib
                _mq_dir = _pathlib.Path("/dev/mqueue")
                if _mq_dir.exists():
                    for _mq_file in _mq_dir.glob("paiton_mq_*"):
                        try:
                            _mq_file.unlink()
                        except OSError:
                            pass
            _dist.barrier()

        self.model = Model(model_so_path)
        self.dtype = self.config.torch_dtype
        # vLLM represents fp8 KV cache as a uint8 buffer and views it as the
        # platform-appropriate fp8 dtype inside attention kernels.
        # Make sure we view the KV cache with the same fp8 dtype (fn vs fnuz),
        # otherwise decode can become degenerate (e.g. repeated tokens) even if
        # logits remain finite.
        cache_dtype_str = vllm_config.cache_config.cache_dtype
        if cache_dtype_str == "auto":
            self.cache_dtype = self.dtype
        elif cache_dtype_str.startswith("fp8"):
            self.cache_dtype = current_platform.fp8_dtype()
        else:
            # Fallback: treat as model dtype. (We can extend this if we add int8 KV.)
            self.cache_dtype = self.dtype
        self.quantized = vllm_config.quant_config is not None
        self.dynamic_quant = self.quantized and getattr(
            vllm_config.quant_config, "activation_scheme", None
        ) == "dynamic"
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
        # cublasLt matmul in some generated kernels is currently unstable under
        # FULL graph capture on CUDA. Keep PIECEWISE capture enabled by default
        # for Paiton models unless explicitly overridden.
        if os.getenv("PAITON_ENABLE_FULL_CUDAGRAPH", "0") != "1":
            mode = getattr(self.compilation_config, "cudagraph_mode", None)
            if mode in (CUDAGraphMode.FULL, CUDAGraphMode.FULL_AND_PIECEWISE):
                self.compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
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
        # Debug flags: cached once at init to avoid per-forward getenv() overhead.
        self._dbg_logits = os.getenv("PAITON_DEBUG_LOGITS", "0") == "1"
        self._dbg_kvcache = os.getenv("PAITON_DEBUG_KVCACHE", "0") == "1"
        self._dbg_decode_meta = os.getenv("PAITON_DEBUG_DECODE_META", "0") == "1"
        self._paiton_debug_step = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # For debugging NaNs, start from a known value to distinguish
        # "not written" from "written as NaN".
        if self._dbg_logits:
            output = torch.zeros(
                [input_ids.shape[0], self.config.vocab_size],
                dtype=torch.float32,
                device="cuda",
            )
        else:
            output = torch.empty(
                [input_ids.shape[0], self.config.vocab_size],
                dtype=torch.float32,
                device="cuda",
            )
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not attn_metadata:
            return output

        attn_metadata = attn_metadata['0'] # It's the same for all layers, so just use the first one
        max_query_len = attn_metadata.max_query_len
        max_seq_len = attn_metadata.max_seq_len

        # IMPORTANT: match the compiled model's expected input dtypes.
        # From the generated C++ interface for this artifact:
        # - input_ids: int32_t*
        # - position_ids: int64_t*
        # - slot_mapping: int64_t*
        # - context_lengths / query_start_locations / block_tables: int32_t*
        #
        # If we pass int64 tensors where the model expects int32 (or vice versa),
        # the runtime will read the wrong memory stride (e.g., treating int64
        # data as int32), which can yield degenerate generation like repeating
        # a single token.
        input_ids_i32 = input_ids.to(dtype=torch.int32, copy=False).contiguous()
        position_ids_i64 = positions.to(dtype=torch.int64, copy=False).contiguous()
        slot_mapping_i64 = attn_metadata.slot_mapping.to(dtype=torch.int64, copy=False).contiguous()
        query_start_loc_i32 = attn_metadata.query_start_loc.to(dtype=torch.int32, copy=False).contiguous()
        seq_lens_i32 = attn_metadata.seq_lens.to(dtype=torch.int32, copy=False).contiguous()
        block_table_i32 = attn_metadata.block_table.to(dtype=torch.int32, copy=False).contiguous()
        # NOTE: On this ROCm/PyTorch build, *all* zero-numel tensors report
        # `data_ptr()==0` even if they are views backed by real storage.
        #
        # The compiled model uses `max_query_len` / `max_seq_len` only for shape
        # metadata (shape=(N,0)). To avoid passing a null pointer into the runtime,
        # we bypass `run_with_tensors()` and pass explicit `PData` with:
        # - a real, non-null backing pointer (1 element)
        # - the expected (N,0) shape
        max_query_len_backing = torch.empty([1], dtype=torch.int32, device="cuda")
        max_seq_len_backing = torch.empty([1], dtype=torch.int32, device="cuda")
        # The paged-attention kernels consume the *value* of max_query_len/max_seq_len.
        # Ensure the backing scalars are initialized with the runtime values.
        max_query_len_backing.fill_(int(max_query_len))
        max_seq_len_backing.fill_(int(max_seq_len))

        inputs = {
            "input_ids": torch_to_paiton_data(input_ids_i32),
            "position_ids": torch_to_paiton_data(position_ids_i64),
            "slot_mapping": torch_to_paiton_data(slot_mapping_i64),
            "query_start_locations": torch_to_paiton_data(query_start_loc_i32),
            "context_lengths": torch_to_paiton_data(seq_lens_i32),
            "block_tables": torch_to_paiton_data(block_table_i32),
            "max_query_len": PData(
                max_query_len_backing.data_ptr(),
                [max_query_len, 0],
                torch_dtype_to_string(torch.int32),
            ),
            "max_seq_len": PData(
                max_seq_len_backing.data_ptr(),
                [max_seq_len, 0],
                torch_dtype_to_string(torch.int32),
            ),
        }
        kv_cache0_for_debug = None
        for i in range(self.num_layers):
            idx = f"kv_cache_{i}"
            kv_cache_i = self.compilation_config.static_forward_context[str(i)].kv_cache[0]
            if i == 0:
                kv_cache0_for_debug = kv_cache_i
            inputs[idx] = torch_to_paiton_data(
                kv_cache_i.view(self.cache_dtype)
            )

        outputs = {"logits": torch_to_paiton_data(output)}

        # Optional KV-cache debug: verify tensor layout and that the current slot
        # actually changes in memory after the compiled forward.
        if self._dbg_kvcache and self.tp_rank == 0 and kv_cache0_for_debug is not None:
            with torch.no_grad():
                if not getattr(self, "_paiton_debug_printed_kvcache_layout", False):
                    self._paiton_debug_printed_kvcache_layout = True
                    print(
                        "[paiton][kvcache] layer0 "
                        f"shape={tuple(kv_cache0_for_debug.shape)} dtype={kv_cache0_for_debug.dtype} "
                        f"stride={kv_cache0_for_debug.stride()} is_contig={kv_cache0_for_debug.is_contiguous()}"
                    )
                if slot_mapping_i64.numel() > 0 and kv_cache0_for_debug.dim() == 5:
                    slot = int(slot_mapping_i64.view(-1)[0].item())
                    if slot >= 0:
                        # IMPORTANT: our compiled kernels treat kv_cache as a flat
                        # buffer with packed K and V layouts. Index the underlying
                        # storage directly (not via tensor indexing) to validate
                        # whether the compiled forward writes the expected bytes.
                        # We sample K and V bytes for (kv_head=0, head_offset=0..7)
                        # at this slot.
                        # vLLM's raw KV-cache layout is expected to be
                        # [2, num_blocks, block_size, num_kv_heads, head_dim] for fp8.
                        # Print the shape above to confirm.
                        num_blocks = int(kv_cache0_for_debug.shape[1])
                        block_size = int(kv_cache0_for_debug.shape[2])
                        num_kv_heads = int(kv_cache0_for_debug.shape[3])
                        head_dim = int(kv_cache0_for_debug.shape[4])
                        block_idx = slot // block_size
                        off = slot % block_size
                        # K layout params (fp8 => 1 byte per elem).
                        x = block_size
                        k_elems = num_blocks * block_size * num_kv_heads * head_dim
                        # Base offsets in bytes for head_offset 0..7 (x_offset 0..7).
                        k_base = (
                            block_idx * num_kv_heads * (head_dim // x) * block_size * x
                            + 0 * (head_dim // x) * block_size * x
                            + 0 * block_size * x
                            + off * x
                        )
                        v_base = k_elems + (
                            block_idx * num_kv_heads * head_dim * block_size
                            + 0 * head_dim * block_size
                            + 0 * block_size
                            + off
                        )
                        flat = kv_cache0_for_debug.view(torch.uint8).reshape(-1)
                        k_bytes = flat[k_base : k_base + 8].tolist()
                        v_bytes = flat[v_base : v_base + 8 * block_size : block_size].tolist()
                        print(
                            f"[paiton][kvcache] slot={slot} block={block_idx} off={off} "
                            f"k_base={k_base} v_base={v_base} "
                            f"K_bytes_before={k_bytes} V_bytes_before={v_bytes}"
                        )

        # Optional decode metadata debug to diagnose "repeats same token" issues.
        # This prints a small summary of the inputs that drive KV-cache updates.
        if self._dbg_decode_meta and self.tp_rank == 0:
            with torch.no_grad():
                def _head(t: torch.Tensor, n: int = 8):
                    # Handle empty tensors safely.
                    if t.numel() == 0:
                        return []
                    return t.flatten()[:n].tolist()

                # Key checks:
                # - positions should increase during decode
                # - slot_mapping for active tokens should be >=0 (otherwise KV writes are skipped)
                # - seq_lens should increase during decode
                sm = slot_mapping_i64.flatten()
                sm_min = int(sm.min().item()) if sm.numel() else 0
                sm_max = int(sm.max().item()) if sm.numel() else 0
                print(
                    "[paiton][decode_meta] "
                    f"num_tokens={int(input_ids_i32.shape[0])} "
                    f"max_query_len={int(max_query_len)} max_seq_len={int(max_seq_len)} "
                    f"slot_mapping(min,max)=({sm_min},{sm_max}) "
                    f"input_ids[:8]={_head(input_ids_i32)} "
                    f"positions[:8]={_head(position_ids_i64)} "
                    f"slot_mapping[:8]={_head(slot_mapping_i64)} "
                    f"seq_lens[:8]={_head(seq_lens_i32)} "
                    f"query_start_loc[:8]={_head(query_start_loc_i32)}"
                )
                if block_table_i32.numel() > 0:
                    bt0 = block_table_i32[0]
                    print(f"[paiton][decode_meta] block_table[0,:8]={bt0[:8].tolist()}")

        # Run on the current PyTorch stream to match vLLM's execution order.
        # IMPORTANT: keep sync=True for correctness, since the Paiton runtime
        # launches kernels outside of PyTorch's awareness; without synchronization
        # (or explicit record_stream bookkeeping), PyTorch may reuse/freeze input
        # buffers early and we can hit GPU memory faults.
        if self._dbg_logits and self.tp_rank == 0:
            self._paiton_debug_step += 1
            with torch.no_grad():
                sample = output[0]
                tok0 = int(input_ids_i32.view(-1)[0].item()) if input_ids_i32.numel() else -1
                pos0 = int(position_ids_i64.view(-1)[0].item()) if position_ids_i64.numel() else -1
                sm0 = int(slot_mapping_i64.view(-1)[0].item()) if slot_mapping_i64.numel() else -1
                sl0 = int(seq_lens_i32.view(-1)[0].item()) if seq_lens_i32.numel() else -1
                print(
                    "[paiton] logits pre-run "
                    f"step={self._paiton_debug_step} tok0={tok0} pos0={pos0} slot0={sm0} seqlen0={sl0} "
                    f"min={sample.min().item():.6f} max={sample.max().item():.6f} "
                    f"mean={sample.mean().item():.6f} finite={int(torch.isfinite(sample).sum().item())}/{int(sample.numel())}"
                )
        stream_ptr = torch.cuda.current_stream().cuda_stream
        self.model.run(inputs, outputs, stream_ptr=stream_ptr, sync=True)

        # Quick NaN audit of KV cache layer 0 after model.run() to
        # narrow down whether NaN originates in the attention path.
        if self._dbg_logits and self.tp_rank == 0 and kv_cache0_for_debug is not None:
            with torch.no_grad():
                flat_i8 = kv_cache0_for_debug.view(torch.int8).reshape(-1)
                nan_bytes = int((flat_i8 == -128).sum().item())
                total = int(flat_i8.numel())
                nonzero = int((flat_i8 != 0).sum().item())
                print(
                    f"[paiton][kv-nan-check] layer0 total_bytes={total} "
                    f"nonzero={nonzero} fnuz_nan_bytes={nan_bytes}"
                )
        if self._dbg_kvcache and self.tp_rank == 0 and kv_cache0_for_debug is not None:
            with torch.no_grad():
                if slot_mapping_i64.numel() > 0 and kv_cache0_for_debug.dim() == 5:
                    slot = int(slot_mapping_i64.view(-1)[0].item())
                    if slot >= 0:
                        num_blocks = int(kv_cache0_for_debug.shape[1])
                        block_size = int(kv_cache0_for_debug.shape[2])
                        num_kv_heads = int(kv_cache0_for_debug.shape[3])
                        head_dim = int(kv_cache0_for_debug.shape[4])
                        block_idx = slot // block_size
                        off = slot % block_size
                        x = block_size
                        k_elems = num_blocks * block_size * num_kv_heads * head_dim
                        k_base = (
                            block_idx * num_kv_heads * (head_dim // x) * block_size * x
                            + 0 * (head_dim // x) * block_size * x
                            + 0 * block_size * x
                            + off * x
                        )
                        v_base = k_elems + (
                            block_idx * num_kv_heads * head_dim * block_size
                            + 0 * head_dim * block_size
                            + 0 * block_size
                            + off
                        )
                        flat = kv_cache0_for_debug.view(torch.uint8).reshape(-1)
                        k_bytes = flat[k_base : k_base + 8].tolist()
                        v_bytes = flat[v_base : v_base + 8 * block_size : block_size].tolist()
                        print(
                            f"[paiton][kvcache] slot={slot} block={block_idx} off={off} "
                            f"k_base={k_base} v_base={v_base} "
                            f"K_bytes_after={k_bytes} V_bytes_after={v_bytes}"
                        )
        # self.model.profile(inputs, outputs, num_iters=10, filename=f"vllm_profile_{attn_metadata.max_decode_seq_len}.json")
        # The compiled LM head gathers logits to TP rank 0 only. Broadcast
        # the result so any rank performing sampling sees valid logits.
        if self.tp_size > 1:
            get_tp_group().broadcast(output, src=0)
        if self._dbg_logits and self.tp_rank == 0:
            with torch.no_grad():
                sample = output[0]
                topk = torch.topk(sample, k=5)
                finite = torch.isfinite(sample)
                num_finite = int(finite.sum().item())
                numel = int(sample.numel())
                tok0 = int(input_ids_i32.view(-1)[0].item()) if input_ids_i32.numel() else -1
                pos0 = int(position_ids_i64.view(-1)[0].item()) if position_ids_i64.numel() else -1
                sm0 = int(slot_mapping_i64.view(-1)[0].item()) if slot_mapping_i64.numel() else -1
                sl0 = int(seq_lens_i32.view(-1)[0].item()) if seq_lens_i32.numel() else -1
                print(
                    "[paiton] logits stats "
                    f"step={self._paiton_debug_step} tok0={tok0} pos0={pos0} slot0={sm0} seqlen0={sl0} "
                    f"min={sample.min().item():.6f} "
                    f"max={sample.max().item():.6f} "
                    f"mean={sample.mean().item():.6f} "
                    f"finite={num_finite}/{numel} "
                    f"topk_ids={topk.indices.tolist()} "
                    f"topk_vals={[float(v) for v in topk.values.tolist()]}"
                )
        return output

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

    # FIXME: add EP support
    def map_pt_params(
        self,
        pt_params: Dict[str, torch.Tensor],
        expected_constant_names: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
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

        # CK MoE requires pre-shuffling the weights for faster GEMM.
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

        def fix_fp8(w: torch.Tensor) -> torch.Tensor:
            """Ensure ROCm-compatible FP8 dtype (e4m3fnuz) and move to GPU."""
            if w.dtype == torch.float8_e4m3fn:
                w_int8 = w.view(torch.int8).cuda()
                # e4m3fn `-0` is `NaN` in e4m3fnuz; map it to `0`.
                w_int8[w_int8 == -128] = 0
                w = w_int8.view(torch.float8_e4m3fnuz)
            return w.cuda()

        def compress_blockscale_if_needed(p: torch.Tensor) -> torch.Tensor:
            """If p is per-element [O,I] and divisible by 128x128, compress to [O/128,I/128]."""
            if p.dim() == 2 and (p.shape[0] % 128 == 0) and (p.shape[1] % 128 == 0):
                ob, ib = p.shape[0] // 128, p.shape[1] // 128
                return p.view(ob, 128, ib, 128)[:, 0, :, 0]
            return p

        # On ROCm gfx94, FP8 uses the "fnuz" variant and vLLM doubles certain
        # scaling factors to match the representable range. Mirror that behavior
        # here; on non-fnuz platforms keep scales unchanged.
        fp8_scale_factor: float = 2.0 if current_platform.is_fp8_fnuz() else 1.0

        params_paiton: Dict[str, torch.Tensor] = {}

        # EP semantics (mirrors vLLM): experts are sharded across an EP group.
        ep_rank = get_ep_group().rank_in_group
        ep_size = get_ep_group().world_size
        assert self.config.num_experts % ep_size == 0, (
            f"EP world_size must divide num_experts (ep_size={ep_size}, "
            f"num_experts={self.config.num_experts})")
        num_local_experts = self.config.num_experts // ep_size
        placement = getattr(self.parallel_config, "expert_placement_strategy", "linear")
        if placement == "round_robin":
            local_expert_ids = list(range(ep_rank, self.config.num_experts, ep_size))
        else:
            expert_start = ep_rank * num_local_experts
            expert_end = expert_start + num_local_experts
            local_expert_ids = list(range(expert_start, expert_end))
        assert len(local_expert_ids) == num_local_experts, (
            f"Unexpected local expert count: {len(local_expert_ids)} vs "
            f"{num_local_experts} (placement={placement}, ep_rank={ep_rank}, "
            f"ep_size={ep_size})")

        # For each layer, save the experts separately for subsequent processing
        layers_experts: List[List[Dict[str, Tensor]]] = [
            [{} for i in range(self.config.num_experts)]
            for j in range(self.config.num_hidden_layers)
        ]
        expert_regex = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.+)")

        for name, param in pt_params.items():
            if "mlp.experts" in name:
                # Handle MoE separately:
                ret = expert_regex.match(name)
                layer_id = int(ret[1])
                expert_id = int(ret[2])
                weight_name = ret[3]
                layers_experts[layer_id][expert_id][weight_name] = param
                continue

            fused_module, unfused_module, full_fused_name = find_fusion(name)
            if fused_module is not None:
                if convert_name(full_fused_name) in params_paiton:
                    continue

                modules_to_fuse = self.packed_modules_mapping[fused_module]
                params_names = [name.replace(unfused_module, module) for module in modules_to_fuse]
                params = [pt_params[param_name] for param_name in params_names]
                name = full_fused_name

                if name.endswith(".weight"):
                    if fused_module == "qkv_proj":
                        head_dim = self.config.head_dim
                        full_q = self.config.num_attention_heads * head_dim
                        full_kv = self.config.num_key_value_heads * head_dim
                        q_is_full = params[0].shape[0] == full_q
                        k_is_full = params[1].shape[0] == full_kv
                        v_is_full = params[2].shape[0] == full_kv
                        split_qkv = q_is_full and k_is_full and v_is_full
                        if split_qkv:
                            params = [self.get_rank_weight(param, dim=0).cuda() for param in params]
                        else:
                            params = [param.cuda() for param in params]
                    else:
                        # gate_up is column parallel, split across dim=0
                        params = [self.get_rank_weight(param, dim=0).cuda() for param in params]

                    # Fuse the weights
                    paiton_param = torch.cat(params, dim=0)
                    if self.dynamic_quant:
                        paiton_param = shuffle_weight(paiton_param)
                elif name.endswith(".bias"):
                    bias = torch.cat(params, dim=0)
                    if fused_module == "qkv_proj":
                        head_dim = self.config.head_dim
                        full_q = self.config.num_attention_heads * head_dim
                        full_kv = self.config.num_key_value_heads * head_dim
                        q_is_full = params[0].shape[0] == full_q
                        k_is_full = params[1].shape[0] == full_kv
                        v_is_full = params[2].shape[0] == full_kv
                        split_qkv = q_is_full and k_is_full and v_is_full
                        paiton_param = self.get_rank_bias(bias) if split_qkv else bias
                    else:
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
                    # Fuse inverse scales. Compiled models expect block scales
                    # (weight_block_size=128x128), so compress if needed.
                    ps = []
                    for param_name in params_names:
                        p = compress_blockscale_if_needed(pt_params[param_name])
                        if fused_module == "qkv_proj":
                            # Scales are block-wise: [O/128, I/128]. Determine whether we
                            # need to TP-shard based on the *scale* shapes.
                            head_dim = self.config.head_dim
                            full_q = self.config.num_attention_heads * head_dim
                            full_kv = self.config.num_key_value_heads * head_dim
                            full_q_scale0 = full_q // 128
                            full_kv_scale0 = full_kv // 128
                            q_is_full = params[0].shape[0] == full_q_scale0
                            k_is_full = params[1].shape[0] == full_kv_scale0
                            v_is_full = params[2].shape[0] == full_kv_scale0
                            split_qkv = q_is_full and k_is_full and v_is_full
                            ps.append(self.get_rank_weight(p, dim=0).cuda() if split_qkv else p.cuda())
                        else:
                            ps.append(self.get_rank_weight(p, dim=0).cuda())
                    paiton_param = (torch.cat(ps, dim=0) * 2).contiguous().float()
            elif name.endswith("down_proj.weight") or name.endswith("o_proj.weight"):
                paiton_param = self.get_rank_weight(param, dim=1) # row parallel, split across dim=1
                if self.dynamic_quant:
                    paiton_param = shuffle_weight(paiton_param)
            elif name.endswith("mlp.gate.weight"):
                # MoE router gate is small and must be replicated across TP ranks.
                # The compiled model expects the full [num_experts, hidden_size] weight.
                paiton_param = param
            elif name.endswith(".bias"):
                paiton_param = self.get_rank_bias(param)
            elif name.endswith("norm.weight"):
                paiton_param = param # we do not split normalization weights across ranks
            elif name.endswith(".input_scale"):
                paiton_param = (param * fp8_scale_factor).float()
            elif name.endswith(".weight_scale_inv"):
                # Compiled models use block-wise inverse scales (typically
                # weight_block_size=(128,128)): [O/128, I/128]. Some checkpoints
                # store full per-element inverse scales [O, I]; compress to
                # one representative value per block.
                p = compress_blockscale_if_needed(param)
                # RowParallelLinear (o_proj, down_proj) shards the weight along
                # dim=1 (input dimension).  The block-wise inverse scale must be
                # split along the same axis so the per-block scale values stay
                # aligned with their corresponding weight blocks.
                # ColumnParallelLinear weights shard along dim=0 (output dimension).
                if "o_proj" in name or "down_proj" in name:
                    split_dim = 1
                else:
                    split_dim = 0
                paiton_param = (self.get_rank_weight(p, dim=split_dim) * fp8_scale_factor).float()
            elif name.endswith(".weight_scale"):
                paiton_param = (param * fp8_scale_factor).float()
            elif name.endswith(".kv_scale"):
                # this is an "old" way of doing things now, so k and v scales must be separate
                k_scale_name = name.replace("kv_scale", "k_scale")
                v_scale_name = name.replace("kv_scale", "v_scale")
                params_paiton[convert_name(k_scale_name)] = (param * fp8_scale_factor).cuda()
                params_paiton[convert_name(v_scale_name)] = (param * fp8_scale_factor).cuda()
                continue
            else:
                paiton_param = self.get_rank_weight(param, dim=0)

            out_name = convert_name(name)
            # Some checkpoints / loaders may materialize scale tensors in FP8.
            # The compiled Paiton model expects scales as float tensors.
            if out_name.endswith(("_weight_scale_inv", "_w1_scale", "_w2_scale")):
                paiton_param = paiton_param.float()

            params_paiton[out_name] = fix_fp8(paiton_param)

        for layer_id, experts in enumerate(layers_experts):
            # Shard experts by EP rank: [num_experts, ...] -> [num_local_experts, ...]
            local_experts = [experts[i] for i in local_expert_ids]

            w1s = [
                torch.cat((e['gate_proj.weight'], e['up_proj.weight']), dim=0)
                for e in local_experts
            ]
            w1 = torch.stack(w1s, dim=0)
            w1 = shuffle_weight(w1, layout=(16, 16))
            params_paiton[convert_name(f"model.layers.{layer_id}.mlp.experts.w1_weight")] = fix_fp8(w1)

            w2s = [e['down_proj.weight'].cuda() for e in local_experts]
            w2 = torch.stack(w2s, dim=0)
            w2 = shuffle_weight(w2, layout=(16, 16))
            params_paiton[convert_name(f"model.layers.{layer_id}.mlp.experts.w2_weight")] = fix_fp8(w2)

            # Local expert mask (global size): used by ck_tile moe_sorting to:
            # - skip non-local experts
            # - remap global expert ids -> dense local ids (0..num_local_experts-1)
            local_mask_name = convert_name(
                f"model.layers.{layer_id}.mlp.experts.local_expert_mask"
            )
            # Some compiled artifacts do not declare local_expert_mask constants.
            # Only emit them when the artifact explicitly expects them.
            if (expected_constant_names is None
                    or local_mask_name in expected_constant_names):
                local_mask = torch.zeros((self.config.num_experts,), dtype=torch.int32)
                local_mask[local_expert_ids] = 1
                params_paiton[local_mask_name] = local_mask.cuda()

            # Compiled FP8 MoE expects expert block scales (dynamic scheme).
            # Always provide them when the model is quantized; otherwise the
            # fused MoE kernels may see nullptr scale pointers and crash.
            if self.quantized:
                w1_scales = []
                for e in local_experts:
                    s = torch.cat(
                        (e["gate_proj.weight_scale_inv"], e["up_proj.weight_scale_inv"]),
                        dim=0,
                    )
                    s = compress_blockscale_if_needed(s)
                    w1_scales.append(s)
                w1_scale = (torch.stack(w1_scales, dim=0) * fp8_scale_factor).contiguous().float()
                params_paiton[convert_name(
                    f"model.layers.{layer_id}.mlp.experts.w1_scale")] = w1_scale.cuda()

                w2_scales = [
                    compress_blockscale_if_needed(e["down_proj.weight_scale_inv"])
                    for e in local_experts
                ]
                w2_scale = (torch.stack(w2_scales, dim=0) * fp8_scale_factor).contiguous().float()
                params_paiton[convert_name(
                    f"model.layers.{layer_id}.mlp.experts.w2_scale")] = w2_scale.cuda()

        return params_paiton

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # vLLM may call `compute_logits` on every TP rank during profiling.
        # Returning `None` on non-zero ranks crashes the dummy sampler
        # (`logits.size(0)`), but we also don't want non-zero TP ranks to run
        # the logits post-processing path (the compiled model already handles
        # TP logits behavior).
        return self.logits_processor(None, hidden_states)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # Same safety as in __init__: ensure we allocate constants on the worker's
        # assigned device.
        local_rank_env = os.environ.get("LOCAL_RANK")
        if local_rank_env is not None:
            torch.cuda.set_device(int(local_rank_env))

        # Important: do NOT materialize all weights as CUDA tensors at once.
        # vLLM loaders may hand us CUDA tensors depending on the execution path,
        # and `dict(weights)` would keep the entire checkpoint alive on GPU while
        # we also create packed/shuffled constants (plus temporary buffers like
        # `permute(...).contiguous()`), easily causing OOM.
        #
        # Instead, move weights to CPU first, then perform packing/shuffling on
        # CPU and only upload the final constants to GPU.
        pt_params = {name: tensor.detach().cpu() for name, tensor in weights}
        _dbg_weights = os.getenv("PAITON_DEBUG_WEIGHTS", "0") == "1"
        _dbg_const_stats = os.getenv("PAITON_DEBUG_CONST_STATS", "0") == "1"
        if _dbg_weights and self.tp_rank == 0:
            sample_keys = [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.o_proj.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight",
                "model.layers.0.mlp.experts.0.up_proj.weight",
                "model.layers.0.mlp.experts.0.down_proj.weight",
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
                "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
            ]
            for key in sample_keys:
                if key in pt_params:
                    print(f"[paiton] weight {key} shape={tuple(pt_params[key].shape)} dtype={pt_params[key].dtype}")
        expected_all = set(self.model.get_constant_names(unbound_constants_only=False))
        mapped = self.map_pt_params(
            pt_params,
            expected_constant_names=expected_all,
        )
        # Debug: verify our mapped constant names line up with what the compiled
        # artifact expects. If names don't match, the runtime will leave them
        # unbound and later kernels will crash.
        mapped_names = set(mapped.keys())
        missing_set = expected_all - mapped_names
        extra = sorted(mapped_names - expected_all)

        fp8_scale_factor: float = 2.0 if current_platform.is_fp8_fnuz() else 1.0

        # Qwen3 FP8 builds may expect per-layer KV-cache scales as constants
        # (`layers_X_self_attn_{k,v}_scale`). Some vLLM loaders don't surface
        # these weights explicitly. Synthesize safe defaults if needed.
        if missing_set:
            kv_scale_missing = {
                n
                for n in missing_set
                if n.endswith("_self_attn_k_scale") or n.endswith("_self_attn_v_scale")
            }
            if kv_scale_missing:
                # Default scale of 1.0 in "Paiton scale units" (match fnuz behavior).
                default_scale = torch.tensor([fp8_scale_factor], dtype=torch.float32, device="cuda")
                for name in kv_scale_missing:
                    mapped[name] = default_scale
                mapped_names = set(mapped.keys())
                missing_set = expected_all - mapped_names

        if _dbg_const_stats and self.tp_rank == 0:
            def _stat(name: str):
                if name not in mapped:
                    # Distinguish between "missing from mapping" vs "required by artifact".
                    if name in expected_all:
                        print(f"[paiton] const {name} MISSING (expected by artifact)")
                    else:
                        print(f"[paiton] const {name} missing from mapped (not expected by artifact)")
                    return
                t = mapped[name]
                with torch.no_grad():
                    finite = torch.isfinite(t)
                    num_finite = int(finite.sum().item())
                    numel = int(t.numel())
                    t_f = t.float()
                    # Avoid calling min/max on all-NaN tensors.
                    if num_finite == 0:
                        print(f"[paiton] const {name} shape={tuple(t.shape)} dtype={t.dtype} finite=0/{numel}")
                        return
                    t_finite = t_f[finite]
                    print(
                        f"[paiton] const {name} shape={tuple(t.shape)} dtype={t.dtype} "
                        f"finite={num_finite}/{numel} "
                        f"min={t_finite.min().item():.6g} max={t_finite.max().item():.6g} mean={t_finite.mean().item():.6g}"
                    )

            # Attention FP8 block scales (most likely to cause NaNs if wrong).
            _stat("layers_0_self_attn_qkv_proj_weight_scale_inv")
            _stat("layers_0_self_attn_o_proj_weight_scale_inv")
            _stat("layers_0_self_attn_k_scale")
            _stat("layers_0_self_attn_v_scale")
            # MoE expert FP8 block scales.
            _stat("layers_0_mlp_experts_w1_scale")
            _stat("layers_0_mlp_experts_w2_scale")

            # Comprehensive NaN/Inf audit: check EVERY constant and report
            # only the problematic ones.
            bad_consts = []
            fp8_nan_consts = []
            for cname, ctensor in mapped.items():
                with torch.no_grad():
                    # torch.isfinite is not implemented on float8 dtypes.
                    # For fnuz fp8, raw byte 0x80 (-128 in int8) encodes NaN.
                    if ctensor.dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
                        raw = ctensor.view(torch.int8)
                        nan_count = int((raw == -128).sum().item())
                        if nan_count > 0:
                            fp8_nan_consts.append(
                                f"  {cname}: {nan_count}/{ctensor.numel()} fnuz-NaN bytes")
                    elif ctensor.is_floating_point():
                        nf = int(torch.isfinite(ctensor).sum().item())
                        if nf < ctensor.numel():
                            bad_consts.append(
                                f"  {cname}: shape={tuple(ctensor.shape)} dtype={ctensor.dtype} "
                                f"finite={nf}/{ctensor.numel()}")
            if bad_consts:
                print(f"[paiton] WARNING: {len(bad_consts)} constants have non-finite values:")
                for line in bad_consts:
                    print(line)
            else:
                print(f"[paiton] All {len(mapped)} constants are fully finite.")
            if fp8_nan_consts:
                print(f"[paiton] WARNING: {len(fp8_nan_consts)} FP8 constants have fnuz-NaN bytes:")
                for line in fp8_nan_consts:
                    print(line)
            else:
                print(f"[paiton] No FP8 fnuz-NaN bytes found in any constant.")

        missing = sorted(missing_set)
        if missing:
            raise RuntimeError(
                "Paiton constants mismatch: we did not provide values for some "
                "expected constants during load_weights(). "
                f"(mapped={len(mapped_names)}, expected={len(expected_all)}). "
                "First 50 missing:\n- " + "\n- ".join(missing[:50])
            )
        if extra:
            raise RuntimeError(
                "Paiton constants mismatch: we produced constant names that the "
                "compiled artifact does not expect during load_weights(). "
                f"(mapped={len(mapped_names)}, expected={len(expected_all)}). "
                "First 50 extra:\n- " + "\n- ".join(extra[:50])
            )

        self.model.set_many_constants_with_tensors(mapped)
        return set()
