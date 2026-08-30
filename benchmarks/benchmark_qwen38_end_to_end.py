"""Warm batch-1 Qwen3.8 Qronos INT4 backend benchmark.

Run each backend in a fresh process. Model loading, engine initialization, and
one warmup request per workload are excluded. Prefix caching is disabled and
all requests use token IDs directly, greedy sampling, and an exact forced
output length so Paiton and public vLLM/AITER exercise identical work.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

# ROCm's matching AMD-SMI binding must be loaded before Torch initializes HIP
# in pip-based ROCm environments. Paiton's platform plugin does not require it.
try:
    import amdsmi  # noqa: F401
except ImportError:
    pass

import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


DEFAULT_WORKLOADS = ((128, 1), (1024, 1), (4096, 1), (128, 128), (4096, 128))
RELEVANT_ENV = (
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_LINEAR",
    "VLLM_ROCM_USE_AITER_LINEAR_HIPBMM",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_ROCM_USE_AITER_MOE_SITUV2_A8W4",
    "VLLM_ROCM_USE_AITER_RMSNORM",
    "VLLM_ROCM_USE_AITER_MLA",
    "VLLM_ROCM_USE_AITER_MHA",
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
    "VLLM_ROCM_USE_AITER_TRITON_ROPE",
    "VLLM_ROCM_USE_AITER_FP8BMM",
    "VLLM_ROCM_USE_AITER_FP4BMM",
    "VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION",
    "VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS",
    "VLLM_ROCM_USE_AITER_TRITON_GEMM",
    "VLLM_ROCM_USE_AITER_CUSTOM_AR",
    "VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT",
    "VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_GDN_DECODE_KERNEL",
    "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE",
    "HIP_VISIBLE_DEVICES",
    "VLLM_ATTENTION_BACKEND",
    "VLLM_USE_PAITON_PLATFORM",
)


def _percentiles(values):
    ordered = sorted(values)

    def at(fraction):
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "p10_seconds": at(0.1),
        "median_seconds": statistics.median(ordered),
        "p90_seconds": at(0.9),
    }


def _prompt(length, variant=0):
    if length < 4:
        raise ValueError("prompt length must be at least four tokens")
    middle = [198 + variant % 7] * (length - 3)
    return TokensPrompt(prompt_token_ids=[151644, 8948, *middle, 151645])


def _workload(value):
    try:
        input_len, output_len = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "workloads must use INPUTxOUTPUT, for example 4096x128"
        ) from error
    if input_len < 4 or output_len < 1:
        raise argparse.ArgumentTypeError(
            "input length must be at least four and output length must be positive"
        )
    return input_len, output_len


def _generate(llm, input_len, output_len, variant):
    result = llm.generate(
        [_prompt(input_len, variant)],
        SamplingParams(
            temperature=0,
            max_tokens=output_len,
            min_tokens=output_len,
            ignore_eos=True,
            detokenize=False,
        ),
        use_tqdm=False,
    )[0].outputs[0]
    if len(result.token_ids) != output_len:
        raise AssertionError(
            f"expected {output_len} output tokens, got {len(result.token_ids)}"
        )
    return list(result.token_ids)


def _aiter_selection():
    try:
        from vllm._aiter_ops import rocm_aiter_ops
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
    except (ImportError, ModuleNotFoundError):
        return None
    return {
        "rdna_aiter": rocm_aiter_ops.is_rdna_aiter_enabled(),
        "rdna_linear": rocm_aiter_ops.is_rdna_linear_enabled(),
        "rdna_gdn": rocm_aiter_ops.is_rdna_gdn_triton_kernels_available(),
        "gdn_module_aiter": qwen_gdn_linear_attn.GDN_AITER_TRITON_AVAILABLE,
    }


def _aiter_version():
    if importlib.util.find_spec("aiter") is None:
        return None
    try:
        return importlib.metadata.version("amd-aiter")
    except importlib.metadata.PackageNotFoundError:
        # Qualification may use an immutable source checkout rather than an
        # installed wheel. AITER's setuptools-scm build writes this module.
        from aiter._version import __version__

        return f"{__version__}+source"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--kv-cache-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--revision")
    parser.add_argument("--attention-backend")
    parser.add_argument(
        "--workload",
        action="append",
        type=_workload,
        dest="workloads",
        help="repeatable INPUTxOUTPUT workload; defaults to the qualification matrix",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 1:
        raise ValueError("samples and warmups must be positive")
    workloads = tuple(args.workloads or DEFAULT_WORKLOADS)
    for input_len, output_len in workloads:
        if input_len + output_len > 8192:
            raise ValueError("workload exceeds the qualified 8192-token context")

    config_bytes = (Path(args.model) / "config.json").read_bytes()
    started = time.perf_counter()
    llm = LLM(
        model=args.model,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        max_num_seqs=1,
        kv_cache_memory_bytes=args.kv_cache_bytes,
        enforce_eager=True,
        enable_prefix_caching=False,
        load_format="safetensors",
        trust_remote_code=False,
        attention_backend=args.attention_backend,
    )
    loaded = time.perf_counter()
    selection = _aiter_selection()
    if os.getenv("VLLM_ROCM_USE_AITER") == "1":
        if not selection or not all(
            selection[key]
            for key in ("rdna_aiter", "rdna_gdn", "gdn_module_aiter")
        ):
            raise RuntimeError(f"requested AITER baseline is inactive: {selection}")

    results = []
    for input_len, output_len in workloads:
        print(
            f"benchmark workload input={input_len} output={output_len}",
            flush=True,
        )
        for warmup in range(args.warmups):
            _generate(llm, input_len, output_len, 100 + warmup)
        torch.cuda.synchronize()
        latencies = []
        token_sha = []
        for sample in range(args.samples):
            torch.cuda.synchronize()
            sample_start = time.perf_counter()
            tokens = _generate(llm, input_len, output_len, sample)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - sample_start)
            token_sha.append(
                hashlib.sha256(
                    bytes().join(token.to_bytes(4, "little") for token in tokens)
                ).hexdigest()
            )
        timing = _percentiles(latencies)
        timing["output_tokens_per_second"] = (
            output_len / timing["median_seconds"]
        )
        results.append(
            {
                "input_tokens": input_len,
                "output_tokens": output_len,
                "timing": timing,
                "sample_seconds": latencies,
                "output_token_sha256": token_sha,
                # The V1 engine owns HIP allocations in a spawned worker, so
                # main-process Torch allocator counters are not meaningful.
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
            }
        )
        print(
            f"completed input={input_len} output={output_len} "
            f"median={timing['median_seconds']:.6f}s",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "backend": args.backend_label,
        "model": str(Path(args.model).resolve()),
        "revision": args.revision,
        "checkpoint_sha256": args.checkpoint_sha256,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "load_and_initialize_seconds_excluded": loaded - started,
        "contract": {
            "tp_size": 1,
            "max_num_seqs": 1,
            "max_model_len": 8192,
            "max_num_batched_tokens": 8192,
            "kv_cache_memory_bytes": args.kv_cache_bytes,
            "prefix_caching": False,
            "enforce_eager": True,
            "attention_backend": args.attention_backend,
            "temperature": 0,
            "ignore_eos": True,
            "detokenize": False,
            "samples": args.samples,
            "warmups_per_workload": args.warmups,
            "workloads": [list(workload) for workload in workloads],
        },
        "environment": {
            "gpu": torch.cuda.get_device_name(),
            "arch": torch.cuda.get_device_properties(0).gcnArchName,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "vllm": importlib.metadata.version("vllm"),
            "aiter": _aiter_version(),
            "variables": {name: os.getenv(name) for name in RELEVANT_ENV},
            "aiter_selection": selection,
        },
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
