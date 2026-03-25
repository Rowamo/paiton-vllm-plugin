"""Offline benchmark entrypoint for the standalone Paiton vLLM plugin."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple offline benchmark for Paiton-compiled or vanilla vLLM models.",
    )
    parser.add_argument(
        "--backend",
        default="paiton",
        choices=("paiton", "vllm"),
        help="Inference backend to benchmark.",
    )
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model identifier or model directory.",
    )
    parser.add_argument(
        "--tp",
        default=1,
        type=int,
        help="Tensor parallel size.",
    )
    parser.add_argument(
        "--compiled-root",
        default="/app/paiton-compiler/tmp",
        help="Root directory containing compiled Paiton model folders.",
    )
    parser.add_argument(
        "--compiled-model-dir",
        default=None,
        help="Explicit compiled model directory. Overrides --compiled-root resolution.",
    )
    parser.add_argument(
        "--num-prompts",
        default=32,
        type=int,
        help="Number of prompts to generate.",
    )
    parser.add_argument(
        "--max-tokens",
        default=256,
        type=int,
        help="Maximum tokens to generate per prompt.",
    )
    parser.add_argument(
        "--warmup-iters",
        default=1,
        type=int,
        help="How many warmup generate() calls to run before timing.",
    )
    parser.add_argument(
        "--measure-iters",
        default=1,
        type=int,
        help="How many timed generate() calls to run.",
    )
    parser.add_argument(
        "--enable-aiter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable ROCm AITER in stock vLLM mode by setting VLLM_ROCM_USE_AITER=1 before importing vLLM.",
    )
    return parser


def resolve_paiton_model_path(model: str, compiled_root: str,
                              compiled_model_dir: str | None) -> str:
    if compiled_model_dir:
        return compiled_model_dir

    model_name = model.rstrip("/").split("/")[-1]
    return str(Path(compiled_root) / model_name)


def build_prompts(num_prompts: int) -> list[str]:
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    repeats = max(1, (num_prompts + len(prompts) - 1) // len(prompts))
    return (prompts * repeats)[:num_prompts]


def configure_environment(args: argparse.Namespace) -> None:
    if args.backend == "vllm":
        os.environ["VLLM_DISABLE_PAITON_PLATFORM"] = "1"
        if args.enable_aiter is not None:
            os.environ["VLLM_ROCM_USE_AITER"] = "1" if args.enable_aiter else "0"
    else:
        os.environ["VLLM_DISABLE_PAITON_PLATFORM"] = "0"


def import_runtime(args: argparse.Namespace):
    from vllm import LLM, SamplingParams
    from vllm.config import CompilationConfig

    if args.backend == "paiton":
        from paiton_vllm_plugin import register_paiton_models

        register_paiton_models()

    return LLM, SamplingParams, CompilationConfig


def count_generated_tokens(outputs) -> int:
    return sum(len(output.outputs[0].token_ids) for output in outputs)


def run_benchmark(args: argparse.Namespace) -> None:
    configure_environment(args)
    LLM, SamplingParams, CompilationConfig = import_runtime(args)

    model_path = (
        resolve_paiton_model_path(
            args.model,
            compiled_root=args.compiled_root,
            compiled_model_dir=args.compiled_model_dir,
        )
        if args.backend == "paiton"
        else args.model
    )

    prompts = build_prompts(args.num_prompts)
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    llm_kwargs = {
        "model": model_path,
        "enforce_eager": False,
        "tensor_parallel_size": args.tp,
        "kv_cache_dtype": "fp8" if "fp8" in args.model.lower() else "auto",
    }
    if args.backend == "paiton":
        llm_kwargs["compilation_config"] = CompilationConfig(
            cudagraph_mode=0,
            cudagraph_capture_sizes=[],
        )

    llm = LLM(**llm_kwargs)

    for _ in range(args.warmup_iters):
        llm.generate(prompts, sampling_params)

    timings_s: list[float] = []
    measured_outputs = None
    for _ in range(args.measure_iters):
        start = time.perf_counter()
        measured_outputs = llm.generate(prompts, sampling_params)
        timings_s.append(time.perf_counter() - start)

    assert measured_outputs is not None
    generated_tokens = count_generated_tokens(measured_outputs)
    avg_latency_s = sum(timings_s) / len(timings_s)
    toks_per_s = generated_tokens / avg_latency_s if avg_latency_s > 0 else 0.0

    print(
        f"backend={args.backend} "
        f"aiter={os.environ.get('VLLM_ROCM_USE_AITER', 'unset')} "
        f"prompts={len(prompts)} max_tokens={args.max_tokens} "
        f"warmup_iters={args.warmup_iters} measure_iters={args.measure_iters}"
    )
    print(
        f"avg_latency_s={avg_latency_s:.4f} "
        f"generated_tokens={generated_tokens} "
        f"generated_toks_per_s={toks_per_s:.2f}"
    )

    for output in measured_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
