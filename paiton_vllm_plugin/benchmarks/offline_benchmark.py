"""Offline benchmark entrypoint for the standalone Paiton vLLM plugin."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig

from paiton_vllm_plugin import register_paiton_models


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


def run_benchmark(args: argparse.Namespace) -> None:
    register_paiton_models()

    if args.backend == "vllm":
        os.environ.setdefault("VLLM_DISABLE_PAITON_PLATFORM", "1")

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

    llm = LLM(
        model=model_path,
        enforce_eager=False,
        compilation_config=CompilationConfig(
            cudagraph_mode=0,
            cudagraph_capture_sizes=[],
        ),
        tensor_parallel_size=args.tp,
        kv_cache_dtype="fp8" if "fp8" in args.model.lower() else "auto",
    )

    outputs = llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()

