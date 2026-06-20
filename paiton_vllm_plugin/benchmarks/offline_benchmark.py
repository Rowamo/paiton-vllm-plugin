"""Offline benchmark entrypoint for the standalone Paiton vLLM plugin."""

from __future__ import annotations

import argparse
import os
import time
from importlib.metadata import entry_points
from pathlib import Path


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

MODEL_PRESETS = {
    "llama-3.1-8b-fp8": {
        "model": "amd/Llama-3.1-8B-Instruct-FP8-KV",
        "prompts": [
            "Hello, my name is",
            "The capital of France is",
            "The future of AI is",
        ],
        "kv_cache_dtype": "fp8",
    },
    "deepseek-v4-flash": {
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "compiled_model_dir": "/app/paiton-compiler/tmp/DeepSeek-V4-Flash",
        "prompts": [
            "Write one sentence about why compilers are useful.",
            "Explain tensor parallelism in one short paragraph.",
            "Name one advantage of using FP4 weights for routed experts.",
        ],
        "kv_cache_dtype": "fp8",
        # Keep the default bring-up path aligned with the smallest compiled
        # artifact. That avoids silently falling back to an older larger-capacity
        # .so when multiple DeepSeek artifacts coexist in the same directory.
        "max_model_len": 8192,
        "max_num_batched_tokens": 512,
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a simple offline benchmark for Paiton-compiled or vanilla vLLM models.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        choices=tuple(MODEL_PRESETS),
        help="Use built-in model defaults, including DeepSeek V4 Flash.",
    )
    parser.add_argument(
        "--backend",
        default="paiton",
        choices=("paiton", "vllm"),
        help="Inference backend to benchmark.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
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
        "--prompt",
        action="append",
        default=None,
        help="Prompt to run. Can be passed multiple times. Overrides preset/default prompts.",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="KV-cache dtype to pass to vLLM. Defaults to preset value, fp8 for fp8/deepseek models, otherwise auto.",
    )
    parser.add_argument(
        "--max-model-len",
        default=None,
        type=int,
        help="Optional vLLM max_model_len override.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        default=None,
        type=int,
        help="Optional vLLM max_num_batched_tokens override.",
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
        default=2,
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


def apply_preset_defaults(args: argparse.Namespace) -> None:
    if args.preset is None:
        return

    preset = MODEL_PRESETS[args.preset]
    if args.model == DEFAULT_MODEL:
        args.model = preset["model"]
    if args.compiled_model_dir is None:
        args.compiled_model_dir = preset.get("compiled_model_dir")
    if args.kv_cache_dtype is None:
        args.kv_cache_dtype = preset.get("kv_cache_dtype")
    if args.max_model_len is None:
        args.max_model_len = preset.get("max_model_len")
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = preset.get("max_num_batched_tokens")


def resolve_paiton_model_path(model: str, compiled_root: str,
                              compiled_model_dir: str | None) -> str:
    if compiled_model_dir:
        return compiled_model_dir

    model_name = model.rstrip("/").split("/")[-1]
    return str(Path(compiled_root) / model_name)


def build_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt:
        prompts = args.prompt
    elif args.preset is not None:
        prompts = MODEL_PRESETS[args.preset]["prompts"]
    else:
        prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        ]
    repeats = max(1, (args.num_prompts + len(prompts) - 1) // len(prompts))
    return (prompts * repeats)[:args.num_prompts]


def configure_environment(args: argparse.Namespace) -> None:
    if args.backend == "vllm":
        os.environ["VLLM_DISABLE_PAITON_PLATFORM"] = "1"
        if args.enable_aiter is not None:
            os.environ["VLLM_ROCM_USE_AITER"] = "1" if args.enable_aiter else "0"
    else:
        require_paiton_plugin_entry_points()
        os.environ["VLLM_DISABLE_PAITON_PLATFORM"] = "0"
        os.environ.setdefault("VLLM_USE_PAITON_PLATFORM", "1")
        enable_vllm_plugin("paiton_platform")
        enable_vllm_plugin("register_paiton_models")


def enable_vllm_plugin(plugin_name: str) -> None:
    configured = os.environ.get("VLLM_PLUGINS")
    if configured is None:
        os.environ["VLLM_PLUGINS"] = plugin_name
        return
    plugins = [p for p in configured.split(",") if p]
    if plugin_name not in plugins:
        plugins.append(plugin_name)
        os.environ["VLLM_PLUGINS"] = ",".join(plugins)


def require_paiton_plugin_entry_points() -> None:
    general_plugins = {
        ep.name for ep in entry_points(group="vllm.general_plugins")
    }
    platform_plugins = {
        ep.name for ep in entry_points(group="vllm.platform_plugins")
    }
    missing = []
    if "register_paiton_models" not in general_plugins:
        missing.append("vllm.general_plugins:register_paiton_models")
    if "paiton_platform" not in platform_plugins:
        missing.append("vllm.platform_plugins:paiton_platform")
    if missing:
        raise RuntimeError(
            "Paiton vLLM plugin entry points are not installed, so vLLM's "
            "EngineCore subprocess cannot register Paiton model architectures. "
            "Install the plugin first:\n\n"
            "  cd /app/paiton-vllm-plugin && python3 -m pip install -e .\n\n"
            "Missing entry points:\n- " + "\n- ".join(missing))


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
    apply_preset_defaults(args)
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

    prompts = build_prompts(args)
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens,
    )

    model_l = args.model.lower()
    kv_cache_dtype = args.kv_cache_dtype
    if kv_cache_dtype is None:
        kv_cache_dtype = "fp8" if ("fp8" in model_l or "deepseek-v4" in model_l
                                   or "deepseek_v4" in model_l) else "auto"

    llm_kwargs = {
        "model": model_path,
        "enforce_eager": False,
        "tensor_parallel_size": args.tp,
        "kv_cache_dtype": kv_cache_dtype,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    if args.max_num_batched_tokens is not None:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.backend == "paiton":
        # Let the Paiton platform handle compilation_config. The platform
        # sets cudagraph_mode=NONE + empty capture sizes, while
        # enforce_eager=False (above) enables vLLM's async scheduler.
        pass

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
    print(f"resolved_model_path={model_path}")
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
