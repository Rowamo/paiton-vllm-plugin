"""Zero-configuration server entrypoint for the public RDNA4 Qwen3.8 image."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path


DEFAULT_MODEL = (
    "EliovpAI/"
    "Qwen3.8-27B-Quark-Qronos-INT4-W4A16-Paiton-RDNA4"
)
DEFAULT_REVISION = "paiton-rdna4-v1"


def build_server_command(extra_args: list[str] | None = None) -> list[str]:
    model = os.environ.get("PAITON_MODEL", DEFAULT_MODEL)
    revision = os.environ.get("PAITON_MODEL_REVISION", DEFAULT_REVISION)
    command = ["vllm", "serve", model]
    if revision and not Path(model).exists():
        command.extend(["--revision", revision])
    command.extend(
        [
            "--served-model-name",
            os.environ.get("PAITON_SERVED_MODEL_NAME", "qwen38"),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "8192",
            "--max-num-batched-tokens",
            "8192",
            "--max-num-seqs",
            "1",
            "--kv-cache-dtype",
            "auto",
            "--kv-cache-memory-bytes",
            "2G",
            "--load-format",
            "safetensors",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--reasoning-parser",
            "qwen3",
            "--host",
            "0.0.0.0",
            "--port",
            os.environ.get("PAITON_PORT", "8000"),
        ]
    )
    command.extend(extra_args or [])
    return command


def main() -> None:
    command = build_server_command(sys.argv[1:])
    print("Starting the pinned Paiton RDNA4 server:", flush=True)
    print("  " + shlex.join(command), flush=True)
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
