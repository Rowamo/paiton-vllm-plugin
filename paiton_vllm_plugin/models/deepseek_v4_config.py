# SPDX-License-Identifier: Apache-2.0
"""Minimal Hugging Face config registration for DeepSeek V4 checkpoints."""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    """Config shim used by vLLM before dispatching to the Paiton runtime."""

    model_type = "deepseek_v4"


def register_deepseek_v4_config() -> None:
    """Allow AutoConfig to parse DeepSeek V4 configs on older Transformers."""
    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
