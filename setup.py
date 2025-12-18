# SPDX-License-Identifier: Apache-2.0
"""
Paiton vLLM Plugin - A platform plugin for running compiled Paiton models in vLLM.
"""

from setuptools import setup, find_packages

setup(
    name="paiton-vllm-plugin",
    version="0.1.0",
    description="vLLM platform plugin for Paiton compiled models",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.8.0",
        "torch>=2.0.0",
    ],
    entry_points={
        "vllm.platform_plugins": [
            "paiton_platform = paiton_vllm_plugin:paiton_platform_plugin"
        ],
        "vllm.general_plugins": [
            "register_paiton_models = paiton_vllm_plugin:register_paiton_models"
        ],
    },
)

