# SPDX-License-Identifier: Apache-2.0
"""Worker hooks required by Paiton's process-local platform settings."""

from vllm.v1.worker.gpu_worker import Worker

from paiton_vllm_plugin.paiton_platform import _apply_numa_memory_policy


class PaitonGPUWorker(Worker):
    """GPU worker that binds its own memory policy after device selection."""

    def init_device(self) -> None:
        super().init_device()
        _apply_numa_memory_policy()
