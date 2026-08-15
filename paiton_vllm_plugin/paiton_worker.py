# SPDX-License-Identifier: Apache-2.0
"""Worker hooks required by Paiton's process-local platform settings."""

from typing import Any

from vllm.v1.worker.gpu_worker import Worker

from paiton_vllm_plugin.paiton_platform import _apply_numa_memory_policy


def _scheduler_physical_block_high_water(scheduler_output: Any) -> int:
    """Return one past the largest physical block ID sent by the scheduler.

    ``SchedulerOutput`` owns CPU lists for new and resumed requests. Tracking
    their maximum in the worker avoids reading the GPU block table from the
    model wrapper. The result is intentionally monotonic when installed on
    the model: freed blocks may be reused, but an already allocated sparse
    cache remains valid for every lower physical ID.
    """

    max_block = -1

    def include_groups(groups: Any) -> None:
        nonlocal max_block
        if groups is None:
            return
        for group in groups:
            if group:
                max_block = max(max_block, max(int(block) for block in group))

    for request in getattr(scheduler_output, "scheduled_new_reqs", ()):
        include_groups(getattr(request, "block_ids", None))

    cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
    cached_groups = (
        getattr(cached, "new_block_ids", ()) if cached is not None else ()
    )
    for groups in cached_groups:
        include_groups(groups)

    # Current vLLM also sends a flat list of freshly allocated blocks to be
    # zeroed. Include it as a compatibility/defence-in-depth source.
    blocks_to_zero = getattr(scheduler_output, "new_block_ids_to_zero", None)
    if blocks_to_zero:
        max_block = max(max_block, max(int(block) for block in blocks_to_zero))

    return max_block + 1


class PaitonGPUWorker(Worker):
    """GPU worker that binds its own memory policy after device selection."""

    def init_device(self) -> None:
        super().init_device()
        _apply_numa_memory_policy()

    def execute_model(self, scheduler_output):  # type: ignore[no-untyped-def]
        high_water = _scheduler_physical_block_high_water(scheduler_output)
        if high_water > 0:
            model = self.model_runner.get_model()
            previous = int(
                getattr(model, "_paiton_physical_block_high_water", 0) or 0
            )
            model._paiton_physical_block_high_water = max(previous, high_water)
        return super().execute_model(scheduler_output)
