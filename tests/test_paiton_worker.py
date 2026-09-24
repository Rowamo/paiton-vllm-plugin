import types

from paiton_vllm_plugin.paiton_worker import (
    _scheduler_physical_block_high_water,
)


def test_scheduler_physical_block_high_water_covers_all_cpu_sources():
    output = types.SimpleNamespace(
        scheduled_new_reqs=[
            types.SimpleNamespace(block_ids=([2, 120], [7])),
        ],
        scheduled_cached_reqs=types.SimpleNamespace(
            new_block_ids=[None, ([9, 300],)],
        ),
        new_block_ids_to_zero=[4, 400],
    )

    assert _scheduler_physical_block_high_water(output) == 401


def test_scheduler_physical_block_high_water_handles_empty_step():
    output = types.SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=types.SimpleNamespace(new_block_ids=[]),
        new_block_ids_to_zero=None,
    )

    assert _scheduler_physical_block_high_water(output) == 0
