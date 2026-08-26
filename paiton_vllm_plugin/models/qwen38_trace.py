# SPDX-License-Identifier: Apache-2.0
"""Opt-in, non-synchronizing causal trace for the Qwen3.8 runtime path."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


_CPROFILE_INSTALLED = False


def install_qwen38_worker_cprofile() -> None:
    """Profile worker execute calls only when a diagnostic directory is set."""

    global _CPROFILE_INSTALLED
    value = os.environ.get("PAITON_QWEN38_CPROFILE_DIR")
    if not value or _CPROFILE_INSTALLED:
        return
    import cProfile
    from functools import wraps

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    destination = Path(value).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    original = GPUModelRunner.execute_model
    call_count = 0

    @wraps(original)
    def profiled_execute_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal call_count
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            return original(self, *args, **kwargs)
        finally:
            profiler.disable()
            path = destination / f"execute-model-{call_count:03d}.pstats"
            call_count += 1
            profiler.dump_stats(path)

    GPUModelRunner.execute_model = profiled_execute_model  # type: ignore[method-assign]
    _CPROFILE_INSTALLED = True


class Qwen38RuntimeTrace:
    """Record host and caller-stream boundaries without synchronizing the path.

    Event results are consumed only at the beginning of a later model call,
    after sampling has necessarily consumed the prior logits.  The feature is
    inactive unless ``PAITON_QWEN38_RUNTIME_TRACE`` names a new output file.
    """

    _EVENTS_PER_CALL = 16
    _MAX_CALLS = 160

    def __init__(self) -> None:
        value = os.environ.get("PAITON_QWEN38_RUNTIME_TRACE")
        self.path = Path(value).resolve() if value else None
        self._event_sets: list[list[torch.cuda.Event]] = []
        self._next_event_set = 0
        self._call_index = 0
        self._current: dict[str, Any] | None = None
        self._pending: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def initialize(self, *, device: torch.device, identity: dict[str, Any]) -> None:
        if not self.enabled:
            return
        assert self.path is not None
        if self.path.exists():
            raise RuntimeError(f"refusing to overwrite Qwen3.8 trace: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with torch.cuda.device(device):
            self._event_sets = [
                [
                    torch.cuda.Event(enable_timing=True)
                    for _ in range(self._EVENTS_PER_CALL)
                ]
                for _ in range(self._MAX_CALLS)
            ]
        self._append(
            {
                "kind": "session",
                "schema": "paiton.qwen38.runtime-causal-trace.v1",
                "pid": os.getpid(),
                "initialized_monotonic_ns": time.monotonic_ns(),
                "device": str(device),
                "identity": identity,
                "event_sets": self._MAX_CALLS,
                "events_per_call": self._EVENTS_PER_CALL,
                "host_sync_in_trace_path": False,
            }
        )

    def _append(self, record: dict[str, Any]) -> None:
        assert self.path is not None
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _flush_ready(self) -> None:
        if not self.enabled:
            return
        retained: list[dict[str, Any]] = []
        for call in self._pending:
            boundaries = call["boundaries"]
            if not boundaries or not boundaries[-1]["event"].query():
                retained.append(call)
                continue
            serialized: list[dict[str, Any]] = []
            previous = None
            for boundary in boundaries:
                item = {
                    key: value
                    for key, value in boundary.items()
                    if key != "event"
                }
                if previous is None:
                    item["host_delta_us"] = 0.0
                    item["device_delta_ms"] = 0.0
                else:
                    item["host_delta_us"] = (
                        boundary["monotonic_ns"] - previous["monotonic_ns"]
                    ) / 1000.0
                    item["device_delta_ms"] = previous["event"].elapsed_time(
                        boundary["event"]
                    )
                serialized.append(item)
                previous = boundary
            self._append(
                {
                    "kind": "call",
                    "call_index": call["call_index"],
                    "tokens": call["tokens"],
                    "allocation_only": call["allocation_only"],
                    "boundaries": serialized,
                }
            )
        self._pending = retained

    def begin(self, *, tokens: int) -> None:
        if not self.enabled:
            return
        self._flush_ready()
        if self._current is not None:
            raise RuntimeError("Qwen3.8 causal trace call is already active")
        if self._next_event_set >= len(self._event_sets):
            raise RuntimeError("Qwen3.8 causal trace event capacity exhausted")
        self._current = {
            "call_index": self._call_index,
            "tokens": tokens,
            "allocation_only": False,
            "events": self._event_sets[self._next_event_set],
            "boundaries": [],
        }
        self._call_index += 1
        self._next_event_set += 1
        self.mark("wrapper_entry")

    def classify_allocation_only(self, value: bool) -> None:
        if self._current is not None:
            self._current["allocation_only"] = value

    def mark(self, name: str, **fields: Any) -> None:
        if self._current is None:
            return
        boundaries = self._current["boundaries"]
        if len(boundaries) >= self._EVENTS_PER_CALL:
            raise RuntimeError("Qwen3.8 causal trace boundary capacity exhausted")
        event = self._current["events"][len(boundaries)]
        event.record()
        boundary = {
            "name": name,
            "monotonic_ns": time.monotonic_ns(),
            "memory_allocated": torch.cuda.memory_allocated(),
            "memory_reserved": torch.cuda.memory_reserved(),
            "event": event,
        }
        boundary.update(fields)
        boundaries.append(boundary)

    def finish(self) -> None:
        if self._current is None:
            return
        self._pending.append(self._current)
        self._current = None


__all__ = ["Qwen38RuntimeTrace", "install_qwen38_worker_cprofile"]
