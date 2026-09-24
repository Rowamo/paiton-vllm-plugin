"""Optional MoE route capture support for compiled model wrappers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch

from paiton_vllm_plugin.runtime.core import torch_to_paiton_data


class MoeTopKCaptureMixin:
    def _configure_moe_topk_capture_state(self) -> None:
        """Initialize diagnostic capture state without allocating tensors."""
        self._moe_topk_capture: Optional[dict] = None
        self._moe_topk_ring: Optional[torch.Tensor] = None
        self._moe_topk_metadata: list[dict] = []
        self._moe_topk_ring_idx: int = 0
        self._moe_topk_step: int = 0
        self._moe_topk_recorded: int = 0
        self._moe_topk_chunk_idx: int = 0
        self._moe_topk_done: bool = False
        self._moe_topk_flush_path: Optional[str] = os.getenv(
            "PAITON_MOE_TOPK_FLUSH_PATH", "")
        self._moe_topk_ring_max = self._positive_capture_env(
            "PAITON_MOE_TOPK_RING_MAX", 256
        )
        self._moe_topk_max_tokens = self._positive_capture_env(
            "PAITON_MOE_TOPK_MAX_TOKENS", 32
        )
        self._moe_topk_capture_steps = self._positive_capture_env(
            "PAITON_MOE_TOPK_CAPTURE_STEPS", 256
        )
        self._moe_topk_current_rank = int(os.getenv("RANK", "0"))
        self._moe_topk_capture_rank = int(
            os.getenv("PAITON_MOE_TOPK_RANK", "0")
        )

    def _ensure_moe_topk_capture_state(self) -> None:
        # Several unit-test and compatibility wrappers construct the model via
        # __new__ and intentionally bypass the heavyweight vLLM initializer.
        if not hasattr(self, "_moe_topk_capture"):
            self._configure_moe_topk_capture_state()

    _TOPK_IDS_RE = re.compile(r"^topk_ids_layer_(\d+)$")
    _TOPK_WEIGHTS_RE = re.compile(r"^topk_weights_layer_(\d+)$")

    @staticmethod
    def _positive_capture_env(name: str, default: int) -> int:
        raw = os.getenv(name, str(default))
        try:
            value = int(raw)
        except ValueError:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}") from None
        if value < 1:
            raise ValueError(f"{name} must be a positive integer, got {raw!r}")
        return value

    def _init_moe_topk_capture(self, device: torch.device) -> bool:
        """Discover MoE route outputs and allocate persistent GPU buffers.

        ``topk_ids_layer_*`` is the legacy minimum capture ABI. New artifacts
        also expose ``topk_weights_layer_*`` so route replay has the exact
        weights passed into the fused MoE.  A legacy IDs-only artifact remains
        readable; a partially emitted weights set is rejected as an ABI error.

        Called once during model initialization (lazy, on first forward).
        The buffer is sized to the artifact's max output shape and reused
        across forwards — no per-forward allocation, no pointer churn.
        Graph replay sees the same data_ptr because the backing tensor
        is never freed or reallocated.
        """
        self._ensure_moe_topk_capture_state()
        if self._moe_topk_capture is not None:
            return bool(self._moe_topk_capture)
        get_output_map = getattr(self.model, "get_output_name_to_index_map", None)
        if get_output_map is None:
            self._moe_topk_capture = {}
            return False
        out_map = get_output_map()
        ids_by_layer = {}
        weights_by_layer = {}
        for name, idx in out_map.items():
            m = self._TOPK_IDS_RE.match(name)
            if m:
                ids_by_layer[int(m.group(1))] = (name, idx)
                continue
            m = self._TOPK_WEIGHTS_RE.match(name)
            if m:
                weights_by_layer[int(m.group(1))] = (name, idx)
        if not ids_by_layer:
            self._moe_topk_capture = {}
            return False
        layer_indices = sorted(ids_by_layer)
        if weights_by_layer and set(weights_by_layer) != set(layer_indices):
            raise RuntimeError(
                "MoE route capture artifact has mismatched topk id/weight "
                "layer outputs"
            )
        id_names = [ids_by_layer[layer_idx][0] for layer_idx in layer_indices]
        weight_names = (
            [weights_by_layer[layer_idx][0] for layer_idx in layer_indices]
            if weights_by_layer
            else []
        )
        names = [*id_names, *weight_names]
        max_shape = self.model.get_output_maximum_shape(id_names[0])
        max_tokens = int(max_shape[0]) if max_shape else 1
        topk = int(max_shape[1]) if len(max_shape) > 1 else 1
        max_tokens = max(max_tokens, 1)
        topk = max(topk, 1)
        num_layers = len(layer_indices)
        self._moe_topk_buffer = torch.empty(
            (num_layers, max_tokens, topk),
            dtype=torch.int32, device=device,
        )
        layer_views = {}
        for i, name in enumerate(id_names):
            layer_views[name] = self._moe_topk_buffer[i]
        self._moe_topk_weights_buffer = None
        if weight_names:
            weight_shape = self.model.get_output_maximum_shape(weight_names[0])
            if tuple(weight_shape) != tuple(max_shape):
                raise RuntimeError(
                    "MoE route capture id/weight output shapes do not match"
                )
            self._moe_topk_weights_buffer = torch.empty(
                (num_layers, max_tokens, topk), dtype=torch.float32, device=device
            )
            for i, name in enumerate(weight_names):
                layer_views[name] = self._moe_topk_weights_buffer[i]
        self._moe_topk_capture = {
            "layer_indices": layer_indices,
            "names": names,
            "id_names": id_names,
            "weight_names": weight_names,
            "layer_views": layer_views,
            "max_tokens": max_tokens,
            "topk": topk,
            "num_layers": num_layers,
        }
        if self._moe_topk_current_rank == self._moe_topk_capture_rank:
            ring_tokens = min(max_tokens, self._moe_topk_max_tokens)
            self._moe_topk_ring = torch.empty(
                (
                    self._moe_topk_ring_max,
                    num_layers,
                    ring_tokens,
                    topk,
                ),
                dtype=torch.int32,
                device=device,
            )
            self._moe_topk_weights_ring = (
                torch.empty(
                    (
                        self._moe_topk_ring_max,
                        num_layers,
                        ring_tokens,
                        topk,
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                if weight_names
                else None
            )
        else:
            self._moe_topk_ring = None
            self._moe_topk_weights_ring = None
        self._moe_topk_metadata = []
        self._moe_topk_ring_idx = 0
        self._moe_topk_step = 0
        self._moe_topk_recorded = 0
        self._moe_topk_chunk_idx = 0
        self._moe_topk_done = False
        return True

    def _bind_moe_topk_outputs(
        self, outputs: dict, num_tokens: int,
    ) -> None:
        """Bind layer views into the outputs dict for the current forward.

        Called every forward (the artifact requires all outputs to be bound).
        The same backing tensor is reused — no allocation, stable pointers.
        """
        cap = self._moe_topk_capture
        if not cap:
            return
        for name in cap["names"]:
            view = cap["layer_views"][name]
            outputs[name] = torch_to_paiton_data(view)

    def _record_moe_topk_capture(
        self, num_tokens: int, step: int, metadata: Optional[dict] = None,
    ) -> None:
        """Copy the contiguous capture buffer into the ring on rank 0.

        Called after selected forwards (not every forward).  Uses one
        async device-to-device copy of the [num_layers, num_tokens, topk]
        slice into a preallocated GPU ring slot.  The ring is flushed to
        a .pt file when full or when _flush_moe_topk_capture is called.
        """
        cap = self._moe_topk_capture
        if (
            not cap
            or self._moe_topk_done
            or self._moe_topk_current_rank != self._moe_topk_capture_rank
            or self._moe_topk_ring is None
        ):
            return
        if num_tokens > self._moe_topk_ring.shape[2]:
            return
        if self._moe_topk_ring_idx >= self._moe_topk_ring.shape[0]:
            self._flush_moe_topk_capture()
        ring_idx = self._moe_topk_ring_idx
        self._moe_topk_ring[ring_idx, :, :num_tokens, :].copy_(
            self._moe_topk_buffer[:, :num_tokens, :], non_blocking=True
        )
        if self._moe_topk_weights_ring is not None:
            self._moe_topk_weights_ring[ring_idx, :, :num_tokens, :].copy_(
                self._moe_topk_weights_buffer[:, :num_tokens, :],
                non_blocking=True,
            )
        self._moe_topk_metadata.append({
            "step": step,
            "num_tokens": num_tokens,
            "metadata": metadata or {},
        })
        self._moe_topk_ring_idx += 1
        self._moe_topk_recorded += 1
        self._moe_topk_step = step + 1
        if (
            self._moe_topk_ring_idx >= self._moe_topk_ring.shape[0]
            or self._moe_topk_recorded >= self._moe_topk_capture_steps
        ):
            self._flush_moe_topk_capture()
        if self._moe_topk_recorded >= self._moe_topk_capture_steps:
            self._moe_topk_done = True

    def _moe_topk_chunk_path(self) -> Path:
        configured = self._moe_topk_flush_path
        base = Path(configured) if configured else Path(
            f"/tmp/moe_topk_capture_rank{self._moe_topk_current_rank}.pt"
        )
        suffix = base.suffix or ".pt"
        stem = base.stem if base.suffix else base.name
        return base.with_name(
            f"{stem}.chunk{self._moe_topk_chunk_idx:05d}{suffix}"
        )

    def _flush_moe_topk_capture(self) -> Optional[str]:
        """Flush the ring to a .pt file and reset the ring index.

        Returns the file path if flushed, None if the ring was empty.
        """
        if (
            self._moe_topk_ring is None
            or self._moe_topk_ring_idx == 0
            or self._moe_topk_current_rank != self._moe_topk_capture_rank
        ):
            return None
        path = self._moe_topk_chunk_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        count = self._moe_topk_ring_idx
        captured = self._moe_topk_ring[:count].cpu()
        payload = {
            "topk_ids": captured,
            "entries": list(self._moe_topk_metadata),
            "layer_indices": list(self._moe_topk_capture["layer_indices"]),
            "topk": self._moe_topk_capture["topk"],
            "artifact_identity": getattr(self, "model_name", ""),
            "rank": self._moe_topk_current_rank,
        }
        if self._moe_topk_weights_ring is not None:
            payload["topk_weights"] = self._moe_topk_weights_ring[:count].cpu()
        torch.save(payload, path)
        self._moe_topk_ring_idx = 0
        self._moe_topk_metadata = []
        self._moe_topk_chunk_idx += 1
        return str(path)

