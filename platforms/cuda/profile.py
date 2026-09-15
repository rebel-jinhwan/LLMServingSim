"""Profiler side of the CUDA platform.

The ``PlatformProfile`` defaults are CUDA's (engine kwargs, TP emulated on
one GPU, vLLM's ``SchedulerOutput``), so this overrides only how a shot is
timed and what identifies the device: ``measure`` runs the forwards under
vLLM's ``layerwise_profile`` and returns one sample per catalog layer,
which is what the ``dense`` / ``per_sequence`` / ``attention`` / ``moe``
CSVs are made of.
"""

from __future__ import annotations

from typing import Any, Callable

from platforms.profile import PlatformProfile


class CudaProfile(PlatformProfile):

    def device_info(self) -> dict[str, Any]:
        out = {"gpu": "unknown", "cuda_version": "unknown"}
        try:
            import torch
            out["cuda_version"] = torch.version.cuda or "unknown"
            if torch.cuda.is_available():
                out["gpu"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
        return out

    def measure(self, run_forward: Callable[[], None], iterations: int,
                catalog_slice: dict) -> list[dict]:
        """layerwise_profile accumulates ``cuda_time_us`` and
        ``invocations`` over every forward inside its context, and
        ``extract_samples`` divides one by the other, so this yields the
        per-call mean."""
        from vllm.profiler.layerwise_profile import layerwise_profile

        from profiler.core.hooks.timings import extract_samples

        with layerwise_profile() as hook:
            for _ in range(iterations):
                run_forward()
        summary = hook.results.convert_stats_to_dict()["summary_stats"]
        return [s.as_dict() for s in extract_samples(summary, catalog_slice)]


PROFILE = CudaProfile()
