"""Profiler side of the CUDA platform.

``measure`` runs the timed forwards under vLLM's ``layerwise_profile`` and
returns one sample per catalog layer, which is the per-kernel data the
``dense`` / ``per_sequence`` / ``attention`` / ``moe`` CSVs are made of.
"""

from __future__ import annotations

from typing import Callable

# Engine kwargs merged over profiler.core.config.HOST_ENGINE_DEFAULTS.
ENGINE_KWARGS: dict = {}

# Every TP degree is profiled on one GPU: the engine boots with
# tensor_parallel_size=1 and per-rank shapes come from dividing SHARD_FIELDS
# by TP. ASTRA-Sim times the collectives.
TP_EMULATION = True


def device_info() -> dict:
    out = {"gpu": "unknown", "cuda_version": "unknown"}
    try:
        import torch
        out["cuda_version"] = torch.version.cuda or "unknown"
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return out


def measure(run_forward: Callable[[], None], iterations: int, catalog_slice: dict) -> list[dict]:
    """Time ``iterations`` forwards; return TimingSample dicts per layer.

    layerwise_profile accumulates ``cuda_time_us`` and ``invocations`` over
    every forward inside its context, and ``extract_samples`` divides one by
    the other, so this yields the per-call mean.
    """
    from vllm.profiler.layerwise_profile import layerwise_profile

    from profiler.core.hooks.timings import extract_samples

    with layerwise_profile() as hook:
        for _ in range(iterations):
            run_forward()
    summary = hook.results.convert_stats_to_dict()["summary_stats"]
    return [s.as_dict() for s in extract_samples(summary, catalog_slice)]
