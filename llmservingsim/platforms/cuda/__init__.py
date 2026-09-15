"""CUDA vLLM: per-layer kernel timings, TP emulated on one GPU, collectives
left to ASTRA-Sim. This is what the simulator did before platforms existed,
and ``PlatformSpec``'s defaults describe it, so the spec overrides little."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from llmservingsim.platforms.spec import PlatformSpec

if TYPE_CHECKING:
    from llmservingsim.platforms.profile import PlatformProfile


class CudaPlatform(PlatformSpec):
    name = "cuda"
    granularity = "layer"

    def is_available(self) -> bool:
        if importlib.util.find_spec("torch") is None:
            return False
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001 - a probe must not raise
            return False

    @property
    def profile_cls(self) -> type[PlatformProfile]:
        from llmservingsim.platforms.cuda.profile import CudaProfile
        return CudaProfile
