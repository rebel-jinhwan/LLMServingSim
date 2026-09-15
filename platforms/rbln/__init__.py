"""Rebellions NPUs through vllm-rbln.

The device runs a compiled graph per padded shape, with the TP collectives
inside it, so a forward is measured whole: one ``step.csv`` row per
(prefill_chunk, kv_prefill, n_decode, kv_decode), TP profiled on real ranks.
The scheduler is vllm-rbln's own, which never mixes prefill and decode in one
step.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from platforms.spec import PlatformSpec

if TYPE_CHECKING:
    from platforms.profile import PlatformProfile


class RBLNPlatform(PlatformSpec):
    name = "rbln"
    granularity = "step"

    def is_available(self) -> bool:
        if importlib.util.find_spec("rebel") is None:
            return False
        try:
            import rebel
            return bool(rebel.get_npu_name(0))
        except Exception:  # noqa: BLE001 - no NPU, or no runtime: not available
            return False

    @property
    def profile_cls(self) -> type[PlatformProfile]:
        from platforms.rbln.profile import RBLNProfile
        return RBLNProfile

    @property
    def scheduler_cls(self) -> type:
        from platforms.rbln.simulator import RBLNVllmScheduler
        return RBLNVllmScheduler
