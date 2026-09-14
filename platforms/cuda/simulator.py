"""Simulator side of the CUDA platform: the in-tree port of vLLM's
scheduler. ``--scheduler vllm`` swaps in ``serving.core.vllm_scheduler``,
which drives upstream vLLM's own ``Scheduler`` instead."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serving.core.scheduler import Scheduler


def scheduler_class() -> type[Scheduler]:
    from serving.core.scheduler import Scheduler
    return Scheduler
