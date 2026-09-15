"""The profiler-side interface every platform implements.

A platform's ``profile.py`` subclasses ``PlatformProfile``, overrides what
differs from the defaults below, and exposes one instance as ``PROFILE``.
The defaults are CUDA vLLM's behaviour, so a platform states only how it
departs from it. Nothing here imports vLLM or torch at module scope: the
simulator imports ``platforms`` too, and the subclasses import what they
need inside the methods that need it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator

if TYPE_CHECKING:
    from profiler.core.config import ProfileArgs
    from profiler.core.engine import RuntimeLimits
    from profiler.core.hooks.batch import Shot


class PlatformProfile(ABC):
    """How the profiler boots, drives and times vLLM on one platform."""

    ENGINE_KWARGS: ClassVar[dict[str, Any]] = {}
    """Engine kwargs merged over ``profiler.core.config.HOST_ENGINE_DEFAULTS``
    and under the CLI's own."""

    TP_EMULATION: ClassVar[bool] = True
    """True: every TP degree boots on one device with ``SHARD_FIELDS``
    divided by TP, and ASTRA-Sim adds the collectives. False: ``tp<N>`` boots
    N real ranks, for a device whose collectives are inside what is timed."""

    def scheduler_output_cls(self) -> type:
        """The ``SchedulerOutput`` class the platform's model runner reads;
        synthetic profiling batches are built as this class."""
        from vllm.v1.core.sched.output import SchedulerOutput
        return SchedulerOutput

    def device_info(self) -> dict[str, Any]:
        """Device identity for ``meta.yaml``. ``gpu`` is the one key every
        platform writes; add whatever else identifies the software stack."""
        return {"gpu": "unknown"}

    @abstractmethod
    def measure(self, run_forward: Callable[[], None], iterations: int,
                catalog_slice: dict) -> list[dict]:
        """Run ``run_forward`` ``iterations`` times and return
        ``TimingSample.as_dict()`` rows: one per catalog layer at layer
        granularity, a single ``step`` row at step granularity."""

    def step_grid(self, args: ProfileArgs, limits: RuntimeLimits) -> Iterator[Shot]:
        """The shots ``step.csv`` is swept over. Required at step
        granularity, where it must visit exactly the shapes the platform's
        runner can produce; unused at layer granularity."""
        raise NotImplementedError(
            f"{type(self).__name__} does not define step_grid(); a "
            f"step-granularity platform must")
