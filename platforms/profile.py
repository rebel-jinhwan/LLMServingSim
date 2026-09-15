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


def _selfcheck() -> None:
    """Every registered platform's profile is a ``PlatformProfile``, and a
    step platform whose profile has no grid is refused. Run by
    ``python -m platforms``."""
    from platforms import registry
    from platforms.spec import PlatformSpec
    # Run as ``python -m``, this file would be __main__ with its own copy of
    # the class; check against the one the platforms import.
    from platforms.profile import PlatformProfile as Base

    for spec in registry().values():
        assert isinstance(spec.profile, Base), (spec.name, type(spec.profile))

    class NoGrid(Base):
        def measure(self, run_forward, iterations, catalog_slice):
            return []

    class NoGridPlatform(PlatformSpec):
        name, granularity = "nogrid", "step"
        profile_cls = NoGrid  # a plain class attribute satisfies the property's contract

    try:
        NoGridPlatform().profile
    except TypeError as e:
        assert "step_grid" in str(e), e
    else:
        raise AssertionError("a step platform without step_grid was accepted")

    # A platform that binds no scheduler takes the in-tree port; one that
    # binds something that is not a class is refused at the assignment.
    from serving.core.scheduler import Scheduler

    class NoScheduler(PlatformSpec):
        name = "nosched"

    class OwnScheduler(PlatformSpec):
        name = "ownsched"

        def bind_scheduler(self):
            self.scheduler = Scheduler  # any class; the point is that it sticks

    class BadScheduler(PlatformSpec):
        name = "badsched"

        def bind_scheduler(self):
            self.scheduler = "serving.core.scheduler.Scheduler"

    assert NoScheduler().scheduler is Scheduler
    own = OwnScheduler()
    assert own.scheduler is Scheduler and own.scheduler is own.scheduler
    try:
        BadScheduler().scheduler
    except TypeError as e:
        assert "must be a class" in str(e), e
    else:
        raise AssertionError("a non-class scheduler binding was accepted")
    print(f"ok: {sorted(registry())} expose PlatformProfile; a step platform without "
          f"step_grid is refused; an unbound scheduler falls back to the in-tree port")


if __name__ == "__main__":
    _selfcheck()
