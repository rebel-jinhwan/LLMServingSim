"""``PlatformSpec``: the one object that describes a hardware platform.

Modelled on LMCache's ``DeviceSpec``. A platform is a subclass of
``PlatformSpec``, built in (a subpackage of ``platforms/``) or out of tree
(a class named by an ``llmservingsim.platforms`` entry point). Either way it
goes through the same registry (``platforms._registry``) and every
capability is read off the spec instance.

Rules a platform must follow, built in or not:
- Importing the spec's module and constructing the spec must not need the
  hardware or its SDK. Probe the hardware only in ``is_available()``, and
  import vLLM, torch or a vendor SDK only inside the members that need them.
- ``name`` is lowercase, unique, and equal to the entry-point name.
- Anything the platform ships as files lives next to the spec's module:
  ``devices/<hardware>.yaml``, ``perf/<hardware>/<model>/<variant>/`` bundles
  and ``models/<model_type>.yaml`` architecture catalogs. The simulator and
  the profiler search those after the in-tree locations.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from platforms.profile import PlatformProfile

GRANULARITIES = ("layer", "step")


class PlatformSpec:
    """Defaults are CUDA vLLM's behaviour; a platform overrides what differs."""

    name: ClassVar[str] = ""
    """Unique lowercase name; the ``--platform`` value and meta.yaml's ``platform``."""

    granularity: ClassVar[str] = "layer"
    """``layer``: per-kernel profile, one trace row per canonical layer, ASTRA-Sim
    adds the collectives. ``step``: one wall-clock time per padded forward, for a
    device that runs a compiled graph with the collectives inside it."""

    def is_available(self) -> bool:
        """Whether this host has the hardware. Cheap, never raises, and only
        consulted where the hardware is actually used (profiler, bench)."""
        return False

    @property
    def profile_cls(self) -> type[PlatformProfile]:
        """The ``PlatformProfile`` subclass that drives the profiler."""
        raise NotImplementedError(f"{type(self).__name__} does not define profile_cls")

    @functools.cached_property
    def profile(self) -> PlatformProfile:
        """One profile instance, checked against the interface."""
        from platforms.profile import PlatformProfile

        prof = self.profile_cls()
        if not isinstance(prof, PlatformProfile):
            raise TypeError(
                f"platform {self.name!r}: profile_cls must be a PlatformProfile "
                f"subclass, got {type(prof).__name__}")
        if self.granularity == "step" and type(prof).step_grid is PlatformProfile.step_grid:
            raise TypeError(
                f"platform {self.name!r} is step-granularity but "
                f"{type(prof).__name__} does not override step_grid()")
        return prof

    @property
    def scheduler_cls(self) -> type:
        """The simulator's scheduler for this platform: the in-tree port of
        vLLM's scheduler unless the platform needs its own."""
        from serving.core.scheduler import Scheduler
        return Scheduler

    @property
    def resource_dir(self) -> Path:
        """Directory holding the platform's ``devices/``, ``perf/`` and ``models/``."""
        return Path(inspect.getfile(type(self))).resolve().parent

    def resources(self, kind: str) -> Path | None:
        """``resource_dir / kind`` when it exists."""
        path = self.resource_dir / kind
        return path if path.is_dir() else None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, granularity={self.granularity!r})"
