"""Platform plugins: how one vLLM hardware platform differs from CUDA.

vLLM supports out-of-tree hardware through ``vllm.platform_plugins`` entry
points; ``vllm-rbln`` is one. This package is the simulator's counterpart.
A platform is a package with two submodules, imported on demand so the
profiler side (which needs vLLM and torch) never loads in the simulator
container and vice versa:

    platforms/<vendor>/__init__.py   NAME, GRANULARITY
    platforms/<vendor>/profile.py    PROFILE, a platforms.profile.PlatformProfile
                                     subclass instance: how a shot is measured
    platforms/<vendor>/simulator.py  which Scheduler the simulator runs

``GRANULARITY`` is ``"layer"`` (per-kernel timings, the CUDA default: the
trace is one row per canonical layer and ASTRA-Sim adds the collectives) or
``"step"`` (one wall-clock time per padded forward, for devices that run a
compiled graph with the collectives inside it: the trace is one row and TP
is profiled on real ranks).

Resolution order for ``load_platform``: the explicit name, then the
``platform`` key of the perf bundle's meta.yaml (the profiler writes it, so
a simulation run needs no flag), then the sole installed non-cuda
``llmservingsim.platforms`` entry point, then cuda. An entry point names a
package with the same three-module layout, so a vendor can ship its
platform inside its own vLLM plugin.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from types import ModuleType
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from platforms.profile import PlatformProfile

ENTRY_POINT_GROUP = "llmservingsim.platforms"
BUILTIN = {"cuda": "platforms.cuda", "rbln": "platforms.rbln"}


class Platform:
    """A resolved platform package. ``profile`` and ``simulator`` import
    their submodule on first access."""

    def __init__(self, pkg: ModuleType) -> None:
        self.pkg = pkg
        self.name: str = pkg.NAME
        self.granularity: str = pkg.GRANULARITY
        if self.granularity not in ("layer", "step"):
            raise ValueError(
                f"platform {self.name!r}: GRANULARITY must be 'layer' or 'step', "
                f"got {self.granularity!r}")

    @property
    def profile(self) -> PlatformProfile:
        """The platform's ``PROFILE``, checked against the interface."""
        from platforms.profile import PlatformProfile

        module = importlib.import_module(self.pkg.__name__ + ".profile")
        prof = getattr(module, "PROFILE", None)
        if not isinstance(prof, PlatformProfile):
            raise TypeError(
                f"platform {self.name!r}: {module.__name__}.PROFILE must be a "
                f"platforms.profile.PlatformProfile instance, got {type(prof).__name__}")
        if self.granularity == "step" and type(prof).step_grid is PlatformProfile.step_grid:
            raise TypeError(
                f"platform {self.name!r} is step-granularity but "
                f"{type(prof).__name__} does not override step_grid()")
        return prof

    @property
    def simulator(self) -> ModuleType:
        return importlib.import_module(self.pkg.__name__ + ".simulator")

    def __repr__(self) -> str:
        return f"Platform({self.name}, granularity={self.granularity})"


def _installed() -> dict[str, str]:
    return {ep.name: ep.value for ep in entry_points(group=ENTRY_POINT_GROUP)}


def load_platform(name: str | None = None, meta: Mapping[str, Any] | None = None) -> Platform:
    """Resolve a platform by name, by the perf bundle's meta, or by discovery."""
    installed = _installed()
    if name is None:
        name = (meta or {}).get("platform")
    if name is None:
        oot = [n for n in installed if n != "cuda"]
        name = oot[0] if len(oot) == 1 else "cuda"
    target = installed.get(name) or BUILTIN.get(name)
    if target is None:
        raise ValueError(
            f"unknown platform {name!r}; built in: {sorted(BUILTIN)}, "
            f"installed via {ENTRY_POINT_GROUP}: {sorted(installed)}")
    return Platform(importlib.import_module(target))
