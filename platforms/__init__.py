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
    platforms/<vendor>/devices/<hardware>.yaml
                                     hardware facts for one device: npu_mem
                                     defaults and the KV dtypes it supports

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

import functools
import importlib
import importlib.util
from importlib.metadata import entry_points
from pathlib import Path
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


def _devices_dirs(target: str) -> list[Path]:
    spec = importlib.util.find_spec(target)
    return [Path(d) / "devices" for d in (spec.submodule_search_locations or [])] if spec else []


@functools.cache
def load_device(hardware: str) -> dict[str, Any] | None:
    """The spec for ``hardware`` from ``platforms/<vendor>/devices/<hardware>.yaml``,
    with ``platform`` set to the vendor it was found under, or None when no
    platform describes that device (a cluster config then has to state
    ``npu_mem`` in full, as before).

    Built-in vendors are searched first and installed entry points only on a
    miss, so looking up built-in hardware never imports a plugin package.
    """
    import yaml

    for group in (BUILTIN, _installed()):
        found = [(name, d / f"{hardware}.yaml") for name, target in group.items()
                 for d in _devices_dirs(target) if (d / f"{hardware}.yaml").is_file()]
        if len(found) > 1:
            raise ValueError(
                f"device {hardware!r} is described by more than one platform: "
                f"{[str(p) for _, p in found]}")
        if found:
            name, path = found[0]
            data = yaml.safe_load(path.read_text()) or {}
            if data.get("name") != hardware:
                raise ValueError(f"{path}: name must be {hardware!r}, got {data.get('name')!r}")
            return {**data, "platform": name}
    return None


def resolve_npu_mem(hardware: str, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """An instance's ``npu_mem``: the device spec's values, overridden by
    whatever the cluster config states."""
    device = load_device(hardware) or {}
    return {**(device.get("npu_mem") or {}), **(given or {})}


def supported_kv_cache_dtypes(hardware: str) -> list[str] | None:
    """The KV cache dtypes ``hardware`` can run, or None when unknown."""
    device = load_device(hardware)
    return list(device["kv_cache_dtypes"]) if device and "kv_cache_dtypes" in device else None


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
