"""``PlatformSpec``: the one object that describes a hardware platform.

A platform is a subclass of
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

import dataclasses
import functools
import inspect
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Mapping

if TYPE_CHECKING:
    from platforms.profile import PlatformProfile

GRANULARITIES = ("layer", "step")

NPU_MEM_KEYS = ("mem_size", "mem_bw", "mem_latency")
"""The device facts a spec states. ``mem_util`` is deliberately not one: it
scales a deployment's share of the card, not the card."""


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


@dataclasses.dataclass(frozen=True)
class DeviceSpec:
    """One piece of hardware, as the platform that runs it describes it.

    A device spec holds facts every deployment of the device shares, and
    nothing about any one deployment. It is data, not behaviour: a platform
    ships one ``devices/<hardware>.yaml`` per device rather than a subclass,
    because a device differs from another device in its numbers, while a
    platform differs from another platform in its code.

    ``name`` is the cluster config's ``hardware``, the ``profiler/perf/<name>/``
    folder and the file's own stem, all the same string, so a run that names
    the hardware resolves the spec, and the spec names the platform.
    """

    name: str
    platform: str
    """The platform that ships this device; how `hardware` resolves a platform."""
    npu_mem: Mapping[str, Any]
    """Defaults for an instance's ``npu_mem``: every key of ``NPU_MEM_KEYS``."""
    kv_cache_dtypes: tuple[str, ...]
    """What the device's attention kernels can actually run."""
    source: Path | None = None
    """The yaml this came from, so an error can name the file to fix."""

    @classmethod
    def from_yaml(cls, path: Path, platform: str) -> DeviceSpec:
        """Read and validate one ``devices/<hardware>.yaml``."""
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: a device spec must be a mapping")
        if data.get("name") != path.stem:
            raise ValueError(f"{path}: name must be {path.stem!r}, got {data.get('name')!r}")

        npu_mem = data.get("npu_mem") or {}
        missing = [k for k in NPU_MEM_KEYS if k not in npu_mem]
        if missing:
            raise ValueError(f"{path}: npu_mem is missing {missing}")
        # An unknown key is a typo or a deployment knob in the wrong file;
        # either way it would be silently ignored, so refuse it.
        unknown = sorted(set(npu_mem) - set(NPU_MEM_KEYS))
        if unknown:
            raise ValueError(
                f"{path}: npu_mem has unknown keys {unknown}; a device spec states "
                f"{list(NPU_MEM_KEYS)} only, and a deployment states the rest in its "
                f"cluster config")

        dtypes = data.get("kv_cache_dtypes")
        if not dtypes or not all(isinstance(d, str) for d in dtypes):
            raise ValueError(f"{path}: kv_cache_dtypes must be a non-empty list of strings")

        return cls(name=data["name"], platform=platform, npu_mem=dict(npu_mem),
                   kv_cache_dtypes=tuple(dtypes), source=path)

    def npu_mem_with(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        """This device's defaults, overridden key by key by one deployment's
        ``npu_mem``, so a cluster config states only what differs."""
        return {**self.npu_mem, **(given or {})}

    def supports_kv_cache_dtype(self, dtype: str) -> bool:
        return dtype in self.kv_cache_dtypes

    def __repr__(self) -> str:
        return (f"DeviceSpec(name={self.name!r}, platform={self.platform!r}, "
                f"kv_cache_dtypes={list(self.kv_cache_dtypes)})")
