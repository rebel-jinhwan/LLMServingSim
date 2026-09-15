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
  and ``cluster/<name>.json`` deployments. The simulator and the
  profiler search those after the in-tree locations. Architecture catalogs
  (``profiler/models/``) are deliberately not among them: a catalog describes
  a model, not the hardware it runs on, so a missing one is contributed
  upstream rather than shipped by a vendor.
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

GRANULARITIES = ("layer", "step")

NPU_MEM_KEYS = ("mem_size", "mem_bw", "mem_latency")
"""The memory facts a device spec states, and the keys of a cluster config's
``npu_mem`` they default. ``mem_util`` is deliberately not one: it scales a
deployment's share of the card, not the card."""


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

    _scheduler: type | None = None
    _scheduler_bound: bool = False

    def bind_scheduler(self) -> None:
        """Bind the platform's own scheduler by assigning ``self.scheduler``.

        A platform whose vLLM plugin schedules differently sets it to a
        ``serving.core.vllm_scheduler.VllmScheduler`` subclass pinned to that
        plugin's scheduler class, so the simulation schedules with the
        vendor's real code. A platform that does not override this, or
        overrides it and binds nothing, runs the in-tree port of vLLM's own
        scheduler, which is what CUDA vLLM does.

        A hook rather than an attribute because binding may have to import
        the vendor's package, which must not happen until a run asks for it.
        """
        return None

    @property
    def scheduler(self) -> type:
        """The scheduler class this platform runs. Reading it binds once:
        ``bind_scheduler()`` runs, and if it assigned nothing the in-tree
        port is bound in its place."""
        if not self._scheduler_bound:
            self._scheduler_bound = True
            self.bind_scheduler()
            if self._scheduler is None:
                from llmservingsim.serving.core.scheduler import Scheduler

                self._scheduler = Scheduler
        assert self._scheduler is not None
        return self._scheduler

    @scheduler.setter
    def scheduler(self, cls: type) -> None:
        if not inspect.isclass(cls):
            raise TypeError(
                f"platform {self.name!r}: scheduler must be a class, got {type(cls).__name__}"
            )
        self._scheduler = cls

    @property
    def resource_dir(self) -> Path:
        """Directory holding the platform's ``devices/``, ``perf/`` and ``cluster/``."""
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
    mem_size: float
    """Device memory, GB."""
    mem_bw: float
    """Device memory bandwidth, GB/s."""
    mem_latency: float
    """Device memory latency, ns."""
    support_fp8: bool = False
    """Whether the device runs fp8 weights (an ``fp8`` variant)."""
    support_fp8_kv: bool = False
    """Whether the device's attention kernel reads an fp8 KV cache. Separate
    from ``support_fp8`` because they come apart in practice: RBLN-CR03 runs
    MiniMax-M2.5's fp8 checkpoint but keeps a bf16 KV cache, the fp8 kernel
    being RBLN-CR13's. A KV dtype of ``auto`` is the model's dtype and needs
    no flag at all."""
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

        if "npu_mem" in data:
            raise ValueError(
                f"{path}: a device spec states mem_size, mem_bw and mem_latency at top "
                f"level, not under npu_mem (that block is the cluster config's)"
            )
        missing = [k for k in NPU_MEM_KEYS if k not in data]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        for key in NPU_MEM_KEYS:
            if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
                raise ValueError(f"{path}: {key} must be a number, got {data[key]!r}")
        # An unknown key is a typo or a deployment knob in the wrong file;
        # either way it would be silently ignored, so refuse it.
        known = {"name", *NPU_MEM_KEYS, "support_fp8", "support_fp8_kv"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"{path}: unknown keys {unknown}; a device spec states {sorted(known)} "
                f"only, and a deployment states the rest (mem_util, ...) in its cluster "
                f"config"
            )

        flags = {}
        for key in ("support_fp8", "support_fp8_kv"):
            flags[key] = data.get(key, False)
            if not isinstance(flags[key], bool):
                raise ValueError(f"{path}: {key} must be true or false, got {flags[key]!r}")
        if flags["support_fp8_kv"] and not flags["support_fp8"]:
            # An fp8 KV cache is fp8 arithmetic in the attention kernel; a
            # device that does not do fp8 at all cannot have one.
            raise ValueError(f"{path}: support_fp8_kv is true but support_fp8 is false")

        return cls(
            name=data["name"],
            platform=platform,
            mem_size=data["mem_size"],
            mem_bw=data["mem_bw"],
            mem_latency=data["mem_latency"],
            source=path,
            **flags,
        )

    @property
    def npu_mem(self) -> dict[str, Any]:
        """The device's memory facts in the shape of a cluster config's
        ``npu_mem`` block, which they default."""
        return {k: getattr(self, k) for k in NPU_MEM_KEYS}

    def npu_mem_with(self, given: Mapping[str, Any] | None) -> dict[str, Any]:
        """This device's defaults, overridden key by key by one deployment's
        ``npu_mem``, so a cluster config states only what differs."""
        return {**self.npu_mem, **(given or {})}

    def supports_kv_cache_dtype(self, dtype: str) -> bool:
        """``auto`` is the model's own dtype and always runs; every fp8 variant
        vLLM accepts (``fp8``, ``fp8_e4m3``, ``fp8_e5m2``, ...) needs the fp8
        attention kernel; nothing else is a KV dtype the simulator models."""
        if dtype == "auto":
            return True
        if dtype.startswith("fp8"):
            return self.support_fp8_kv
        return False

    def __repr__(self) -> str:
        return (
            f"DeviceSpec(name={self.name!r}, platform={self.platform!r}, "
            f"support_fp8={self.support_fp8}, support_fp8_kv={self.support_fp8_kv})"
        )
