"""Platforms: how one vLLM hardware platform differs from CUDA.

vLLM supports out-of-tree hardware through ``vllm.platform_plugins`` entry
points (``vllm-rbln`` is one). This package is the simulator's counterpart,
built the same way:

- ``platforms.spec.PlatformSpec``: one class per platform, safe to construct
  without the hardware; every capability is read off it.
- ``platforms.profile.PlatformProfile``: the profiler-side interface a spec's
  ``profile_cls`` implements.
- ``platforms._registry``: built-ins (subpackages of ``platforms/``) and
  out-of-tree specs (the ``llmservingsim.platforms`` entry-point group) in
  one registry.

Layout of a platform package, in tree or in its own distribution::

    <pkg>/__init__.py              class <Vendor>Platform(PlatformSpec)
    <pkg>/profile.py               class <Vendor>Profile(PlatformProfile)
    <pkg>/devices/<hardware>.yaml  npu_mem defaults, kv_cache_dtypes
    <pkg>/perf/<hardware>/...      step or layer perf bundles (optional)
    <pkg>/models/<model_type>.yaml architecture catalogs (optional)

``load_platform`` resolves, first hit wins: the explicit name, the perf
bundle's recorded ``platform``, ``$LLMSERVINGSIM_PLATFORM``, the platform
whose ``devices/`` describes the instance's hardware, the one platform whose
``is_available()`` is true (only when ``detect=True``, which the profiler and
bench pass because they run on the hardware), and finally ``cuda``. The
simulator never detects: it models a platform, it does not run on one.
"""

from __future__ import annotations

import functools
import logging
import os
from pathlib import Path
from typing import Any, Mapping

from platforms._registry import ENTRY_POINT_GROUP, ENV_VAR, registry, validate_spec
from platforms.spec import GRANULARITIES, NPU_MEM_KEYS, DeviceSpec, PlatformSpec

__all__ = [
    "ENTRY_POINT_GROUP", "ENV_VAR", "GRANULARITIES", "NPU_MEM_KEYS", "DeviceSpec",
    "PlatformSpec", "detect_platform", "devices", "load_device", "load_platform",
    "registry", "resolve_npu_mem", "resource_dirs", "supported_kv_cache_dtypes",
    "validate_spec",
]

logger = logging.getLogger("llmservingsim.platforms")


def detect_platform() -> PlatformSpec | None:
    """The one platform whose hardware this host has, or None. Raises when
    more than one is available: pick with ``$LLMSERVINGSIM_PLATFORM``."""
    available = []
    for spec in registry().values():
        try:
            ok = spec.is_available()
        except Exception as e:  # noqa: BLE001 - a probe must not break discovery
            logger.warning("platform %r is_available() raised %s: %s", spec.name, type(e).__name__, e)
            ok = False
        if ok:
            available.append(spec)
    if len(available) > 1:
        raise RuntimeError(
            f"several platforms are available on this host: {[s.name for s in available]}; "
            f"choose one with {ENV_VAR} or --platform")
    return available[0] if available else None


def load_platform(name: str | None = None, meta: Mapping[str, Any] | None = None, *,
                  hardware: str | None = None, detect: bool = False) -> PlatformSpec:
    """Resolve a platform; see the module docstring for the order."""
    reg = registry()
    chosen = name or (meta or {}).get("platform") or os.environ.get(ENV_VAR)
    if chosen is None and hardware is not None:
        device = load_device(hardware)
        chosen = device.platform if device else None
    if chosen is None and detect:
        spec = detect_platform()
        chosen = spec.name if spec is not None else None
    chosen = chosen or "cuda"
    if chosen not in reg:
        raise ValueError(
            f"unknown platform {chosen!r}; registered: {sorted(reg)} (built in under "
            f"platforms/, or installed through the {ENTRY_POINT_GROUP!r} entry-point group)")
    return reg[chosen]


def resource_dirs(kind: str) -> list[Path]:
    """Every registered platform's ``<resource_dir>/<kind>`` that exists, in
    registry order (built-ins first)."""
    return [p for spec in registry().values() if (p := spec.resources(kind)) is not None]


@functools.cache
def devices() -> dict[str, DeviceSpec]:
    """Every device any registered platform describes, by name.

    Built once per process, so a device yaml that does not validate fails
    here rather than at the point of use, and a device two platforms both
    claim is caught once instead of per lookup.
    """
    found: dict[str, DeviceSpec] = {}
    for spec in registry().values():
        directory = spec.resources("devices")
        for path in sorted(directory.glob("*.yaml")) if directory else []:
            device = DeviceSpec.from_yaml(path, spec.name)
            if device.name in found:
                raise ValueError(
                    f"device {device.name!r} is described by more than one platform: "
                    f"{found[device.name].source} and {path}")
            found[device.name] = device
    return found


def load_device(hardware: str) -> DeviceSpec | None:
    """The spec for ``hardware``, or None when no platform describes it (a
    cluster config then states ``npu_mem`` in full)."""
    return devices().get(hardware)


def resolve_npu_mem(hardware: str, given: Mapping[str, Any] | None) -> dict[str, Any]:
    """An instance's ``npu_mem``: the device spec's values, overridden by
    whatever the cluster config states."""
    device = load_device(hardware)
    return device.npu_mem_with(given) if device else dict(given or {})


def supported_kv_cache_dtypes(hardware: str) -> list[str] | None:
    """The KV cache dtypes ``hardware`` can run, or None when unknown."""
    device = load_device(hardware)
    return list(device.kv_cache_dtypes) if device else None
