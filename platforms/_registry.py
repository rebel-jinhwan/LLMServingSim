"""Discovery of built-in and out-of-tree platforms, modelled on LMCache's
``_device_detect``.

- Built-in platforms are ``PlatformSpec`` subclasses defined in a subpackage
  of ``platforms/`` (its ``__init__``); no registration list.
- Out-of-tree platforms are classes named by the ``llmservingsim.platforms``
  entry-point group, for example in a vendor package's pyproject.toml::

      [project.entry-points."llmservingsim.platforms"]
      example = "llmservingsim_example:ExamplePlatform"

  The target must be a ``PlatformSpec`` subclass (not an instance), must
  construct with no arguments, and its ``name`` must equal the entry-point
  name.
- Built-ins register first, then plugins in (name, value) order. The first
  spec with a given name wins; later ones are logged and ignored, so a plugin
  cannot silently replace a built-in.
- A broken plugin (import error, failed validation) is logged and skipped;
  it does not stop the others.
- The registry is built once per process: installing a plugin needs a new
  process.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import logging
import pkgutil
import re
from importlib.metadata import entry_points

from platforms.spec import GRANULARITIES, PlatformSpec

ENTRY_POINT_GROUP = "llmservingsim.platforms"
ENV_VAR = "LLMSERVINGSIM_PLATFORM"

logger = logging.getLogger("llmservingsim.platforms")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


def validate_spec(obj: object, expected_name: str | None = None) -> PlatformSpec:
    """Instantiate and check a platform class; raise with the reason if it
    does not meet the rules in ``platforms.spec``."""
    if not (inspect.isclass(obj) and issubclass(obj, PlatformSpec) and obj is not PlatformSpec):
        raise TypeError(f"{obj!r} is not a PlatformSpec subclass")
    spec = obj()
    if not _NAME.match(spec.name):
        raise ValueError(f"{obj.__name__}.name must be lowercase [a-z0-9_], got {spec.name!r}")
    if expected_name is not None and spec.name != expected_name:
        raise ValueError(
            f"entry point name {expected_name!r} does not match {obj.__name__}.name {spec.name!r}")
    if spec.granularity not in GRANULARITIES:
        raise ValueError(
            f"{obj.__name__}.granularity must be one of {GRANULARITIES}, got {spec.granularity!r}")
    return spec


def _builtin_specs() -> list[PlatformSpec]:
    import platforms

    specs = []
    for mod in sorted(pkgutil.iter_modules(platforms.__path__), key=lambda m: m.name):
        if not mod.ispkg:
            continue
        modname = f"platforms.{mod.name}"
        try:
            module = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001 - one broken platform must not hide the rest
            logger.warning("skipping built-in platform %s: %s: %s", modname, type(e).__name__, e)
            continue
        for obj in vars(module).values():
            if inspect.isclass(obj) and issubclass(obj, PlatformSpec) \
                    and obj is not PlatformSpec and obj.__module__ == modname:
                try:
                    specs.append(validate_spec(obj))
                except Exception as e:  # noqa: BLE001
                    logger.warning("skipping built-in platform %s.%s: %s", modname, obj.__name__, e)
    return specs


def _plugin_specs() -> list[PlatformSpec]:
    specs = []
    for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: (e.name, e.value)):
        try:
            specs.append(validate_spec(ep.load(), expected_name=ep.name))
        except Exception as e:  # noqa: BLE001 - a broken plugin is skipped, not fatal
            logger.warning("skipping platform plugin %s = %s: %s: %s",
                           ep.name, ep.value, type(e).__name__, e)
    return specs


@functools.cache
def registry() -> dict[str, PlatformSpec]:
    """Every platform, by name: built-ins first, then plugins."""
    reg: dict[str, PlatformSpec] = {}
    for spec in [*_builtin_specs(), *_plugin_specs()]:
        if spec.name in reg:
            logger.warning("ignoring platform %r from %s: the name is already registered by %s",
                           spec.name, type(spec).__module__, type(reg[spec.name]).__module__)
            continue
        reg[spec.name] = spec
    return reg
