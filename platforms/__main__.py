"""``python -m platforms``: check the registry, device specs, resources and profiles.

Needs only pyyaml. Out-of-tree discovery is exercised with fake entry points,
the way LMCache's device-plugin tests do, so no plugin has to be installed:
a valid plugin with its own devices/, perf/ and models/; a name that does not
match its entry point; a target that is not a PlatformSpec; a plugin that
fails to import; and a plugin reusing a built-in name.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import types
from pathlib import Path

import platforms
from platforms import _registry
from platforms.profile import _selfcheck as profile_selfcheck
from platforms.spec import PlatformSpec


class _EntryPoint:
    def __init__(self, name, value, target=None, error=None):
        self.name, self.value, self._target, self._error = name, value, target, error

    def load(self):
        if self._error:
            raise self._error
        return self._target


def _reset():
    _registry.registry.cache_clear()
    platforms.load_device.cache_clear()


def _check_builtin_devices() -> list[str]:
    seen = []
    for spec in platforms.registry().values():
        devices = spec.resources("devices")
        for path in sorted(devices.glob("*.yaml")) if devices else []:
            device = platforms.load_device(path.stem)
            assert device["platform"] == spec.name, (path, device["platform"])
            assert set(device["npu_mem"]) == {"mem_size", "mem_bw", "mem_latency"}, path
            assert device["kv_cache_dtypes"], path
            seen.append(path.stem)
    assert seen, "no device specs found"
    merged = platforms.resolve_npu_mem("RTX4090", {"mem_size": 20, "mem_util": 0.8})
    assert merged == {"mem_size": 20, "mem_bw": 1008, "mem_latency": 0, "mem_util": 0.8}, merged
    assert platforms.resolve_npu_mem("NO-SUCH-DEVICE", {"mem_size": 1}) == {"mem_size": 1}
    assert platforms.load_device("NO-SUCH-DEVICE") is None
    assert platforms.supported_kv_cache_dtypes("RTX4090") == ["auto", "fp8"]
    assert platforms.supported_kv_cache_dtypes("NO-SUCH-DEVICE") is None
    return seen


def _check_plugins() -> None:
    root = Path(tempfile.mkdtemp(prefix="platform_plugin_"))
    (root / "devices").mkdir()
    (root / "devices" / "ACME-X1.yaml").write_text(
        "name: ACME-X1\nnpu_mem: {mem_size: 32, mem_bw: 500, mem_latency: 0}\nkv_cache_dtypes: [auto]\n")
    (root / "perf" / "ACME-X1" / "org" / "model" / "bf16").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "models" / "acme_arch.yaml").write_text("catalog: {}\n")
    (root / "configs" / "cluster").mkdir(parents=True)
    (root / "configs" / "cluster" / "acme_one_node.json").write_text("{}\n")

    acme = types.ModuleType("acme_plugin")

    class AcmePlatform(PlatformSpec):
        name, granularity = "acme", "layer"

        @property
        def resource_dir(self):
            return root

    class Misnamed(PlatformSpec):
        name = "other"

    class Duplicate(PlatformSpec):
        name = "cuda"

    acme.AcmePlatform = AcmePlatform
    fake = [
        _EntryPoint("acme", "acme_plugin:AcmePlatform", AcmePlatform),
        _EntryPoint("wrong", "acme_plugin:Misnamed", Misnamed),
        _EntryPoint("notaspec", "acme_plugin:thing", object()),
        _EntryPoint("broken", "missing_pkg:Spec", error=ImportError("No module named 'missing_pkg'")),
        _EntryPoint("cuda", "acme_plugin:Duplicate", Duplicate),
    ]

    warnings: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())
    logger = logging.getLogger("llmservingsim.platforms")
    logger.addHandler(handler)
    real = _registry.entry_points
    _registry.entry_points = lambda group: fake if group == _registry.ENTRY_POINT_GROUP else []
    _reset()
    try:
        reg = platforms.registry()
        assert "acme" in reg and isinstance(reg["acme"], AcmePlatform), sorted(reg)
        assert "other" not in reg and "notaspec" not in reg and "broken" not in reg, sorted(reg)
        assert type(reg["cuda"]).__module__ == "platforms.cuda", "a plugin replaced a built-in"
        joined = "\n".join(warnings)
        for needle in ("does not match", "not a PlatformSpec subclass", "missing_pkg", "already registered"):
            assert needle in joined, (needle, warnings)

        device = platforms.load_device("ACME-X1")
        assert device["platform"] == "acme" and device["npu_mem"]["mem_size"] == 32, device
        assert platforms.load_platform(hardware="ACME-X1").name == "acme"
        assert root / "perf" in platforms.resource_dirs("perf")
        assert root / "models" in platforms.resource_dirs("models")
        assert root / "configs" in platforms.resource_dirs("configs")
        try:
            from serving.core.config_builder import resolve_cluster_config
        except ImportError:
            pass  # the simulator's own dependencies are absent; this check needs only pyyaml
        else:
            found = resolve_cluster_config("acme_one_node.json")
            assert found == str(root / "configs" / "cluster" / "acme_one_node.json"), found
            assert resolve_cluster_config("configs/cluster/single_node_single_instance.json") \
                == "../configs/cluster/single_node_single_instance.json"
            assert resolve_cluster_config("/abs/path.json") == "/abs/path.json"
            assert resolve_cluster_config("no_such_config.json") == "../no_such_config.json"
            # A path that names a directory is a location, never a lookup.
            assert resolve_cluster_config("elsewhere/acme_one_node.json") \
                == "../elsewhere/acme_one_node.json"

        os.environ[_registry.ENV_VAR] = "acme"
        assert platforms.load_platform().name == "acme"
        del os.environ[_registry.ENV_VAR]
        assert platforms.load_platform().name == "cuda"
        try:
            platforms.load_platform("nope")
        except ValueError as e:
            assert "registered" in str(e), e
        else:
            raise AssertionError("an unknown platform name was accepted")

        AcmePlatform.is_available = lambda self: True
        assert platforms.detect_platform().name == "acme"
        assert platforms.load_platform(detect=True).name == "acme"
        type(reg["cuda"]).is_available = lambda self: True
        try:
            platforms.detect_platform()
        except RuntimeError as e:
            assert _registry.ENV_VAR in str(e), e
        else:
            raise AssertionError("two available platforms were not refused")
    finally:
        _registry.entry_points = real
        logger.removeHandler(handler)
        os.environ.pop(_registry.ENV_VAR, None)
        for cls in (type(platforms.registry().get("cuda")),):
            if "is_available" in vars(cls) and cls.__module__ == "platforms.cuda":
                del cls.is_available  # restore the class's own probe
        _reset()
    print("ok: plugins register through entry points with their own devices/, perf/, models/ and configs/; "
          "misnamed, non-spec, broken and duplicate plugins are skipped with a warning; "
          "selection by name, env var, device and detection")


def main() -> None:
    seen = _check_builtin_devices()
    print(f"ok: {len(seen)} device specs ({', '.join(seen)}); npu_mem overrides merge key by key")
    _check_plugins()
    profile_selfcheck()


if __name__ == "__main__":
    sys.exit(main())
