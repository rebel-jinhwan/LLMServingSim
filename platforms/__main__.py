"""``python -m platforms``: check the registry, device specs, resources and profiles.

Needs only pyyaml. Out-of-tree discovery is exercised with fake entry points,
so no plugin has to be installed:
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
    platforms.devices.cache_clear()


def _check_builtin_devices() -> None:
    seen = []
    for name, device in platforms.devices().items():
        assert device.name == name, (name, device)
        assert device.platform in platforms.registry(), device
        assert set(device.npu_mem) == set(platforms.NPU_MEM_KEYS), device
        assert device.kv_cache_dtypes, device
        assert device.source is not None and device.source.stem == name, device
        seen.append(name)
    assert seen, "no device specs found"

    rtx = platforms.load_device("RTX4090")
    assert rtx.supports_kv_cache_dtype("fp8") and not rtx.supports_kv_cache_dtype("fp4")
    # A deployment's npu_mem overrides the device's, key by key, and may add
    # mem_util, which is a deployment knob rather than a device fact.
    merged = platforms.resolve_npu_mem("RTX4090", {"mem_size": 20, "mem_util": 0.8})
    assert merged == {"mem_size": 20, "mem_bw": 1008, "mem_latency": 0, "mem_util": 0.8}, merged
    assert rtx.npu_mem_with(None) == dict(rtx.npu_mem)
    assert platforms.resolve_npu_mem("NO-SUCH-DEVICE", {"mem_size": 1}) == {"mem_size": 1}
    assert platforms.load_device("NO-SUCH-DEVICE") is None
    assert platforms.supported_kv_cache_dtypes("RTX4090") == ["auto", "fp8"]
    assert platforms.supported_kv_cache_dtypes("NO-SUCH-DEVICE") is None
    _check_device_validation()
    print(f"ok: {len(seen)} device specs ({', '.join(seen)}); npu_mem overrides merge key by key")


def _check_device_validation() -> None:
    """A device yaml that would be silently misread is refused instead."""
    from platforms.spec import DeviceSpec

    root = Path(tempfile.mkdtemp(prefix="device_spec_"))
    cases = [
        ("name: OTHER\nnpu_mem: {mem_size: 1, mem_bw: 1, mem_latency: 0}\n"
         "kv_cache_dtypes: [auto]\n", "name must be"),
        ("name: D1\nnpu_mem: {mem_size: 1, mem_bw: 1}\nkv_cache_dtypes: [auto]\n",
         "missing"),
        ("name: D1\nnpu_mem: {mem_size: 1, mem_bw: 1, mem_latency: 0, mem_util: 0.9}\n"
         "kv_cache_dtypes: [auto]\n", "unknown keys"),
        ("name: D1\nnpu_mem: {mem_size: 1, mem_bw: 1, mem_latency: 0}\n"
         "kv_cache_dtypes: []\n", "non-empty"),
    ]
    for text, needle in cases:
        path = root / "D1.yaml"
        path.write_text(text)
        try:
            DeviceSpec.from_yaml(path, "example")
        except ValueError as e:
            assert needle in str(e), (needle, e)
        else:
            raise AssertionError(f"a device spec with {needle!r} was accepted: {text!r}")


def _check_plugins() -> None:
    root = Path(tempfile.mkdtemp(prefix="platform_plugin_"))
    (root / "devices").mkdir()
    (root / "devices" / "EXAMPLE-D1.yaml").write_text(
        "name: EXAMPLE-D1\nnpu_mem: {mem_size: 32, mem_bw: 500, mem_latency: 0}\nkv_cache_dtypes: [auto]\n")
    (root / "perf" / "EXAMPLE-D1" / "org" / "model" / "bf16").mkdir(parents=True)
    (root / "models").mkdir()
    (root / "models" / "example_arch.yaml").write_text("catalog: {}\n")
    (root / "configs" / "cluster").mkdir(parents=True)
    (root / "configs" / "cluster" / "example_one_node.json").write_text("{}\n")

    example = types.ModuleType("example_plugin")

    class ExamplePlatform(PlatformSpec):
        name, granularity = "example", "layer"

        @property
        def resource_dir(self):
            return root

    class Misnamed(PlatformSpec):
        name = "other"

    class Duplicate(PlatformSpec):
        name = "cuda"

    example.ExamplePlatform = ExamplePlatform
    fake = [
        _EntryPoint("example", "example_plugin:ExamplePlatform", ExamplePlatform),
        _EntryPoint("wrong", "example_plugin:Misnamed", Misnamed),
        _EntryPoint("notaspec", "example_plugin:thing", object()),
        _EntryPoint("broken", "missing_pkg:Spec", error=ImportError("No module named 'missing_pkg'")),
        _EntryPoint("cuda", "example_plugin:Duplicate", Duplicate),
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
        assert "example" in reg and isinstance(reg["example"], ExamplePlatform), sorted(reg)
        assert "other" not in reg and "notaspec" not in reg and "broken" not in reg, sorted(reg)
        assert type(reg["cuda"]).__module__ == "platforms.cuda", "a plugin replaced a built-in"
        joined = "\n".join(warnings)
        for needle in ("does not match", "not a PlatformSpec subclass", "missing_pkg", "already registered"):
            assert needle in joined, (needle, warnings)

        device = platforms.load_device("EXAMPLE-D1")
        assert device.platform == "example" and device.npu_mem["mem_size"] == 32, device
        assert device.supports_kv_cache_dtype("auto"), device
        assert platforms.load_platform(hardware="EXAMPLE-D1").name == "example"
        assert root / "perf" in platforms.resource_dirs("perf")
        assert root / "models" in platforms.resource_dirs("models")
        assert root / "configs" in platforms.resource_dirs("configs")
        try:
            from serving.core.config_builder import resolve_cluster_config
        except ImportError:
            pass  # the simulator's own dependencies are absent; this check needs only pyyaml
        else:
            found = resolve_cluster_config("example_one_node.json")
            assert found == str(root / "configs" / "cluster" / "example_one_node.json"), found
            assert resolve_cluster_config("configs/cluster/single_node_single_instance.json") \
                == "../configs/cluster/single_node_single_instance.json"
            assert resolve_cluster_config("/abs/path.json") == "/abs/path.json"
            assert resolve_cluster_config("no_such_config.json") == "../no_such_config.json"
            # A path that names a directory is a location, never a lookup.
            assert resolve_cluster_config("elsewhere/example_one_node.json") \
                == "../elsewhere/example_one_node.json"

        os.environ[_registry.ENV_VAR] = "example"
        assert platforms.load_platform().name == "example"
        del os.environ[_registry.ENV_VAR]
        assert platforms.load_platform().name == "cuda"
        try:
            platforms.load_platform("nope")
        except ValueError as e:
            assert "registered" in str(e), e
        else:
            raise AssertionError("an unknown platform name was accepted")

        ExamplePlatform.is_available = lambda self: True
        assert platforms.detect_platform().name == "example"
        assert platforms.load_platform(detect=True).name == "example"
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
    _check_builtin_devices()
    _check_plugins()
    profile_selfcheck()


if __name__ == "__main__":
    sys.exit(main())
