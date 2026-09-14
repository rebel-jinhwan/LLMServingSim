"""``python -m platforms``: check the device specs and the profile interface.

Every built-in ``devices/*.yaml`` must load under its own name with a full
``npu_mem`` and a KV dtype list; a cluster config's ``npu_mem`` must win
over the spec key by key; an unknown device must fall back to the config
alone. Then the ``PlatformProfile`` checks in ``platforms.profile`` run.
"""

from platforms import BUILTIN, _devices_dirs, load_device, resolve_npu_mem, supported_kv_cache_dtypes
from platforms.profile import _selfcheck as profile_selfcheck


def main() -> None:
    seen = []
    for name, target in BUILTIN.items():
        for d in _devices_dirs(target):
            for path in sorted(d.glob("*.yaml")):
                spec = load_device(path.stem)
                assert spec["platform"] == name, (path, spec["platform"])
                assert set(spec["npu_mem"]) == {"mem_size", "mem_bw", "mem_latency"}, path
                assert spec["kv_cache_dtypes"], path
                seen.append(path.stem)
    assert seen, "no device specs found"

    merged = resolve_npu_mem("RTX4090", {"mem_size": 20, "mem_util": 0.8})
    assert merged == {"mem_size": 20, "mem_bw": 1008, "mem_latency": 0, "mem_util": 0.8}, merged
    assert resolve_npu_mem("NO-SUCH-DEVICE", {"mem_size": 1}) == {"mem_size": 1}
    assert load_device("NO-SUCH-DEVICE") is None
    assert supported_kv_cache_dtypes("RBLN-CR03") == ["auto"]
    assert supported_kv_cache_dtypes("NO-SUCH-DEVICE") is None
    print(f"ok: {len(seen)} device specs ({', '.join(seen)}); npu_mem overrides merge key by key")

    profile_selfcheck()


if __name__ == "__main__":
    main()
