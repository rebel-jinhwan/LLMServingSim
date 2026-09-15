---
title: Platforms
sidebar_position: 8
---

# Platforms

vLLM runs on hardware other than NVIDIA GPUs through out-of-tree
platform plugins: a package registers itself under the
`vllm.platform_plugins` entry-point group and overrides the platform,
scheduler, model runner and attention backend. `vllm-rbln` is one.
LLMServingSim models vLLM, so it has the same seam. A **platform** says
how a piece of hardware differs from CUDA in the three places the
simulator would otherwise assume the CUDA answer:

| Where | CUDA | A compiled-graph accelerator |
| --- | --- | --- |
| Measuring a shot | vLLM `layerwise_profile()`, one time per kernel | wall-clock of the whole forward between device syncs |
| Shape of the profile | per-layer CSVs, TP emulated on one GPU, ASTRA-Sim adds the collectives | `step.csv`: one time per padded forward, TP on real ranks, collectives inside the measured time |
| Scheduler | the in-tree port of vLLM's `Scheduler` | the vLLM plugin's own scheduler, run through vLLM's scheduler classes |

`cuda` is built in, under `platforms/cuda/`. Everything else is a separate
distribution that registers itself, exactly as its vLLM plugin does. The
worked example is
[`llmservingsim-rbln`](https://github.com/rebel-jinhwan/llmservingsim-rbln),
the platform for Rebellions NPUs: it ships the `rbln` platform, the
`RBLN-CR03` device spec, profiled step bundles and four calibrated
end-to-end examples, and LLMServingSim carries none of it.

## How a platform is chosen

Every platform is a `PlatformSpec` subclass, and one registry holds them
all: the subpackages of `platforms/` plus every class named by an
installed `llmservingsim.platforms` entry point. `platforms.load_platform`
resolves in this order and stops at the first hit:

1. `--platform <name>` on `python -m profiler` or `python -m serving`.
2. The `platform` key in the perf bundle's `meta.yaml`. The profiler
   writes it, so a simulation over a bundle needs no flag: a cluster
   config that names the hardware folder is enough.
3. `$LLMSERVINGSIM_PLATFORM`.
4. The platform whose `devices/` describes the instance's `hardware`.
5. The one platform whose `is_available()` is true, and only where that
   makes sense. The profiler and bench ask for it because they run on the
   hardware; the simulator never does, because it models a platform
   rather than running on one. Two available platforms is an error that
   names `$LLMSERVINGSIM_PLATFORM`.
6. `cuda`.

Built-ins register before plugins and the first spec with a given name
wins, so a plugin cannot silently replace a built-in. A plugin that fails
to import or fails validation is logged and skipped; the others still
register.

## Layer vs step granularity

A CUDA bundle has `granularity: layer`: the trace generator walks the
architecture yaml and emits one row per canonical layer, looked up in the
per-category CSVs, and ASTRA-Sim simulates the TP and EP collectives
between them.

A `step` bundle comes from a device that runs a compiled graph per padded
shape: it pads every prefill to `max_num_batched_tokens` query tokens and
every decode batch up to one of a fixed set of buckets, so the
latency is a step function of the padded shape and there is no per-kernel
view of the graph. The profiler therefore sweeps exactly the shapes the
runner can produce, lone prefills over the `kv_prefill` axis and decode
batches at each bucket over the `kv_decode` axis, and times each forward
whole into `tp<N>/step.csv`, keyed like `attention.csv`. The trace is then
a single `step` row per iteration with no communication row, and the
lookup snaps the batch to the profiled shapes before interpolating along
the kv axes: a prefill chunk to the profiled chunk, a decode batch up to
the next profiled `n_decode`. Those shapes are read off the CSV itself,
so profiling more buckets needs no simulator change.

With pipeline parallelism the profile is taken at the deployment's
pipeline depth: every worker times its own `execute_model`, a stage's
time is cumulative up to it, so `step.csv` carries one row per `stage`
with the difference, and the trace emits one row per stage with the
hidden state on both sides of each cut. Prefill/decode disaggregation
works with a step bundle: each stage row of a prefill instance carries
that stage's KV bytes, and the converter sends them to the paired decode
NPU. PIM attention offloading is refused, because it hangs off per-layer
rows.

## Calibrating a step bundle against a bench run

The profile times `execute_model` inside the worker. A real step also
pays the scheduler, the executor round trip and output handling, so a
raw step bundle runs a few percent fast. Two knobs carry that host time,
calibrated against `python -m bench` the way `mem_util` is calibrated
against `num_gpu_blocks`: `--step-overhead-us` on every step and
`--prefill-step-overhead-us` on top for a step that carries a prefill
chunk. Both can be set per instance in the cluster config.

They are per deployment, not per platform: a single-device decode step of
5 ms carries proportionally far more host time than a four-device MoE
step of 30 ms. Measured deployments have needed anything from 700 to 2200
us per step and 2600 to 9000 us per prefill step, moving TTFT and TPOT
mean from tens of percent low to under one percent.

Fit each knob from one measurement rather than a grid search. With
prefill/decode disaggregation the servers' Prometheus metrics split the
latency into exactly the parts the knobs cover:

| Part of the real run | Knob it sets |
| --- | --- |
| Decode inter-token latency, against the profiled decode step | `step_overhead_us` on the decode instance |
| Prefill server, arrival to done, against the bundle's prefill time | `prefill_step_overhead_us` on the prefill instance |
| `nixl_bytes_transferred` over `nixl_xfer_time_seconds` | `link_bw` |
| The remainder of TTFT | `link_latency` |

NIXL moves whole KV blocks once a request's prefill is done, so the
simulator sends each request's KV on its final prefill step, rounded up
to blocks. `link_latency` counts about three times toward TTFT, since the
link also carries the output hand-off, so fit it last and against TTFT
rather than setting the measured wait directly. The client's first token
comes from the decode server, so TTFT spans both servers plus the proxy.

A calibrated example of each shape, with the fitted numbers and the
metrics they came from, lives in
[`llmservingsim-rbln`](https://github.com/rebel-jinhwan/llmservingsim-rbln):
tensor-parallel MoE, a single-device MXFP4 model, a pipeline-parallel run
kept as ground truth only, and a two-server NIXL pair. `bench/examples`'s
`run.sh` and `validate.sh` take `EXAMPLES_DIR`, so those examples run
through the in-tree runner from wherever they live.

## Running vLLM's scheduler instead of the port

`serving/core/scheduler.py` is a port of vLLM's V1 scheduler. The
`VllmScheduler` in `serving/core/vllm_scheduler.py` instead drives vLLM's
own scheduler classes, doing on the host side what `EngineCore` does:
build the engine config from `configs/model/<model>.json`, let
`scheduler_config.get_scheduler_cls()` pick the class, size a
`KVCacheConfig` from the memory model's block count, then alternate
`schedule()` and `update_from_output()` around the simulated forward.
`--scheduler vllm` selects it for any platform, and a platform whose vLLM
plugin schedules differently assigns its own subclass in
`bind_scheduler()`, so the plugin's scheduling rules and `VLLM_*`
environment apply verbatim.

It needs vLLM importable in the simulator container. The CPU wheel is
enough:

```bash
pip install vllm==0.24.0 --extra-index-url https://wheels.vllm.ai/0.24.0/cpu
```

`python -m serving.core.vllm_scheduler` runs the port and the vLLM-driven
scheduler over the same ShareGPT requests with a constant step time. The
batches are identical while nothing is preempted. Under KV pressure vLLM
keeps block 0 of its `BlockPool` as the null block, so it has one usable
block fewer than the port at the same block count and preempts one step
earlier; with that block returned the runs are identical.

Prefill/decode disaggregation follows vLLM's NIXL flow. A prefill
instance runs each request with `max_tokens=1`, as vLLM's disaggregation
proxy does, and hands it on without recording a token, because the proxy
discards that token. A decode instance carries vLLM's
`DecodeBenchConnector`, whose scheduler side reports every prompt token
but the last as already present. That is where NIXL leaves a decode
request once its KV has arrived, so decode's first step computes one
token and emits the first output token. NIXL's asynchronous wait for one
more step before the request resumes is not modelled.

Not modelled by the vLLM-driven scheduler: `--prefix-storage`, a KV
connector in vLLM that sits below the scheduler. The port models it
directly.

## Devices

A platform can describe the devices it runs on, one yaml per device in its
`devices/` directory, named exactly as the cluster config's `hardware` and
the `profiler/perf/<hardware>/` folder. Each one is read into a `DeviceSpec`.
A device is data rather than behaviour, so there is no subclass to write: a
device differs from another device in its numbers, while a platform differs
from another platform in its code.

```yaml
# platforms/cuda/devices/RTX4090.yaml
name: RTX4090
npu_mem:
  mem_size: 24       # GB
  mem_bw: 1008       # GB/s
  mem_latency: 0     # ns
kv_cache_dtypes: [auto, fp8]
```

A spec holds hardware facts only, the ones every deployment of the device
shares. Its `npu_mem` values are the defaults for any instance naming the
device, and the instance's own `npu_mem` overrides them key by key, so a
cluster config states only what differs for that deployment. Both the
simulator and the profiler refuse a `kv_cache_dtype` outside
`kv_cache_dtypes`, the profiler before an engine boots. A device with no
spec keeps working when its cluster config states `npu_mem` in full.

Every spec is validated when the registry is built, not when it is first
used, so a mistake names its own file straight away. A `name` that does not
match the filename, a missing `npu_mem` key, an empty `kv_cache_dtypes` and a
device two platforms both claim are all refused. So is an unknown `npu_mem`
key: `mem_util` scales one deployment's share of the card rather than
describing the card, so it belongs in the cluster config, and silently
ignoring it there would be worse than failing.

| Device | Platform | `mem_size` | `mem_bw` | KV cache dtypes |
| --- | --- | --- | --- | --- |
| `RTX4090` | cuda | 24 | 1008 | auto, fp8 |
| `RTXPRO6000` | cuda | 96 | 1597 | auto, fp8 |
| `H100` | cuda | 80 | 3350 | auto, fp8 |

An installed platform's devices join the same table: `load_device` searches
every registered platform and reports which one owns the device, which is
how a cluster config that names only `hardware` resolves its platform.
`python -m platforms` checks every spec it can see and the merge.

## Writing a platform

A platform is a `PlatformSpec` subclass. The same layout works in tree
(a subpackage of `platforms/`) and out of tree (its own distribution),
and nothing but the entry point differs:

```
<pkg>/__init__.py               class <Vendor>Platform(PlatformSpec)
<pkg>/profile.py                class <Vendor>Profile(PlatformProfile)
<pkg>/devices/<hardware>.yaml   npu_mem defaults, kv_cache_dtypes
<pkg>/perf/<hardware>/...       perf bundles the platform ships, optional
<pkg>/models/<type>.yaml        architecture catalogs, optional
<pkg>/configs/cluster/*.json    cluster configs, found by name, optional
```

```python
from platforms.spec import PlatformSpec

class ExamplePlatform(PlatformSpec):
    name = "example"
    granularity = "step"

    def is_available(self):
        return importlib.util.find_spec("example_sdk") is not None

    @property
    def profile_cls(self):
        from example_pkg.profile import ExampleProfile
        return ExampleProfile
```

| Member | Default | Override when |
| --- | --- | --- |
| `name` | `""` | Always: lowercase, unique, equal to the entry-point name |
| `granularity` | `"layer"` | The device runs a compiled graph per padded shape |
| `is_available()` | `False` | The platform can probe for its hardware |
| `profile_cls` | raises | Always, to profile on the hardware |
| `bind_scheduler()` | binds nothing, so the in-tree port of vLLM's `Scheduler` runs | The platform's vLLM plugin schedules differently: assign `self.scheduler` |
| `resource_dir` | the spec module's directory | Never, in practice |

Each hook has a resolved counterpart that callers read rather than the hook
itself: `spec.profile` builds and checks `profile_cls`, and reading
`spec.scheduler` runs `bind_scheduler()` once, which assigns `self.scheduler`
or leaves it alone, in which case the in-tree port is bound. Binding is a
method and not an attribute because it may have to import the vendor's
package, which must not happen until a run asks for it.

Two rules make the spec safe to hold without the hardware, which is what
lets the simulator model a platform it cannot run on: importing the
module and constructing the spec must touch neither the hardware nor its
SDK, and everything heavy is imported inside the member that needs it.
`is_available()` is the only place that probes.

Out of tree, one entry point registers the whole thing:

```toml
[project.entry-points."llmservingsim.platforms"]
example = "example_pkg:ExamplePlatform"
```

The entry-point name must equal the spec's `name`. Anything the platform
ships as files lives next to its module: `devices/`, `perf/`, `models/`
and `configs/` are searched after the in-tree locations, so a plugin can
ship its own device specs, perf bundles, architecture catalogs and
cluster configs without touching LLMServingSim.

Cluster configs resolve by name, so a deployment the platform was
calibrated for is run without a path:

```bash
python -m serving --cluster-config example_llama_tp4.json ...
```

A path is a location, absolute or relative to the repo root. A bare name
is a lookup: the in-tree `configs/cluster/` first, then each registered
platform's. A path that names a directory is never searched for by name,
so a mistyped directory fails where it was typed.

### The profiler side

`platforms.profile.PlatformProfile` is the profiler-side interface a
spec's `profile_cls` implements. Its defaults are CUDA vLLM's, so a
subclass overrides only what differs:

| Member | Default | Override when |
| --- | --- | --- |
| `ENGINE_KWARGS` | `{}` | The platform needs engine kwargs of its own, merged under the CLI's |
| `TP_EMULATION` | `True` | TP cannot be emulated on one device, because the collectives are inside what is timed |
| `scheduler_output_cls()` | vLLM's `SchedulerOutput` | The model runner reads a subclass of it |
| `device_info()` | `{"gpu": "unknown"}` | Always: it identifies the device in `meta.yaml` |
| `measure(run_forward, iterations, catalog_slice)` | abstract | Always: how a shot is timed |
| `step_grid(args, limits)` | raises | The platform is step-granularity, where it is required |

```python
from platforms.profile import PlatformProfile

class ExampleProfile(PlatformProfile):
    TP_EMULATION = False

    def device_info(self):
        return {"gpu": example_sdk.device_name(0)}

    def measure(self, run_forward, iterations, catalog_slice):
        ...  # time run_forward() and return one TimingSample("step", ...) dict

    def step_grid(self, args, limits):
        ...  # yield the Shots the runner can actually execute
```

The spec refuses a `profile_cls` that is not a `PlatformProfile`, and a
step-granularity platform whose profile leaves `step_grid` at the
default.
