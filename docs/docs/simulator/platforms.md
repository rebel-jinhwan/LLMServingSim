---
title: Platforms
sidebar_position: 8
---

# Platforms

vLLM runs on hardware other than NVIDIA GPUs through out-of-tree
platform plugins: a package registers itself under the
`vllm.platform_plugins` entry-point group and overrides the platform,
scheduler, model runner and attention backend. `vllm-rbln` is one.
LLMServingSim models vLLM, so it has the same seam. A **platform** is a
package under `platforms/<vendor>/` that says how that hardware differs
from CUDA in the three places the simulator would otherwise assume the
CUDA answer:

| Where | CUDA (`platforms/cuda`) | RBLN (`platforms/rbln`) |
| --- | --- | --- |
| Measuring a shot | vLLM `layerwise_profile()`, one time per kernel | wall-clock of the whole forward between device syncs |
| Shape of the profile | per-layer CSVs, TP emulated on one GPU, ASTRA-Sim adds the collectives | `step.csv`: one time per padded forward, TP on real ranks, collectives inside the measured time |
| Scheduler | the in-tree port of vLLM's `Scheduler` | vllm-rbln's `RBLNScheduler`, run through vLLM's own scheduler classes |

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
shape. vllm-rbln pads every prefill to `max_num_batched_tokens` query
tokens and every decode batch up to one of a fixed set of buckets, so the
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

The first bundle, MiniMax-M2.5 on four RBLN-CR03 (tp4 + EP, vllm-rbln
0.26), against a 24-request random workload:

| Knobs | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- |
| raw bundle | -7.3% | -3.9% | -4.1% |
| 1300 us per step | -5.6% | +0.3% | -0.1% |
| 1100 us per step, +9000 us per prefill step | +0.0% | +0.7% | +0.6% |

The second bundle, gpt-oss-120b (MXFP4) on one RBLN-CR03 with four
decode buckets (1, 2, 4, 8), against 32 random-length requests:

| Knobs | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- |
| raw bundle | -11.0% | -31.7% | -29.9% |
| MiniMax's 1100 / 9000 us | +0.7% | -16.8% | -15.3% |
| 2200 us per step, +7000 us per prefill step | -0.2% | -0.9% | -0.8% |

The knobs are per deployment, not per platform: a single-device decode
step of 5 ms carries proportionally far more host time than a four-device
MoE step of 30 ms. Both examples live under `bench/examples/RBLN-CR03/`
and need vLLM and `vllm-rbln` importable to re-run, since the simulation
drives vllm-rbln's own scheduler. The gpt-oss config raises
`npu_mem.mem_size` above the card's 140 GB because the memory model sizes
an MXFP4 checkpoint at 8 bits per weight, twice its footprint; the
number is chosen so the pool holds vLLM's 227 blocks. The pp4 MiniMax
run is kept as ground truth only (`MiniMax-M2.5-pp4/NOTE.md`): the
profiler's per-rank timing does not measure a pipeline's latency yet.

### Prefill/decode disaggregation over NIXL

The third case splits Llama-3.2-1B-Instruct across two RBLN-CR03, prefill
on one and decode on the other, with vllm-rbln's `RblnNixlConnector`
behind vLLM's disaggregation proxy. The real run used the host-bounce
path over upstream NIXL and UCX. Both servers' Prometheus metrics split
its latency into parts, and each part maps to one knob:

| Part of the real run | Measured | Knob it sets |
| --- | --- | --- |
| Decode inter-token latency | 3.41 ms, against a profiled step of about 2.7 ms | `step_overhead_us` 700 on the decode instance |
| Prefill server, arrival to done | 20.92 ms per request, against 15.66 ms profiled over 2 chunks | `prefill_step_overhead_us` 2630 on the prefill instance |
| NIXL transfer | 44.7 MB per request in 2.42 ms | `link_bw` 18.5 GB/s |
| Decode's wait for remote KV, and the proxy hop | the remainder of TTFT | `link_latency` 5.3 ms |

NIXL moves whole KV blocks once a request's prefill is done, so the
simulator sends each request's KV on its final prefill step, rounded up
to blocks. `link_latency` counts about three times toward TTFT, since the
link also carries the output hand-off, so fit it against TTFT rather
than setting the measured wait directly.

| Llama-3.2-1B-Instruct PD, 48 requests | TTFT mean | TPOT mean | Latency mean |
| --- | --- | --- | --- |
| raw bundle | -50.8% | -19.7% | -22.9% |
| decode overhead only | -48.2% | +0.9% | -4.3% |
| all four knobs | -0.6% | +0.9% | +0.8% |

The example lives under `bench/examples/RBLN-CR03/Llama-3.2-1B-Instruct-pd`.
Its `vllm/` directory carries both servers' metrics next to the
per-request results.

## Running vLLM's scheduler instead of the port

`serving/core/scheduler.py` is a port of vLLM's V1 scheduler. The
`VllmScheduler` in `serving/core/vllm_scheduler.py` instead drives vLLM's
own scheduler classes, doing on the host side what `EngineCore` does:
build the engine config from `configs/model/<model>.json`, let
`scheduler_config.get_scheduler_cls()` pick the class, size a
`KVCacheConfig` from the memory model's block count, then alternate
`schedule()` and `update_from_output()` around the simulated forward. The
`rbln` platform always uses it, pinned to
`vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler`, so vllm-rbln's
scheduling rules run verbatim and its `VLLM_RBLN_*` environment applies.
`--scheduler vllm` selects it for any platform.

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

A platform can describe the devices it runs on, one yaml per device under
`platforms/<vendor>/devices/`, named exactly as the cluster config's
`hardware` and the `profiler/perf/<hardware>/` folder:

```yaml
# platforms/rbln/devices/RBLN-CR03.yaml
name: RBLN-CR03
npu_mem:
  mem_size: 140      # GB
  mem_bw: 2000       # GB/s, a placeholder until measured
  mem_latency: 0     # ns
kv_cache_dtypes: [auto]
```

A spec holds hardware facts only, the ones every deployment of the device
shares. Its `npu_mem` values are the defaults for any instance naming the
device, and the instance's own `npu_mem` overrides them key by key, so a
cluster config states only what differs for that deployment. Both the
simulator and the profiler refuse a `kv_cache_dtype` outside
`kv_cache_dtypes`, the profiler before an engine boots. A device with no
spec keeps working when its cluster config states `npu_mem` in full.

| Device | Platform | `mem_size` | `mem_bw` | KV cache dtypes |
| --- | --- | --- | --- | --- |
| `RTX4090` | cuda | 24 | 1008 | auto, fp8 |
| `RTXPRO6000` | cuda | 96 | 1597 | auto, fp8 |
| `H100` | cuda | 80 | 3350 | auto, fp8 |
| `RBLN-CR03` | rbln | 140 | 2000, placeholder | auto |

`python -m platforms` checks every built-in spec and the merge.

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
```

```python
from platforms.spec import PlatformSpec

class AcmePlatform(PlatformSpec):
    name = "acme"
    granularity = "step"

    def is_available(self):
        return importlib.util.find_spec("acme_sdk") is not None

    @property
    def profile_cls(self):
        from acme_pkg.profile import AcmeProfile
        return AcmeProfile
```

| Member | Default | Override when |
| --- | --- | --- |
| `name` | `""` | Always: lowercase, unique, equal to the entry-point name |
| `granularity` | `"layer"` | The device runs a compiled graph per padded shape |
| `is_available()` | `False` | The platform can probe for its hardware |
| `profile_cls` | raises | Always, to profile on the hardware |
| `scheduler_cls` | the in-tree port of vLLM's `Scheduler` | The platform's vLLM plugin schedules differently |
| `resource_dir` | the spec module's directory | Never, in practice |

Two rules make the spec safe to hold without the hardware, which is what
lets the simulator model a platform it cannot run on: importing the
module and constructing the spec must touch neither the hardware nor its
SDK, and everything heavy is imported inside the member that needs it.
`is_available()` is the only place that probes.

Out of tree, one entry point registers the whole thing:

```toml
[project.entry-points."llmservingsim.platforms"]
acme = "acme_pkg:AcmePlatform"
```

The entry-point name must equal the spec's `name`. Anything the platform
ships as files lives next to its module: `devices/`, `perf/` and
`models/` are searched after the in-tree locations, so a plugin can ship
its own device specs, perf bundles and architecture catalogs without
touching LLMServingSim.

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
