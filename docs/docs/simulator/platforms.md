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

`platforms.load_platform` resolves in this order and stops at the first
hit:

1. `--platform <name>` on `python -m profiler` or `python -m serving`.
2. The `platform` key in the perf bundle's `meta.yaml`. The profiler
   writes it, so a simulation run over an RBLN bundle needs no flag: a
   cluster config that names the hardware folder is enough.
3. Exactly one installed `llmservingsim.platforms` entry point that is
   not `cuda`. This mirrors vLLM's own one-plugin-at-a-time rule and is
   how a vendor ships its platform inside its own vLLM plugin package.
4. `cuda`.

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

Pipeline parallelism, prefill/decode disaggregation and PIM attention
offloading are refused with a step bundle: all three hang off per-layer
rows.

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

Not modelled by the vLLM-driven scheduler: prefill/decode disaggregation
and `--prefix-storage`. Both are KV connectors in vLLM and sit below the
scheduler; the port models them directly.

## Writing a platform

```
platforms/<vendor>/__init__.py    NAME = "<vendor>"; GRANULARITY = "layer" | "step"
platforms/<vendor>/profile.py     ENGINE_KWARGS, TP_EMULATION, device_info(), measure(), step_grid()
platforms/<vendor>/simulator.py   scheduler_class()
```

`profile.py` may import vLLM and torch; `simulator.py` may import
`serving`. Neither is imported until the profiler or the simulator asks
for it. To ship the platform inside a vLLM plugin package, give it the
same three modules and register it:

```toml
[project.entry-points."llmservingsim.platforms"]
<vendor> = "<package>.<module>"
```
