"""Profiler side of the RBLN platform.

vllm-rbln compiles a fixed shape per step: a prefill step is one request
padded to ``max_num_batched_tokens`` query tokens, a decode step is the
batch padded up to a bucket (``vllm_rbln/v1/worker/dp_utils.py``,
``determine_batch_execution_and_padding``). Latency is a step function of
the padded shape and there is no per-kernel view of the compiled graph, so
the sweep visits exactly the shapes the runner can produce and times each
forward whole.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Iterator

from profiler.core.hooks.batch import Shot
from profiler.core.hooks.timings import TimingSample

if TYPE_CHECKING:
    from profiler.core.config import ProfileArgs
    from profiler.core.engine import RuntimeLimits

# Merged over profiler.core.config.HOST_ENGINE_DEFAULTS. enforce_eager would
# bypass the compiled graph that is being measured, and vLLM's dummy loader
# draws its random weights with a torch Generator on the model's device,
# which torch-rbln does not provide ("Expected a 'cpu' device type for
# generator but found 'rbln'"), so the real checkpoint is loaded.
ENGINE_KWARGS: dict = {"enforce_eager": False, "load_format": "auto"}

# The collectives are inside the compiled graph, so tp<N> is measured on N
# real ranks; there is nothing for ASTRA-Sim to add.
TP_EMULATION = False


def scheduler_output_cls():
    # The RBLN runner reads kv_cache_copy_ops off every step.
    from vllm_rbln.v1.core.rbln_scheduler import RBLNSchedulerOutput
    return RBLNSchedulerOutput


def device_info() -> dict:
    out = {"gpu": "unknown", "rebel_version": "unknown"}
    try:
        import rebel
        out["rebel_version"] = getattr(rebel, "__version__", "unknown")
        out["gpu"] = rebel.get_npu_name(0) or "unknown"
    except Exception:
        pass
    return out


def decode_buckets(max_num_seqs: int) -> list[int]:
    """The decode batch sizes the runner pads to, from vllm-rbln's own
    bucketing manager and ``VLLM_RBLN_DECODE_BATCH_BUCKET_*`` when the
    plugin is installed; powers of two plus the cap otherwise."""
    try:
        from vllm_rbln import envs
        from vllm_rbln.v1.worker.bucketing import get_bucketing_manager
        mgr = get_bucketing_manager(
            envs.VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY,
            max_batch_size=max_num_seqs,
            min_batch_size=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_MIN,
            step=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_STEP,
            limit=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT,
            manual_buckets=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS,
        )
        return sorted(set(int(b) for b in mgr.decode_batch_buckets))
    except ImportError:
        buckets, b = [], 1
        while b < max_num_seqs:
            buckets.append(b)
            b *= 2
        return buckets + [max_num_seqs]


def step_grid(args: ProfileArgs, limits: RuntimeLimits) -> Iterator[Shot]:
    """Shots for step.csv: lone prefills at the padded chunk over the
    kv_prefill axis, and decode batches at each bucket over the kv_decode
    axis. Never mixed, matching the scheduler."""
    from profiler.core.categories import _ATTN_KV_START, _geometric_grid

    def aligned(n: int) -> int:
        # A request holds whole blocks; vllm-rbln deployments run large ones.
        return -(-n // limits.block_size) * limits.block_size

    kv_cap = min(args.attention_max_kv, limits.max_model_len)
    kv_vals = _geometric_grid(kv_cap, _ATTN_KV_START, factor=args.attention_kv_factor)
    chunk = limits.max_num_batched_tokens

    for kp in kv_vals:
        if chunk + kp + 1 > limits.max_model_len:
            continue
        if aligned(chunk + kp) > limits.num_cache_tokens:
            continue
        yield Shot.attention(prefill_chunk=chunk, kv_prefill=kp, n_decode=0, kv_decode=0)

    # The runner buckets the per-stage decode batch, max_num_seqs // pp.
    pp = int((args.engine_kwargs or {}).get("pipeline_parallel_size", 1))
    for n in decode_buckets(max(1, limits.max_num_seqs // pp)):
        for kd in kv_vals:
            if kd == 0 or 1 + kd + 1 > limits.max_model_len:
                continue
            if n * aligned(1 + kd) > limits.num_cache_tokens:
                continue
            yield Shot.attention(prefill_chunk=0, kv_prefill=0, n_decode=n, kv_decode=kd)


def measure(run_forward: Callable[[], None], iterations: int, catalog_slice: dict) -> list[dict]:
    """Wall-clock the forwards between device syncs; one ``step`` sample."""
    sync: Callable[[], None]
    try:
        import torch
        sync = torch.rbln.synchronize
    except (ImportError, AttributeError):
        def sync() -> None:
            pass  # execute_model already copies the sampled ids to host
    sync()
    t0 = time.perf_counter()
    for _ in range(iterations):
        run_forward()
    sync()
    us = (time.perf_counter() - t0) * 1e6 / max(1, iterations)
    return [TimingSample(layer="step", microseconds=us).as_dict()]
