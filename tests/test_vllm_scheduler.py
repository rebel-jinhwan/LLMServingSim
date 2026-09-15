"""The in-tree port of vLLM's scheduler against vLLM's own, on the same
ShareGPT requests with a constant step time. Was
``python -m serving.core.vllm_scheduler``; needs vLLM importable.
"""

import unittest

from llmservingsim.serving.core.scheduler import Scheduler
from llmservingsim.serving.core.vllm_scheduler import VllmScheduler


def _require_upstream_vllm():
    """The comparison is against upstream vLLM's scheduler, so an installed
    platform plugin makes it meaningless: the plugin's own
    check_and_update_config runs and may refuse the port's configuration
    outright (vllm-rbln rejects block_size 16 against its prefix_block_size)."""
    try:
        import vllm  # noqa: F401
        from vllm.platforms import current_platform
    except ImportError as e:
        raise unittest.SkipTest(f"vLLM is not importable: {e}") from e
    plugin = type(current_platform).__module__.split(".")[0]
    if plugin != "vllm":
        raise unittest.SkipTest(
            f"the {plugin} vLLM platform plugin is active; this check compares "
            f"against upstream vLLM's own scheduler")


def test_vllm_scheduler():
    """Run the port and the vLLM-driven scheduler on the same ShareGPT
    requests with a constant step time and require the same batches.
    Identical batches while nothing is preempted; under KV pressure vLLM's
    BlockPool keeps block 0 as the null block, so it has one usable block
    fewer and preempts one step earlier."""
    import json
    import os
    _require_upstream_vllm()
    from llmservingsim.serving.core.logger import configure_logger
    configure_logger(level="ERROR")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    model = "meta-llama/Llama-3.1-8B"
    rows = [json.loads(l) for l in open(
        f"{repo}/workloads/sharegpt-llama-3.1-8b-300-sps10.jsonl")][:20]

    def run(cls, mem_util, prefix):
        s = cls(model, 0, 0, 128, 2048, 1, 1, 1, 24, 64, 0, None, 16, 16, len(rows),
                prefix, False, None, None, True, 0, npu_memory_utilization=mem_util)
        for i, r in enumerate(rows):
            req = [i, model, r["input_toks"], r["input_toks"] + r["output_toks"], r["arrival_time_ns"], 0]
            s.add_request(req + ([r["input_tok_ids"], r["output_tok_ids"]] if prefix else []))
        arrivals = sorted(r["arrival_time_ns"] for r in rows)
        now, log, idle = 0, [], 0
        while not s.is_request_empty():
            batch = s.schedule(now, 0)
            if batch is None:
                later = [a for a in arrivals if a > now]
                idle += 1
                assert later or idle < 3, f"{cls.__name__} stalled at {now}"
                now = later[0] if later else now
                continue
            idle = 0
            log.append(tuple(sorted(batch.scheduled_tokens.items())))
            now += 20_000_000
            s.add_done(batch.batch_id + 1, 0, now)
        assert len(s.done) == len(rows), f"{cls.__name__}: {len(s.done)} of {len(rows)} done"
        return log, s.num_preemptions

    class NullBlockReturned(VllmScheduler):
        # vLLM's BlockPool keeps block 0 as the null block, so it has one
        # usable block fewer than the port at the same num_blocks. Handing it
        # one more isolates that: everything else must then agree exactly.
        def _create_scheduler(self):
            self.memory.npu_pool.num_blocks += 1
            try:
                return super()._create_scheduler()
            finally:
                self.memory.npu_pool.num_blocks -= 1

    def same(port, vllm, label):
        first = next((i for i, (a, b) in enumerate(zip(port, vllm)) if a != b), None)
        assert port == vllm, (
            f"{label}: batches diverge at step {first}: "
            f"port={port[first][:4] if first is not None else None} "
            f"vllm={vllm[first][:4] if first is not None else None}")

    from vllm.v1.core.sched.scheduler import Scheduler as UpstreamScheduler

    for mem_util, prefix in ((0.9, False), (0.9, True), (0.7, False), (0.7, True)):
        label = f"mem_util={mem_util} prefix_caching={prefix}"
        port, p_pre = run(Scheduler, mem_util, prefix)
        vllm, v_pre = run(VllmScheduler, mem_util, prefix)
        installed = VllmScheduler._last_scheduler_cls
        if installed is not UpstreamScheduler:
            # A vLLM platform plugin is active and installed its own
            # scheduler; the port mirrors upstream, so no equality holds.
            print(f"note: {label}: {installed.__module__}.{installed.__name__} ran "
                  f"({len(vllm)} steps, {v_pre} preemptions) vs the port's {len(port)} "
                  f"steps; equality is only expected against upstream vLLM")
            continue
        if p_pre == 0:
            same(port, vllm, label)
            print(f"ok: {label}: {len(port)} identical steps, no preemption")
        else:
            # Under KV pressure the null block moves the first preemption one
            # step earlier; with it returned the runs are identical.
            assert port != vllm, f"{label}: expected the null block to show"
            same(port, run(NullBlockReturned, mem_util, prefix)[0], label + " (+1 block)")
            print(f"ok: {label}: {p_pre} vs {v_pre} preemptions from vLLM's null block; "
                  f"identical with the block returned")
