"""Drive vLLM's own scheduler instead of the in-tree port.

``VllmScheduler`` does what ``vllm.v1.engine.core.EngineCore`` does on the
host side, and nothing the model executor does: build the ``VllmConfig``
from ``EngineArgs``, let ``scheduler_config.get_scheduler_cls()`` pick the
class (upstream ``Scheduler``, or whatever a platform plugin such as
vllm-rbln installs), hand it a ``KVCacheConfig`` sized from the memory
model, then alternate ``schedule()`` and ``update_from_output()`` with the
simulated forward in between. The ``ModelRunnerOutput`` fed back is the
one the real runner would produce: one sampled token for every request
that reached the end of what it has, none for a request still mid-prefill.

The class keeps the port's outward shape (``schedule`` / ``add_done`` /
``add_request`` / ``is_request_empty`` and the reporting helpers), so the
main loop, DP barrier and trace generator are unchanged. It needs vLLM
importable in the simulator container; the CPU wheel is enough.

P/D disaggregation follows vLLM's NIXL flow. A prefill instance runs each
request with ``max_tokens=1``, as vLLM's disaggregation proxy does, and hands
it on once that step completes; the token it sampled is discarded, so it
records no TTFT. A decode instance carries vLLM's ``DecodeBenchConnector``,
whose scheduler side treats every prompt token but the last as already
present, which is where NIXL leaves a decode request once its KV has arrived
(``_update_waiting_for_remote_kv`` backs off one token on a full-prompt hit):
the first decode step computes that last token and emits the first output
token. The KV transfer itself is the prefill trace's send to the paired
decode NPU. Not modelled: NIXL's asynchronous wait
(``WAITING_FOR_REMOTE_KVS`` for one more step), which calibration absorbs.

Not supported here: a lower prefix tier (``--prefix-storage``), a KV
connector in vLLM that sits below the scheduler; the port models it directly.
"""

from __future__ import annotations

import atexit
import bisect
import json
import shutil
import tempfile
from typing import Any

from .request import Batch, Request, RequestStatus
from .scheduler import Scheduler


class VllmScheduler(Scheduler):
    # Extra EngineArgs a platform pins, e.g. its scheduler_cls. A subclass in
    # platforms/<vendor>/simulator.py sets this.
    ENGINE_ARGS: dict[str, Any] = {}
    _last_scheduler_cls: type | None = None  # what get_scheduler_cls() resolved to, for the self-check

    def __init__(self, *args, **kwargs):
        # The KV dtype vLLM should see: 'auto' follows the activation dtype,
        # which is not the weight dtype the memory model sizes with.
        self._kv_cache_dtype = kwargs.get("kv_cache_dtype", "auto")
        super().__init__(*args, **kwargs)
        if self.memory.storage_pool is not None:
            raise NotImplementedError(
                "--prefix-storage is a KV connector in vLLM and is not modelled by "
                "the vLLM-driven scheduler; use the in-tree scheduler")

        self._vllm_config = self._create_engine_config()
        self._core = self._create_scheduler()
        # Sim requests that have not arrived yet, sorted like the port's waiting.
        self._arrivals: list[Request] = []
        self._sim: dict[str, Request] = {}        # request_id -> sim Request
        self._vreq: dict[str, Any] = {}           # request_id -> vLLM Request
        # batch_id -> (SchedulerOutput, {request_id: emits a token this step})
        self._outputs: dict[int, tuple[Any, dict[str, bool]]] = {}
        self.logger.info("vLLM scheduler: %s", type(self._core).__name__)

    # ==================== vLLM bring-up ====================

    def _create_engine_config(self):
        from vllm import EngineArgs

        # Materialise configs/model/<model>.json as a model directory so
        # ModelConfig loads offline, the way the profiler's engine does.
        tmpdir = tempfile.mkdtemp(prefix="servingsim_model_")
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
        with open(f"{tmpdir}/config.json", "w") as f:
            json.dump(self.config, f)

        kwargs: dict[str, Any] = dict(
            model=tmpdir,
            skip_tokenizer_init=True,
            load_format="dummy",
            # No model runs here, so vLLM's default compilation settings are
            # left alone; vllm-rbln rejects enforce_eager for gpt-oss.
            seed=0,
            # The activation dtype, which vLLM reads off the checkpoint. The
            # simulator's --dtype is the weight footprint (fp8 / int8 for a
            # quantized checkpoint) and is not what a quantization method
            # accepts here.
            dtype="auto",
            kv_cache_dtype=self._kv_cache_dtype,
            max_model_len=self.config['max_position_embeddings'],
            max_num_seqs=self.max_num_seqs,
            max_num_batched_tokens=self.max_num_batched_tokens,
            enable_chunked_prefill=self.enable_chunked_prefill,
            long_prefill_token_threshold=self.long_prefill_token_threshold,
            enable_prefix_caching=self.enable_prefix_caching,
            block_size=self.memory.block_size,
            pipeline_parallel_size=self.pp_size,
            # The main loop keeps one batch per pipeline stage in flight
            # itself, and the trace has no async output to overlap.
            async_scheduling=False,
        )
        kwargs.update(self.ENGINE_ARGS)
        if self.pd_type == "decode":
            # All but the last prompt token arrive from the prefill instance.
            from vllm.config import KVTransferConfig
            kwargs["kv_transfer_config"] = KVTransferConfig(
                kv_connector="DecodeBenchConnector", kv_role="kv_both")
        return EngineArgs(**kwargs).create_engine_config()

    def _create_scheduler(self):
        import torch
        from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec,
        )
        from vllm.v1.structured_output import StructuredOutputManager

        cfg = self._vllm_config
        mc, pc = cfg.model_config, cfg.parallel_config
        # The memory model already turned mem_size * mem_util - weights into a
        # block count; vLLM's profiling run would do the same on a device.
        num_blocks = self.memory.npu_pool.num_blocks
        spec = FullAttentionSpec(
            block_size=self.memory.block_size,
            num_kv_heads=mc.get_num_kv_heads(pc),
            head_size=mc.get_head_size(),
            dtype=torch.float8_e4m3fn if self._kv_cache_dtype == "fp8" else mc.dtype,
        )
        layers = [f"layers.{i}" for i in range(mc.get_num_layers(pc))]
        kv_cache_config = KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec(layers, spec)])
        cfg.cache_config.num_gpu_blocks = num_blocks
        block_size, hash_block_size = resolve_kv_cache_block_sizes(kv_cache_config, cfg)

        # EngineCore.__init__, minus the executor.
        scheduler_cls = cfg.scheduler_config.get_scheduler_cls()
        VllmScheduler._last_scheduler_cls = scheduler_cls
        # A vLLM platform plugin's check_and_update_config runs inside
        # create_engine_config() and may assign scheduler_config.scheduler_cls
        # itself, which silently outranks the one a platform pinned here. The
        # simulation then schedules with a class the deployment does not run,
        # and nothing in the results says so.
        pinned = self.ENGINE_ARGS.get("scheduler_cls")
        resolved = f"{scheduler_cls.__module__}.{scheduler_cls.__qualname__}"
        if isinstance(pinned, str) and pinned != resolved:
            self.logger.warning(
                "scheduler_cls %s was overridden by the vLLM platform plugin, which "
                "installed %s instead; the plugin's environment is not the one this "
                "platform expects", pinned, resolved)
        self._block_hasher = None
        if cfg.cache_config.enable_prefix_caching:
            from vllm.utils.hashing import get_hash_fn_by_name
            from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
            hash_fn = get_hash_fn_by_name(cfg.cache_config.prefix_caching_hash_algo)
            init_none_hash(hash_fn)
            self._block_hasher = get_request_block_hasher(hash_block_size, hash_fn)
        return scheduler_cls(
            vllm_config=cfg,
            kv_cache_config=kv_cache_config,
            structured_output_manager=StructuredOutputManager(cfg),
            block_size=block_size,
            hash_block_size=hash_block_size,
            include_finished_set=False,
            log_stats=False,
        )

    def _admit(self, current):
        """Hand every request that has arrived by ``current`` to vLLM."""
        from vllm import SamplingParams
        from vllm.v1.request import Request as VllmRequest

        while self._arrivals and self._arrivals[0].arrival <= current:
            req = self._arrivals.pop(0)
            request_id = str(req.id)
            # Real token ids when the workload carries them (prefix caching
            # then hits exactly where vLLM would); a private range otherwise.
            prompt = list(req.input_hash_ids) if req.input_hash_ids else \
                [req.id * (1 << 20) + i for i in range(req.input)]
            vreq = VllmRequest(
                request_id=request_id,
                prompt_token_ids=prompt,
                sampling_params=SamplingParams(
                    # A prefill instance computes the prompt and one token, as
                    # the disaggregation proxy asks it to.
                    max_tokens=1 if self.pd_type == "prefill" else max(1, req.output - req.input),
                    ignore_eos=True),
                pooling_params=None,
                arrival_time=req.arrival / 1e9,
                block_hasher=self._block_hasher,
            )
            self._sim[request_id] = req
            self._vreq[request_id] = vreq
            self._core.add_request(vreq)

    def _next_token(self, req):
        k = req.num_tokens_reached - req.input
        if req.output_hash_ids and k < len(req.output_hash_ids):
            return int(req.output_hash_ids[k])
        return 0

    def _refresh_views(self):
        # The main loop's progress line reads these two lists.
        self.running = [self._sim[r.request_id] for r in self._core.running]
        self.waiting = self._arrivals + [
            self._sim[r.request_id] for r in self._core.waiting]

    # ==================== scheduling ====================

    def schedule(self, current, sys, batch_id=-1):
        if sys != self.start_npu:
            return self._schedule_existing(sys, batch_id)
        existing = self._schedule_existing(sys, batch_id)
        if existing is not None:
            return existing
        if len(self.inflight) >= self.pp_size:
            return None

        self._admit(current)
        if not self._core.has_requests():
            return None
        out = self._core.schedule()
        self._refresh_views()
        if out.total_num_scheduled_tokens == 0:
            return None
        return self._build_batch(current, sys, out)

    def _build_batch(self, current, sys, out):
        """Turn a SchedulerOutput into the Batch the trace generator reads.

        ``num_computed_tokens`` was already advanced by vLLM's
        ``_update_after_schedule``, so the pre-step value is that minus this
        step's tokens. Prefill-vs-decode is by scheduled token count, as in
        the port.
        """
        total_len = kv_len = num_prefill = num_decode = 0
        pd_kv_send_tokens = 0
        q_list, k_list = [], []
        prefill_q_list, prefill_k_list, decode_k_list = [], [], []
        emits: dict[str, bool] = {}
        reqs = []
        for request_id, num_new in out.num_scheduled_tokens.items():
            vreq = self._vreq[request_id]
            req = self._sim[request_id]
            computed_after = vreq.num_computed_tokens
            computed_before = computed_after - num_new
            if self.pd_type == "prefill" and computed_after >= vreq.num_prompt_tokens:
                # NIXL moves a request's KV once its prefill is done, and moves
                # whole blocks: the decode side pulls every block the prompt
                # occupies (measured on RBLN-CR03, 44.7 MB per request for
                # Llama-3.2-1B, about 1.3 blocks of 1024 tokens).
                block = self.memory.block_size
                pd_kv_send_tokens += -(-vreq.num_prompt_tokens // block) * block
            if req.queuing_delay < 0:
                # First time scheduled. What vLLM already counts as computed
                # on a brand-new request is its prefix-cache hit.
                req.prefix_cache_hit = req.npu_cache_hit = req.storage_cache_hit = computed_before
                if req.is_init:
                    req.set_que_delay(current)
            req.num_computed_tokens = computed_after
            # The runner samples for a request that has caught up to its
            # length; a chunk that leaves it mid-prompt yields no token.
            emits[request_id] = computed_after >= vreq.num_tokens

            total_len += num_new
            q_list.append(num_new)
            k_list.append(computed_before)
            if num_new > 1:
                num_prefill += 1
                prefill_q_list.append(num_new)
                prefill_k_list.append(computed_before)
            else:
                num_decode += 1
                kv_len += computed_before
                decode_k_list.append(computed_before)
            reqs.append(req)

        pool = self._core.kv_cache_manager.block_pool
        kv_used = (pool.num_gpu_blocks - pool.get_num_free_blocks()) * self.memory.npu_pool.bytes_per_block
        batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list,
                      num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list,
                      current, kv_used, 0, 0, pd_kv_send_tokens=pd_kv_send_tokens)
        batch.fired.append(sys)
        batch.requests.extend(reqs)
        # Keyed by the sim request id like the port's, for whoever reads it.
        batch.scheduled_tokens = {int(k): v for k, v in out.num_scheduled_tokens.items()}
        batch.write_through = 0
        self.inflight.append(batch)
        self._outputs[batch.batch_id] = (out, emits)
        self.recompute_tokens += sum(
            out.num_scheduled_tokens[r] for r in out.scheduled_cached_reqs.resumed_req_ids)
        self.logger.info("Scheduling new batch #%d to NPU[%d]", batch.batch_id, sys)
        return batch

    # ==================== completion ====================

    def add_done(self, id, sys, finish):
        prompt_t = gen_t = 0
        end_reqs = []
        if not self.inflight:
            return prompt_t, gen_t, end_reqs

        batch, idx = None, 0
        id -= 1
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch, idx = b, i
        if batch is None or sys in batch.end:
            return prompt_t, gen_t, end_reqs
        batch.end.append(sys)
        # A prefill instance also waits for its paired decode NPUs, which
        # receive the KV it ships.
        last_npu = self.num_npus * (2 if self.pd_type == "prefill" else 1) - 1
        if self.start_npu not in batch.end or (self.start_npu + last_npu) not in batch.end:
            return prompt_t, gen_t, end_reqs
        self.logger.info("Batch #%d is done", batch.batch_id)

        step = self._outputs.pop(batch.batch_id, None)
        if step is not None:  # a DP dummy batch has no vLLM step behind it
            out, emits = step
            prompt_t, gen_t, end_reqs = self._complete_step(out, emits, batch, finish)
        del self.inflight[idx]
        self._refresh_views()
        return prompt_t, gen_t, end_reqs

    def _complete_step(self, out, emits, batch, finish):
        from vllm.v1.outputs import ModelRunnerOutput

        prompt_t = gen_t = 0
        end_reqs = []
        request_ids = list(out.num_scheduled_tokens)
        sampled = []
        for request_id in request_ids:
            req = self._sim[request_id]
            if emits[request_id] and req.status != RequestStatus.FINISHED:
                sampled.append([self._next_token(req)])
            else:
                sampled.append([])
        self._core.update_from_output(out, ModelRunnerOutput(
            req_ids=request_ids,
            req_id_to_index={r: i for i, r in enumerate(request_ids)},
            sampled_token_ids=sampled,
        ))

        for request_id, tokens in zip(request_ids, sampled):
            req = self._sim[request_id]
            vreq = self._vreq[request_id]
            if req.status == RequestStatus.FINISHED:
                continue  # pp_size > 1: finished on an earlier in-flight batch
            num_new = out.num_scheduled_tokens[request_id]
            computed_before = req.num_computed_tokens - num_new if req.num_computed_tokens >= num_new else 0
            # Prompt tokens this step, plus the prefix hit the first time. A
            # decode instance's prompt was already counted where it was prefilled.
            if computed_before < req.input and self.pd_type != "decode":
                prompt_t += min(num_new, req.input - computed_before)
                if computed_before <= req.prefix_cache_hit:
                    prompt_t += req.prefix_cache_hit
            if self.pd_type == "prefill":
                if vreq.is_finished():
                    # Prompt computed: its KV ships to a decode instance, and the
                    # token sampled here is discarded, so no TTFT is recorded.
                    self.logger.info("Request #%d is prefill done, sent to decode instance", req.id)
                    req.status = RequestStatus.FINISHED
                    end_reqs.append(req)
                continue
            if tokens:
                req.num_tokens_reached += 1
                gen_t += 1
                if req.is_init:
                    req.is_init = False
                    req.set_ttft(finish)
                else:
                    req.add_itl(finish)
            if vreq.is_finished():
                self.logger.info("Request #%d is done", req.id)
                req.add_latency(finish)
                req.status = RequestStatus.FINISHED
                self.done.append(req)
                end_reqs.append(req)
        self.num_preemptions = sum(v.num_preemptions for v in self._vreq.values())
        return prompt_t, gen_t, end_reqs

    # ==================== queue management ====================

    def add_request(self, req, is_init=True):
        new_req = Request(*req, is_init=is_init)
        bisect.insort(self._arrivals, new_req, key=lambda r: (r.arrival, r.id))
        self.waiting = list(self._arrivals)

    def add_decode(self, req):
        """Take over a request whose prompt KV a prefill instance just shipped.

        It is admitted on the next ``schedule()`` like any arrival; the
        DecodeBenchConnector then reports all but its last prompt token as
        present, so the first step computes one token.
        """
        req.instance_id = self.instance_id
        req.status = RequestStatus.WAITING
        req.num_computed_tokens = 0
        bisect.insort(self._arrivals, req, key=lambda r: (r.arrival, r.id))
        self._refresh_views()

    def is_request_empty(self):
        # Not has_requests(): that also counts requests finished in the last
        # update_from_output until the next schedule() clears them.
        return (not self._arrivals and not self.inflight
                and self._core.get_num_unfinished_requests() == 0)
