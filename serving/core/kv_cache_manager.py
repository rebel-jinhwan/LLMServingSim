"""Tiered KV cache manager over per-tier :class:`BlockPool`s.

Port of vLLM v0.19.0's ``vllm/v1/core/kv_cache_manager.py`` plus the parts of
``single_type_kv_cache_manager.py`` that a full-attention, single-group model
needs, extended with a memory hierarchy.

The tiers share **one key space**. Block hashes are chained once at the NPU
block size (``hash(parent_hash, block_tokens)``, as vLLM does), and a lower tier
whose blocks are ``factor`` times larger keys on every ``factor``-th hash -- the
last fine hash of each coarse block. Because the chain is cumulative that single
hash identifies the whole prefix up to that point, so no concatenation and no
second hash function is needed. This is exactly
``offloading/scheduler.py::_get_block_hashes``.

Consequences of that shared key space, all deliberate:

* one walk over a request's hashes yields both the NPU hit and the lower-tier
  hit, instead of comparing two independent tree traversals
* the recoverable prefix is prefix-shaped: the walk stops at the first miss, so
  freeing a request's blocks in reverse order (tail first) keeps the head --
  and therefore the recoverable prefix -- alive longest
* only full blocks are indexed, so a resumed request always recomputes the
  remainder past its last complete block boundary: up to ``block_size`` tokens
  from the NPU, up to the coarse chunk from a lower tier

Traffic model. Recall (lower tier -> NPU) is on the critical path and is
charged. The write-through (NPU -> lower tier) is not: vLLM's own
``OffloadingConnector`` defers it to the next engine step and runs it on a
dedicated stream precisely so it cannot delay token generation, so its bytes are
reported for energy accounting only. Eviction from the NPU costs nothing --
either the data is a finished request's cache, or a copy already exists below.
"""

from .block_pool import NONE_HASH, BlockPool, Device


def cdiv(a, b):
    return -(-a // b)


def request_block_hashes(req, block_size):
    """Chained block hashes over ``input_hash_ids + output_hash_ids``.

    Computed once and cached on the request. The simulator knows the whole token
    sequence up front, so unlike vLLM -- which extends the chain each time an
    output token is appended -- the entire chain can be built in one pass. That
    is not future information leaking into scheduling: indexing into the chain
    is gated by ``num_computed_tokens`` at every call site, so a block only
    becomes insertable after its tokens have actually been computed.

    Returns an empty list for workloads that carry no token ids, which disables
    prefix caching for those requests without disabling allocation.
    """
    cached = req.block_hashes
    if cached is not None:
        return cached

    tokens = req.input_hash_ids
    if not tokens:
        req.block_hashes = []
        return req.block_hashes
    if req.output_hash_ids:
        tokens = tokens + req.output_hash_ids

    hashes = []
    parent = NONE_HASH
    for start in range(0, len(tokens) - block_size + 1, block_size):
        parent = hash((parent, tuple(tokens[start:start + block_size])))
        hashes.append(parent)
    req.block_hashes = hashes
    return hashes


class TieredKVCacheManager:
    """Owns the per-request NPU block tables and the tier hierarchy.

    Args:
        block_size: NPU block size, and the granularity of the hash chain.
        npu_pool: the tier requests actually compute against.
        lower_pools: victim tiers in order (CPU, then CXL), each with a
            ``block_size`` that is a multiple of ``block_size``. Empty when
            ``--prefix-storage`` is not set, which is the default and matches
            plain vLLM: there is no host KV tier, so a preempted request
            recovers only what is still resident and recomputes the rest.
        enable_caching: mirrors ``--enable-prefix-caching``.
    """

    def __init__(self, block_size, npu_pool, lower_pools=(), enable_caching=True):
        self.block_size = block_size
        self.npu_pool = npu_pool
        self.lower_pools = list(lower_pools)
        self.enable_caching = enable_caching and npu_pool.enable_caching

        for pool in self.lower_pools:
            if pool.block_size % block_size != 0:
                raise ValueError(
                    f"[TieredKVCacheManager] {pool.tier}: block_size "
                    f"{pool.block_size} must be a multiple of the NPU block_size "
                    f"{block_size} -- the lower tier keys on every Nth hash of the "
                    f"same chain, so a non-integer factor has no key to use"
                )

        # request id -> its NPU blocks, in sequence order
        self.req_to_blocks = {}
        # request id -> how many of those blocks are already indexed
        self.num_cached_block = {}

        # Drained once per scheduling step by take_traffic().
        self._recall_bytes = 0
        self._write_through_bytes = 0

    # -------------------- helpers --------------------

    def _factor(self, pool):
        return pool.block_size // self.block_size

    def _coarse_hash(self, hashes, pool, coarse_idx):
        """Key for coarse block ``coarse_idx`` of ``pool``: the last fine hash
        it covers. ``offloading/scheduler.py::_get_block_hashes``."""
        factor = self._factor(pool)
        fine_idx = factor * coarse_idx + factor - 1
        if fine_idx >= len(hashes):
            return None
        return hashes[fine_idx]

    # -------------------- lookup --------------------

    def get_computed_blocks(self, req):
        """Longest recoverable prefix for ``req``, split by tier.

        Returns ``(npu_blocks, num_npu_hit_tokens, num_lower_hit_tokens)``.
        ``num_lower_hit_tokens`` are tokens whose KV has to be recalled into
        fresh NPU blocks; the recall bytes are charged in
        :meth:`allocate_slots`, once the allocation is known to succeed.
        """
        if not self.enable_caching:
            return [], 0, 0

        hashes = request_block_hashes(req, self.block_size)
        if not hashes:
            return [], 0, 0

        # vLLM caps the hit at num_tokens - 1: at least one token must be
        # recomputed to produce logits. Combined with block alignment this is
        # why a fully resident resume still costs up to one block.
        max_hit_len = req.num_tokens_reached - 1
        max_blocks = max_hit_len // self.block_size

        npu_blocks = []
        for block_hash in hashes[:max_blocks]:
            block = self.npu_pool.get_cached_block(block_hash)
            if block is None:
                # The chain is cumulative, so a miss means everything after it
                # is uncomputed or gone. Stop.
                break
            npu_blocks.append(block)
        num_npu_hit = len(npu_blocks) * self.block_size

        num_lower_hit = self._lookup_lower(req, hashes, num_npu_hit)
        return npu_blocks, num_npu_hit, num_lower_hit

    def _lookup_lower(self, req, hashes, num_computed_tokens):
        """Extra tokens recoverable from a victim tier beyond the NPU hit.

        Mirrors ``get_num_new_matched_tokens``: round the NPU hit down to a
        coarse boundary, count consecutive coarse hits from there, and bail
        unless at least one whole coarse block is gained. The coarse block
        straddling the boundary is re-fetched -- transfers are block-granular.
        """
        for pool in self.lower_pools:
            coarse = pool.block_size
            num_coarse_blocks = req.num_tokens_reached // coarse
            full_block_tokens = coarse * num_coarse_blocks
            if full_block_tokens - num_computed_tokens < coarse:
                continue

            start_idx = num_computed_tokens // coarse
            hits = 0
            for idx in range(start_idx, num_coarse_blocks):
                block_hash = self._coarse_hash(hashes, pool, idx)
                if block_hash is None or pool.get_cached_block(block_hash) is None:
                    break
                hits += 1
            if hits == 0:
                continue

            gained = coarse * (start_idx + hits) - num_computed_tokens
            if gained < coarse:
                continue
            req.storage_hit_pool = pool
            req.storage_hit_blocks = hits
            return gained
        req.storage_hit_pool = None
        req.storage_hit_blocks = 0
        return 0

    # -------------------- allocation --------------------

    def _num_blocks_to_allocate(self, req, num_tokens, new_computed_blocks):
        """NPU blocks the pool would have to hand over to give ``req`` slots for
        ``num_tokens`` in total. Port of
        ``single_type_kv_cache_manager.get_num_blocks_to_allocate``.
        """
        req_blocks = self.req_to_blocks.get(req.id, ())
        num_required_blocks = cdiv(num_tokens, self.block_size)
        if req.id in self.num_cached_block:
            # Fast path: a running request has no new prefix hits to account for.
            return max(num_required_blocks - len(req_blocks), 0)
        num_known_blocks = len(new_computed_blocks) + len(req_blocks)
        num_new_blocks = max(num_required_blocks - num_known_blocks, 0)
        # A hit block sitting in the free list leaves it when touched, so it
        # consumes free capacity too and must be counted here.
        num_evictable = sum(1 for b in new_computed_blocks if b.ref_cnt == 0)
        return num_new_blocks + num_evictable

    def can_fit_full_sequence(self, req, new_computed_blocks=None,
                              num_new_computed_tokens=0, num_lower_tier_tokens=0):
        """Would ``req``'s whole sequence fit, not just this step's chunk?

        vLLM's admission gate (``kv_cache_manager.can_fit_full_sequence``, called
        from ``schedule()`` under ``scheduler_reserve_full_isl``, which defaults
        to True). Without it, chunked prefill admits a request on the strength of
        its first chunk, the request then grows past what the pool can hold, and
        something has to be preempted -- the "over-admission and KV cache
        thrashing" the vLLM config docstring names.

        Measured on the RTX 4090 replay, the absence of this gate was worth
        ~230 preemptions and ~145k recomputed tokens at *every* utilization from
        0.8 to 1.0, i.e. it is not a capacity effect.
        """
        new_computed_blocks = new_computed_blocks or []
        total_computed = (req.num_computed_tokens + num_new_computed_tokens
                          + num_lower_tier_tokens)
        # The length reached so far, which for a request being admitted for the
        # first time is its whole prompt. num_tokens_reached rather than
        # num_computed_tokens: preemption resets the latter to 0.
        full_num_tokens = max(req.num_tokens_reached, total_computed)
        return self._num_blocks_to_allocate(
            req, full_num_tokens, new_computed_blocks
        ) <= self.npu_pool.get_num_free_blocks()

    def allocate_slots(self, req, num_new_tokens, new_computed_blocks=None,
                       num_new_computed_tokens=0, num_lower_tier_tokens=0):
        """Give ``req`` slots for ``num_new_tokens`` more tokens.

        Returns the newly allocated blocks, or **None** when the pool cannot
        satisfy the request -- and in that case nothing has been mutated. That
        all-or-nothing property in the same call that can fail is the point of
        the block pool: the previous allocator decided whether to preempt
        against one estimate and charged memory later against another.
        """
        new_computed_blocks = new_computed_blocks or []
        req_blocks = self.req_to_blocks.setdefault(req.id, [])

        num_local_computed = req.num_computed_tokens + num_new_computed_tokens
        total_computed = num_local_computed + num_lower_tier_tokens
        num_tokens_need_slot = total_computed + num_new_tokens

        if req.id in self.num_cached_block and (new_computed_blocks or num_lower_tier_tokens):
            # Running request: it cannot pick up new prefix hits mid-flight.
            raise RuntimeError(
                f"[TieredKVCacheManager] request {req.id} is running but was "
                f"given new computed blocks"
            )

        if self._num_blocks_to_allocate(
                req, num_tokens_need_slot, new_computed_blocks
        ) > self.npu_pool.get_num_free_blocks():
            return None

        # ---- past this point the allocation is guaranteed to succeed ----

        if new_computed_blocks or num_lower_tier_tokens:
            self.npu_pool.touch(new_computed_blocks)
            req_blocks.extend(new_computed_blocks)
            self.num_cached_block[req.id] = len(req_blocks)
            if num_lower_tier_tokens:
                self._charge_recall(req)

        new_blocks = self._allocate_new_blocks(req, num_tokens_need_slot)

        if self.enable_caching:
            # Cap at the reached length: only finalised tokens may be indexed.
            self.cache_blocks(
                req, min(total_computed + num_new_tokens, req.num_tokens_reached))
        return new_blocks

    def _allocate_new_blocks(self, req, num_tokens):
        req_blocks = self.req_to_blocks[req.id]
        num_new = cdiv(num_tokens, self.block_size) - len(req_blocks)
        if num_new <= 0:
            return []
        new_blocks = self.npu_pool.get_new_blocks(num_new)
        req_blocks.extend(new_blocks)
        return new_blocks

    def _charge_recall(self, req):
        """Bytes moved from a victim tier into the NPU for a resume or a hit.

        Whole coarse blocks move, including the one straddling the NPU-hit
        boundary, so the charge is ``hits x bytes_per_block`` of that tier.
        """
        pool = req.storage_hit_pool
        if pool is None or req.storage_hit_blocks <= 0:
            return
        self._recall_bytes += req.storage_hit_blocks * pool.bytes_per_block
        req.storage_hit_pool = None
        req.storage_hit_blocks = 0

    # -------------------- caching --------------------

    def cache_blocks(self, req, num_tokens):
        """Index every block of ``req`` that is now full, then write down.

        Called from ``allocate_slots`` and again when a chunk completes, so it
        must be idempotent -- ``num_cached_block`` is what makes it so.
        """
        if not self.enable_caching:
            return
        hashes = request_block_hashes(req, self.block_size)
        if not hashes:
            return

        num_cached = self.num_cached_block.get(req.id, 0)
        num_full = min(num_tokens // self.block_size, len(hashes))
        if num_cached >= num_full:
            return

        blocks = self.req_to_blocks[req.id]
        if len(blocks) < num_full:
            raise RuntimeError(
                f"[TieredKVCacheManager] request {req.id}: {num_full} full blocks "
                f"to cache but only {len(blocks)} allocated"
            )
        self.npu_pool.cache_full_blocks(hashes, blocks, num_cached, num_full)
        self.num_cached_block[req.id] = num_full
        self._write_down(hashes, num_cached, num_full)

    def _write_down(self, hashes, num_cached, num_full):
        """Inclusive write-through into every victim tier.

        A coarse block is written once it is fully covered by indexed NPU
        blocks. The bytes are recorded for energy accounting only -- see the
        module docstring on why the latency is assumed overlapped.
        """
        for pool in self.lower_pools:
            factor = self._factor(pool)
            first = num_cached // factor
            last = num_full // factor          # exclusive; only complete coarse blocks
            for idx in range(first, last):
                block_hash = self._coarse_hash(hashes, pool, idx)
                if block_hash is None:
                    break
                if pool.cache_copy(block_hash):
                    self._write_through_bytes += pool.bytes_per_block

    # -------------------- release --------------------

    def free(self, req):
        """Release a finished request's blocks, keeping their hashes.

        Reverse order so the tail is reused first and the head -- the
        recoverable prefix -- survives longest.
        """
        blocks = self.req_to_blocks.pop(req.id, [])
        self.num_cached_block.pop(req.id, None)
        if blocks:
            self.npu_pool.free_blocks(reversed(blocks))

    def preempt(self, req):
        """Release a *running* request's blocks so someone else can use them.

        Identical to :meth:`free`; kept as a separate call site because the
        request lives on and the scheduler counts these. Nothing here touches
        ``num_computed_tokens``: the request is re-admitted through the normal
        path, and :meth:`get_computed_blocks` re-derives where it got to. That
        is vLLM's behaviour, not a deviation from it -- with a victim tier the
        recovery is a recall, and without one it is a recompute.
        """
        self.free(req)

    # -------------------- accounting --------------------

    def take_traffic(self):
        """``(recall_bytes, write_through_bytes)`` since the last call."""
        recall, write_through = self._recall_bytes, self._write_through_bytes
        self._recall_bytes = 0
        self._write_through_bytes = 0
        return recall, write_through

    def npu_used_bytes(self):
        return self.npu_pool.used_bytes()

    def usage(self):
        return self.npu_pool.usage()

    def is_free(self):
        """Every tier back to a fully free list. End-of-run leak check."""
        return all(p.is_free() for p in [self.npu_pool] + self.lower_pools)
