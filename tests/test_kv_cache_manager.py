"""Tiered KV cache manager: block hashing, allocation across tiers, and what
survives a preemption. Was ``python -m llmservingsim.serving.core.kv_cache_manager``."""

from llmservingsim.serving.core.block_pool import BlockPool, Device
from llmservingsim.serving.core.kv_cache_manager import TieredKVCacheManager, request_block_hashes


def test_kv_cache_manager():
    B, KB = 16, 1024
    BYTES = B * 128 * KB

    class Req:
        _next = 0

        def __init__(self, prompt, output=()):
            Req._next += 1
            self.id = Req._next
            self.input_hash_ids = list(prompt)
            self.output_hash_ids = list(output)
            self.original_input = len(prompt)
            self.num_computed_tokens = 0
            self.num_tokens_reached = len(prompt)
            self.block_hashes = None
            self.storage_hit_pool = None
            self.storage_hit_blocks = 0

    def new_mgr(npu_blocks=64, lower=None, caching=True):
        npu = BlockPool(Device.NPU, npu_blocks, B, BYTES, enable_caching=caching)
        pools = []
        if lower:
            pools.append(BlockPool(Device.CPU, lower, 256, 16 * BYTES,
                                   enable_caching=caching))
        return TieredKVCacheManager(B, npu, pools, enable_caching=caching)

    # chained hashes: same tail tokens, different prefix -> different hash
    a, b = Req(list(range(0, 16)) + list(range(100, 116))), Req(list(range(50, 66)) + list(range(100, 116)))
    ha, hb = request_block_hashes(a, B), request_block_hashes(b, B)
    assert len(ha) == 2 and ha[1] != hb[1], "unchained hashes would collide here"

    # cold miss allocates everything; the hit path allocates nothing for the prefix
    m = new_mgr()
    r1 = Req(list(range(64)))
    blocks, npu_hit, low_hit = m.get_computed_blocks(r1)
    assert (blocks, npu_hit, low_hit) == ([], 0, 0)
    assert m.allocate_slots(r1, 64) is not None
    assert len(m.req_to_blocks[r1.id]) == 4
    r1.num_computed_tokens = 64

    r2 = Req(list(range(64)))                       # same prompt
    blocks, npu_hit, low_hit = m.get_computed_blocks(r2)
    assert npu_hit == 48 and low_hit == 0, (npu_hit, low_hit)   # capped at 64-1 -> 3 blocks
    free_before = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r2, 64 - npu_hit, blocks, npu_hit) is not None
    assert m.npu_pool.get_num_free_blocks() == free_before - 1, "only the tail block is new"

    # allocation failure returns None and mutates nothing
    m = new_mgr(npu_blocks=4)
    r = Req(list(range(64)))
    assert m.allocate_slots(r, 64) is not None
    r2 = Req(list(range(100, 164)))
    free_before = m.npu_pool.get_num_free_blocks()
    assert m.allocate_slots(r2, 64) is None
    assert m.npu_pool.get_num_free_blocks() == free_before
    assert m.req_to_blocks[r2.id] == []

    # preempt then resume with everything resident: no recall, and the recompute
    # is only the block-aligned remainder
    m = new_mgr(npu_blocks=64)
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    r.num_computed_tokens = 64
    m.preempt(r)
    r.num_computed_tokens = 0
    blocks, npu_hit, low_hit = m.get_computed_blocks(r)
    assert low_hit == 0 and npu_hit == 48, (npu_hit, low_hit)
    assert m.take_traffic() == (0, 0), "resident resume must be free"
    m.allocate_slots(r, 64 - npu_hit, blocks, npu_hit)
    assert len(m.req_to_blocks[r.id]) == 4

    # preempt, force reuse, no lower tier -> full recompute, still no transfer
    m = new_mgr(npu_blocks=8)
    r = Req(list(range(128)))
    m.allocate_slots(r, 128)
    r.num_computed_tokens = 128
    m.preempt(r)
    r.num_computed_tokens = 0
    other = Req(list(range(500, 628)))
    m.allocate_slots(other, 128)                    # evicts every one of r's blocks
    m.req_to_blocks[r.id] = []
    m.num_cached_block.pop(r.id, None)
    _, npu_hit, low_hit = m.get_computed_blocks(r)
    assert (npu_hit, low_hit) == (0, 0), (npu_hit, low_hit)
    assert m.take_traffic() == (0, 0)

    # with a lower tier the same resume is a recall, not a recompute
    m = new_mgr(npu_blocks=64, lower=8)
    r = Req(list(range(512)))
    m.allocate_slots(r, 512)
    r.num_computed_tokens = 512
    _, wt = m.take_traffic()
    assert wt == 2 * 16 * BYTES, wt                 # two 256-token coarse blocks
    m.preempt(r)
    r.num_computed_tokens = 0
    m.req_to_blocks[r.id] = []
    m.num_cached_block.pop(r.id, None)
    fillers = []                                    # push every NPU block out...
    for i in range(16):
        f = Req(list(range(9000 + 100 * i, 9064 + 100 * i)))
        if m.allocate_slots(f, 64) is not None:
            fillers.append(f)
    _, npu_hit, low_hit = m.get_computed_blocks(r)
    assert npu_hit == 0 and low_hit == 512, (npu_hit, low_hit)
    for f in fillers:                               # ...then make room to resume
        m.free(f)
    m.take_traffic()
    assert m.allocate_slots(r, 1, [], 0, low_hit) is not None
    recall, _ = m.take_traffic()
    assert recall == 2 * 16 * BYTES, recall

    # coarse tier only hits on its own boundary
    m = new_mgr(npu_blocks=64, lower=8)
    r = Req(list(range(200)))                       # < one 256-token chunk
    m.allocate_slots(r, 200)
    r.num_computed_tokens = 200
    _, wt = m.take_traffic()
    assert wt == 0, "no complete coarse block yet"

    # caching disabled: no index, no hits, resume is a full recompute
    m = new_mgr(caching=False)
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    r.num_computed_tokens = 64
    m.preempt(r)
    r.num_computed_tokens = 0
    assert m.get_computed_blocks(r) == ([], 0, 0)
    assert m.npu_pool.is_free()

    # can_fit_full_sequence: the gate refuses a request whose first chunk fits
    # but whose whole prompt does not
    m = new_mgr(npu_blocks=8)                       # 8 blocks = 128 tokens
    r = Req(list(range(2048)))                      # 128 full blocks wanted
    _, npu_hit, low_hit = m.get_computed_blocks(r)
    assert m.can_fit_full_sequence(r, [], npu_hit, low_hit) is False
    assert m.allocate_slots(r, 64) is not None, "a single chunk still fits"
    m.free(r)
    r2 = Req(list(range(100)))                      # 6 full blocks + tail
    assert m.can_fit_full_sequence(r2, [], 0, 0) is True
    # a hit already on the NPU does not have to be re-reserved
    m = new_mgr(npu_blocks=8)
    a = Req(list(range(64)))
    m.allocate_slots(a, 64); a.num_computed_tokens = 64
    b = Req(list(range(64)))
    blocks, npu_hit, low_hit = m.get_computed_blocks(b)
    assert npu_hit == 48
    assert m.can_fit_full_sequence(b, blocks, npu_hit, low_hit) is True

    # no leaks after a normal finish
    m = new_mgr()
    r = Req(list(range(64)))
    m.allocate_slots(r, 64)
    m.free(r)
    assert m.is_free(), m.npu_pool

