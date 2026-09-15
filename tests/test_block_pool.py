"""Per-tier KV block pool: allocation order, the prefix-cache index, and the
byte ledger. Was ``python -m serving.core.block_pool``."""

from serving.core.block_pool import BlockPool, Device


def test_block_pool():
    KB = 1024

    def new_pool(n=8, caching=True):
        return BlockPool(Device.NPU, n, block_size=16,
                         bytes_per_block=16 * 128 * KB, enable_caching=caching)

    # allocate / free round-trip restores the free count
    p = new_pool()
    blocks = p.get_new_blocks(3)
    assert p.get_num_free_blocks() == 5, p
    p.free_blocks(reversed(blocks))
    assert p.is_free(), p

    # a freed block is reused LAST, so a just-preempted request survives a
    # small reclaim
    p = new_pool()
    first = p.get_new_blocks(2)
    p.free_blocks(reversed(first))
    order = [b.block_id for b in p.free_block_queue.get_all_free_blocks()]
    assert order == [2, 3, 4, 5, 6, 7, 1, 0], order

    # an indexed block popped for reuse loses its hash; a live one is untouched
    p = new_pool(n=3)
    blocks = p.get_new_blocks(2)
    p.cache_full_blocks([111, 222], blocks, 0, 2)
    assert p.get_cached_block(111) is blocks[0]
    p.free_blocks(reversed(blocks))
    assert p.get_cached_block(222) is blocks[1], "hash must survive a free"
    p.get_new_blocks(3)  # forces both cached blocks out
    assert p.get_cached_block(111) is None and p.get_cached_block(222) is None
    assert len(p.cached_block_hash_to_block) == 0

    # touch() pulls a cached, unpinned block back out of the free list
    p = new_pool()
    blocks = p.get_new_blocks(1)
    p.cache_full_blocks([777], blocks, 0, 1)
    p.free_blocks(blocks)
    assert p.get_num_free_blocks() == 8
    hit = p.get_cached_block(777)
    p.touch([hit])
    assert hit.ref_cnt == 1 and p.get_num_free_blocks() == 7

    # shared prefix: two requests on one block, freed once each
    p = new_pool()
    shared = p.get_new_blocks(1)
    p.cache_full_blocks([42], shared, 0, 1)
    p.touch(shared)
    assert shared[0].ref_cnt == 2
    p.free_blocks(shared)
    assert shared[0].ref_cnt == 1 and p.get_num_free_blocks() == 7
    p.free_blocks(shared)
    assert p.is_free()

    # used_bytes counts a cached-but-unpinned block as occupied
    p = new_pool()
    blocks = p.get_new_blocks(2)
    p.cache_full_blocks([1, 2], blocks, 0, 2)
    assert p.used_bytes() == 2 * p.bytes_per_block
    p.free_blocks(reversed(blocks))
    assert p.used_bytes() == 2 * p.bytes_per_block, "still holding data"

    # caching disabled: allocation is identical, the index stays empty
    p = new_pool(caching=False)
    blocks = p.get_new_blocks(4)
    p.cache_full_blocks([1, 2, 3, 4], blocks, 0, 4)
    assert len(p.cached_block_hash_to_block) == 0
    assert p.get_cached_block(1) is None
    p.free_blocks(reversed(blocks))
    assert p.is_free()

    # over-allocation raises rather than silently wrapping
    p = new_pool(n=2)
    p.get_new_blocks(2)
    try:
        p.get_new_blocks(1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("over-allocation must raise")

    # a lower tier at a coarser granularity is the same class
    cpu = BlockPool(Device.CPU, 4, block_size=256,
                    bytes_per_block=256 * 128 * KB, enable_caching=True)
    assert cpu.block_size == 16 * 16
    assert cpu.bytes_per_block == 16 * p.bytes_per_block

    # cache_copy: inclusive victim cache, unpinned, LRU-dropped on overflow
    assert cpu.cache_copy(1001) is True
    assert cpu.cache_copy(1001) is False, "already resident, must not re-charge"
    assert cpu.get_cached_block(1001).ref_cnt == 0, "copies stay unpinned"
    assert cpu.is_free(), "a copy occupies the index, not the free list"
    for h in (1002, 1003, 1004):
        assert cpu.cache_copy(h) is True
    assert len(cpu.cached_block_hash_to_block) == 4
    assert cpu.cache_copy(1005) is True                  # forces an eviction
    assert cpu.get_cached_block(1001) is None, "oldest copy dropped first"
    assert len(cpu.cached_block_hash_to_block) == 4

    print("block_pool self-test: all checks passed")
