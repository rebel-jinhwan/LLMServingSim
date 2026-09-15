# tests

Unit checks for the parts that are not covered by simulating: the block pool,
the tiered KV cache manager, the platform registry, model config loading, and
the in-tree scheduler against vLLM's own.

```bash
python tests/run.py                     # everything
python tests/run.py test_platforms      # one module
pytest tests                            # same files, if you have pytest
```

Each `test_*.py` holds plain `test_*()` functions that assert. There is no
framework and no dependency: `run.py` imports each module and calls its
functions. A check that cannot run in this environment raises
`unittest.SkipTest` and is reported as a skip, not a failure.

`demo_logger.py` is not a test. It prints every logger surface at once, for
looking at.

Simulator *behaviour* is validated separately: `./serving/validate.sh` runs
every scenario against recorded clocks and regenerates the `bench/examples`
comparisons. Run that after any change under `serving/`.

## Environment

| Check | Needs |
| --- | --- |
| `test_block_pool`, `test_kv_cache_manager`, `test_utils` | nothing beyond the simulator's own dependencies |
| `test_platforms`, `test_platform_profile` | pyyaml; an installed platform plugin joins the registry and is checked too |
| `test_vllm_scheduler` | vLLM importable, and **no** vLLM platform plugin active, since it compares against upstream vLLM's scheduler. With a plugin installed, run it as `VLLM_PLUGINS= python tests/run.py test_vllm_scheduler` |
