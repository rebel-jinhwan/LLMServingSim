# tests

Unit checks for the parts that simulating does not cover: the block pool, the
tiered KV cache manager, the platform registry, model config loading, and the
in-tree scheduler against vLLM's own.

```bash
pip install -e '.[dev]'
pytest                      # everything
pytest tests/test_platforms.py
```

`.github/workflows/tests.yml` runs the same `pytest` on every push and pull
request.

Plain `test_*()` functions that assert — no fixtures, no helpers, no base
classes. `conftest.py` puts the repository root on `sys.path`, so no
`PYTHONPATH` is needed. A check that cannot run in this environment raises
`unittest.SkipTest` and is reported as a skip, not a failure.

`demo_logger.py` is not a test. It prints every logger surface at once, for
looking at: `python tests/demo_logger.py`.

Simulator *behaviour* is validated separately: `./serving/validate.sh` runs
every scenario against recorded clocks and regenerates the `bench/examples`
comparisons. Run that after any change under `serving/`.

## Environment

| Check | Needs |
| --- | --- |
| `test_block_pool`, `test_kv_cache_manager`, `test_utils` | nothing beyond the simulator's own dependencies |
| `test_platforms` | pyyaml; an installed platform plugin joins the registry and is checked too |
| `test_vllm_scheduler` | vLLM importable, and **no** vLLM platform plugin active, since it compares against upstream vLLM's scheduler. With a plugin installed, run it as `VLLM_PLUGINS= pytest tests/test_vllm_scheduler.py` |
