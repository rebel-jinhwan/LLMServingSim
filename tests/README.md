# tests

Unit checks for the parts that simulating does not cover: the block pool, the
tiered KV cache manager, and model config loading.

```bash
pip install pytest
pytest                      # everything
pytest tests/test_block_pool.py
```

Plain `test_*()` functions that assert — no fixtures, no helpers, no base
classes. `conftest.py` puts the repository root on `sys.path`, so no
`PYTHONPATH` is needed.

`demo_logger.py` is not a test. It prints every logger surface at once, for
looking at: `python tests/demo_logger.py`.

Simulator *behaviour* is validated separately: `./serving/validate.sh` runs
every scenario against recorded clocks and regenerates the `bench/examples`
comparisons. Run that after any change under `serving/`.
