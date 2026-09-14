# MiniMax-M2.5 on 4x RBLN-CR03, pp4: ground truth only

`vllm/` is a real `python -m bench` run (vllm-rbln at `a2088f6`, vLLM 0.26,
pipeline_parallel_size 4, block_size 8192, mnbt 1024, msq 4, bf16 KV,
146 KV blocks) over `workloads/random-minimax-m2.5-24-sps0.2.jsonl`:
TTFT mean 710 ms, TPOT mean 20.5 ms, latency mean 3.7 s.

There is no `outputs/` because the step profile for this shape is not
trustworthy yet. The profiler times each worker's `execute_model` +
`sample_tokens` for N forwards inside `collective_rpc`; with pipeline
parallelism the ranks block on one another's sends and receives, and the
resulting per-rank wall-clocks do not measure the pipeline latency of one
forward. On this run they gave a prefill step of 1.3 s per forward where
vLLM's single-request TTFT is 0.34 s, and a decode step of 5.2 ms where
vLLM's TPOT is 22 ms. Simulating over that bundle put TTFT at +102% and
TPOT at -50%.

What a pp step profile needs instead: the latency of one forward through
the whole pipeline (drive one shot at a time from the host and time it
there, or timestamp the first stage's start and the last stage's
`sample_tokens` end), split across stages as `get_pp_indices` splits the
blocks. The per-stage trace rows and the `stage` column in `step.csv` are
already in place for that.
