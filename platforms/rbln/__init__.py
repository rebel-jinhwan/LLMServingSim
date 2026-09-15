"""Rebellions NPUs through vllm-rbln.

The device runs a compiled graph per padded shape, with the TP collectives
inside it, so a forward is measured whole: one ``step.csv`` row per
(prefill_chunk, kv_prefill, n_decode, kv_decode), TP profiled on real ranks.
The scheduler never mixes prefill and decode in one step.
"""

NAME = "rbln"
GRANULARITY = "step"
