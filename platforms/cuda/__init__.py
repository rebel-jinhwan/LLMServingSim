"""CUDA vLLM: per-layer kernel timings, TP emulated on one GPU, collectives
left to ASTRA-Sim. This is what the simulator did before platforms existed."""

NAME = "cuda"
GRANULARITY = "layer"
