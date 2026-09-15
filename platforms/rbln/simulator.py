"""Simulator side of the RBLN platform: vllm-rbln's own scheduler.

``VllmScheduler`` builds the ``VllmConfig`` the way ``EngineCore`` does and
instantiates whatever ``scheduler_cls`` says, so pinning it to
``RBLNScheduler`` runs vllm-rbln's scheduling code (no mixed batching,
sub-block prefix caching, its admission caps) verbatim. The plugin's own
``VLLM_RBLN_*`` environment and ``additional_config`` apply as they would
in a real engine. Requires ``vllm-rbln`` importable in the simulator
environment; the import error otherwise names the missing module.
"""

from __future__ import annotations

from serving.core.vllm_scheduler import VllmScheduler


class RBLNVllmScheduler(VllmScheduler):
    ENGINE_ARGS = {"scheduler_cls": "vllm_rbln.v1.core.rbln_scheduler.RBLNScheduler"}


