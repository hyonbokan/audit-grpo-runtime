"""Scoped compatibility for the pinned V2 sampler; restore every modified call site."""

from __future__ import annotations

import importlib.metadata
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from grpo.sampling import filter_logits

# Reviewed builds of the pinned release; the local tag names only the CUDA build.
SUPPORTED_VLLM_BUILDS = frozenset({"0.31.0", "0.31.0+cu129"})


@contextmanager
def native_sampling_scope() -> Iterator[None]:
    require_pinned_vllm()
    # V2 resolves filtering through both globals; changing one leaves replay inconsistent.
    from vllm.v1.worker.gpu.sample import sampler, states

    modules = (sampler, states)
    originals = [module.apply_top_k_top_p for module in modules]
    try:
        for module in modules:
            module.apply_top_k_top_p = filter_logits
        yield
    finally:
        for module, original in zip(modules, originals, strict=True):
            module.apply_top_k_top_p = original


def require_pinned_vllm() -> str:
    """Return the installed vLLM version, or raise unless it is a reviewed build."""
    installed = importlib.metadata.version("vllm")
    if installed not in SUPPORTED_VLLM_BUILDS:
        raise ValueError(
            f"Compatibility requires pinned vLLM layout: installed {installed}, "
            f"allowed {sorted(SUPPORTED_VLLM_BUILDS)}"
        )
    return installed


@contextmanager
def native_engine_sampling(engine: Any) -> Iterator[None]:
    """Use public worker RPC so each model process owns its compatibility scope."""
    try:
        engine.collective_rpc("enable_native_sampling")
        yield
    finally:
        engine.collective_rpc("disable_native_sampling")


class NativeSamplingWorker:
    """vLLM worker extension: bind hook lifetime to one explicitly scoped operation."""

    def enable_native_sampling(self) -> dict[str, str]:
        if getattr(self, "_native_sampling_context", None) is not None:
            raise ValueError("Native sampler scope is already active")
        context = native_sampling_scope()
        context.__enter__()
        self._native_sampling_context = context
        return {"filter": "native_torch", "vllm_version": importlib.metadata.version("vllm")}

    def disable_native_sampling(self) -> None:
        context = getattr(self, "_native_sampling_context", None)
        if context is not None:
            try:
                context.__exit__(None, None, None)
            finally:
                self._native_sampling_context = None
