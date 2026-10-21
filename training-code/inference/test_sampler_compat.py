"""Real V2 call sites must use the shared filter and restore after failures."""

import math
import re

import pytest
import torch

import sampler_compat


def test_installed_v2_call_sites_filter_and_restore_after_exception():
    from vllm.v1.worker.gpu.sample import sampler, states

    original_sampler = sampler.apply_top_k_top_p
    original_states = states.apply_top_k_top_p
    with pytest.raises(RuntimeError, match="abort"), sampler_compat.native_sampling_scope():
        scores = torch.tensor([[3.0, 2.0, 2.0, 0.0]])
        assert sampler.apply_top_k_top_p(scores, torch.tensor([2]), torch.tensor([1.0]))[
            0
        ].tolist() == [3.0, 2.0, 2.0, -math.inf]
        assert states.apply_top_k_top_p is sampler.apply_top_k_top_p
        raise RuntimeError("abort")
    assert sampler.apply_top_k_top_p is original_sampler
    assert states.apply_top_k_top_p is original_states


def test_nested_scope_preserves_outer_scope_until_both_exit():
    from vllm.v1.worker.gpu.sample import sampler

    original = sampler.apply_top_k_top_p
    with sampler_compat.native_sampling_scope():
        outer = sampler.apply_top_k_top_p
        with sampler_compat.native_sampling_scope():
            assert sampler.apply_top_k_top_p is outer
        assert sampler.apply_top_k_top_p is outer
    assert sampler.apply_top_k_top_p is original


def test_worker_rpc_scope_restores_real_filter_when_operation_fails():
    from vllm.v1.worker.gpu.sample import sampler

    module = sampler_compat
    worker = module.NativeSamplingWorker()

    class LocalWorkerEngine:
        def collective_rpc(self, method):
            return [getattr(worker, method)()]

    original = sampler.apply_top_k_top_p
    with (
        pytest.raises(RuntimeError, match="operation failed"),
        module.native_engine_sampling(LocalWorkerEngine()),
    ):
        scores = torch.tensor([[3.0, 2.0, 2.0, 0.0]])
        filtered = sampler.apply_top_k_top_p(scores, torch.tensor([2]), torch.tensor([1.0]))
        assert torch.isfinite(filtered).tolist() == [[True, True, True, False]]
        raise RuntimeError("operation failed")
    assert sampler.apply_top_k_top_p is original
    worker.disable_native_sampling()


@pytest.mark.parametrize("installed", ["0.31.0", "0.31.0+cu129"])
def test_reviewed_builds_of_the_pinned_release_are_accepted(monkeypatch, installed):
    monkeypatch.setattr(sampler_compat.importlib.metadata, "version", lambda _name: installed)
    assert sampler_compat.require_pinned_vllm() == installed


@pytest.mark.parametrize("installed", ["0.31.1", "0.31.0+cpu", "0.31.0rc1"])
def test_other_vllm_builds_are_rejected_with_the_allowed_values(monkeypatch, installed):
    monkeypatch.setattr(sampler_compat.importlib.metadata, "version", lambda _name: installed)
    allowed = r"allowed \['0\.31\.0', '0\.31\.0\+cu129'\]"
    with pytest.raises(ValueError, match=rf"installed {re.escape(installed)}, {allowed}"):
        sampler_compat.require_pinned_vllm()


def test_worker_receipt_reports_the_installed_vllm_build():
    import importlib.metadata

    worker = sampler_compat.NativeSamplingWorker()
    try:
        receipt = worker.enable_native_sampling()
    finally:
        worker.disable_native_sampling()
    assert receipt == {"filter": "native_torch", "vllm_version": importlib.metadata.version("vllm")}


def test_unsupported_version_is_rejected_before_installing_hooks(monkeypatch):
    from vllm.v1.worker.gpu.sample import sampler

    module = sampler_compat
    original = sampler.apply_top_k_top_p
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _name: "999.0.0")
    with pytest.raises(ValueError, match="pinned"), module.native_sampling_scope():
        pytest.fail("Unsupported layout was accepted")
    assert sampler.apply_top_k_top_p is original
