import pytest

import tabulus.mineru.backends as backends
from tabulus.mineru.backends import GPUCapability


def test_pipeline_is_default_backend():
    assert backends.DEFAULT_BACKEND == "pipeline"


def test_pipeline_does_not_require_gpu():
    backend, capability = backends.resolve_backend("pipeline")

    assert backend == "pipeline"
    assert capability is None


def test_hybrid_uses_suitable_gpu(monkeypatch):
    capability = GPUCapability(
        available=True,
        suitable_for_hybrid=True,
        device_name="Test GPU",
        vram_gb=24.0,
        compute_capability=(8, 0),
    )

    monkeypatch.setattr(
        backends,
        "check_hybrid_gpu",
        lambda: capability,
    )

    backend, result = backends.resolve_backend("hybrid-engine")

    assert backend == "hybrid-engine"
    assert result == capability


def test_hybrid_falls_back_without_gpu(monkeypatch):
    capability = GPUCapability(
        available=False,
        suitable_for_hybrid=False,
        reason="CUDA is not available.",
    )

    monkeypatch.setattr(
        backends,
        "check_hybrid_gpu",
        lambda: capability,
    )

    backend, result = backends.resolve_backend("hybrid-engine")

    assert backend == "pipeline"
    assert result == capability
    assert result.reason == "CUDA is not available."


def test_hybrid_falls_back_for_insufficient_vram(monkeypatch):
    capability = GPUCapability(
        available=True,
        suitable_for_hybrid=False,
        device_name="Small GPU",
        vram_gb=4.0,
        compute_capability=(8, 0),
        reason="GPU has 4.0 GB VRAM; 8 GB is required.",
    )

    monkeypatch.setattr(
        backends,
        "check_hybrid_gpu",
        lambda: capability,
    )

    backend, result = backends.resolve_backend("hybrid-engine")

    assert backend == "pipeline"
    assert result == capability


def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        backends.resolve_backend("something-else")
