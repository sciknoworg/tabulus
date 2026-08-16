from __future__ import annotations

from dataclasses import dataclass


PIPELINE_BACKEND = "pipeline"
HYBRID_BACKEND = "hybrid-engine"

SUPPORTED_BACKENDS = (
    PIPELINE_BACKEND,
    HYBRID_BACKEND,
)

DEFAULT_BACKEND = PIPELINE_BACKEND

HYBRID_MIN_VRAM_GB = 8.0
HYBRID_MIN_COMPUTE_CAPABILITY = (7, 0)


@dataclass(frozen=True)
class GPUCapability:
    """GPU capability information relevant to MinerU hybrid execution."""

    available: bool
    suitable_for_hybrid: bool
    device_name: str | None = None
    vram_gb: float | None = None
    compute_capability: tuple[int, int] | None = None
    reason: str | None = None


def check_hybrid_gpu() -> GPUCapability:
    """
    Check whether the current Python environment exposes a GPU suitable
    for MinerU's hybrid-engine backend.

    The check respects CUDA_VISIBLE_DEVICES because PyTorch only sees
    devices exposed to the current process.
    """

    try:
        import torch
    except ImportError:
        return GPUCapability(
            available=False,
            suitable_for_hybrid=False,
            reason="PyTorch is not installed.",
        )

    if not torch.cuda.is_available():
        return GPUCapability(
            available=False,
            suitable_for_hybrid=False,
            reason="CUDA is not available.",
        )

    if torch.cuda.device_count() < 1:
        return GPUCapability(
            available=False,
            suitable_for_hybrid=False,
            reason="No CUDA GPU is visible to the current process.",
        )

    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)

    vram_gb = properties.total_memory / (1024 ** 3)
    compute_capability = (
        properties.major,
        properties.minor,
    )

    if compute_capability < HYBRID_MIN_COMPUTE_CAPABILITY:
        return GPUCapability(
            available=True,
            suitable_for_hybrid=False,
            device_name=properties.name,
            vram_gb=vram_gb,
            compute_capability=compute_capability,
            reason=(
                "GPU architecture is older than NVIDIA Volta "
                "(compute capability 7.0)."
            ),
        )

    if vram_gb < HYBRID_MIN_VRAM_GB:
        return GPUCapability(
            available=True,
            suitable_for_hybrid=False,
            device_name=properties.name,
            vram_gb=vram_gb,
            compute_capability=compute_capability,
            reason=(
                f"GPU has {vram_gb:.1f} GB VRAM; "
                f"{HYBRID_MIN_VRAM_GB:.0f} GB is required."
            ),
        )

    return GPUCapability(
        available=True,
        suitable_for_hybrid=True,
        device_name=properties.name,
        vram_gb=vram_gb,
        compute_capability=compute_capability,
    )


def resolve_backend(
    requested_backend: str,
) -> tuple[str, GPUCapability | None]:
    """
    Resolve a MinerU backend.

    pipeline:
        Always accepted because it can run on CPU.

    hybrid-engine:
        Used only when a suitable CUDA GPU is available. Otherwise
        Tabulus falls back to pipeline.
    """

    if requested_backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported MinerU backend: {requested_backend}. "
            f"Choose from: {', '.join(SUPPORTED_BACKENDS)}"
        )

    if requested_backend == PIPELINE_BACKEND:
        return PIPELINE_BACKEND, None

    capability = check_hybrid_gpu()

    if capability.suitable_for_hybrid:
        return HYBRID_BACKEND, capability

    return PIPELINE_BACKEND, capability
