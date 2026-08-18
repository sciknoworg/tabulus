from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol


TableOCRStatus = Literal["ok", "empty", "error"]


class TableOCRDependencyError(RuntimeError):
    """Raised when an optional OCR adapter dependency is unavailable."""


@dataclass(frozen=True)
class TableOCRCapabilities:
    """Static device capabilities exposed by a table OCR adapter."""

    name: str
    display_name: str
    cpu_supported: bool
    gpu_supported: bool

    def supports_device(self, device: str) -> bool:
        normalized = device.strip().lower()

        if normalized.startswith("cpu"):
            return self.cpu_supported

        if normalized.startswith("gpu"):
            return self.gpu_supported

        return False


@dataclass(frozen=True)
class TableOCRInput:
    """One normalized table crop passed to a reconstruction adapter."""

    table_id: int
    image_path: Path
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "image_path", Path(self.image_path))

    @classmethod
    def from_crop_record(
        cls,
        crop_root: Path,
        record: dict[str, Any],
    ) -> "TableOCRInput":
        table_id = record.get("table_id")
        image = record.get("image")

        if not isinstance(table_id, int):
            raise ValueError("Table crop record has no valid integer table_id.")

        if not isinstance(image, str) or not image.strip():
            raise ValueError("Table crop record has no valid image path.")

        return cls(
            table_id=table_id,
            image_path=Path(crop_root) / image,
            provenance=dict(record),
        )


@dataclass
class TableOCRResult:
    """Adapter-neutral result for one input table crop."""

    table_id: int
    adapter_name: str
    device: str
    source_image: Path
    status: TableOCRStatus
    provenance: dict[str, Any] = field(default_factory=dict)
    adapter_version: str | None = None
    model_version: str | None = None
    result_count: int = 0
    native_json: list[Any] = field(default_factory=list)
    native_markdown: list[Any] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_image"] = str(self.source_image)
        return data


class TableOCRAdapter(Protocol):
    """Common contract implemented by table reconstruction adapters."""

    @property
    def capabilities(self) -> TableOCRCapabilities:
        ...

    def extract(self, table: TableOCRInput) -> TableOCRResult:
        ...
