from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from tabulus.table_ocr.base import (
    TableOCRCapabilities,
    TableOCRDependencyError,
    TableOCRInput,
    TableOCRResult,
)
from tabulus.table_ocr.parsing import parse_table_text


VALIDATED_PYTHON_MAJOR_MINOR = (3, 12)
VALIDATED_TRANSFORMERS_VERSION = "5.6.1"
VALIDATED_ACCELERATE_VERSION = "1.12.0"
VALIDATED_ALBUMENTATIONS_VERSION = "2.0.8"
VALIDATED_TIMM_VERSION = "1.0.22"
VALIDATED_EINOPS_VERSION = "0.8.2"
VALIDATED_OPEN_CLIP_TORCH_VERSION = "3.3.0"
VALIDATED_OPENCV_VERSION = "5.0.0.93"
VALIDATED_BEAUTIFULSOUP_VERSION = "4.15.0"

MODEL_REPO = "nvidia/NVIDIA-Nemotron-Parse-v1.2"
MODEL_REVISION = "2bd0189bffd6cdded6280d9f22a4077b25a504e3"
MODEL_VERSION = f"{MODEL_REPO}@{MODEL_REVISION}"

CRADIO_REPO = "nvidia/C-RADIOv2-H"
CRADIO_REVISION = "0d8f4c18c877166eda07ddae1386bcad256b7a6a"

MODEL_CLASS = "NemotronParseForConditionalGeneration"
ENCODER_CLASS = "RADIOModel"
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"
MODEL_IMAGE_SIZE = [2048, 1664]

PROMPT = (
    "</s><s><predict_bbox><predict_classes>"
    "<output_markdown><predict_no_text_in_pic>"
)

MAX_NEW_TOKENS = 9000
GENERATION_DO_SAMPLE = False
GENERATION_NUM_BEAMS = 1
GENERATION_REPETITION_PENALTY = 1.1

TABLE_PREFIX = r"\begin{tabular}"
REPETITION_MAX_REPETITIONS = 10
REPETITION_NGRAM_SIZES = [3, 4, 5, 6]
REPETITION_WINDOW_SIZE = 500

HELPER_FILES = (
    "hf_logits_processor.py",
    "postprocessing.py",
    "latex2html.py",
)


@dataclass(frozen=True)
class _NemotronParseRuntime:
    torch: Any
    image_module: Any
    tokenizer: Any
    processor: Any
    generation_config: Any
    model: Any
    table_insertion_processor_class: Any
    repetition_stop_processor_class: Any
    extract_classes_bboxes: Any
    postprocess_text: Any
    model_device: str
    model_dtype: str
    model_class: str
    encoder_class: str
    encoder_module: str
    encoder_source: str
    cradio_revision_verified: bool
    attention_implementation: str | None
    image_size: list[int] | None
    transformers_version: str | None
    accelerate_version: str | None
    albumentations_version: str | None
    timm_version: str | None
    einops_version: str | None
    open_clip_torch_version: str | None
    opencv_version: str | None
    beautifulsoup_version: str | None
    torch_version: str | None
    torchvision_version: str | None
    pillow_version: str | None
    huggingface_hub_version: str | None
    safetensors_version: str | None
    helper_dir: str


RuntimeLoader = Callable[[str], Any]
InferenceRunner = Callable[[Path, Any], dict[str, Any]]


def _installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _require_validated_version(
    package_name: str,
    expected: str,
) -> str:
    found = _installed_package_version(package_name)

    if found != expected:
        actual = found or "not installed"
        raise TableOCRDependencyError(
            "The nemotron-parse-v1-2 adapter is validated against "
            f"{package_name}=={expected}; found {actual}."
        )

    return found


def _nemotron_device(device: str) -> str:
    normalized = device.strip().lower()

    if normalized == "gpu":
        return "cuda:0"

    if normalized.startswith("gpu:"):
        index = normalized.split(":", maxsplit=1)[1]
        if index.isdigit():
            return f"cuda:{index}"

    raise ValueError(
        "Use device='gpu' or 'gpu:<index>' for nemotron-parse-v1-2."
    )


def _dependency_error() -> TableOCRDependencyError:
    return TableOCRDependencyError(
        "The nemotron-parse-v1-2 adapter requires Python 3.12, "
        "transformers==5.6.1, accelerate==1.12.0, "
        "albumentations==2.0.8, timm==1.0.22, einops==0.8.2, "
        "open-clip-torch==3.3.0, opencv-python-headless==5.0.0.93, "
        "beautifulsoup4==4.15.0, PyTorch, torchvision, Pillow, "
        "huggingface_hub, and safetensors."
    )


def _load_module_from_file(
    module_name: str,
    path: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load helper module from {path}."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def _load_pinned_helpers(
    huggingface_hub: Any,
) -> tuple[ModuleType, ModuleType, Path]:
    helper_paths = [
        Path(
            huggingface_hub.hf_hub_download(
                repo_id=MODEL_REPO,
                revision=MODEL_REVISION,
                filename=filename,
                local_files_only=True,
            )
        )
        for filename in HELPER_FILES
    ]

    helper_dir = helper_paths[0].parent

    if any(path.parent != helper_dir for path in helper_paths):
        raise RuntimeError(
            "Pinned Nemotron helper files resolved to different snapshots."
        )

    latex_module = _load_module_from_file(
        "tabulus_nemotron_parse_v1_2_latex2html",
        helper_dir / "latex2html.py",
    )

    sentinel = object()
    previous_latex = sys.modules.get("latex2html", sentinel)
    sys.modules["latex2html"] = latex_module

    try:
        logits_module = _load_module_from_file(
            "tabulus_nemotron_parse_v1_2_logits",
            helper_dir / "hf_logits_processor.py",
        )
        postprocessing_module = _load_module_from_file(
            "tabulus_nemotron_parse_v1_2_postprocessing",
            helper_dir / "postprocessing.py",
        )
    finally:
        if previous_latex is sentinel:
            sys.modules.pop("latex2html", None)
        else:
            sys.modules["latex2html"] = previous_latex

    return logits_module, postprocessing_module, helper_dir


def _verify_cradio_encoder(
    model: Any,
) -> tuple[str, str, str]:
    encoder = model.encoder.model_encoder
    encoder_class = type(encoder)
    encoder_module = encoder_class.__module__
    encoder_source = inspect.getfile(encoder_class)

    verified = (
        CRADIO_REVISION in encoder_module
        or CRADIO_REVISION in encoder_source
    )

    if not verified:
        raise RuntimeError(
            "Nemotron Parse loaded C-RADIO remote code from an "
            "unexpected revision. Expected "
            f"{CRADIO_REPO}@{CRADIO_REVISION}; "
            f"module={encoder_module!r}, source={encoder_source!r}."
        )

    return (
        encoder_class.__name__,
        encoder_module,
        encoder_source,
    )


def _validate_generation_config(
    generation_config: Any,
) -> None:
    expected = {
        "do_sample": GENERATION_DO_SAMPLE,
        "num_beams": GENERATION_NUM_BEAMS,
        "repetition_penalty": GENERATION_REPETITION_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
    }

    for key, value in expected.items():
        found = getattr(generation_config, key, None)

        if found != value:
            raise RuntimeError(
                "Unexpected Nemotron Parse generation configuration: "
                f"{key}={found!r}; expected {value!r}."
            )


def _default_runtime_loader(
    model_device: str,
) -> _NemotronParseRuntime:
    if sys.version_info[:2] != VALIDATED_PYTHON_MAJOR_MINOR:
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise TableOCRDependencyError(
            "The nemotron-parse-v1-2 adapter is validated against "
            f"Python 3.12; found {found}."
        )

    transformers_version = _require_validated_version(
        "transformers",
        VALIDATED_TRANSFORMERS_VERSION,
    )
    accelerate_version = _require_validated_version(
        "accelerate",
        VALIDATED_ACCELERATE_VERSION,
    )
    albumentations_version = _require_validated_version(
        "albumentations",
        VALIDATED_ALBUMENTATIONS_VERSION,
    )
    timm_version = _require_validated_version(
        "timm",
        VALIDATED_TIMM_VERSION,
    )
    einops_version = _require_validated_version(
        "einops",
        VALIDATED_EINOPS_VERSION,
    )
    open_clip_torch_version = _require_validated_version(
        "open-clip-torch",
        VALIDATED_OPEN_CLIP_TORCH_VERSION,
    )
    opencv_version = _require_validated_version(
        "opencv-python-headless",
        VALIDATED_OPENCV_VERSION,
    )
    beautifulsoup_version = _require_validated_version(
        "beautifulsoup4",
        VALIDATED_BEAUTIFULSOUP_VERSION,
    )

    try:
        torch = importlib.import_module("torch")
        image_module = importlib.import_module("PIL.Image")
        transformers = importlib.import_module("transformers")
        huggingface_hub = importlib.import_module("huggingface_hub")

        importlib.import_module("torchvision")
        importlib.import_module("accelerate")
        importlib.import_module("albumentations")
        importlib.import_module("timm")
        importlib.import_module("einops")
        importlib.import_module("open_clip")
        importlib.import_module("cv2")
        importlib.import_module("bs4")
        importlib.import_module("safetensors")
    except ImportError as exc:
        raise _dependency_error() from exc

    if not bool(torch.cuda.is_available()):
        raise RuntimeError(
            "nemotron-parse-v1-2 requires CUDA in the validated "
            "Tabulus configuration, but CUDA is not available."
        )

    logits_module, postprocessing_module, helper_dir = (
        _load_pinned_helpers(huggingface_hub)
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )

    processor = transformers.AutoProcessor.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    )

    generation_config = (
        transformers.GenerationConfig.from_pretrained(
            MODEL_REPO,
            revision=MODEL_REVISION,
            trust_remote_code=True,
        )
    )
    _validate_generation_config(generation_config)

    model = transformers.AutoModel.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(model_device).eval()

    resolved_device = str(next(model.parameters()).device)

    if resolved_device != model_device:
        raise RuntimeError(
            "NVIDIA Nemotron Parse v1.2 loaded on an unexpected "
            f"device: requested {model_device}, resolved "
            f"{resolved_device}."
        )

    encoder_class, encoder_module, encoder_source = (
        _verify_cradio_encoder(model)
    )

    resolved_attention = getattr(
        model.config.decoder,
        "_attn_implementation",
        None,
    )

    if resolved_attention != ATTENTION_IMPLEMENTATION:
        raise RuntimeError(
            "NVIDIA Nemotron Parse v1.2 loaded with an unexpected "
            f"decoder attention implementation: {resolved_attention!r}."
        )

    image_size_value = getattr(
        model.config,
        "image_size",
        None,
    )
    image_size = (
        [int(value) for value in image_size_value]
        if image_size_value is not None
        else None
    )

    if image_size != MODEL_IMAGE_SIZE:
        raise RuntimeError(
            "NVIDIA Nemotron Parse v1.2 loaded with an unexpected "
            f"image size: {image_size!r}; expected "
            f"{MODEL_IMAGE_SIZE!r}."
        )

    return _NemotronParseRuntime(
        torch=torch,
        image_module=image_module,
        tokenizer=tokenizer,
        processor=processor,
        generation_config=generation_config,
        model=model,
        table_insertion_processor_class=(
            logits_module.TableInsertionLogitsProcessor
        ),
        repetition_stop_processor_class=(
            logits_module.RepetitionStopProcessor
        ),
        extract_classes_bboxes=(
            postprocessing_module.extract_classes_bboxes
        ),
        postprocess_text=postprocessing_module.postprocess_text,
        model_device=model_device,
        model_dtype=str(next(model.parameters()).dtype),
        model_class=type(model).__name__,
        encoder_class=encoder_class,
        encoder_module=encoder_module,
        encoder_source=encoder_source,
        cradio_revision_verified=True,
        attention_implementation=str(resolved_attention),
        image_size=image_size,
        transformers_version=transformers_version,
        accelerate_version=accelerate_version,
        albumentations_version=albumentations_version,
        timm_version=timm_version,
        einops_version=einops_version,
        open_clip_torch_version=open_clip_torch_version,
        opencv_version=opencv_version,
        beautifulsoup_version=beautifulsoup_version,
        torch_version=_installed_package_version("torch"),
        torchvision_version=_installed_package_version(
            "torchvision"
        ),
        pillow_version=_installed_package_version("Pillow"),
        huggingface_hub_version=_installed_package_version(
            "huggingface_hub"
        ),
        safetensors_version=_installed_package_version(
            "safetensors"
        ),
        helper_dir=str(helper_dir),
    )


def _serializable_bbox(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [
            float(item) if hasattr(item, "__float__") else item
            for item in value
        ]

    return value


def _default_inference_runner(
    image_path: Path,
    runtime: _NemotronParseRuntime,
) -> dict[str, Any]:
    with runtime.image_module.open(image_path) as opened_image:
        image = opened_image.convert("RGB")
        source_width, source_height = image.size

    inputs = runtime.processor(
        images=[image],
        text=PROMPT,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(runtime.model_device)

    prompt_tokens = int(inputs["input_ids"].shape[-1])

    table_processor = runtime.table_insertion_processor_class(
        tokenizer=runtime.tokenizer,
        table_prefix=TABLE_PREFIX,
    )
    repetition_processor = runtime.repetition_stop_processor_class(
        tokenizer=runtime.tokenizer,
        max_repetitions=REPETITION_MAX_REPETITIONS,
        ngram_sizes=REPETITION_NGRAM_SIZES,
        window_size=REPETITION_WINDOW_SIZE,
    )

    try:
        with runtime.torch.inference_mode():
            outputs = runtime.model.generate(
                **inputs,
                generation_config=runtime.generation_config,
                logits_processor=[
                    table_processor,
                    repetition_processor,
                ],
            )
    finally:
        table_processor.reset()
        repetition_processor.reset()

    raw_output = runtime.processor.batch_decode(
        outputs,
        skip_special_tokens=False,
    )[0]

    clean_output = runtime.processor.batch_decode(
        outputs,
        skip_special_tokens=True,
    )[0]

    classes, bboxes, texts = runtime.extract_classes_bboxes(
        clean_output
    )

    objects: list[dict[str, Any]] = []
    html_tables: list[str] = []

    for class_name, bbox, text in zip(
        classes,
        bboxes,
        texts,
    ):
        class_value = str(class_name)
        text_value = str(text)

        objects.append(
            {
                "class": class_value,
                "bbox": _serializable_bbox(bbox),
                "text": text_value,
            }
        )

        if class_value != "Table":
            continue

        html = runtime.postprocess_text(
            text_value,
            cls=class_value,
            table_format="HTML",
            text_format="markdown",
            blank_text_in_figures=False,
        )

        if isinstance(html, str) and html.strip():
            html_tables.append(html)

    return {
        "raw_output": raw_output,
        "clean_output": clean_output,
        "source_image_size": [
            int(source_width),
            int(source_height),
        ],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(outputs.shape[-1]),
        "raw_output_chars": len(raw_output),
        "clean_output_chars": len(clean_output),
        "objects": objects,
        "html_tables": html_tables,
    }


class NemotronParseV12Adapter:
    """NVIDIA Nemotron Parse v1.2 reconstruction for canonical crops."""

    NAME = "nemotron-parse-v1-2"
    DISPLAY_NAME = "NVIDIA Nemotron Parse v1.2"
    MODEL_VERSION = MODEL_VERSION

    _CAPABILITIES = TableOCRCapabilities(
        name=NAME,
        display_name=DISPLAY_NAME,
        cpu_supported=False,
        gpu_supported=True,
    )

    def __init__(
        self,
        *,
        device: str = "gpu:0",
        runtime_loader: RuntimeLoader | None = None,
        inference_runner: InferenceRunner | None = None,
    ) -> None:
        self.device = device
        self._model_device = _nemotron_device(device)
        self._runtime_loader = (
            runtime_loader or _default_runtime_loader
        )
        self._inference_runner = (
            inference_runner or _default_inference_runner
        )
        self._runtime: Any | None = None

    @property
    def capabilities(self) -> TableOCRCapabilities:
        return self._CAPABILITIES

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            self._runtime = self._runtime_loader(
                self._model_device
            )

        return self._runtime

    def _result(
        self,
        table: TableOCRInput,
        *,
        status: str,
        adapter_version: str | None = None,
        result_count: int = 0,
        native_json: list[Any] | None = None,
        native_markdown: list[Any] | None = None,
        error: str | None = None,
    ) -> TableOCRResult:
        return TableOCRResult(
            table_id=table.table_id,
            adapter_name=self.NAME,
            adapter_version=adapter_version,
            model_version=self.MODEL_VERSION,
            device=self.device,
            source_image=table.image_path,
            status=status,  # type: ignore[arg-type]
            provenance=dict(table.provenance),
            result_count=result_count,
            native_json=native_json or [],
            native_markdown=native_markdown or [],
            error=error,
        )

    def extract(
        self,
        table: TableOCRInput,
    ) -> TableOCRResult:
        image_path = Path(table.image_path)

        if not image_path.is_file():
            return self._result(
                table,
                status="error",
                error=f"Table crop image not found: {image_path}",
            )

        try:
            runtime = self._get_runtime()
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                error=(
                    "Could not initialize NVIDIA Nemotron Parse "
                    f"v1.2: {exc}"
                ),
            )

        adapter_version = getattr(
            runtime,
            "transformers_version",
            None,
        )

        try:
            reconstruction = self._inference_runner(
                image_path,
                runtime,
            )
        except TableOCRDependencyError:
            raise
        except Exception as exc:
            return self._result(
                table,
                status="error",
                adapter_version=adapter_version,
                error=(
                    "NVIDIA Nemotron Parse v1.2 table "
                    f"reconstruction failed: {exc}"
                ),
            )

        raw_value = reconstruction.get("raw_output", "")
        clean_value = reconstruction.get("clean_output", "")

        raw_output = (
            raw_value
            if isinstance(raw_value, str)
            else str(raw_value or "")
        )
        clean_output = (
            clean_value
            if isinstance(clean_value, str)
            else str(clean_value or "")
        )

        html_tables = [
            str(value)
            for value in reconstruction.get("html_tables", [])
            if isinstance(value, str) and value.strip()
        ]

        parsed_tables = []
        parser_errors: list[str] = []

        for html in html_tables:
            try:
                parsed_tables.extend(parse_table_text(html))
            except Exception as exc:
                parser_errors.append(str(exc))

        native = {
            "nemotron_parse_v1_2": {
                "model_repo": MODEL_REPO,
                "model_revision": MODEL_REVISION,
                "model_class": getattr(
                    runtime,
                    "model_class",
                    MODEL_CLASS,
                ),
                "model_dtype": MODEL_DTYPE,
                "resolved_model_dtype": getattr(
                    runtime,
                    "model_dtype",
                    None,
                ),
                "image_size": getattr(
                    runtime,
                    "image_size",
                    MODEL_IMAGE_SIZE,
                ),
                "attention_implementation": getattr(
                    runtime,
                    "attention_implementation",
                    ATTENTION_IMPLEMENTATION,
                ),
                "cradio_repo": CRADIO_REPO,
                "cradio_revision": CRADIO_REVISION,
                "cradio_revision_verified": getattr(
                    runtime,
                    "cradio_revision_verified",
                    False,
                ),
                "encoder_class": getattr(
                    runtime,
                    "encoder_class",
                    ENCODER_CLASS,
                ),
                "encoder_module": getattr(
                    runtime,
                    "encoder_module",
                    None,
                ),
                "encoder_source": getattr(
                    runtime,
                    "encoder_source",
                    None,
                ),
                "prompt": PROMPT,
                "native_format": (
                    "grounded_semantic_objects_with_latex_tables"
                ),
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_do_sample": GENERATION_DO_SAMPLE,
                "generation_num_beams": GENERATION_NUM_BEAMS,
                "generation_repetition_penalty": (
                    GENERATION_REPETITION_PENALTY
                ),
                "generation_processors": [
                    {
                        "name": "TableInsertionLogitsProcessor",
                        "table_prefix": TABLE_PREFIX,
                    },
                    {
                        "name": "RepetitionStopProcessor",
                        "max_repetitions": (
                            REPETITION_MAX_REPETITIONS
                        ),
                        "ngram_sizes": REPETITION_NGRAM_SIZES,
                        "window_size": REPETITION_WINDOW_SIZE,
                    },
                ],
                "execution_device": getattr(
                    runtime,
                    "model_device",
                    self._model_device,
                ),
                "python_version": (
                    f"{sys.version_info.major}."
                    f"{sys.version_info.minor}."
                    f"{sys.version_info.micro}"
                ),
                "transformers_version": adapter_version,
                "accelerate_version": getattr(
                    runtime,
                    "accelerate_version",
                    None,
                ),
                "albumentations_version": getattr(
                    runtime,
                    "albumentations_version",
                    None,
                ),
                "timm_version": getattr(
                    runtime,
                    "timm_version",
                    None,
                ),
                "einops_version": getattr(
                    runtime,
                    "einops_version",
                    None,
                ),
                "open_clip_torch_version": getattr(
                    runtime,
                    "open_clip_torch_version",
                    None,
                ),
                "opencv_version": getattr(
                    runtime,
                    "opencv_version",
                    None,
                ),
                "beautifulsoup_version": getattr(
                    runtime,
                    "beautifulsoup_version",
                    None,
                ),
                "torch_version": getattr(
                    runtime,
                    "torch_version",
                    None,
                ),
                "torchvision_version": getattr(
                    runtime,
                    "torchvision_version",
                    None,
                ),
                "pillow_version": getattr(
                    runtime,
                    "pillow_version",
                    None,
                ),
                "huggingface_hub_version": getattr(
                    runtime,
                    "huggingface_hub_version",
                    None,
                ),
                "safetensors_version": getattr(
                    runtime,
                    "safetensors_version",
                    None,
                ),
                "helper_revision": MODEL_REVISION,
                "helper_dir": getattr(
                    runtime,
                    "helper_dir",
                    None,
                ),
                "source_image_size": reconstruction.get(
                    "source_image_size",
                    [],
                ),
                "prompt_tokens": reconstruction.get(
                    "prompt_tokens",
                    0,
                ),
                "generated_tokens": reconstruction.get(
                    "generated_tokens",
                    0,
                ),
                "raw_output_chars": reconstruction.get(
                    "raw_output_chars",
                    len(raw_output),
                ),
                "clean_output_chars": reconstruction.get(
                    "clean_output_chars",
                    len(clean_output),
                ),
                "raw_output": raw_output,
                "clean_output": clean_output,
                "objects": reconstruction.get(
                    "objects",
                    [],
                ),
                "table_objects": sum(
                    item.get("class") == "Table"
                    for item in reconstruction.get(
                        "objects",
                        [],
                    )
                    if isinstance(item, dict)
                ),
                "html_tables": html_tables,
                "normalization": (
                    "pinned NVIDIA postprocessing.py:"
                    "postprocess_text(table_format='HTML')"
                ),
                "parser": (
                    "tabulus.table_ocr.parsing:"
                    "parse_table_text"
                ),
                "structured_tables_detected": len(
                    parsed_tables
                ),
                "parser_errors": parser_errors,
                "input_policy": "canonical_mineru_crop",
                "image_preprocessing": {
                    "external": "rgb_conversion_only",
                    "processor": "AutoProcessor",
                    "model_internal_resize": True,
                },
                "generated_bbox_usage": "provenance_only",
                "layout_redetection": False,
                "recropping": False,
                "external_recropping": False,
                "semantic_repair": False,
                "continued_table_merging": False,
            }
        }

        if not clean_output.strip():
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "NVIDIA Nemotron Parse v1.2 returned no "
                    "generated output."
                ),
            )

        table_objects = native["nemotron_parse_v1_2"][
            "table_objects"
        ]

        if table_objects == 0:
            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                error=(
                    "NVIDIA Nemotron Parse v1.2 generated no "
                    "Table-class object."
                ),
            )

        if not parsed_tables:
            error = (
                "NVIDIA Nemotron Parse v1.2 Table-class output "
                "did not yield a usable structured table."
            )

            if parser_errors:
                error += " Shared parser errors: " + "; ".join(
                    parser_errors
                )

            return self._result(
                table,
                status="empty",
                adapter_version=adapter_version,
                native_json=[native],
                native_markdown=html_tables,
                error=error,
            )

        return self._result(
            table,
            status="ok",
            adapter_version=adapter_version,
            result_count=len(parsed_tables),
            native_json=[native],
            native_markdown=html_tables,
        )
