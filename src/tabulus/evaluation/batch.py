"""Resumable, deterministic CPU evaluation of explicitly paired physical grids."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from functools import lru_cache
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import tempfile

from tabulus.evaluation.grid_inputs import FragmentPair, GridInput, load_grid
from tabulus.evaluation.grid_scoring import (
    aggregate_grid_results, prepare_grid, score_table_grids, select_metrics,
)
from tabulus.evaluation.rms import DEFAULT_NUMBER_THRESHOLD, DEFAULT_TEXT_THRESHOLD


def content_hash(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path is not None and path.is_file() else None


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def atomic_json(path: Path, value) -> None:
    """Publish complete JSON with a same-filesystem atomic replacement."""
    atomic_text(path, (_canonical(value) + b"\n").decode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    """Atomically publish a text report in the parent process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def implementation_provenance() -> dict:
    directory = Path(__file__).parent
    packages = {}
    for package in ("numpy", "scipy", "rapidfuzz"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(), "dependencies": packages,
        "sources": {
            name: content_hash(directory / name) for name in (
                "rms.py", "grid_scoring.py", "grid_inputs.py", "batch.py",
                "table_reconstruction.py", "step7_physical.py",
                "__init__.py", "../resolved_export/csv_io.py", "../table_ocr/parsing.py",
            )
        },
    }


def _source_identity(source: GridInput | None, hashes: dict) -> dict | None:
    if source is None:
        return None
    value = asdict(source)
    if source.path is not None:
        value["path"] = str(source.path)
        if source.path not in hashes:
            hashes[source.path] = content_hash(source.path)
        value["sha256"] = hashes[source.path]
    return value


@contextmanager
def _run_lock(out: Path):
    handle = (out / ".lock").open("a+b")
    try:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"Evaluation output directory is already in use: {out}") from error
        yield
    finally:
        handle.close()  # OS locks are released even after a process is killed.


def _worker_init():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"


@lru_cache(maxsize=1024)
def _prepared_gold(source: GridInput, expected_hash: str | None):
    if content_hash(source.path) != expected_hash:
        raise ValueError(f"Gold input changed during evaluation: {source.path}")
    prepared = prepare_grid(load_grid(source))
    if content_hash(source.path) != expected_hash:
        raise ValueError(f"Gold input changed during preparation: {source.path}")
    return prepared


def _score_pair(job):
    pair, identity, settings = job
    if content_hash(pair.gold.path) != identity["gold"].get("sha256"):
        raise ValueError(f"Gold input changed during evaluation: {pair.gold.path}")
    gold = _prepared_gold(pair.gold, identity["gold"].get("sha256"))
    prediction = []
    status, error = "missing", None
    source = pair.prediction
    if pair.prediction_status != "ok":
        status, error = "unusable", f"Archived prediction status is {pair.prediction_status}."
    elif source is not None:
        expected_hash = identity["prediction"].get("sha256")
        if content_hash(source.path) != expected_hash:
            raise ValueError(f"Prediction changed during evaluation: {source.path}")
        if source.inline_grid is not None or (source.path is not None and source.path.is_file()):
            try:
                prediction = load_grid(source)
                status = "scored" if any(cell.strip() for row in prediction for cell in row) else "empty"
            except (csv.Error, UnicodeError) as exc:
                status, error = "unusable", str(exc)
            if content_hash(source.path) != expected_hash:
                raise ValueError(f"Prediction changed during preparation: {source.path}")
    result = score_table_grids(
        gold, prediction, metrics=settings["metrics"],
        text_threshold=settings["text_threshold"], number_threshold=settings["number_threshold"],
        status=status,
    )
    return {
        **result, "fragment_id": pair.fragment_id, "group": pair.group,
        "provenance": pair.provenance, "inputs": identity, "error": error,
    }


def summarize_results(results: list[dict]) -> dict:
    results = sorted(results, key=lambda row: row["fragment_id"])
    groups = defaultdict(list)
    for row in results:
        groups[row["group"]].append(row)
    return {
        "schema_version": 1, "score_scale": "[0,100]",
        "groups": {key: aggregate_grid_results(rows) for key, rows in sorted(groups.items())},
        "aggregate": aggregate_grid_results(results),
    }


def _write_csv(path: Path, results: list[dict]):
    descriptor, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            fields = [
                "fragment_id", "group", "status", "metric", "precision", "recall", "f1",
                "matched_credit", "gold_count", "predicted_count", "orientation", "unit",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in results:
                for metric, scores in row["metrics"].items():
                    stats = row["statistics"][metric]
                    writer.writerow({
                        **{key: row[key] for key in ("fragment_id", "group", "status")},
                        "metric": metric, **scores,
                        **{key: stats.get(key) for key in fields[7:]},
                    })
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def evaluate_pairs(
    pairs: list[FragmentPair], *, metrics=("rms", "exact-mapping", "exact"),
    text_threshold=DEFAULT_TEXT_THRESHOLD, number_threshold=DEFAULT_NUMBER_THRESHOLD,
    workers: int = 1, output_dir: Path | None = None, resume: bool = False,
    on_result=None, on_complete=None,
) -> dict:
    """Score explicit pairs, saving each completion atomically in the parent process."""
    metrics = select_metrics(metrics)
    if workers < 1:
        raise ValueError("workers must be positive.")
    if not 0 <= text_threshold <= 1 or not 0 <= number_threshold <= 1:
        raise ValueError("RMS thresholds must be between zero and one.")
    if not pairs or any(not isinstance(p.fragment_id, str) or not p.fragment_id for p in pairs):
        raise ValueError("Supply nonempty fragment identities.")
    if len({p.fragment_id for p in pairs}) != len(pairs):
        raise ValueError("Duplicate fragment identity.")
    pairs = sorted(pairs, key=lambda pair: pair.fragment_id)
    hashes = {}
    identities = {
        p.fragment_id: {
            "gold": _source_identity(p.gold, hashes), "prediction": _source_identity(p.prediction, hashes),
            "group": p.group, "status": p.prediction_status, "provenance": p.provenance,
        } for p in pairs
    }
    settings = {
        "metrics": list(metrics), "text_threshold": text_threshold, "number_threshold": number_threshold,
    }
    configuration = {
        "schema_version": 1, "settings": settings, "implementation": implementation_provenance(),
        "preparation_implementations": sorted({
            pair.provenance["benchmark_preparation_sha256"] for pair in pairs
            if pair.provenance.get("benchmark_preparation_sha256")
        }),
        "pairs": identities,
    }
    fingerprint = _digest(configuration)
    out = Path(output_dir).expanduser().resolve() if output_dir is not None else None
    if resume and out is None:
        raise ValueError("Resume requires an output directory.")
    if out is not None:
        if any(path.is_relative_to(out) for path in hashes):
            raise ValueError("Evaluation output cannot contain input artifacts.")
        if out.exists() and any(out.iterdir()) and not resume:
            raise ValueError(f"Evaluation output directory must be empty: {out}")
        if resume and not (out / "run.json").is_file():
            raise ValueError("Resume requires an existing run.json.")
        out.mkdir(parents=True, exist_ok=True)
    results = {}
    with _run_lock(out) if out is not None else nullcontext():
        if out is not None:
            run_path = out / "run.json"
            if resume:
                saved = json.loads(run_path.read_text(encoding="utf-8"))
                if (
                    saved.get("fingerprint") != fingerprint
                    or _digest({k: v for k, v in saved.items() if k != "fingerprint"}) != fingerprint
                ):
                    raise ValueError("Stale checkpoint: inputs, correspondence, settings or implementation changed.")
            else:
                atomic_json(run_path, {"fingerprint": fingerprint, **configuration})
            checkpoints = out / "fragments"
            checkpoints.mkdir(exist_ok=True)
        jobs = []
        for pair in pairs:
            checkpoint = out / "fragments" / f"{_digest(pair.fragment_id)}.json" if out else None
            if checkpoint is not None and checkpoint.exists():
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if (
                    saved.get("fingerprint") != fingerprint
                    or not isinstance(saved.get("result"), dict)
                    or saved["result"].get("fragment_id") != pair.fragment_id
                    or saved.get("result_sha256") != _digest(saved["result"])
                ):
                    raise ValueError(f"Stale or corrupted fragment checkpoint: {pair.fragment_id}")
                results[pair.fragment_id] = saved["result"]
            else:
                jobs.append((pair, identities[pair.fragment_id], settings))

        def accept(result):
            # Match the representation restored from JSON checkpoints exactly.
            result = json.loads(_canonical(result))
            key = result["fragment_id"]
            if out is not None:
                atomic_json(out / "fragments" / f"{_digest(key)}.json", {
                    "fingerprint": fingerprint, "result_sha256": _digest(result), "result": result,
                })
            results[key] = result
            if on_result is not None:
                on_result(result)

        if workers == 1:
            for job in jobs:
                accept(_score_pair(job))
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
                futures = [executor.submit(_score_pair, job) for job in jobs]
                for future in as_completed(futures):
                    accept(future.result())
        ordered = [results[p.fragment_id] for p in pairs]
        summary = {**summarize_results(ordered), "settings": settings, "fingerprint": fingerprint}
        if out is not None:
            atomic_json(out / "results.json", ordered)
            atomic_json(out / "summary.json", summary)
            _write_csv(out / "fragments.csv", ordered)
        if on_complete is not None:
            on_complete({**summary, "results": ordered})
    return {**summary, "results": ordered}


def aggregate_evaluation_runs(run_dirs: list[Path], output_dir: Path) -> dict:
    """Aggregate disjoint completed tasks after validating settings and provenance."""
    out = Path(output_dir).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Aggregation requires an empty output directory.")
    results, seen = [], set()
    expected = None
    for directory in run_dirs:
        directory = Path(directory).expanduser().resolve()
        if out.is_relative_to(directory) or directory.is_relative_to(out):
            raise ValueError("Aggregation output must be independent of its input runs.")
        config = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        signature = (config["settings"], config["implementation"], config.get("preparation_implementations", []))
        if _digest({k: v for k, v in config.items() if k != "fingerprint"}) != config["fingerprint"]:
            raise ValueError(f"Corrupted run configuration: {directory}")
        if expected is not None and signature != expected:
            raise ValueError("Cannot aggregate runs with different metric settings or implementations.")
        expected = signature
        rows = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if summary["fingerprint"] != config["fingerprint"] or {r["fragment_id"] for r in rows} != set(config["pairs"]):
            raise ValueError(f"Run is incomplete or inconsistent: {directory}")
        for row in rows:
            key = row["fragment_id"]
            checkpoint = json.loads((directory / "fragments" / f"{_digest(key)}.json").read_text(encoding="utf-8"))
            if key in seen or _digest(row) != checkpoint["result_sha256"] or checkpoint["fingerprint"] != config["fingerprint"]:
                raise ValueError(f"Duplicate or corrupted fragment result: {key}")
            seen.add(key)
            results.append(row)
    if not results:
        raise ValueError("Supply completed evaluation runs.")
    results.sort(key=lambda row: row["fragment_id"])
    report = {**summarize_results(results), "settings": expected[0],
              "implementation": expected[1], "runs": [str(p) for p in run_dirs]}
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", report)
    atomic_json(out / "results.json", results)
    _write_csv(out / "fragments.csv", results)
    return report
