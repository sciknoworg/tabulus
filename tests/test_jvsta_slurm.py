from __future__ import annotations

import json
from pathlib import Path

from evaluation.jvsta.run_profiling_experiment import summarize_canonical_crops
from evaluation.jvsta.slurm.submit_matrix import build_sbatch_command


def test_summarize_canonical_crops_aggregates_papers(tmp_path: Path) -> None:
    for name, count in (("paper-a", 2), ("paper-b", 3)):
        root = tmp_path / name
        root.mkdir()
        (root / "tables_index.json").write_text(
            json.dumps(
                {
                    "tables_found": count,
                    "crops_saved": count,
                    "refs_start_page": 10,
                    "tables": [{"table_id": i} for i in range(count)],
                }
            ),
            encoding="utf-8",
        )

    result = summarize_canonical_crops(tmp_path)
    assert result["paper_crop_roots"] == 2
    assert result["tables_found"] == 5
    assert result["crops_saved"] == 5


def test_build_sbatch_command_for_l40s_reconstruction(tmp_path: Path) -> None:
    row = {
        "stage": "reconstruction",
        "corpus": "ald",
        "adapter": "internvl3-5-8b",
        "conda_env": "tabulus-internvl3-5-8b",
        "partition": "p_48G",
        "gres": "gpu:l40s:1",
        "nodelist": "",
        "cpus": "8",
        "memory": "32G",
        "time_limit": "12:00:00",
        "input_path": "/crops/ald",
        "run_root": "/runs",
        "device": "gpu:0",
    }
    command = build_sbatch_command(row, repo_root=tmp_path, expected_commit="abc123")
    assert command[:3] == ["sbatch", "-p", "p_48G"]
    assert command[command.index("--gres") + 1] == "gpu:l40s:1"
    assert "--nodelist" not in command
    assert "evaluation.jvsta.run_experiment" in command
    assert "abc123" in command
    wrapper = str(tmp_path / "evaluation" / "jvsta" / "slurm" / "run_job.sh")
    wrapper_index = command.index(wrapper)
    assert command[wrapper_index + 1] == str(tmp_path)
    assert command[wrapper_index + 2] == "tabulus-internvl3-5-8b"


def test_build_sbatch_command_for_cpu_pipeline_uses_no_gpu(tmp_path: Path) -> None:
    row = {
        "stage": "profiling",
        "corpus": "asd",
        "backend": "pipeline",
        "conda_env": "tabulus-mineru",
        "partition": "p_48G",
        "gres": "",
        "nodelist": "gpu-l40s-01",
        "cpus": "8",
        "memory": "32G",
        "time_limit": "12:00:00",
        "input_path": "/papers/asd",
        "run_root": "/runs",
        "method": "auto",
        "effort": "high",
    }
    command = build_sbatch_command(row, repo_root=tmp_path, expected_commit="abc123")
    assert "--gres" not in command
    assert command[command.index("--nodelist") + 1] == "gpu-l40s-01"
    assert "evaluation.jvsta.run_profiling_experiment" in command
    assert "pipeline" in command
