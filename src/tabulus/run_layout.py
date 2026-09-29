from __future__ import annotations

from pathlib import Path


def step3_output_dir_for_reconstruction(
    reconstruction_dir: Path,
) -> Path | None:
    """Return the canonical paper-local Step 3 output directory.

    Recognized controlled input layouts:

      tabulus_runs/stage2/run_XX/<adapter>/reconstruction/
      tabulus_runs/step2/run_XX/<adapter>/reconstruction/

    map to:

      tabulus_runs/step3/controlled/run_XX/<adapter>/

    Recognized production input layouts:

      tabulus_runs/stage2/production/run_XX/<adapter>/reconstruction/
      tabulus_runs/step2/production/run_XX/<adapter>/reconstruction/

    map to:

      tabulus_runs/step3/production/run_XX/<adapter>/

    ``stage2`` remains supported while Step 2 has not yet been renamed.
    ``step2`` is accepted already so the mapping survives that later
    filesystem migration unchanged.

    Non-TabulusBench or unrecognized layouts return ``None``.
    """

    reconstruction_dir = Path(reconstruction_dir)
    parts = reconstruction_dir.parts

    try:
        tabulus_runs_index = parts.index("tabulus_runs")
    except ValueError:
        return None

    suffix = parts[tabulus_runs_index + 1 :]

    if (
        not suffix
        or suffix[0] not in {"stage2", "step2"}
        or suffix[-1] != "reconstruction"
    ):
        return None

    if len(suffix) == 4:
        _, run_id, adapter_name, _ = suffix
        track = "controlled"

    elif len(suffix) == 5 and suffix[1] == "production":
        _, _, run_id, adapter_name, _ = suffix
        track = "production"

    else:
        return None

    tabulus_runs_root = Path(
        *parts[: tabulus_runs_index + 1]
    )

    return (
        tabulus_runs_root
        / "step3"
        / track
        / run_id
        / adapter_name
    )
