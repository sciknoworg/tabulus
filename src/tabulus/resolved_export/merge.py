from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tabulus.table_continuations import (
    continuation_links_from_records,
)


MERGED_TABLES_DIR_NAME = "merged"
MERGED_ORIGIN_COLUMN = "tabulus_physical_table_id"


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    path = Path(path).expanduser()

    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} is not valid JSON: {path}"
        ) from error

    if not isinstance(value, dict):
        raise ValueError(
            f"{label} must contain a JSON object: {path}"
        )

    return value


def _read_csv(path: Path) -> list[list[str]]:
    with Path(path).open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        return [
            list(row)
            for row in csv.reader(handle)
        ]


def resolve_tables_index_path(
    reconstruction_dir: Path,
    explicit_path: Path | None,
) -> Path:
    """
    Resolve the Step 1 index required for continuation merging.

    Canonical reconstruction output is
    ``<crop-root>/reconstructions/<adapter>/``.
    """

    if explicit_path is not None:
        path = Path(explicit_path).expanduser()

        if not path.is_file():
            raise FileNotFoundError(
                "Step 1 tables_index.json not found: "
                f"{path}"
            )

        return path

    inferred = (
        Path(reconstruction_dir)
        .parent
        .parent
        / "tables_index.json"
    )

    if inferred.is_file():
        return inferred

    raise FileNotFoundError(
        "Step 7 continuation merging requires the "
        "Step 1 tables_index.json artifact. It could "
        "not be inferred from reconstruction directory "
        f"{reconstruction_dir}."
    )


def _normalize_header_row(
    row: list[str],
) -> tuple[str, ...]:
    return tuple(
        " ".join(str(cell).split()).casefold()
        for cell in row
    )


def _rectangular_width(
    rows: list[list[str]],
    table_id: int,
) -> tuple[int | None, str | None]:
    if not rows:
        return (
            None,
            f"physical table {table_id} has no rows",
        )

    width = len(rows[0])

    if width == 0:
        return (
            None,
            f"physical table {table_id} has zero columns",
        )

    for row_index, row in enumerate(rows):
        if len(row) != width:
            return (
                None,
                "physical table "
                f"{table_id} is ragged at row "
                f"{row_index}: expected {width} "
                f"columns, found {len(row)}",
            )

    return width, None


def _load_continuation_records(
    tables_index_path: Path,
) -> tuple[
    list[dict[str, Any]],
    dict[int, Any],
    dict[int, int],
]:
    payload = _load_json_object(
        tables_index_path,
        label="Step 1 tables_index.json artifact",
    )

    records = payload.get("tables")

    if not isinstance(records, list):
        raise ValueError(
            "Step 1 tables_index.json must contain "
            "a tables list."
        )

    if any(
        not isinstance(record, dict)
        for record in records
    ):
        raise ValueError(
            "Step 1 table-index entries must be objects."
        )

    positions: dict[int, int] = {}

    for position, record in enumerate(records):
        table_id = record.get("table_id")

        if not isinstance(table_id, int):
            raise ValueError(
                "Step 1 table-index record is missing "
                "an integer table_id at position "
                f"{position}."
            )

        if table_id in positions:
            raise ValueError(
                f"Duplicate Step 1 table_id: {table_id}"
            )

        positions[table_id] = position

    links = continuation_links_from_records(
        records
    )

    # Validate symbolic topology before using it to merge data.
    for child_id, link in links.items():
        parent_id = link.parent_table_id
        root_id = link.root_table_id

        if parent_id not in positions:
            raise ValueError(
                "Step 1 continuation table "
                f"{child_id} references missing parent "
                f"table {parent_id}."
            )

        if root_id not in positions:
            raise ValueError(
                "Step 1 continuation table "
                f"{child_id} references missing root "
                f"table {root_id}."
            )

        if positions[parent_id] >= positions[child_id]:
            raise ValueError(
                "Step 1 continuation parent must "
                "precede child table "
                f"{child_id}."
            )

        if positions[root_id] > positions[parent_id]:
            raise ValueError(
                "Step 1 continuation root must not "
                "occur after parent table "
                f"{parent_id}."
            )

        if parent_id in links:
            parent_root = (
                links[parent_id].root_table_id
            )

            if parent_root != root_id:
                raise ValueError(
                    "Inconsistent Step 1 continuation "
                    f"roots: table {child_id} has root "
                    f"{root_id}, but parent {parent_id} "
                    f"has root {parent_root}."
                )

    return records, links, positions


def _group_continuation_tables(
    records: list[dict[str, Any]],
    links: dict[int, Any],
) -> list[tuple[int, tuple[int, ...]]]:
    """Return logical groups in physical-document order."""

    record_order = [
        record["table_id"]
        for record in records
    ]

    members_by_root: dict[int, set[int]] = {}

    for child_id, link in links.items():
        root_id = link.root_table_id

        members = members_by_root.setdefault(
            root_id,
            {root_id},
        )

        members.add(child_id)
        members.add(link.parent_table_id)

    groups: list[
        tuple[int, tuple[int, ...]]
    ] = []

    for root_id, members in (
        members_by_root.items()
    ):
        ordered = tuple(
            table_id
            for table_id in record_order
            if table_id in members
        )

        if (
            len(ordered) >= 2
            and ordered[0] == root_id
        ):
            groups.append(
                (root_id, ordered)
            )

    order = {
        table_id: position
        for position, table_id
        in enumerate(record_order)
    }

    groups.sort(
        key=lambda item: order[item[0]]
    )

    return groups


def _plan_one_group(
    root_table_id: int,
    physical_table_ids: tuple[int, ...],
    plan_by_table_id: dict[
        int,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    exported_ids = tuple(
        table_id
        for table_id in physical_table_ids
        if table_id in plan_by_table_id
    )

    missing_ids = tuple(
        table_id
        for table_id in physical_table_ids
        if table_id not in plan_by_table_id
    )

    base = {
        "root_table_id": root_table_id,
        "physical_table_ids": list(
            physical_table_ids
        ),
        "exported_table_ids": list(
            exported_ids
        ),
        "missing_table_ids": list(
            missing_ids
        ),
        "merged_csv": None,
        "dropped_repeated_header_table_ids": [],
        "alignment": {},
    }

    if not exported_ids:
        return {
            **base,
            "merge_status": "not_applicable",
            "reason": (
                "No physical table in this "
                "continuation group was exported."
            ),
        }

    # Do not merge only some fragments of a known logical chain.
    if missing_ids:
        return {
            **base,
            "merge_status": "incomplete",
            "reason": (
                "The known continuation group is "
                "incomplete in the Step 7 physical "
                "exports; subset merging is not "
                "permitted."
            ),
        }

    widths: dict[int, int] = {}

    for table_id in physical_table_ids:
        plan = plan_by_table_id[table_id]

        width, error = _rectangular_width(
            plan["rows"],
            table_id,
        )

        if error is not None:
            return {
                **base,
                "merge_status": "incompatible",
                "reason": error,
            }

        assert width is not None
        widths[table_id] = width

    if len(set(widths.values())) != 1:
        description = ", ".join(
            f"{table_id}:{widths[table_id]}"
            for table_id
            in physical_table_ids
        )

        return {
            **base,
            "merge_status": "incompatible",
            "reason": (
                "Continuation fragments have "
                "different scientific-column counts "
                f"({description})."
            ),
        }

    root_plan = plan_by_table_id[
        root_table_id
    ]

    root_header_index = root_plan[
        "header_row_index"
    ]

    if (
        root_header_index is not None
        and root_header_index != 0
    ):
        return {
            **base,
            "merge_status": "incompatible",
            "reason": (
                "Root table has a recognized header "
                "that is not the first physical row; "
                "automatic merging is ambiguous."
            ),
        }

    root_header: tuple[str, ...] | None = (
        None
    )

    if root_header_index == 0:
        root_header = _normalize_header_row(
            root_plan["rows"][0]
        )

    drop_headers: list[int] = []
    alignment: dict[str, str] = {
        str(root_table_id): "root"
    }

    for table_id in physical_table_ids[1:]:
        plan = plan_by_table_id[table_id]

        header_index = plan[
            "header_row_index"
        ]

        if (
            header_index is not None
            and header_index != 0
        ):
            return {
                **base,
                "merge_status": "incompatible",
                "reason": (
                    "Continuation table "
                    f"{table_id} has a recognized "
                    "header that is not its first "
                    "physical row."
                ),
            }

        first_row = (
            _normalize_header_row(
                plan["rows"][0]
            )
            if plan["rows"]
            else None
        )

        if header_index == 0:
            if root_header is None:
                return {
                    **base,
                    "merge_status": "incompatible",
                    "reason": (
                        "Continuation table "
                        f"{table_id} has a recognized "
                        "header but the root table "
                        "does not."
                    ),
                }

            if first_row != root_header:
                return {
                    **base,
                    "merge_status": "incompatible",
                    "reason": (
                        "Recognized repeated header "
                        "in continuation table "
                        f"{table_id} conflicts with "
                        "the root-table header."
                    ),
                }

            drop_headers.append(table_id)

            alignment[str(table_id)] = (
                "exact_repeated_header"
            )

            continue

        # Exact scientific-row equality is also sufficient
        # evidence for a repeated physical header.
        if (
            root_header is not None
            and first_row == root_header
        ):
            drop_headers.append(table_id)

            alignment[str(table_id)] = (
                "exact_repeated_header"
            )

            continue

        # Explicit continuation + equal rectangular width
        # defines a unique positional column mapping.
        alignment[str(table_id)] = (
            "positional_same_width"
        )

    return {
        **base,
        "merge_status": "ready",
        "reason": (
            "Explicit Step 1 continuation topology "
            "and deterministic same-width column "
            "alignment."
        ),
        "dropped_repeated_header_table_ids": (
            drop_headers
        ),
        "alignment": alignment,
    }


def plan_continuation_merges(
    tables_index_path: Path,
    plans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records, links, _positions = (
        _load_continuation_records(
            tables_index_path
        )
    )

    plan_by_table_id = {
        plan["table_id"]: plan
        for plan in plans
    }

    groups = _group_continuation_tables(
        records,
        links,
    )

    return [
        _plan_one_group(
            root_table_id,
            physical_table_ids,
            plan_by_table_id,
        )
        for (
            root_table_id,
            physical_table_ids,
        ) in groups
        if any(
            table_id in plan_by_table_id
            for table_id
            in physical_table_ids
        )
    ]


def materialize_continuation_merges(
    merge_plans: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    table_results: tuple[
        dict[str, Any],
        ...
    ],
    output_dir: Path,
) -> tuple[dict[str, Any], ...]:
    """
    Materialize only groups already validated as safe.

    Physical resolved CSV files are retained unchanged.
    """

    plan_by_table_id = {
        plan["table_id"]: plan
        for plan in plans
    }

    result_by_table_id = {
        item["table_id"]: item
        for item in table_results
    }

    final_groups: list[
        dict[str, Any]
    ] = []

    for merge_plan in merge_plans:
        if (
            merge_plan["merge_status"]
            != "ready"
        ):
            final_groups.append(
                dict(merge_plan)
            )
            continue

        root_table_id = merge_plan[
            "root_table_id"
        ]

        physical_table_ids = tuple(
            merge_plan[
                "physical_table_ids"
            ]
        )

        drop_headers = set(
            merge_plan[
                "dropped_repeated_header_table_ids"
            ]
        )

        merged_dir = (
            Path(output_dir)
            / MERGED_TABLES_DIR_NAME
        )

        merged_path = (
            merged_dir
            / (
                f"table_{root_table_id:03d}"
                "_merged_resolved.csv"
            )
        )

        merged_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rows_written = 0

        rows_by_physical_table: dict[
            str,
            int,
        ] = {}

        with merged_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.writer(handle)

            for table_id in (
                physical_table_ids
            ):
                physical_result = (
                    result_by_table_id[
                        table_id
                    ]
                )

                physical_path = Path(
                    physical_result[
                        "resolved_csv"
                    ]
                )

                resolved_rows = _read_csv(
                    physical_path
                )

                source_plan = (
                    plan_by_table_id[
                        table_id
                    ]
                )

                if len(resolved_rows) != len(
                    source_plan["rows"]
                ):
                    raise ValueError(
                        "Physical resolved CSV "
                        "row count changed before "
                        "continuation merge for "
                        f"table {table_id}."
                    )

                start_index = (
                    1
                    if table_id
                    in drop_headers
                    else 0
                )

                table_rows_written = 0

                for row_index in range(
                    start_index,
                    len(resolved_rows),
                ):
                    row = list(
                        resolved_rows[
                            row_index
                        ]
                    )

                    if (
                        table_id
                        == root_table_id
                        and row_index
                        == source_plan[
                            "header_row_index"
                        ]
                    ):
                        origin_value = (
                            MERGED_ORIGIN_COLUMN
                        )
                    else:
                        origin_value = str(
                            table_id
                        )

                    writer.writerow(
                        row
                        + [origin_value]
                    )

                    rows_written += 1
                    table_rows_written += 1

                rows_by_physical_table[
                    str(table_id)
                ] = table_rows_written

        final_groups.append(
            {
                **merge_plan,
                "merge_status": "merged",
                "reason": (
                    "Merged after deterministic "
                    "Step 1 continuation and "
                    "column-compatibility validation."
                ),
                "merged_csv": str(
                    merged_path
                ),
                "rows": rows_written,
                "rows_by_physical_table": (
                    rows_by_physical_table
                ),
                "origin_column": (
                    MERGED_ORIGIN_COLUMN
                ),
            }
        )

    return tuple(final_groups)
