"""Export every machine-readable table from normalized_results.json to CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLES = (
    "benchmark_bks",
    "fixed_iteration",
    "symmetric_wall_clock",
    "runtime_budget_baselines",
    "runtime_budgets",
    "sensitivity",
    "focused_mechanism",
    "qa_summary",
    "reported_statistics",
    "component_statistics",
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot export empty table: {path.stem}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"Inconsistent columns in {path.stem}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/results/normalized_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/results/csv"))
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    for table in TABLES:
        rows = payload.get(table)
        if not isinstance(rows, list):
            raise ValueError(f"Missing result table: {table}")
        write_csv(args.output_dir / f"{table}.csv", rows)
        print(f"{table}: {len(rows)} rows")


if __name__ == "__main__":
    main()
