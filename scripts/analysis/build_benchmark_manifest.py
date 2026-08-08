"""Create a checksum/BKS manifest for locally obtained Dang and Chao files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


DOI = {"Dang": "10.1016/j.ejor.2013.02.049", "Chao": "10.1016/0377-2217(94)00289-4"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--results", type=Path, default=Path("data/results/normalized_results.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark_manifest_check.csv"))
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    bks = {(row["Benchmark"], row["Instance"]): row["BKS"] for row in payload["benchmark_bks"]}
    rows_by_key = {}
    for path in sorted(args.benchmark_root.rglob("*.txt"), key=lambda item: str(item).lower()):
        instance = path.stem
        benchmark = "Chao" if instance.lower().startswith(("p4.", "p5.", "p6.", "p7.")) else "Dang"
        key = (benchmark, instance)
        if key not in bks:
            continue
        row = {
            "Benchmark": benchmark,
            "Set": path.parent.name if benchmark == "Chao" else "Dang benchmark",
            "Instance": instance,
            "BKS": bks[key],
            "Bytes": path.stat().st_size,
            "SHA256": sha256(path),
            "Source_DOI": DOI[benchmark],
            "Expected_Relative_Path": (
                f"{path.parent.name}/{path.name}" if benchmark == "Chao" else path.name
            ),
        }
        if key in rows_by_key and rows_by_key[key]["SHA256"] != row["SHA256"]:
            raise ValueError(f"Conflicting duplicate benchmark file for {key}")
        rows_by_key[key] = row
    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    found = {(row["Benchmark"], row["Instance"]) for row in rows}
    if found != set(bks):
        raise ValueError(f"Benchmark mismatch; missing={sorted(set(bks) - found)}, extra={sorted(found - set(bks))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} benchmark records to {args.output}")


if __name__ == "__main__":
    main()
