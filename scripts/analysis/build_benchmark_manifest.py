"""Create a checksum/BKS manifest for a locally obtained Dang-82 dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark_dir", type=Path)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/results/normalized_results.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/benchmark_manifest.csv")
    )
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    bks = {row["Instance"]: row["BKS"] for row in payload["benchmark_bks"]}
    files = sorted(args.benchmark_dir.glob("*.txt"), key=lambda path: path.stem.lower())
    if {path.stem for path in files} != set(bks):
        missing = sorted(set(bks) - {path.stem for path in files})
        extra = sorted({path.stem for path in files} - set(bks))
        raise ValueError(f"Benchmark mismatch; missing={missing}, extra={extra}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("instance", "bks", "bytes", "sha256", "source_doi"),
        )
        writer.writeheader()
        for path in files:
            writer.writerow(
                {
                    "instance": path.stem,
                    "bks": bks[path.stem],
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "source_doi": "10.1016/j.ejor.2013.02.049",
                }
            )
    print(f"Wrote {len(files)} benchmark records to {args.output}")


if __name__ == "__main__":
    main()
