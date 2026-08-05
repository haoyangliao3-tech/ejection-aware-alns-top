"""Discovery helpers for locally downloaded TOP benchmark instances."""

from __future__ import annotations

from pathlib import Path


def discover_instances(
    benchmark_root: str | Path,
    dataset: str = "all",
    pattern: str = "*.txt",
) -> list[Path]:
    """Return Chao or Dang instance files in deterministic path order."""
    root = Path(benchmark_root)
    directories: list[Path] = []
    if dataset in {"all", "chao"}:
        directories.append(root / "Chao et al., (1996)")
    if dataset in {"all", "dang"}:
        directories.append(root / "Dang et al., (2013)")
    return sorted(
        (
            path
            for directory in directories
            if directory.exists()
            for path in directory.rglob(pattern)
            if path.is_file()
        ),
        key=lambda path: str(path).lower(),
    )
