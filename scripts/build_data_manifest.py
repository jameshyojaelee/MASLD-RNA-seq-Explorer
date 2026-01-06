#!/usr/bin/env python3
"""Build a manifest of bundled DEG files (size, sha256, rows, cols)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_sep(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".tsv") or name.endswith(".tsv.gz"):
        return "\t"
    return ","


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(data_dir.glob("*")):
        if path.is_dir():
            continue
        if path.name == "manifest.tsv":
            continue
        sep = detect_sep(path)
        df = pd.read_csv(path, sep=sep)
        rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256sum(path),
                "rows": len(df),
                "cols": len(df.columns),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(data_dir / "manifest.tsv", sep="\t", index=False)
    print(f"Wrote {data_dir / 'manifest.tsv'} with {len(rows)} files")


if __name__ == "__main__":
    main()
