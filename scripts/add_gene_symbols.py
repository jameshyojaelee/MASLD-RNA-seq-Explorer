#!/usr/bin/env python3
"""Add gene_symbol column to bundled DEG tables."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "streamlit_deg_explorer" / "data"


def strip_version(gene_id: str) -> str:
    return gene_id.split(".")[0] if isinstance(gene_id, str) else gene_id


def load_human_symbol_map(paths: list[Path]) -> dict[str, str]:
    symbol_map: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        gene_col = cols.get("ensembl_gene_id")
        gene_ver_col = cols.get("ensembl_gene_id_versioned")
        symbol_col = cols.get("external_gene_name")
        if gene_col is None or symbol_col is None:
            continue
        for row in df.itertuples(index=False):
            gene_id = getattr(row, gene_col)
            symbol = getattr(row, symbol_col)
            if isinstance(gene_id, str) and isinstance(symbol, str) and symbol:
                symbol_map.setdefault(strip_version(gene_id), symbol)
            if gene_ver_col is not None:
                gene_ver = getattr(row, gene_ver_col)
                if isinstance(gene_ver, str) and isinstance(symbol, str) and symbol:
                    symbol_map.setdefault(strip_version(gene_ver), symbol)
    return symbol_map


def load_mouse_symbol_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    symbol_map: dict[str, str] = {}
    with path.open() as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene_id, gene_name = parts[0], parts[1]
            if gene_id and gene_name:
                symbol_map.setdefault(strip_version(gene_id), gene_name)
    return symbol_map


def add_symbols(path: Path, human_map: dict[str, str], mouse_map: dict[str, str]) -> bool:
    sep = "\t" if path.name.endswith(".tsv.gz") else ","
    df = pd.read_csv(path, sep=sep)
    if "gene_id" not in df.columns:
        return False
    gene_ids = df["gene_id"].astype(str)
    symbols = []
    for gid in gene_ids:
        gid_norm = strip_version(gid)
        symbol = mouse_map.get(gid_norm)
        if symbol is None:
            symbol = human_map.get(gid_norm, "")
        symbols.append(symbol)
    if "gene_symbol" in df.columns:
        df["gene_symbol"] = symbols
    else:
        insert_at = df.columns.get_loc("gene_id") + 1
        df.insert(insert_at, "gene_symbol", symbols)
    df.to_csv(path, sep=sep, index=False, compression="gzip")
    return True


def main() -> None:
    human_map = load_human_symbol_map(
        [
            REPO_ROOT
            / "RNA-seq"
            / "patient_RNAseq"
            / "results"
            / "GSE130970"
            / "edgeR_results"
            / "edgeR_optimized_20250622_073427"
            / "results"
            / "gene_annotations.csv",
            REPO_ROOT
            / "RNA-seq"
            / "patient_RNAseq"
            / "results"
            / "GSE135251"
            / "edgeR_results"
            / "edgeR_with_gene_names_20250621_171101"
            / "results"
            / "gene_annotations.csv",
        ]
    )
    mouse_map = load_mouse_symbol_map(
        REPO_ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "reference" / "indices" / "star" / "geneInfo.tab"
    )

    updated = 0
    for path in DATA_DIR.glob("*.gz"):
        if add_symbols(path, human_map, mouse_map):
            updated += 1

    print(f"Added gene_symbol to {updated} bundled DEG tables.")


if __name__ == "__main__":
    main()
