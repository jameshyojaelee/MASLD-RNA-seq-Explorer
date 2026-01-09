#!/usr/bin/env python3
"""Add mean TPM (across all samples) to bundled DEG tables.

Computes TPM from featureCounts gene counts using gene lengths.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "streamlit_deg_explorer" / "data"


def strip_version(gene_id: str) -> str:
    return gene_id.split(".")[0] if isinstance(gene_id, str) else gene_id


def read_featurecounts(path: Path) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    df = pd.read_csv(path, sep="\t", comment="#")
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("geneid") or cols.get("gene_id") or cols.get("gene")
    length_col = cols.get("length")
    if gene_col is None or length_col is None:
        raise ValueError(f"Missing Geneid/Length columns in {path}")
    meta_cols = {gene_col, length_col}
    for key in ("chr", "start", "end", "strand"):
        col = cols.get(key)
        if col is not None:
            meta_cols.add(col)
    sample_cols = [c for c in df.columns if c not in meta_cols]
    gene_ids = df[gene_col].astype(str)
    lengths = pd.to_numeric(df[length_col], errors="coerce")
    counts = df[sample_cols].apply(pd.to_numeric, errors="coerce")
    counts.index = gene_ids
    lengths.index = gene_ids
    return gene_ids, lengths, counts


def read_gene_counts_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    gene_col = df.columns[0]
    df[gene_col] = df[gene_col].astype(str)
    counts = df.set_index(gene_col)
    counts = counts.apply(pd.to_numeric, errors="coerce")
    return counts


def read_gene_lengths_from_counts(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#")
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("geneid") or cols.get("gene_id") or cols.get("gene")
    length_col = cols.get("length")
    if gene_col is None or length_col is None:
        raise ValueError(f"Missing Geneid/Length columns in {path}")
    lengths = pd.to_numeric(df[length_col], errors="coerce")
    lengths.index = df[gene_col].astype(str)
    return lengths


def compute_tpm_mean(counts: pd.DataFrame, lengths: pd.Series) -> pd.Series:
    lengths = lengths.dropna()
    lengths = lengths[lengths > 0]
    shared = counts.index.intersection(lengths.index)
    counts = counts.loc[shared]
    lengths = lengths.loc[shared]
    length_kb = lengths / 1000.0
    rpk = counts.div(length_kb, axis=0)
    scale = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scale, axis=1)
    tpm_mean = tpm.mean(axis=1, skipna=True)
    return tpm_mean


def add_tpm_column(path: Path, tpm_map: dict[str, float]) -> None:
    tpm_map_stripped = {strip_version(k): v for k, v in tpm_map.items()}
    sep = "\t" if path.name.endswith(".tsv.gz") else ","
    df = pd.read_csv(path, sep=sep)
    if "gene_id" not in df.columns:
        raise ValueError(f"gene_id column missing in {path}")
    gene_ids = df["gene_id"].astype(str)
    tpm_series = gene_ids.map(tpm_map)
    missing = tpm_series.isna()
    if missing.any():
        tpm_series.loc[missing] = gene_ids.loc[missing].map(lambda x: tpm_map_stripped.get(strip_version(x)))
    if "tpm_mean" in df.columns:
        df["tpm_mean"] = tpm_series
    else:
        insert_at = df.columns.get_loc("padj") + 1 if "padj" in df.columns else len(df.columns)
        df.insert(insert_at, "tpm_mean", tpm_series)
    df.to_csv(path, sep=sep, index=False, compression="gzip")


def main() -> None:
    # In-house MCD TPM (mouse)
    inhouse_counts = REPO_ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "counts" / "featurecounts" / "gene_counts.txt"
    _, inhouse_lengths, inhouse_counts_df = read_featurecounts(inhouse_counts)
    inhouse_tpm = compute_tpm_mean(inhouse_counts_df, inhouse_lengths)

    # External MCD TPM (mouse)
    gse156918_counts = REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "counts" / "featurecounts" / "gene_counts.txt"
    _, gse156918_lengths, gse156918_counts_df = read_featurecounts(gse156918_counts)
    gse156918_tpm = compute_tpm_mean(gse156918_counts_df, gse156918_lengths)

    gse205974_counts = REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "counts" / "featurecounts" / "gene_counts.txt"
    _, gse205974_lengths, gse205974_counts_df = read_featurecounts(gse205974_counts)
    gse205974_tpm = compute_tpm_mean(gse205974_counts_df, gse205974_lengths)

    # Patient GSE TPM (human)
    gse130970_counts = REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE130970" / "counts" / "gene_counts_matrix.txt"
    gse130970_counts_df = read_gene_counts_matrix(gse130970_counts)
    gse130970_length_source = next((REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE130970" / "counts" / "individual").glob("*_counts.txt"))
    gse130970_lengths = read_gene_lengths_from_counts(gse130970_length_source)
    gse130970_tpm = compute_tpm_mean(gse130970_counts_df, gse130970_lengths)

    gse135251_counts = REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE135251" / "counts" / "gene_counts_matrix.txt"
    gse135251_counts_df = read_gene_counts_matrix(gse135251_counts)
    gse135251_length_source = next((REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE135251" / "counts" / "individual").glob("*_counts.txt"))
    gse135251_lengths = read_gene_lengths_from_counts(gse135251_length_source)
    gse135251_tpm = compute_tpm_mean(gse135251_counts_df, gse135251_lengths)

    # Apply to bundled DEG tables
    inhouse_files = [
        DATA_DIR / "mcd_week1.tsv.gz",
        DATA_DIR / "mcd_week2.tsv.gz",
        DATA_DIR / "mcd_week3.tsv.gz",
        DATA_DIR / "mcd_week_pooled_combined.tsv.gz",
    ]
    for path in inhouse_files:
        add_tpm_column(path, inhouse_tpm.to_dict())

    add_tpm_column(DATA_DIR / "other_mcd_gse156918.tsv.gz", gse156918_tpm.to_dict())
    add_tpm_column(DATA_DIR / "other_mcd_gse205974.tsv.gz", gse205974_tpm.to_dict())

    gse130970_files = [
        DATA_DIR / "gse130970_nas_high.csv.gz",
        DATA_DIR / "gse130970_nas_low.csv.gz",
        DATA_DIR / "gse130970_fibrosis.csv.gz",
    ]
    for path in gse130970_files:
        add_tpm_column(path, gse130970_tpm.to_dict())

    gse135251_files = [
        DATA_DIR / "gse135251_nas_high.csv.gz",
        DATA_DIR / "gse135251_nas_low.csv.gz",
        DATA_DIR / "gse135251_fibrosis.csv.gz",
    ]
    for path in gse135251_files:
        add_tpm_column(path, gse135251_tpm.to_dict())

    print("TPM columns added to bundled DEG tables.")


if __name__ == "__main__":
    main()
