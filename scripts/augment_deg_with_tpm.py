#!/usr/bin/env python3
"""Add mean TPM (disease-only samples) to bundled DEG tables.

Computes TPM from featureCounts gene counts using gene lengths.
For human cohorts, uses MASLD-only samples; for mouse, uses MCD-only samples.
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


def _normalize_sample_ids(sample_ids: list[str] | None) -> list[str]:
    if not sample_ids:
        return []
    cleaned = [str(s).strip() for s in sample_ids if str(s).strip()]
    return sorted(set(cleaned))


def _featurecounts_sample_id(col: str) -> str:
    path = Path(str(col))
    name = path.name
    if ".Aligned" in name:
        return name.split(".Aligned", 1)[0]
    parts = path.parts
    if len(parts) >= 2:
        return parts[-2]
    return name.split(".", 1)[0]


def _subset_featurecounts_counts(counts: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    sample_ids = _normalize_sample_ids(sample_ids)
    if not sample_ids:
        return counts
    wanted = set(sample_ids)
    selected = [col for col in counts.columns if _featurecounts_sample_id(col) in wanted]
    if not selected:
        return counts
    return counts[selected]


def _subset_counts_matrix(counts: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    sample_ids = _normalize_sample_ids(sample_ids)
    if not sample_ids:
        return counts
    wanted = set(sample_ids)
    selected = [col for col in counts.columns if col in wanted]
    if not selected:
        return counts
    return counts[selected]


def load_mcd_sample_ids(metadata_path: Path) -> list[str]:
    if not metadata_path.exists():
        return []
    df = pd.read_csv(metadata_path, sep="\t")
    if "sample_id" not in df.columns:
        return []
    if "diet" not in df.columns:
        return df["sample_id"].astype(str).tolist()
    mask = df["diet"].astype(str).str.contains("mcd", case=False, na=False)
    if not mask.any():
        return df["sample_id"].astype(str).tolist()
    return df.loc[mask, "sample_id"].astype(str).tolist()


def load_masld_sample_ids(dataset: str) -> list[str]:
    samplesheet = (
        REPO_ROOT
        / "RNA-seq"
        / "patient_RNAseq"
        / "data"
        / "samplesheets"
        / f"{dataset}_samplesheet.csv"
    )
    if not samplesheet.exists():
        return []
    df = pd.read_csv(samplesheet)
    if "sample" not in df.columns:
        return []
    if dataset == "GSE135251":
        if "disease" in df.columns:
            mask = df["disease"].astype(str).str.lower().ne("control")
        elif "group_in_paper" in df.columns:
            mask = df["group_in_paper"].astype(str).str.lower().ne("control")
        else:
            mask = pd.Series(True, index=df.index)
    else:
        if "nafld_activity_score" in df.columns:
            score = pd.to_numeric(df["nafld_activity_score"], errors="coerce")
            mask = score > 0
            if not mask.any():
                mask = pd.Series(True, index=df.index)
        elif "steatosis_grade" in df.columns:
            score = pd.to_numeric(df["steatosis_grade"], errors="coerce")
            mask = score > 0
        else:
            mask = pd.Series(True, index=df.index)
    return df.loc[mask, "sample"].astype(str).tolist()


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
    inhouse_samples = load_mcd_sample_ids(
        REPO_ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "metadata" / "samples.tsv"
    )
    _, inhouse_lengths, inhouse_counts_df = read_featurecounts(inhouse_counts)
    inhouse_counts_df = _subset_featurecounts_counts(inhouse_counts_df, inhouse_samples)
    inhouse_tpm = compute_tpm_mean(inhouse_counts_df, inhouse_lengths)

    # External MCD TPM (mouse)
    gse156918_counts = REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "counts" / "featurecounts" / "gene_counts.txt"
    gse156918_samples = load_mcd_sample_ids(
        REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "metadata" / "samples.tsv"
    )
    _, gse156918_lengths, gse156918_counts_df = read_featurecounts(gse156918_counts)
    gse156918_counts_df = _subset_featurecounts_counts(gse156918_counts_df, gse156918_samples)
    gse156918_tpm = compute_tpm_mean(gse156918_counts_df, gse156918_lengths)

    gse205974_counts = REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "counts" / "featurecounts" / "gene_counts.txt"
    gse205974_samples = load_mcd_sample_ids(
        REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "metadata" / "samples.tsv"
    )
    _, gse205974_lengths, gse205974_counts_df = read_featurecounts(gse205974_counts)
    gse205974_counts_df = _subset_featurecounts_counts(gse205974_counts_df, gse205974_samples)
    gse205974_tpm = compute_tpm_mean(gse205974_counts_df, gse205974_lengths)

    # Patient GSE TPM (human)
    gse130970_counts = REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE130970" / "counts" / "gene_counts_matrix.txt"
    gse130970_counts_df = read_gene_counts_matrix(gse130970_counts)
    gse130970_samples = load_masld_sample_ids("GSE130970")
    gse130970_counts_df = _subset_counts_matrix(gse130970_counts_df, gse130970_samples)
    gse130970_length_source = next((REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE130970" / "counts" / "individual").glob("*_counts.txt"))
    gse130970_lengths = read_gene_lengths_from_counts(gse130970_length_source)
    gse130970_tpm = compute_tpm_mean(gse130970_counts_df, gse130970_lengths)

    gse135251_counts = REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE135251" / "counts" / "final_count_matrix_all_216_samples.txt"
    if not gse135251_counts.exists():
        gse135251_counts = REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE135251" / "counts" / "gene_counts_matrix.txt"
    gse135251_counts_df = read_gene_counts_matrix(gse135251_counts)
    gse135251_samples = load_masld_sample_ids("GSE135251")
    gse135251_counts_df = _subset_counts_matrix(gse135251_counts_df, gse135251_samples)
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
