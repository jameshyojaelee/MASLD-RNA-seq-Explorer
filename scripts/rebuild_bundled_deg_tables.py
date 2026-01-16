#!/usr/bin/env python3
"""Rebuild bundled DEG tables from local STAR/featureCounts DESeq2 outputs.

This script pre-computes TPM values and embeds them in the bundled files so that
the Streamlit app can filter by TPM even when running in bundled mode (e.g., Streamlit Cloud).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "streamlit_deg_explorer" / "data"

INHOUSE_SOURCES = {
    "mcd_week1.tsv.gz": REPO_ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v5_week1MCD_vs_controls"
    / "deseq2_results.tsv",
    "mcd_week2.tsv.gz": REPO_ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v5_week2MCD_vs_controls"
    / "deseq2_results.tsv",
    "mcd_week3.tsv.gz": REPO_ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v5_week3MCD_vs_controls"
    / "deseq2_results.tsv",
    "mcd_week_pooled_combined.tsv.gz": REPO_ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v1_weekPooled_combined"
    / "deseq2_results.tsv",
}

EXTERNAL_SOURCES = {
    "other_mcd_gse156918.tsv.gz": REPO_ROOT
    / "RNA-seq"
    / "other_MCD_RNAseq"
    / "GSE156918"
    / "analysis_mcd_vs_control"
    / "deseq2_mcd_vs_control.tsv",
    "other_mcd_gse205974.tsv.gz": REPO_ROOT
    / "RNA-seq"
    / "other_MCD_RNAseq"
    / "GSE205974"
    / "analysis_mcd_vs_control"
    / "deseq2_mcd_vs_control.tsv",
}

# featureCounts paths for mouse datasets
INHOUSE_FEATURECOUNTS = REPO_ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "counts" / "featurecounts" / "gene_counts.txt"
EXTERNAL_FEATURECOUNTS = {
    "other_mcd_gse156918.tsv.gz": REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "counts" / "featurecounts" / "gene_counts.txt",
    "other_mcd_gse205974.tsv.gz": REPO_ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "counts" / "featurecounts" / "gene_counts.txt",
}

PATIENT_DATASETS = ("GSE130970", "GSE135251")
REQUIRED_PATIENT_REL = [
    Path("nas_high/deseq2_results/NAS_4plus_vs_NAS_0/differential_expression.csv"),
    Path("nas_low/deseq2_results/NAS_1to3_vs_NAS_0/differential_expression.csv"),
    Path("fibrosis_strict/deseq2_results/Fibrosis_vs_Healthy/differential_expression.csv"),
]


def strip_version(gene_id: str) -> str:
    """Strip version suffix from Ensembl gene ID."""
    return gene_id.split(".")[0] if isinstance(gene_id, str) else gene_id


def read_featurecounts(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Read featureCounts output and return (lengths, counts) DataFrames."""
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
    return lengths, counts


def compute_tpm_mean(counts: pd.DataFrame, lengths: pd.Series) -> pd.Series:
    """Compute mean TPM across all samples."""
    lengths = lengths.dropna()
    lengths = lengths[lengths > 0]
    shared = counts.index.intersection(lengths.index)
    counts = counts.loc[shared]
    lengths = lengths.loc[shared]
    length_kb = lengths / 1000.0
    rpk = counts.div(length_kb, axis=0)
    scale = rpk.sum(axis=0) / 1e6
    tpm = rpk.div(scale, axis=1)
    return tpm.mean(axis=1, skipna=True)


def load_tpm_map_from_featurecounts(path: Path) -> dict[str, float]:
    """Load TPM map from featureCounts file."""
    if not path.exists():
        print(f"  Warning: featureCounts file not found: {path}")
        return {}
    lengths, counts = read_featurecounts(path)
    tpm = compute_tpm_mean(counts, lengths)
    return {strip_version(k): v for k, v in tpm.to_dict().items()}


def load_tpm_map_from_counts_matrix(counts_path: Path, length_source: Path) -> dict[str, float]:
    """Load TPM map from count matrix + length source."""
    if not counts_path.exists() or not length_source.exists():
        print(f"  Warning: counts or length file not found: {counts_path}, {length_source}")
        return {}
    counts_df = pd.read_csv(counts_path, sep="\t")
    gene_col = counts_df.columns[0]
    counts = counts_df.set_index(gene_col)
    counts = counts.apply(pd.to_numeric, errors="coerce")

    lengths_df = pd.read_csv(length_source, sep="\t", comment="#")
    cols = {c.lower(): c for c in lengths_df.columns}
    gene_col = cols.get("geneid") or cols.get("gene_id") or cols.get("gene")
    length_col = cols.get("length")
    if gene_col is None or length_col is None:
        return {}
    lengths = pd.to_numeric(lengths_df[length_col], errors="coerce")
    lengths.index = lengths_df[gene_col].astype(str)
    tpm = compute_tpm_mean(counts, lengths)
    return {strip_version(k): v for k, v in tpm.to_dict().items()}


def latest_run_with_files(base_dir: Path, required: list[Path]) -> Path | None:
    if not base_dir.exists():
        return None
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name[:1].isdigit()]
    if not run_dirs:
        return None
    for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
        if all((run_dir / rel).exists() for rel in required):
            return run_dir
    return None


def write_mouse_table(src: Path, dest: Path, tpm_map: dict[str, float]) -> None:
    df = pd.read_csv(src, sep="\t")
    gene_col = "gene" if "gene" in df.columns else "gene_id"
    symbol_col = "gene_name" if "gene_name" in df.columns else "gene_symbol"
    symbols = df[symbol_col].astype(str) if symbol_col in df.columns else ""
    gene_ids = df[gene_col].astype(str)
    tpm_values = gene_ids.map(lambda x: tpm_map.get(strip_version(x), np.nan))
    out = pd.DataFrame(
        {
            "gene_id": gene_ids,
            "gene_symbol": symbols,
            "log2FoldChange": pd.to_numeric(df["log2FoldChange"], errors="coerce"),
            "padj": pd.to_numeric(df["padj"], errors="coerce"),
            "tpm_mean": tpm_values,
        }
    )
    out.to_csv(dest, sep="\t", index=False, compression="gzip")
    tpm_count = out["tpm_mean"].notna().sum()
    print(f"  Wrote {dest.name} ({len(out)} genes, {tpm_count} with TPM)")


def write_patient_table(src: Path, dest: Path, tpm_map: dict[str, float]) -> None:
    df = pd.read_csv(src)
    required = ["gene_id", "gene_symbol", "log2FoldChange", "padj"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {src}: {missing}")
    gene_ids = df["gene_id"].astype(str)
    tpm_values = gene_ids.map(lambda x: tpm_map.get(strip_version(x), np.nan))
    out = df[required].copy()
    out["tpm_mean"] = tpm_values
    out.to_csv(dest, index=False, compression="gzip")
    tpm_count = out["tpm_mean"].notna().sum()
    print(f"  Wrote {dest.name} ({len(out)} genes, {tpm_count} with TPM)")


def main() -> None:
    print("Loading in-house MCD TPM map...")
    inhouse_tpm_map = load_tpm_map_from_featurecounts(INHOUSE_FEATURECOUNTS)
    print(f"  Loaded {len(inhouse_tpm_map)} genes")

    print("\nProcessing in-house MCD datasets...")
    for out_name, src in INHOUSE_SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing in-house source: {src}")
        write_mouse_table(src, DATA_DIR / out_name, inhouse_tpm_map)

    print("\nProcessing external MCD datasets...")
    for out_name, src in EXTERNAL_SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing external source: {src}")
        fc_path = EXTERNAL_FEATURECOUNTS.get(out_name)
        if fc_path:
            tpm_map = load_tpm_map_from_featurecounts(fc_path)
        else:
            tpm_map = {}
        write_mouse_table(src, DATA_DIR / out_name, tpm_map)

    print("\nProcessing patient datasets...")
    for dataset in PATIENT_DATASETS:
        print(f"\n  {dataset}:")
        base_dir = (
            REPO_ROOT
            / "RNA-seq"
            / "patient_RNAseq"
            / "results"
            / dataset
            / "deseq2_results"
        )
        run_dir = latest_run_with_files(base_dir, REQUIRED_PATIENT_REL)
        if run_dir is None:
            raise FileNotFoundError(f"No complete DESeq2 run found for {dataset}")
        
        # Load TPM map for patient dataset
        counts_dir = REPO_ROOT / "RNA-seq" / "patient_RNAseq" / "results" / dataset / "counts"
        counts_matrix = counts_dir / "gene_counts_matrix.txt"
        length_source = next(counts_dir.glob("individual/*_counts.txt"), None)
        if length_source:
            print(f"    Loading TPM from {counts_matrix.name}...")
            tpm_map = load_tpm_map_from_counts_matrix(counts_matrix, length_source)
            print(f"    Loaded {len(tpm_map)} genes with TPM")
        else:
            print(f"    Warning: No length source found for {dataset}")
            tpm_map = {}
        
        sources = {
            "nas_high": run_dir / REQUIRED_PATIENT_REL[0],
            "nas_low": run_dir / REQUIRED_PATIENT_REL[1],
            "fibrosis": run_dir / REQUIRED_PATIENT_REL[2],
        }
        for key, src in sources.items():
            dest = DATA_DIR / f"{dataset.lower()}_{key}.csv.gz"
            write_patient_table(src, dest, tpm_map)

    print("\n✓ Bundled DEG tables refreshed with TPM values.")


if __name__ == "__main__":
    main()

