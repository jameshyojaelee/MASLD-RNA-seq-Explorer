#!/usr/bin/env python3
"""Rebuild bundled DEG tables from local STAR/featureCounts DESeq2 outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

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

PATIENT_DATASETS = ("GSE130970", "GSE135251")
REQUIRED_PATIENT_REL = [
    Path("nas_high/deseq2_results/NAS_4plus_vs_NAS_0/differential_expression.csv"),
    Path("nas_low/deseq2_results/NAS_1to3_vs_NAS_0/differential_expression.csv"),
    Path("fibrosis_strict/deseq2_results/Fibrosis_vs_Healthy/differential_expression.csv"),
]


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


def write_mouse_table(src: Path, dest: Path) -> None:
    df = pd.read_csv(src, sep="\t")
    gene_col = "gene" if "gene" in df.columns else "gene_id"
    symbol_col = "gene_name" if "gene_name" in df.columns else "gene_symbol"
    symbols = df[symbol_col].astype(str) if symbol_col in df.columns else ""
    out = pd.DataFrame(
        {
            "gene_id": df[gene_col].astype(str),
            "gene_symbol": symbols,
            "log2FoldChange": pd.to_numeric(df["log2FoldChange"], errors="coerce"),
            "padj": pd.to_numeric(df["padj"], errors="coerce"),
        }
    )
    out.to_csv(dest, sep="\t", index=False, compression="gzip")


def write_patient_table(src: Path, dest: Path) -> None:
    df = pd.read_csv(src)
    required = ["gene_id", "gene_symbol", "log2FoldChange", "padj"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {src}: {missing}")
    df[required].to_csv(dest, index=False, compression="gzip")


def main() -> None:
    for out_name, src in INHOUSE_SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing in-house source: {src}")
        write_mouse_table(src, DATA_DIR / out_name)

    for out_name, src in EXTERNAL_SOURCES.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing external source: {src}")
        write_mouse_table(src, DATA_DIR / out_name)

    for dataset in PATIENT_DATASETS:
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
        sources = {
            "nas_high": run_dir / REQUIRED_PATIENT_REL[0],
            "nas_low": run_dir / REQUIRED_PATIENT_REL[1],
            "fibrosis": run_dir / REQUIRED_PATIENT_REL[2],
        }
        for key, src in sources.items():
            dest = DATA_DIR / f"{dataset.lower()}_{key}.csv.gz"
            write_patient_table(src, dest)

    print("Bundled DEG tables refreshed.")


if __name__ == "__main__":
    main()
