"""Streamlit app to count upregulated DEGs by adjustable cutoffs.
Usage: streamlit run streamlit_deg_explorer/app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable
import numpy as np
import math
import matplotlib.pyplot as plt

import pandas as pd
import streamlit as st
import seaborn as sns
from math import pi
from discordance import filter_master_matrix, plot_tug_of_war, plot_barcode_heatmap, plot_radar


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR
for parent in APP_DIR.parents:
    if (parent / "RNA-seq").exists():
        ROOT = parent
        break

if (APP_DIR / "data").exists():
    DATA_DIR = APP_DIR / "data"
else:
    DATA_DIR = None

INHOUSE_MCD_PATHS = {
    "MCD Week 1": ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v5_week1MCD_vs_controls"
    / "deseq2_results.tsv",
    "MCD Week 2": ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v5_week2MCD_vs_controls"
    / "deseq2_results.tsv",
    "MCD Week 3": ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v5_week3MCD_vs_controls"
    / "deseq2_results.tsv",
    "MCD Week pooled (combined)": ROOT
    / "RNA-seq"
    / "in-house_MCD_RNAseq"
    / "analysis_v1_weekPooled_combined"
    / "deseq2_results.tsv",
}

EXTERNAL_MCD_PATHS = {
    "GSE156918 (external MCD)": ROOT
    / "RNA-seq"
    / "other_MCD_RNAseq"
    / "GSE156918"
    / "analysis_mcd_vs_control"
    / "deseq2_mcd_vs_control.tsv",
    "GSE205974 (external MCD)": ROOT
    / "RNA-seq"
    / "other_MCD_RNAseq"
    / "GSE205974"
    / "analysis_mcd_vs_control"
    / "deseq2_mcd_vs_control.tsv",
}

PATIENT_DATASETS = ["GSE130970", "GSE135251"]
PATIENT_CROSS_DATASET_PAIR = ("GSE135251", "GSE130970")
PATIENT_CROSS_COMPARISONS = {
    "NAS high": "nas_high",
    "NAS low": "nas_low",
    "Fibrosis": "fibrosis",
}

INHOUSE_MCD_WEEK_LABELS = {"MCD Week 1", "MCD Week 2", "MCD Week 3"}
BUNDLED_INHOUSE_MCD_FILES = {
    "MCD Week 1": "mcd_week1.tsv.gz",
    "MCD Week 2": "mcd_week2.tsv.gz",
    "MCD Week 3": "mcd_week3.tsv.gz",
    "MCD Week pooled (combined)": "mcd_week_pooled_combined.tsv.gz",
}

BUNDLED_EXTERNAL_MCD_FILES = {
    "GSE156918 (external MCD)": "other_mcd_gse156918.tsv.gz",
    "GSE205974 (external MCD)": "other_mcd_gse205974.tsv.gz",
}

BUNDLED_PATIENT_FILES = {
    "GSE130970": {
        "nas_high": "gse130970_nas_high.csv.gz",
        "nas_low": "gse130970_nas_low.csv.gz",
        "fibrosis": "gse130970_fibrosis.csv.gz",
    },
    "GSE135251": {
        "nas_high": "gse135251_nas_high.csv.gz",
        "nas_low": "gse135251_nas_low.csv.gz",
        "fibrosis": "gse135251_fibrosis.csv.gz",
    },
}

ORTHOLOG_FILENAME = "mouse_human_orthologs.tsv.gz"
ORTHOLOG_ENV = os.environ.get("DEG_ORTHOLOG_MAP")
if ORTHOLOG_ENV:
    ORTHOLOG_PATH = Path(ORTHOLOG_ENV).expanduser().resolve()
elif DATA_DIR is not None and (DATA_DIR / ORTHOLOG_FILENAME).exists():
    ORTHOLOG_PATH = DATA_DIR / ORTHOLOG_FILENAME
else:
    ORTHOLOG_PATH = None

BIOTYPE_FILENAME = "ensembl_gene_biotypes.tsv.gz"
BIOTYPE_ENV = os.environ.get("DEG_BIOTYPE_MAP")
if BIOTYPE_ENV:
    BIOTYPE_PATH = Path(BIOTYPE_ENV).expanduser().resolve()
elif DATA_DIR is not None and (DATA_DIR / BIOTYPE_FILENAME).exists():
    BIOTYPE_PATH = DATA_DIR / BIOTYPE_FILENAME
else:
    BIOTYPE_PATH = None

GWAS_CLOSEST_GENES_FILENAME = "Closest_genes.csv"
GWAS_GENE_BIOTYPE_FILENAME = "gwas_gene_biotype_analysis.csv"

GWAS_CLOSEST_GENES_PATH = None
if DATA_DIR is not None and (DATA_DIR / GWAS_CLOSEST_GENES_FILENAME).exists():
    GWAS_CLOSEST_GENES_PATH = DATA_DIR / GWAS_CLOSEST_GENES_FILENAME
elif (ROOT / "GWAS" / GWAS_CLOSEST_GENES_FILENAME).exists():
    GWAS_CLOSEST_GENES_PATH = ROOT / "GWAS" / GWAS_CLOSEST_GENES_FILENAME

GWAS_GENE_BIOTYPE_PATH = None
if DATA_DIR is not None and (DATA_DIR / GWAS_GENE_BIOTYPE_FILENAME).exists():
    GWAS_GENE_BIOTYPE_PATH = DATA_DIR / GWAS_GENE_BIOTYPE_FILENAME
elif (ROOT / "GWAS" / GWAS_GENE_BIOTYPE_FILENAME).exists():
    GWAS_GENE_BIOTYPE_PATH = ROOT / "GWAS" / GWAS_GENE_BIOTYPE_FILENAME
GWAS_GENE_ANNOTATION_PATHS = [
    ROOT
    / "RNA-seq"
    / "patient_RNAseq"
    / "results"
    / "GSE130970"
    / "edgeR_results"
    / "edgeR_optimized_20250622_073427"
    / "results"
    / "gene_annotations.csv",
    ROOT
    / "RNA-seq"
    / "patient_RNAseq"
    / "results"
    / "GSE135251"
    / "edgeR_results"
    / "edgeR_with_gene_names_20250621_171101"
    / "results"
    / "gene_annotations.csv",
]


def _latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name[:1].isdigit()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda p: p.name)[-1]


def _latest_run_with_files(base_dir: Path, required: list[Path]) -> Path | None:
    if not base_dir.exists():
        return None
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name[:1].isdigit()]
    if not run_dirs:
        return None
    for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
        if all((run_dir / rel).exists() for rel in required):
            return run_dir
    return None


@st.cache_data(show_spinner=False)
def find_latest_run(base_dir: Path) -> Path | None:
    return _latest_run_dir(base_dir)


def _local_star_results_available() -> bool:
    if not all(path.exists() for path in INHOUSE_MCD_PATHS.values()):
        return False
    if not all(path.exists() for path in EXTERNAL_MCD_PATHS.values()):
        return False
    for dataset in PATIENT_DATASETS:
        base_dir = ROOT / "RNA-seq" / "patient_RNAseq" / "results" / dataset / "deseq2_results"
        required = [
            Path("nas_high/deseq2_results/NAS_4plus_vs_NAS_0/differential_expression.csv"),
            Path("nas_low/deseq2_results/NAS_1to3_vs_NAS_0/differential_expression.csv"),
            Path("fibrosis/deseq2_results/F1to4_vs_F0/differential_expression.csv"),
        ]
        latest = _latest_run_with_files(base_dir, required)
        if latest is None:
            return False
    return True


_force_bundled = os.environ.get("DEG_FORCE_BUNDLED", "").strip() == "1"
_force_local = os.environ.get("DEG_FORCE_LOCAL", "").strip() == "1"
if _force_local:
    USE_BUNDLED_RESULTS = False
else:
    USE_BUNDLED_RESULTS = DATA_DIR is not None and (
        _force_bundled or not _local_star_results_available()
    )


@st.cache_data(show_spinner=False)
def load_mcd_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return normalize_df(df)


@st.cache_data(show_spinner=False)
def load_patient_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return normalize_df(df)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("gene") or cols.get("gene_id")
    lfc_col = cols.get("log2foldchange")
    padj_col = cols.get("padj")
    tpm_col = cols.get("tpm_mean") or cols.get("tpm")
    if gene_col is None or lfc_col is None or padj_col is None:
        missing = [k for k, v in {"gene": gene_col, "log2FoldChange": lfc_col, "padj": padj_col}.items() if v is None]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    rename_map = {gene_col: "gene_id", lfc_col: "log2FoldChange", padj_col: "padj"}
    if tpm_col is not None:
        rename_map[tpm_col] = "tpm_mean"
    df = df.rename(columns=rename_map)
    keep_cols = ["gene_id", "log2FoldChange", "padj"]
    if "tpm_mean" in df.columns:
        keep_cols.append("tpm_mean")
    df = df[keep_cols]
    df = df.dropna(subset=["log2FoldChange", "padj", "gene_id"])
    return df


def strip_version(gene_id: str) -> str:
    return gene_id.split(".")[0] if isinstance(gene_id, str) else gene_id


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
def load_mouse_symbol_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    symbol_map: dict[str, str] = {}
    with path.open() as fh:
        first = fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            gene_id, gene_name = parts[0], parts[1]
            if gene_id and gene_name:
                symbol_map.setdefault(strip_version(gene_id), gene_name)
    return symbol_map


@st.cache_data(show_spinner=False)
def load_symbol_maps_from_bundled(data_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    human_map: dict[str, str] = {}
    mouse_map: dict[str, str] = {}
    for path in data_dir.glob("*.gz"):
        sep = "\t" if path.name.endswith(".tsv.gz") else ","
        try:
            df = pd.read_csv(path, sep=sep, usecols=["gene_id", "gene_symbol"])
        except Exception:
            continue
        for gid, symbol in zip(df["gene_id"].astype(str), df["gene_symbol"].astype(str)):
            if not symbol or symbol == "nan":
                continue
            gid_norm = strip_version(gid)
            if gid_norm.startswith("ENSG"):
                human_map.setdefault(gid_norm, symbol)
            elif gid_norm.startswith("ENSMUSG"):
                mouse_map.setdefault(gid_norm, symbol)
    return human_map, mouse_map


@st.cache_data(show_spinner=False)
def load_symbol_to_ensembl_map(paths: list[Path]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for path in paths:
        if path is None or not path.exists():
            continue
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        symbol_col = cols.get("external_gene_name") or cols.get("gene_symbol_clean") or cols.get("gene_symbol")
        ensembl_col = cols.get("ensembl_gene_id") or cols.get("ensembl_id")
        if symbol_col is None or ensembl_col is None:
            continue
        for symbol, ensembl in zip(df[symbol_col].astype(str), df[ensembl_col].astype(str)):
            symbol = symbol.strip()
            ensembl = ensembl.strip()
            if not symbol or symbol == "nan" or not ensembl or ensembl == "nan":
                continue
            mapping.setdefault(symbol, set()).add(strip_version(ensembl))
    return mapping


@st.cache_data(show_spinner=False)
def load_gwas_closest_genes(
    path: Path, mapping_paths: list[Path]
) -> tuple[set[str], dict[str, str], set[str]]:
    gene_ids: set[str] = set()
    id_to_symbol: dict[str, str] = {}
    unmapped: set[str] = set()
    if path is None or not path.exists():
        return gene_ids, id_to_symbol, unmapped
    symbol_map = load_symbol_to_ensembl_map(mapping_paths)
    with path.open() as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            symbol = raw
            if not symbol:
                continue
            if symbol.startswith("ENSG"):
                gid = strip_version(symbol)
                gene_ids.add(gid)
                id_to_symbol.setdefault(gid, symbol)
                continue
            mapped = symbol_map.get(symbol)
            if mapped:
                for gid in mapped:
                    gid_norm = strip_version(gid)
                    gene_ids.add(gid_norm)
                    id_to_symbol.setdefault(gid_norm, symbol)
            else:
                gene_ids.add(symbol)
                id_to_symbol.setdefault(symbol, symbol)
                unmapped.add(symbol)
    return gene_ids, id_to_symbol, unmapped


def add_tpm_from_map(df: pd.DataFrame, tpm_map: dict[str, float] | None) -> pd.DataFrame:
    if not tpm_map or "tpm_mean" in df.columns:
        return df
    tpm_map_stripped = {strip_version(k): v for k, v in tpm_map.items()}
    gene_ids = df["gene_id"].astype(str)
    tpm_series = gene_ids.map(tpm_map)
    missing = tpm_series.isna()
    if missing.any():
        tpm_series.loc[missing] = gene_ids.loc[missing].map(lambda x: tpm_map_stripped.get(strip_version(x)))
    df = df.copy()
    df["tpm_mean"] = tpm_series
    return df


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
    return tpm.mean(axis=1, skipna=True)


def _normalize_sample_ids(sample_ids: Iterable[str] | None) -> tuple[str, ...] | None:
    if not sample_ids:
        return None
    cleaned = [str(s).strip() for s in sample_ids if str(s).strip()]
    if not cleaned:
        return None
    return tuple(sorted(set(cleaned)))


def _featurecounts_sample_id(col: str) -> str:
    path = Path(str(col))
    name = path.name
    if ".Aligned" in name:
        return name.split(".Aligned", 1)[0]
    parts = path.parts
    if len(parts) >= 2:
        return parts[-2]
    return name.split(".", 1)[0]


def _subset_featurecounts_counts(
    counts: pd.DataFrame, sample_ids: Iterable[str] | None
) -> pd.DataFrame:
    sample_ids = _normalize_sample_ids(sample_ids)
    if not sample_ids:
        return counts
    wanted = set(sample_ids)
    selected = [col for col in counts.columns if _featurecounts_sample_id(col) in wanted]
    if not selected:
        return counts
    return counts[selected]


def _subset_counts_matrix(counts: pd.DataFrame, sample_ids: Iterable[str] | None) -> pd.DataFrame:
    sample_ids = _normalize_sample_ids(sample_ids)
    if not sample_ids:
        return counts
    wanted = set(sample_ids)
    selected = [col for col in counts.columns if col in wanted]
    if not selected:
        return counts
    return counts[selected]


def load_mcd_sample_ids(metadata_path: Path) -> set[str]:
    if not metadata_path.exists():
        return set()
    df = pd.read_csv(metadata_path, sep="\t")
    if "sample_id" not in df.columns:
        return set()
    if "diet" not in df.columns:
        return set(df["sample_id"].astype(str))
    mask = df["diet"].astype(str).str.contains("mcd", case=False, na=False)
    if not mask.any():
        return set(df["sample_id"].astype(str))
    return set(df.loc[mask, "sample_id"].astype(str))


def load_masld_sample_ids(dataset: str) -> set[str]:
    samplesheet = (
        ROOT
        / "RNA-seq"
        / "patient_RNAseq"
        / "data"
        / "samplesheets"
        / f"{dataset}_samplesheet.csv"
    )
    if not samplesheet.exists():
        return set()
    df = pd.read_csv(samplesheet)
    if "sample" not in df.columns:
        return set()
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
    return set(df.loc[mask, "sample"].astype(str))


def read_featurecounts_counts(path: Path) -> tuple[pd.Series, pd.DataFrame]:
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


@st.cache_data(show_spinner=False)
def load_tpm_map_from_featurecounts(
    path: Path, sample_ids: Iterable[str] | None = None
) -> dict[str, float] | None:
    if not path.exists():
        return None
    lengths, counts = read_featurecounts_counts(path)
    counts = _subset_featurecounts_counts(counts, sample_ids)
    tpm = compute_tpm_mean(counts, lengths)
    return tpm.to_dict()


@st.cache_data(show_spinner=False)
def load_tpm_map_from_counts_matrix(
    counts_path: Path, length_source: Path, sample_ids: Iterable[str] | None = None
) -> dict[str, float] | None:
    if not counts_path.exists() or not length_source.exists():
        return None
    counts_df = pd.read_csv(counts_path, sep="\t")
    gene_col = counts_df.columns[0]
    counts = counts_df.set_index(gene_col)
    counts = counts.apply(pd.to_numeric, errors="coerce")
    counts = _subset_counts_matrix(counts, sample_ids)

    lengths_df = pd.read_csv(length_source, sep="\t", comment="#")
    cols = {c.lower(): c for c in lengths_df.columns}
    gene_col = cols.get("geneid") or cols.get("gene_id") or cols.get("gene")
    length_col = cols.get("length")
    if gene_col is None or length_col is None:
        return None
    lengths = pd.to_numeric(lengths_df[length_col], errors="coerce")
    lengths.index = lengths_df[gene_col].astype(str)
    tpm = compute_tpm_mean(counts, lengths)
    return tpm.to_dict()


@st.cache_data(show_spinner=False)
def load_ortholog_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    mouse_col = cols.get("mouse_ensembl_gene_id") or cols.get("ensembl_gene_id") or cols.get("mouse_gene_id")
    human_col = cols.get("human_ensembl_gene_id") or cols.get("hsapiens_homolog_ensembl_gene")
    ortho_col = cols.get("orthology_type") or cols.get("hsapiens_homolog_orthology_type")
    if mouse_col is None or human_col is None:
        raise ValueError("Ortholog map missing required mouse/human columns.")
    df = df.rename(
        columns={
            mouse_col: "mouse_ensembl_gene_id",
            human_col: "human_ensembl_gene_id",
            ortho_col: "orthology_type",
        }
    )
    df = df[["mouse_ensembl_gene_id", "human_ensembl_gene_id", "orthology_type"]]
    df = df.dropna(subset=["mouse_ensembl_gene_id", "human_ensembl_gene_id"])
    df["mouse_ensembl_gene_id"] = df["mouse_ensembl_gene_id"].map(strip_version)
    df["human_ensembl_gene_id"] = df["human_ensembl_gene_id"].map(strip_version)
    return df


def build_mouse_to_human_map(df: pd.DataFrame, one2one_only: bool) -> dict[str, set[str]]:
    if one2one_only:
        df = df[df["orthology_type"].str.contains("one2one", case=False, na=False)]
    mapping: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        mapping.setdefault(row.mouse_ensembl_gene_id, set()).add(row.human_ensembl_gene_id)
    return mapping


def build_human_to_mouse_map(df: pd.DataFrame, one2one_only: bool) -> dict[str, set[str]]:
    if one2one_only:
        df = df[df["orthology_type"].str.contains("one2one", case=False, na=False)]
    mapping: dict[str, set[str]] = {}
    for row in df.itertuples(index=False):
        mapping.setdefault(row.human_ensembl_gene_id, set()).add(row.mouse_ensembl_gene_id)
    return mapping


def mapped_counts(gene_set: set[str], mapping: dict[str, set[str]]) -> tuple[int, int, int]:
    mapped = 0
    unmapped = 0
    targets: set[str] = set()
    for gid in gene_set:
        gid_norm = strip_version(gid)
        mapped_to = mapping.get(gid_norm)
        if mapped_to:
            mapped += 1
            targets.update(mapped_to)
        else:
            unmapped += 1
    return mapped, unmapped, len(targets)


@st.cache_data(show_spinner=False)
def load_biotype_map(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("ensembl_gene_id") or cols.get("gene_id")
    biotype_col = cols.get("gene_biotype") or cols.get("gene_type")
    species_col = cols.get("species")
    if gene_col is None or biotype_col is None or species_col is None:
        raise ValueError("Biotype map missing required columns.")
    df = df.rename(
        columns={
            gene_col: "ensembl_gene_id",
            biotype_col: "gene_biotype",
            species_col: "species",
        }
    )
    df = df[["ensembl_gene_id", "gene_biotype", "species"]]
    df = df.dropna(subset=["ensembl_gene_id", "gene_biotype", "species"])
    df["ensembl_gene_id"] = df["ensembl_gene_id"].map(strip_version)
    df["species"] = df["species"].str.lower()
    return df


def canonicalize_human_set(gene_set: set[str]) -> set[str]:
    return {strip_version(gid) for gid in gene_set}


def canonicalize_mouse_set(
    gene_set: set[str],
    mapping: dict[str, set[str]],
    include_unmapped: bool,
) -> set[str]:
    canonical = set()
    for gid in gene_set:
        gid_norm = strip_version(gid)
        mapped = mapping.get(gid_norm)
        if mapped:
            canonical.update(mapped)
        elif include_unmapped:
            canonical.add(f"MOUSE:{gid_norm}")
    return canonical


def tpm_mask(df: pd.DataFrame, tpm_cutoff: float) -> pd.Series:
    if tpm_cutoff <= 0 or "tpm_mean" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["tpm_mean"] >= tpm_cutoff


def log2fc_series(
    df: pd.DataFrame,
    padj_cutoff: float,
    tpm_cutoff: float,
    log2fc_cutoff: float | None = None,
) -> pd.Series:
    mask = df["padj"] < padj_cutoff
    if log2fc_cutoff is not None:
        mask &= df["log2FoldChange"] > log2fc_cutoff
    mask &= tpm_mask(df, tpm_cutoff)
    return df.loc[mask, "log2FoldChange"]


def upregulated_set(
    df: pd.DataFrame,
    padj_cutoff: float,
    log2fc_cutoff: float,
    tpm_cutoff: float,
) -> set[str]:
    mask = (df["padj"] < padj_cutoff) & (df["log2FoldChange"] > log2fc_cutoff)
    mask &= tpm_mask(df, tpm_cutoff)
    return set(df.loc[mask, "gene_id"].astype(str))


def lfc_positive_set(
    df: pd.DataFrame,
    log2fc_cutoff: float,
    *,
    strip_versions: bool = False,
) -> set[str]:
    genes = df.loc[df["log2FoldChange"] > log2fc_cutoff, "gene_id"].astype(str)
    if strip_versions:
        genes = genes.map(strip_version)
    return set(genes)


def intersection_count_from_sets(sets: Iterable[set[str]]) -> int:
    sets = list(sets)
    if not sets:
        return 0
    return len(set.intersection(*sets))


def top_right_set(
    nas_df: pd.DataFrame,
    other_df: pd.DataFrame,
    nas_log2fc_cutoff: float,
    padj_cutoff: float,
    tpm_cutoff: float,
) -> set[str]:
    nas_mask = (nas_df["padj"] < padj_cutoff) & (nas_df["log2FoldChange"] > nas_log2fc_cutoff)
    nas_mask &= tpm_mask(nas_df, tpm_cutoff)
    other_mask = (other_df["padj"] < padj_cutoff) & tpm_mask(other_df, tpm_cutoff)
    nas_set = set(nas_df.loc[nas_mask, "gene_id"].astype(str))
    other_set = set(other_df.loc[other_mask, "gene_id"].astype(str))
    return nas_set & other_set


def cross_dataset_top_right_set(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    padj_cutoff: float,
    log2fc_cutoff: float,
    tpm_cutoff: float,
) -> set[str]:
    return upregulated_set(df_a, padj_cutoff, log2fc_cutoff, tpm_cutoff) & upregulated_set(
        df_b, padj_cutoff, log2fc_cutoff, tpm_cutoff
    )


def top_right_count(
    nas_df: pd.DataFrame,
    other_df: pd.DataFrame,
    nas_log2fc_cutoff: float,
    padj_cutoff: float,
    tpm_cutoff: float,
) -> int:
    return len(top_right_set(nas_df, other_df, nas_log2fc_cutoff, padj_cutoff, tpm_cutoff))


def build_combo_counts(
    sets: dict[str, set[str]],
    labels: list[str],
) -> tuple[dict[tuple[str, ...], int], dict[str, set[str]], set[str]]:
    selected = {label: sets[label] for label in labels if label in sets}
    if not selected:
        return {}, selected, set()
    union = set.union(*selected.values())
    combo_counts: dict[tuple[str, ...], int] = {}
    ordered_labels = [label for label in labels if label in selected]
    for gene in union:
        combo = tuple(label for label in ordered_labels if gene in selected[label])
        combo_counts[combo] = combo_counts.get(combo, 0) + 1
    return combo_counts, selected, union


def build_contribution_table(
    selected_sets: dict[str, set[str]],
    combo_counts: dict[tuple[str, ...], int],
    union_size: int,
) -> pd.DataFrame:
    rows = []
    for label, genes in selected_sets.items():
        total_in_set = len(genes)
        unique_only = combo_counts.get((label,), 0)
        shared_any = total_in_set - unique_only
        rows.append(
            {
                "set": label,
                "total_in_set": total_in_set,
                "unique_only": unique_only,
                "shared_any": shared_any,
                "unique_%_of_union": (unique_only / union_size) if union_size else 0.0,
                "total_%_of_union": (total_in_set / union_size) if union_size else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["unique_%_of_union"] = df["unique_%_of_union"].map(lambda x: f"{x:.2%}")
        df["total_%_of_union"] = df["total_%_of_union"].map(lambda x: f"{x:.2%}")
    return df.sort_values(["unique_only", "total_in_set"], ascending=False)


def build_combo_table(
    combo_counts: dict[tuple[str, ...], int],
    union_size: int,
) -> pd.DataFrame:
    rows = []
    for combo, count in combo_counts.items():
        rows.append(
            {
                "combination": " + ".join(combo),
                "num_sets": len(combo),
                "count": count,
                "percent_of_union": (count / union_size) if union_size else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["percent_of_union"] = df["percent_of_union"].map(lambda x: f"{x:.2%}")
    return df.sort_values(["num_sets", "count"], ascending=[False, False])




def get_inhouse_mcd_paths() -> dict[str, Path]:
    if not USE_BUNDLED_RESULTS:
        return INHOUSE_MCD_PATHS
    return {label: DATA_DIR / filename for label, filename in BUNDLED_INHOUSE_MCD_FILES.items()}


def get_external_mcd_paths() -> dict[str, Path]:
    if not USE_BUNDLED_RESULTS:
        return EXTERNAL_MCD_PATHS
    return {label: DATA_DIR / filename for label, filename in BUNDLED_EXTERNAL_MCD_FILES.items()}


def patient_paths(dataset: str) -> dict[str, Path] | None:
    if USE_BUNDLED_RESULTS:
        files = BUNDLED_PATIENT_FILES.get(dataset)
        if not files:
            return None
        return {
            "run_label": "bundled",
            "nas_high": DATA_DIR / files["nas_high"],
            "nas_low": DATA_DIR / files["nas_low"],
            "fibrosis": DATA_DIR / files["fibrosis"],
        }
    base_dir = ROOT / "RNA-seq" / "patient_RNAseq" / "results" / dataset / "deseq2_results"
    required = [
        Path("nas_high/deseq2_results/NAS_4plus_vs_NAS_0/differential_expression.csv"),
        Path("nas_low/deseq2_results/NAS_1to3_vs_NAS_0/differential_expression.csv"),
        Path("fibrosis/deseq2_results/F1to4_vs_F0/differential_expression.csv"),
    ]
    latest = _latest_run_with_files(base_dir, required)
    if latest is None:
        return None
    return {
        "run": latest,
        "nas_high": latest / required[0],
        "nas_low": latest / required[1],
        "fibrosis": latest / required[2],
    }


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def make_label(group: str, dataset: str, analysis: str) -> str:
    return f"{group} | {dataset} | {analysis}"


def render_table(df: pd.DataFrame) -> None:
    df_display = df.copy()
    for col in ("padj", "log2FC", "tpm"):
        if col in df_display.columns:
            df_display[col] = df_display[col].astype(str)
    st.dataframe(df_display, hide_index=True, width="stretch")


def get_cutoffs(key: str, default_padj: float, default_lfc: float) -> tuple[float, float]:
    use_override = st.session_state.get(f"{key}_override", False)
    if not use_override:
        return default_padj, default_lfc
    padj = st.session_state.get(f"{key}_padj", default_padj)
    lfc = st.session_state.get(f"{key}_lfc", default_lfc)
    return padj, lfc


def get_topright_padj(key: str, default_padj: float) -> float:
    use_override = st.session_state.get(f"{key}_override", False)
    if not use_override:
        return default_padj
    return st.session_state.get(f"{key}_top_padj", default_padj)


def get_tpm_cutoff(key: str, default_tpm: float) -> float:
    use_override = st.session_state.get(f"{key}_override", False)
    if not use_override:
        return default_tpm
    return st.session_state.get(f"{key}_tpm", default_tpm)


def _sync_number_to_slider(key_base: str) -> None:
    value = st.session_state.get(f"{key_base}_number")
    st.session_state[f"{key_base}_val"] = value
    st.session_state[f"{key_base}_slider"] = value


def _sync_slider_to_number(key_base: str) -> None:
    value = st.session_state.get(f"{key_base}_slider")
    st.session_state[f"{key_base}_val"] = value
    st.session_state[f"{key_base}_number"] = value


def synced_cutoff(
    label: str,
    min_value: float,
    max_value: float,
    default: float,
    step: float,
    key_base: str,
    format_str: str | None = None,
) -> float:
    if f"{key_base}_val" not in st.session_state:
        st.session_state[f"{key_base}_val"] = default
    col_num, col_slider = st.columns([1, 3])
    with col_num:
        st.number_input(
            f"{label} (value)",
            min_value,
            max_value,
            step=step,
            value=st.session_state[f"{key_base}_val"],
            key=f"{key_base}_number",
            format=format_str,
            on_change=_sync_number_to_slider,
            args=(key_base,),
        )
    with col_slider:
        st.slider(
            label,
            min_value,
            max_value,
            step=step,
            value=st.session_state[f"{key_base}_val"],
            key=f"{key_base}_slider",
            on_change=_sync_slider_to_number,
            args=(key_base,),
        )
    return st.session_state[f"{key_base}_val"]


st.set_page_config(page_title="Cas13 MASLD Library Explorer", layout="wide")

st.markdown(
    """
<style>
  .mouse-widget {
    position: fixed;
    top: 12px;
    right: 80px;
    z-index: 999999;
    pointer-events: none;
  }
  .mouse-scale {
    transform: scale(3);
    transform-origin: top right;
  }
  .mouse {
    width: 36px;
    height: 24px;
    background: #C8B4A2;
    border-radius: 18px 18px 14px 14px;
    position: relative;
    box-shadow: 0 2px 6px rgba(0,0,0,0.18);
    animation: mouse-walk 3.2s ease-in-out infinite;
  }
  .mouse::after {
    content: "";
    position: absolute;
    right: -6px;
    top: 8px;
    width: 6px;
    height: 6px;
    background: #B29786;
    border-radius: 50%;
  }
  .ear {
    width: 10px;
    height: 10px;
    background: #D8C5B6;
    border-radius: 50%;
    position: absolute;
    top: -4px;
    box-shadow: inset 0 0 0 2px #C2AA98;
  }
  .ear-left { left: 4px; animation: ear-twitch 2.4s ease-in-out infinite; }
  .ear-right { right: 6px; animation: ear-twitch 2.4s ease-in-out infinite 0.6s; }
  .eye {
    width: 3px;
    height: 3px;
    background: #3A2F28;
    border-radius: 50%;
    position: absolute;
    top: 8px;
  }
  .eye-left { left: 10px; }
  .eye-right { left: 17px; }
  .snout {
    width: 6px;
    height: 4px;
    background: #B7908A;
    border-radius: 50%;
    position: absolute;
    left: -2px;
    top: 12px;
  }
  .tail {
    position: absolute;
    right: -14px;
    top: 12px;
    width: 18px;
    height: 2px;
    background: #BBA190;
    border-radius: 2px;
    transform-origin: left center;
    animation: tail-wag 1.4s ease-in-out infinite;
  }
  @keyframes mouse-walk {
    0% { transform: translateX(0) translateY(0); }
    25% { transform: translateX(-6px) translateY(1px); }
    50% { transform: translateX(0) translateY(0); }
    75% { transform: translateX(6px) translateY(1px); }
    100% { transform: translateX(0) translateY(0); }
  }
  @keyframes tail-wag {
    0% { transform: rotate(8deg); }
    50% { transform: rotate(-12deg); }
    100% { transform: rotate(8deg); }
  }
  @keyframes ear-twitch {
    0%, 80%, 100% { transform: scale(1); }
    85% { transform: scale(0.85); }
    90% { transform: scale(1); }
  }
</style>
<div class="mouse-widget" aria-hidden="true">
  <div class="mouse-scale">
    <div class="mouse">
      <div class="ear ear-left"></div>
      <div class="ear ear-right"></div>
      <div class="eye eye-left"></div>
      <div class="eye eye-right"></div>
      <div class="snout"></div>
      <div class="tail"></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.title("Cas13 MASLD Library Explorer")
st.markdown(
    """
**Overview**
- **Mouse (in-house MCD)**: Week 1/2/3 MCD vs control contrasts from the in-house MCD diet study.
- **Mouse (external MCD GEO)**: **GSE156918** and **GSE205974** Control vs MCD contrasts.
- **Patient (human)**: GEO datasets **GSE130970** and **GSE135251** (NAFLD/NASH/MASLD cohorts).
- **GWAS (human, optional)**: liver disease GWAS SNPs with the closest genes to each SNP.
- This app reports **upregulated DEGs only** (log2FC > cutoff) and lets you adjust padj/log2FC cutoffs globally or per-dataset.
- TPM filtering uses **mean TPM per gene from MASLD-only patients and MCD-only mice** (if available).
- **Patient (cross-dataset)**: top-right quadrant overlaps using the global padj/log2FC cutoffs.

**Citations / datasets**: GEO **GSE156918**, **GSE205974**, **GSE130970**, **GSE135251**, and in-house MCD RNA-seq (week 1–3 diet contrasts).
"""
)

# Load MCD data (in-house + external)
inhouse_mcd_frames = {}
inhouse_mcd_sample_ids = tuple(
    sorted(load_mcd_sample_ids(ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "metadata" / "samples.tsv"))
)
inhouse_tpm_map = load_tpm_map_from_featurecounts(
    ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "counts" / "featurecounts" / "gene_counts.txt",
    sample_ids=inhouse_mcd_sample_ids,
)
for label, path in get_inhouse_mcd_paths().items():
    if not path.exists():
        st.warning(f"Missing in-house MCD file: {path}")
        continue
    df = load_mcd_tsv(path)
    df = add_tpm_from_map(df, inhouse_tpm_map)
    inhouse_mcd_frames[label] = df

external_mcd_frames = {}
external_mcd_sample_ids = {
    "GSE156918 (external MCD)": tuple(
        sorted(
            load_mcd_sample_ids(
                ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "metadata" / "samples.tsv"
            )
        )
    ),
    "GSE205974 (external MCD)": tuple(
        sorted(
            load_mcd_sample_ids(
                ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "metadata" / "samples.tsv"
            )
        )
    ),
}
external_tpm_maps: dict[str, dict[str, float] | None] = {
    "GSE156918 (external MCD)": load_tpm_map_from_featurecounts(
        ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "counts" / "featurecounts" / "gene_counts.txt",
        sample_ids=external_mcd_sample_ids.get("GSE156918 (external MCD)"),
    ),
    "GSE205974 (external MCD)": load_tpm_map_from_featurecounts(
        ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "counts" / "featurecounts" / "gene_counts.txt",
        sample_ids=external_mcd_sample_ids.get("GSE205974 (external MCD)"),
    ),
}
for label, path in get_external_mcd_paths().items():
    if not path.exists():
        st.warning(f"Missing external MCD file: {path}")
        continue
    df = load_mcd_tsv(path)
    df = add_tpm_from_map(df, external_tpm_maps.get(label))
    external_mcd_frames[label] = df

# Load patient data for both datasets
patient_data = {}
gse130970_counts = ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE130970" / "counts"
gse135251_counts = ROOT / "RNA-seq" / "patient_RNAseq" / "results" / "GSE135251" / "counts"
gse130970_length_source = next(gse130970_counts.glob("individual/*_counts.txt"), None)
gse135251_length_source = next(gse135251_counts.glob("individual/*_counts.txt"), None)
patient_masld_sample_ids = {
    dataset: tuple(sorted(load_masld_sample_ids(dataset))) for dataset in PATIENT_DATASETS
}
patient_tpm_maps: dict[str, dict[str, float] | None] = {
    "GSE130970": load_tpm_map_from_counts_matrix(
        gse130970_counts / "gene_counts_matrix.txt",
        gse130970_length_source,
        sample_ids=patient_masld_sample_ids.get("GSE130970"),
    )
    if gse130970_length_source
    else None,
    "GSE135251": load_tpm_map_from_counts_matrix(
        gse135251_counts / "gene_counts_matrix.txt",
        gse135251_length_source,
        sample_ids=patient_masld_sample_ids.get("GSE135251"),
    )
    if gse135251_length_source
    else None,
}
for dataset in PATIENT_DATASETS:
    paths = patient_paths(dataset)
    if paths is None:
        patient_data[dataset] = {"paths": None, "error": "No run folder found."}
        continue
    missing = [k for k, p in paths.items() if k not in ("run", "run_label") and not p.exists()]
    if missing:
        patient_data[dataset] = {"paths": paths, "error": f"Missing files: {', '.join(missing)}"}
        continue
    nas_high = load_patient_csv(paths["nas_high"])
    nas_low = load_patient_csv(paths["nas_low"])
    fibrosis = load_patient_csv(paths["fibrosis"])
    tpm_map = patient_tpm_maps.get(dataset)
    nas_high = add_tpm_from_map(nas_high, tpm_map)
    nas_low = add_tpm_from_map(nas_low, tpm_map)
    fibrosis = add_tpm_from_map(fibrosis, tpm_map)
    patient_data[dataset] = {
        "paths": paths,
        "nas_high": nas_high,
        "nas_low": nas_low,
        "fibrosis": fibrosis,
    }

gwas_genes: set[str] = set()
gwas_symbol_map: dict[str, str] = {}
gwas_unmapped: set[str] = set()
gwas_label: str | None = None
gwas_symbol_map_paths = list(GWAS_GENE_ANNOTATION_PATHS)
if GWAS_GENE_BIOTYPE_PATH is not None and GWAS_GENE_BIOTYPE_PATH.exists():
    gwas_symbol_map_paths.append(GWAS_GENE_BIOTYPE_PATH)
if GWAS_CLOSEST_GENES_PATH is not None and GWAS_CLOSEST_GENES_PATH.exists():
    gwas_genes, gwas_symbol_map, gwas_unmapped = load_gwas_closest_genes(
        GWAS_CLOSEST_GENES_PATH, gwas_symbol_map_paths
    )
    if gwas_genes:
        gwas_label = make_label("GWAS", "human", "Closest genes")

# ===== TPM slider bounds =====
def gather_tpm_max() -> float:
    candidates = []
    for df in list(inhouse_mcd_frames.values()) + list(external_mcd_frames.values()):
        if "tpm_mean" in df.columns:
            candidates.append(df["tpm_mean"].max())
    for info in patient_data.values():
        if info.get("error") or info.get("paths") is None:
            continue
        for key in ("nas_high", "nas_low", "fibrosis"):
            df = info.get(key)
            if isinstance(df, pd.DataFrame) and "tpm_mean" in df.columns:
                candidates.append(df["tpm_mean"].max())
    candidates = [v for v in candidates if pd.notna(v)]
    if not candidates:
        return 100.0
    tpm_max = max(candidates)
    return max(10.0, min(1000.0, math.ceil(tpm_max / 10.0) * 10.0))

tpm_slider_max = gather_tpm_max()

# ===== Selection UI =====
st.info("Select the analyses to include in the findings.")
active_labels = []

col_mouse, col_human = st.columns(2)

def selection_component(label, options, key_prefix):
    # options: list of (display_label, full_key)
    st.markdown(f"**{label}**")
    try:
        # Check if st.pills exists (Streamlit 1.40+)
        if hasattr(st, "pills"):
            # Map display label back to full key
            display_map = {opt[0]: opt[1] for opt in options}
            selection = st.pills(label, list(display_map.keys()), selection_mode="multi", key=f"pills_{key_prefix}", label_visibility="collapsed")
            return [display_map[s] for s in selection]
    except Exception:
        pass
    
    # Fallback to checkboxes
    selected = []
    for disp, val in options:
        if st.checkbox(disp, key=f"chk_{val}"):
            selected.append(val)
    return selected

with col_mouse:
    st.subheader("Mouse")
    # In-house
    opts = []
    for label in inhouse_mcd_frames:
        # Shorten label: "MCD Week 1" -> "Week 1"
        short = label.replace("MCD ", "")
        opts.append((short, make_label("MCD (in-house)", "mouse", label)))
    
    if any(l in inhouse_mcd_frames for l in INHOUSE_MCD_WEEK_LABELS):
        opts.append(("Week 1/2/3 Intersection", make_label("MCD (in-house)", "mouse", "MCD Week1/2/3 Intersection")))
    
    if opts:
        active_labels.extend(selection_component("In-house MCD", opts, "inhouse"))

    # External
    opts = []
    for label in external_mcd_frames:
        # Shorten: "GSE156918 (external MCD)" -> "GSE156918"
        short = label.split(" ")[0]
        opts.append((short, make_label("MCD (external)", "mouse", label)))
    
    if opts:
        active_labels.extend(selection_component("External MCD", opts, "external"))

with col_human:
    st.subheader("Human")
    # Patient
    opts = []
    for dataset, info in patient_data.items():
        if not info.get("error") and info.get("paths"):
            opts.append((f"{dataset} NAS High", make_label("Patient", dataset, "NAS high (upregulated)")))
            opts.append((f"{dataset} Fibrosis", make_label("Patient", dataset, "Fibrosis (upregulated)")))
            opts.append((f"{dataset} NAS High vs Fibrosis", make_label("Patient", dataset, "NAS high vs Fibrosis (top-right)")))
            opts.append((f"{dataset} NAS High vs Low", make_label("Patient", dataset, "NAS high vs NAS low (top-right)")))
    
    if opts:
        active_labels.extend(selection_component("Patient Cohorts", opts, "patient"))

    # Cross-dataset
    opts = []
    pair_a, pair_b = PATIENT_CROSS_DATASET_PAIR
    info_a = patient_data.get(pair_a)
    info_b = patient_data.get(pair_b)
    if info_a and info_b and not info_a.get("error") and not info_b.get("error") and info_a.get("paths") and info_b.get("paths"):
        for comp_label in PATIENT_CROSS_COMPARISONS:
             opts.append((f"{comp_label} (Cross-dataset)", make_label("Patient (cross-dataset)", f"{pair_a} vs {pair_b}", f"{comp_label} (top-right)")))
    
    if opts:
        active_labels.extend(selection_component("Cross-dataset", opts, "cross"))

    # GWAS
    if gwas_genes and gwas_label is not None:
        active_labels.extend(
            selection_component("GWAS", [("Closest genes", gwas_label)], "gwas")
        )
        if gwas_unmapped:
            st.caption(
                f"GWAS list includes {len(gwas_unmapped)} symbols without Ensembl IDs; keeping them as symbols."
            )

# ===== Global Sliders + Inputs =====
padj_cutoff = synced_cutoff(
    "Global padj cutoff (MCD + NAS high) (default 0.05)",
    0.0,
    0.2,
    0.05,
    0.005,
    "global_padj_005",
    format_str="%.4f",
)
log2fc_cutoff = synced_cutoff(
    "Global log2FC cutoff (upregulated only)",
    0.0,
    5.0,
    0.0,
    0.1,
    "global_log2fc",
    format_str="%.3f",
)
tpm_cutoff = synced_cutoff(
    "Global TPM cutoff (mean TPM per gene per dataset; disease-only samples)",
    0.0,
    float(tpm_slider_max),
    0.0,
    0.1,
    "global_tpm",
    format_str="%.2f",
)

st.caption(
    "Within-dataset top-right uses a dataset-specific padj cutoff (defaults to the global padj unless overridden), "
    "and log2FC cutoff applies only to NAS high. Cross-dataset top-right uses the global padj/log2FC cutoffs. "
    "TPM cutoff is applied to all sets where TPM data is available (mean TPM per gene uses MASLD-only patients and MCD-only mice when metadata is available)."
)

missing_tpm = []
for label, df in inhouse_mcd_frames.items():
    key = f"inhouse_mcd_{slugify(label)}"
    tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
    if tpm_eff > 0 and "tpm_mean" not in df.columns:
        missing_tpm.append(label)
for label, df in external_mcd_frames.items():
    key = f"external_mcd_{slugify(label)}"
    tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
    if tpm_eff > 0 and "tpm_mean" not in df.columns:
        missing_tpm.append(label)
for dataset, info in patient_data.items():
    if info.get("error") or info.get("paths") is None:
        continue
    key = f"patient_{slugify(dataset)}"
    tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
    if tpm_eff > 0 and "tpm_mean" not in info["nas_high"].columns:
        missing_tpm.append(dataset)
if missing_tpm:
    st.warning(
        "TPM cutoff is enabled but TPM data is missing for: "
        + ", ".join(sorted(set(missing_tpm)))
    )

# ===== Dataset specific overrides =====
with st.expander("Dataset-specific overrides (Advanced)"):
    st.caption("Override the global cutoffs for specific datasets.")

    if inhouse_mcd_frames:
        st.markdown("#### MCD (In-house)")
        for label in inhouse_mcd_frames:
            key = f"inhouse_mcd_{slugify(label)}"
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            with c1:
                use_override = st.checkbox(f"{label}", key=f"{key}_override")
            if use_override:
                with c2:
                    st.number_input(f"padj", 0.0, 1.0, padj_cutoff, 0.005, key=f"{key}_padj")
                with c3:
                    st.number_input(f"log2FC", 0.0, 10.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
                with c4:
                    st.number_input(f"TPM", 0.0, float(tpm_slider_max), tpm_cutoff, 0.1, key=f"{key}_tpm")

    if external_mcd_frames:
        st.markdown("#### MCD (External)")
        for label in external_mcd_frames:
            key = f"external_mcd_{slugify(label)}"
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            with c1:
                use_override = st.checkbox(f"{label}", key=f"{key}_override")
            if use_override:
                with c2:
                    st.number_input(f"padj", 0.0, 1.0, padj_cutoff, 0.005, key=f"{key}_padj")
                with c3:
                    st.number_input(f"log2FC", 0.0, 10.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
                with c4:
                    st.number_input(f"TPM", 0.0, float(tpm_slider_max), tpm_cutoff, 0.1, key=f"{key}_tpm")

    if patient_data:
        st.markdown("#### Patient Datasets")
        for dataset, info in patient_data.items():
            if info.get("error") or info.get("paths") is None:
                continue
            key = f"patient_{slugify(dataset)}"
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 2, 2])
            with c1:
                use_override = st.checkbox(f"{dataset}", key=f"{key}_override")
            if use_override:
                with c2:
                    st.number_input(f"NAS high padj", 0.0, 1.0, padj_cutoff, 0.005, key=f"{key}_padj")
                with c3:
                    st.number_input(f"NAS high log2FC", 0.0, 10.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
                with c4:
                    st.number_input(f"Top-right global padj", 0.0, 1.0, 0.05, 0.005, key=f"{key}_top_padj")
                with c5:
                    st.number_input(f"TPM", 0.0, float(tpm_slider_max), tpm_cutoff, 0.1, key=f"{key}_tpm")

    st.markdown("#### Patient (Cross-dataset)")
    key = "patient_cross_dataset"
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        use_override = st.checkbox(f"Cross-dataset pair", key=f"{key}_override")
    if use_override:
        with c2:
            st.number_input(f"padj", 0.0, 1.0, padj_cutoff, 0.005, key=f"{key}_padj")
        with c3:
            st.number_input(f"log2FC", 0.0, 10.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
        with c4:
            st.number_input(f"TPM", 0.0, float(tpm_slider_max), tpm_cutoff, 0.1, key=f"{key}_tpm")

# ===== Top summary =====
st.subheader("Overall summary")

summary_rows = []
summary_sets: dict[str, dict[str, object]] = {}
raw_sets: dict[str, set[str]] = {}
dedup_sets: dict[str, set[str]] = {}
cross_dataset_rows: list[dict[str, object]] = []
cross_dataset_sets: dict[str, set[str]] = {}

inhouse_mcd_week_sets = []
for label, df in inhouse_mcd_frames.items():
    key = f"inhouse_mcd_{slugify(label)}"
    padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
    gene_set = upregulated_set(df, padj_eff, lfc_eff, tpm_eff)
    if label in INHOUSE_MCD_WEEK_LABELS:
        inhouse_mcd_week_sets.append(gene_set)
    summary_rows.append(
        {
            "group": "MCD (in-house)",
            "dataset": "mouse",
            "analysis": label,
            "padj": padj_eff,
            "log2FC": lfc_eff,
            "tpm": tpm_eff if "tpm_mean" in df.columns else "missing",
            "count": len(gene_set),
        }
    )
    summary_sets[make_label("MCD (in-house)", "mouse", label)] = {"species": "mouse", "genes": gene_set}

if inhouse_mcd_week_sets:
    mcd_intersection = set.intersection(*inhouse_mcd_week_sets)
    summary_rows.append(
        {
            "group": "MCD (in-house)",
            "dataset": "mouse",
            "analysis": "MCD Week1/2/3 Intersection",
            "padj": "varies",
            "log2FC": "varies",
            "tpm": "varies",
            "count": len(mcd_intersection),
        }
    )
    summary_sets[make_label("MCD (in-house)", "mouse", "MCD Week1/2/3 Intersection")] = {
        "species": "mouse",
        "genes": mcd_intersection,
    }

for label, df in external_mcd_frames.items():
    key = f"external_mcd_{slugify(label)}"
    padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
    gene_set = upregulated_set(df, padj_eff, lfc_eff, tpm_eff)
    summary_rows.append(
        {
            "group": "MCD (external)",
            "dataset": "mouse",
            "analysis": label,
            "padj": padj_eff,
            "log2FC": lfc_eff,
            "tpm": tpm_eff if "tpm_mean" in df.columns else "missing",
            "count": len(gene_set),
        }
    )
    summary_sets[make_label("MCD (external)", "mouse", label)] = {"species": "mouse", "genes": gene_set}

for dataset, info in patient_data.items():
    if info.get("paths") is None:
        summary_rows.append(
            {
                "group": "Patient",
                "dataset": dataset,
                "analysis": "No data",
                "padj": "-",
                "log2FC": "-",
                "tpm": "-",
                "count": 0,
            }
        )
        continue
    if info.get("error"):
        summary_rows.append(
            {
                "group": "Patient",
                "dataset": dataset,
                "analysis": info["error"],
                "padj": "-",
                "log2FC": "-",
                "tpm": "-",
                "count": 0,
            }
        )
        continue

    key = f"patient_{slugify(dataset)}"
    nas_padj, nas_lfc = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    top_padj = get_topright_padj(key, default_padj=padj_cutoff)
    tpm_eff = get_tpm_cutoff(key, tpm_cutoff)

    nas_high_df = info["nas_high"]
    nas_low_df = info["nas_low"]
    fibrosis_df = info["fibrosis"]

    nas_high_set = upregulated_set(nas_high_df, nas_padj, nas_lfc, tpm_eff)
    fibrosis_set = upregulated_set(fibrosis_df, nas_padj, nas_lfc, tpm_eff)
    nas_high_vs_fibrosis_set = top_right_set(
        nas_high_df, fibrosis_df, nas_lfc, padj_cutoff=top_padj, tpm_cutoff=tpm_eff
    )
    nas_high_vs_nas_low_set = top_right_set(
        nas_high_df, nas_low_df, nas_lfc, padj_cutoff=top_padj, tpm_cutoff=tpm_eff
    )

    summary_rows.append(
        {
            "group": "Patient",
            "dataset": dataset,
            "analysis": "NAS high (upregulated)",
            "padj": nas_padj,
            "log2FC": nas_lfc,
            "tpm": tpm_eff if "tpm_mean" in nas_high_df.columns else "missing",
            "count": len(nas_high_set),
        }
    )
    summary_sets[make_label("Patient", dataset, "NAS high (upregulated)")] = {
        "species": "human",
        "genes": nas_high_set,
    }
    summary_rows.append(
        {
            "group": "Patient",
            "dataset": dataset,
            "analysis": "Fibrosis (upregulated)",
            "padj": nas_padj,
            "log2FC": nas_lfc,
            "tpm": tpm_eff if "tpm_mean" in fibrosis_df.columns else "missing",
            "count": len(fibrosis_set),
        }
    )
    summary_sets[make_label("Patient", dataset, "Fibrosis (upregulated)")] = {
        "species": "human",
        "genes": fibrosis_set,
    }
    summary_rows.append(
        {
            "group": "Patient",
            "dataset": dataset,
            "analysis": "NAS high vs Fibrosis (top-right)",
            "padj": top_padj,
            "log2FC": nas_lfc,
            "tpm": tpm_eff if "tpm_mean" in nas_high_df.columns else "missing",
            "count": len(nas_high_vs_fibrosis_set),
        }
    )
    summary_sets[make_label("Patient", dataset, "NAS high vs Fibrosis (top-right)")] = {
        "species": "human",
        "genes": nas_high_vs_fibrosis_set,
    }
    summary_rows.append(
        {
            "group": "Patient",
            "dataset": dataset,
            "analysis": "NAS high vs NAS low (top-right)",
            "padj": top_padj,
            "log2FC": nas_lfc,
            "tpm": tpm_eff if "tpm_mean" in nas_high_df.columns else "missing",
            "count": len(nas_high_vs_nas_low_set),
        }
    )
    summary_sets[make_label("Patient", dataset, "NAS high vs NAS low (top-right)")] = {
        "species": "human",
        "genes": nas_high_vs_nas_low_set,
    }

if gwas_genes and gwas_label is not None:
    summary_rows.append(
        {
            "group": "GWAS",
            "dataset": "human",
            "analysis": "Closest genes",
            "padj": "-",
            "log2FC": "-",
            "tpm": "-",
            "count": len(gwas_genes),
        }
    )
    summary_sets[gwas_label] = {"species": "human", "genes": gwas_genes}

pair_a, pair_b = PATIENT_CROSS_DATASET_PAIR
info_a = patient_data.get(pair_a)
info_b = patient_data.get(pair_b)
if info_a and info_b and not info_a.get("error") and not info_b.get("error") and info_a.get("paths") and info_b.get("paths"):
    for comp_label, comp_key in PATIENT_CROSS_COMPARISONS.items():
        df_a = info_a.get(comp_key)
        df_b = info_b.get(comp_key)
        if df_a is None or df_b is None:
            continue
        
        key = "patient_cross_dataset"
        padj_cd, lfc_cd = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
        tpm_cd = get_tpm_cutoff(key, tpm_cutoff)
        
        cross_set = cross_dataset_top_right_set(df_a, df_b, padj_cd, lfc_cd, tpm_cd)
        cross_dataset_rows.append(
            {
                "group": "Patient (cross-dataset)",
                "dataset": f"{pair_a} vs {pair_b}",
                "analysis": f"{comp_label} (top-right)",
                "padj": padj_cd,
                "log2FC": f">{lfc_cd}",
                "tpm": tpm_cd if "tpm_mean" in df_a.columns else "missing",
                "count": len(cross_set),
            }
        )
        summary_sets[make_label("Patient (cross-dataset)", f"{pair_a} vs {pair_b}", f"{comp_label} (top-right)")] = {
            "species": "human",
            "genes": cross_set,
        }
        cross_dataset_sets[comp_label] = cross_set

if cross_dataset_rows:
    summary_rows.extend(cross_dataset_rows)

if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_df["label"] = (
        summary_df["group"].astype(str)
        + " | "
        + summary_df["dataset"].astype(str)
        + " | "
        + summary_df["analysis"].astype(str)
    )
    raw_sets = {label: entry["genes"] for label, entry in summary_sets.items()}
    raw_sets = {label: entry["genes"] for label, entry in summary_sets.items()}

    if not active_labels:
        st.info("No datasets selected yet. Pick one or more above to populate the downstream analyses.")

    dedup_sets = {}
    if ORTHOLOG_PATH is not None:
        use_one2one = st.checkbox("Use one-to-one orthologs only", value=True)
        include_unmapped_human = st.checkbox("Include human genes without mouse homologs (keep all)", value=True)
        include_unmapped = st.checkbox("Include unmapped mouse genes", value=True)
        ortholog_df = load_ortholog_map(ORTHOLOG_PATH)
        mouse_to_human = build_mouse_to_human_map(ortholog_df, one2one_only=use_one2one)
        human_to_mouse = build_human_to_mouse_map(ortholog_df, one2one_only=use_one2one)
        for label, entry in summary_sets.items():
            species = entry["species"]
            genes = entry["genes"]
            if species == "human":
                if include_unmapped_human:
                    dedup_sets[label] = canonicalize_human_set(genes)
                else:
                    human_mapped = set()
                    for gid in genes:
                        gid_norm = strip_version(gid)
                        if gid_norm in human_to_mouse:
                            human_mapped.add(gid_norm)
                    dedup_sets[label] = canonicalize_human_set(human_mapped)
            else:
                dedup_sets[label] = canonicalize_mouse_set(genes, mouse_to_human, include_unmapped)
        if dedup_sets and active_labels:
            selected_labels = [label for label in active_labels if label in dedup_sets]
            # combo_counts, selected_sets, union_genes = build_combo_counts(dedup_sets, selected_labels)
            dedup_total = 1 # dummy placeholder to enter block

            if dedup_total:
                if BIOTYPE_PATH is not None and BIOTYPE_PATH.exists():
                    biotype_df = load_biotype_map(BIOTYPE_PATH)
                    human_biotype_map = (
                        biotype_df[biotype_df["species"] == "human"]
                        .set_index("ensembl_gene_id")["gene_biotype"]
                        .to_dict()
                    )
                    mouse_biotype_map = (
                        biotype_df[biotype_df["species"] == "mouse"]
                        .set_index("ensembl_gene_id")["gene_biotype"]
                        .to_dict()
                    )
                
                # Deduplication strategy selection
                st.write("---")
                col_strat, col_metric = st.columns([2, 2])
                with col_strat:
                    dedup_strategy = st.radio(
                        "Deduplication Strategy",
                        ["Ensembl ID (Strict)", "Gene Symbol (Functional)"],
                        index=0,
                        help="Ensembl ID preserves distinct loci (e.g. Y_RNA copies). Gene Symbol merges them."
                    )
                
                # Apply strategy
                human_symbol_map, mouse_symbol_map = load_symbol_maps_from_bundled(DATA_DIR)
                if gwas_symbol_map:
                    for gid, symbol in gwas_symbol_map.items():
                        human_symbol_map.setdefault(strip_version(gid), symbol)
                        
                final_sets = {}
                for label, genes in dedup_sets.items():
                    # Genes are already canonicalized (version stripped) IDs
                    if dedup_strategy == "Ensembl ID (Strict)":
                        final_sets[label] = genes
                    else:
                        # Map to symbols
                        mapped_genes = set()
                        species = summary_sets[label]["species"]
                        sym_map = human_symbol_map if species == "human" else mouse_symbol_map
                        for gid in genes:
                            # If mapped to a symbol, use it. Else keep ID.
                            sym = sym_map.get(gid)
                            if sym:
                                mapped_genes.add(sym)
                            else:
                                mapped_genes.add(gid)
                        final_sets[label] = mapped_genes

                combo_counts, selected_sets, union_genes = build_combo_counts(final_sets, selected_labels)
                dedup_total = len(union_genes)

                with col_metric:
                    st.metric(f"Total DEGs ({dedup_strategy})", dedup_total)

                with st.expander("Detailed Counts & Overlaps"):
                    st.write(f"**Union Size**: {dedup_total}")
                    st.write("Individual Set Sizes (after filters):")
                    total_raw_sum = 0
                    for label in selected_labels:
                        count = len(final_sets[label])
                        total_raw_sum += count
                        st.write(f"- **{label}**: {count}")
                    
                    overlap_diff = total_raw_sum - dedup_total
                    st.info(f"Overlap Reduction: {overlap_diff} genes merged across datasets.")

                gene_to_sets: dict[str, list[str]] = {}
                for label, genes in selected_sets.items():
                    for gid in genes:
                        gene_to_sets.setdefault(gid, []).append(label)
                rows = []
                rows = []
                is_strict_mode = (dedup_strategy == "Ensembl ID (Strict)")
                for unit in sorted(union_genes):
                    # unit is either an Ensembl ID or a Symbol (if Functional strategy used)
                    
                    # Logic to determine if 'unit' is an Ensembl ID
                    # If strict mode: ALL are IDs.
                    # If functional mode: IDs are those that failed mapping (start with ENS... or MOUSE:) 
                    # while successful mappings are Symbols (e.g. "IL32", "Y_RNA").
                    
                    is_id_format = (
                        unit.startswith("ENS") 
                        or unit.startswith("MOUSE:") 
                        or unit.startswith("TCONS") # rare but possible
                        or (is_strict_mode) # In strict mode, everything is an ID
                    )

                    if is_id_format:
                        gid = unit
                        if gid.startswith("MOUSE:"):
                            raw_id = gid.split("MOUSE:", 1)[1]
                            species = "mouse"
                            symbol = mouse_symbol_map.get(strip_version(raw_id), "")
                            biotype = mouse_biotype_map.get(strip_version(raw_id), "")
                        else:
                            raw_id = gid
                            species = "human"
                            symbol = human_symbol_map.get(strip_version(raw_id), "")
                            biotype = human_biotype_map.get(strip_version(raw_id), "")
                    else:
                        # unit is already a Symbol (Functional mode)
                        species = "human" # Default assumption for symbol-only, or check context?
                        # Actually, symbols are ambiguous for species if we don't track it, 
                        # but here we mixed mouse/human symbols?
                        # In the deduplication logic above:
                        # "sym_map = human_symbol_map if species == 'human' else mouse_symbol_map"
                        # We used consistent mapping.
                        # For display, we can leave species ambiguous or inferred.
                        gid = unit # The 'ID' of this merged entity is the symbol
                        symbol = unit
                        raw_id = unit
                        biotype = "Merged/Functional"
                        
                        # Try to infer species/biotype from map reverse lookup? Too expensive.
                        # Simplification: If it looks like a symbol, we treat it as a symbol key.
                    
                    analyses = gene_to_sets.get(unit, [])
                    rows.append(
                        {
                            "species": species,
                            "gene_id": gid,
                            "gene_symbol": symbol,
                            "biotype": biotype,
                            "overlap_count": len(analyses),
                            "analyses": "; ".join(analyses),
                        }
                    )
                dedup_df = pd.DataFrame(rows)
                dedup_df = dedup_df[
                    ["species", "gene_id", "gene_symbol", "biotype", "overlap_count", "analyses"]
                ]
                csv_bytes = dedup_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download deduplicated genes (CSV)",
                    data=csv_bytes,
                    file_name="deduplicated_degs.csv",
                    mime="text/csv",
                )
                st.caption(
                    "Mouse genes are mapped to human Ensembl orthologs for cross-species de-duplication."
                )
                st.markdown("**DEG contribution breakdown**")
                if dedup_total == 0:
                    st.info("No genes available for contribution breakdown at current cutoffs.")
                else:
                    contrib_df = build_contribution_table(selected_sets, combo_counts, dedup_total)

                    # Enhance with cutoff info from summary_df
                    if not summary_df.empty:
                        cutoff_map = summary_df.set_index("label")[["padj", "log2FC", "tpm"]].to_dict("index")
                        contrib_df["padj"] = contrib_df["set"].map(lambda x: cutoff_map.get(x, {}).get("padj", "-"))
                        contrib_df["log2FC"] = contrib_df["set"].map(lambda x: cutoff_map.get(x, {}).get("log2FC", "-"))
                        contrib_df["tpm"] = contrib_df["set"].map(lambda x: cutoff_map.get(x, {}).get("tpm", "-"))

                    st.markdown("**Per-set contribution to the deduplicated union**")
                    render_table(contrib_df)

                    bar_df = contrib_df[["set", "unique_only", "shared_any"]].copy()
                    bar_df = bar_df.set_index("set")
                    st.markdown("**Unique vs shared contributions (stacked)**")
                    # Use Altair for themed stacked bar chart
                    bar_long = bar_df.reset_index().melt(id_vars="set", var_name="Type", value_name="Count")
                    try:
                        import altair as alt
                        chart = (
                            alt.Chart(bar_long)
                            .mark_bar()
                            .encode(
                                x=alt.X("Count", title="Number of Genes"),
                                y=alt.Y("set", title="Set", sort="-x"),
                                color=alt.Color(
                                    "Type",
                                    scale=alt.Scale(domain=["unique_only", "shared_any"], range=["#C23B75", "#F2A45E"]),
                                    legend=alt.Legend(title="Type"),
                                ),
                                tooltip=["set", "Type", "Count"],
                            )
                        )
                        st.altair_chart(chart, use_container_width=True)
                    except Exception:
                        st.bar_chart(bar_df, stack=True, width="stretch")

                    plot_sources: dict[str, dict[str, object]] = {}
                    for label, df in inhouse_mcd_frames.items():
                        key = f"inhouse_mcd_{slugify(label)}"
                        padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
                        tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
                        plot_label = make_label("MCD (in-house)", "mouse", label)
                        plot_sources[plot_label] = {
                            "df": df,
                            "padj": padj_eff,
                            "log2fc": lfc_eff,
                            "tpm": tpm_eff,
                            "species": "mouse",
                            "genes": raw_sets.get(plot_label, set()),
                        }
                    for label, df in external_mcd_frames.items():
                        key = f"external_mcd_{slugify(label)}"
                        padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
                        tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
                        plot_label = make_label("MCD (external)", "mouse", label)
                        plot_sources[plot_label] = {
                            "df": df,
                            "padj": padj_eff,
                            "log2fc": lfc_eff,
                            "tpm": tpm_eff,
                            "species": "mouse",
                            "genes": raw_sets.get(plot_label, set()),
                        }
                    for dataset, info in patient_data.items():
                        if info.get("error") or info.get("paths") is None:
                            continue
                        key = f"patient_{slugify(dataset)}"
                        nas_padj, nas_lfc = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
                        tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
                        nas_label = make_label("Patient", dataset, "NAS high (upregulated)")
                        plot_sources[nas_label] = {
                            "df": info["nas_high"],
                            "padj": nas_padj,
                            "log2fc": nas_lfc,
                            "tpm": tpm_eff,
                            "species": "human",
                            "genes": raw_sets.get(nas_label, set()),
                        }
                        fib_label = make_label("Patient", dataset, "Fibrosis (upregulated)")
                        plot_sources[fib_label] = {
                            "df": info["fibrosis"],
                            "padj": nas_padj,
                            "log2fc": nas_lfc,
                            "tpm": tpm_eff,
                            "species": "human",
                            "genes": raw_sets.get(fib_label, set()),
                        }
                    plot_labels = [label for label in active_labels if label in plot_sources]
                    missing_plot_labels = [label for label in active_labels if label not in plot_sources]

                # ===== Distributions =====
                with st.expander("Analysis plots", expanded=False):
                    plot_width = 5.2
                    plot_height = 3.2
                    table_height = 260
                    selected_labels = set(active_labels)

                    if plot_labels:
                        st.markdown("**Per-dataset plots (volcano, expression, top genes)**")
                        if missing_plot_labels:
                            st.caption(
                                "Per-dataset plots are available only for base DEG tables; "
                                "skipping: " + ", ".join(missing_plot_labels)
                            )

                        control_left, control_right = st.columns([1, 2], gap="small")
                        with control_left:
                            top_n = st.slider("Top N genes", min_value=5, max_value=50, value=20, step=5)
                        with control_right:
                            rank_by = st.radio(
                                "Rank by",
                                ["log2FC", "-log10(padj)"],
                                horizontal=True,
                                key="top_gene_rank",
                            )

                        tab_labels = [label.split(" | ", 1)[-1] for label in plot_labels]
                        tabs = st.tabs(tab_labels)
                        for tab, label in zip(tabs, plot_labels):
                            source = plot_sources[label]
                            df = source["df"]
                            padj_eff = source["padj"]
                            lfc_eff = source["log2fc"]
                            tpm_eff = source["tpm"]
                            genes = source.get("genes", set())
                            with tab:
                                base_df = df[["log2FoldChange", "padj"]].copy()
                                base_df["log2FoldChange"] = pd.to_numeric(
                                    base_df["log2FoldChange"], errors="coerce"
                                )
                                base_df["padj"] = pd.to_numeric(base_df["padj"], errors="coerce")
                                base_df = base_df.dropna(subset=["log2FoldChange", "padj"])
                                tpm_ok = tpm_mask(df, tpm_eff).reindex(base_df.index, fill_value=True)
                                base_df["pass"] = (
                                    (base_df["padj"] < padj_eff)
                                    & (base_df["log2FoldChange"] > lfc_eff)
                                    & tpm_ok
                                )
                                base_df["neg_log10_padj"] = -np.log10(base_df["padj"].clip(lower=1e-300))

                                col_volcano, col_expr = st.columns(2, gap="small")
                                with col_volcano:
                                    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
                                    ax.scatter(
                                        base_df.loc[~base_df["pass"], "log2FoldChange"],
                                        base_df.loc[~base_df["pass"], "neg_log10_padj"],
                                        s=8,
                                        alpha=0.35,
                                        color="#B9B3AD",
                                        label="Other",
                                    )
                                    ax.scatter(
                                        base_df.loc[base_df["pass"], "log2FoldChange"],
                                        base_df.loc[base_df["pass"], "neg_log10_padj"],
                                        s=10,
                                        alpha=0.7,
                                        color="#C23B75",
                                        label="Pass cutoffs",
                                    )
                                    if padj_eff > 0:
                                        ax.axhline(-np.log10(padj_eff), color="#444444", linestyle="--", linewidth=1)
                                    ax.axvline(lfc_eff, color="#444444", linestyle="--", linewidth=1)
                                    ax.set_xlabel("log2FoldChange")
                                    ax.set_ylabel("-log10(padj)")
                                    ax.set_title("Volcano")
                                    ax.legend(loc="upper right", fontsize=8)
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close(fig)

                                with col_expr:
                                    if "tpm_mean" not in df.columns:
                                        st.info("TPM mean not available for this dataset.")
                                    else:
                                        expr_df = base_df.copy()
                                        expr_df["tpm_mean"] = pd.to_numeric(
                                            df.loc[expr_df.index, "tpm_mean"], errors="coerce"
                                        )
                                        expr_df = expr_df.dropna(subset=["tpm_mean"])
                                        if expr_df.empty:
                                            st.info("Expression plot unavailable (no TPM values).")
                                        else:
                                            expr_df["log2_tpm"] = np.log2(expr_df["tpm_mean"] + 1.0)
                                            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
                                            ax.scatter(
                                                expr_df.loc[~expr_df["pass"], "log2_tpm"],
                                                expr_df.loc[~expr_df["pass"], "log2FoldChange"],
                                                s=8,
                                                alpha=0.35,
                                                color="#B9B3AD",
                                                label="Other",
                                            )
                                            ax.scatter(
                                                expr_df.loc[expr_df["pass"], "log2_tpm"],
                                                expr_df.loc[expr_df["pass"], "log2FoldChange"],
                                                s=10,
                                                alpha=0.7,
                                                color="#C23B75",
                                                label="Pass cutoffs",
                                            )
                                            if tpm_eff > 0:
                                                ax.axvline(
                                                    np.log2(tpm_eff + 1.0),
                                                    color="#444444",
                                                    linestyle="--",
                                                    linewidth=1,
                                                )
                                            ax.axhline(lfc_eff, color="#444444", linestyle="--", linewidth=1)
                                            ax.set_xlabel("log2(TPM + 1)")
                                            ax.set_ylabel("log2FoldChange")
                                            ax.set_title("Expression vs log2FC")
                                            ax.legend(loc="upper right", fontsize=8)
                                            st.pyplot(fig, use_container_width=True)
                                            plt.close(fig)

                                st.markdown("**Top genes**")
                                if not genes:
                                    st.info("No genes available at current cutoffs for this dataset.")
                                else:
                                    subset = df[df["gene_id"].astype(str).isin(genes)].copy()
                                    subset["gene_id"] = subset["gene_id"].astype(str)
                                    subset["log2FoldChange"] = pd.to_numeric(
                                        subset["log2FoldChange"], errors="coerce"
                                    )
                                    subset["padj"] = pd.to_numeric(subset["padj"], errors="coerce")
                                    subset = subset.dropna(subset=["log2FoldChange", "padj"])
                                    if subset.empty:
                                        st.info("No genes available after filtering.")
                                    else:
                                        symbol_map = human_symbol_map if source["species"] == "human" else mouse_symbol_map
                                        subset["gene_symbol"] = subset["gene_id"].map(
                                            lambda gid: symbol_map.get(strip_version(gid), "")
                                        )

                                        if rank_by == "log2FC":
                                            subset["rank_value"] = subset["log2FoldChange"]
                                            x_label = "log2FoldChange"
                                        else:
                                            subset["rank_value"] = -np.log10(subset["padj"].clip(lower=1e-300))
                                            x_label = "-log10(padj)"

                                        subset = subset.sort_values("rank_value", ascending=False).head(top_n)
                                        plot_df = subset.sort_values("rank_value", ascending=True)
                                        labels = [
                                            symbol if symbol else gid
                                            for symbol, gid in zip(plot_df["gene_symbol"], plot_df["gene_id"])
                                        ]

                                        col_plot, col_table = st.columns([2, 1], gap="small")
                                        with col_plot:
                                            fig, ax = plt.subplots(figsize=(plot_width, plot_height))
                                            ax.barh(labels, plot_df["rank_value"], color="#C23B75")
                                            ax.set_xlabel(x_label)
                                            ax.set_title("Top genes")
                                            ax.grid(axis="x", alpha=0.2)
                                            st.pyplot(fig, use_container_width=True)
                                            plt.close(fig)

                                        with col_table:
                                            table_df = plot_df.rename(columns={"log2FoldChange": "log2FC"}).copy()
                                            table_df["padj"] = table_df["padj"].map(
                                                lambda x: f"{x:.2e}" if pd.notna(x) else ""
                                            )
                                            display_cols = ["gene_symbol", "gene_id", "log2FC", "padj"]
                                            if "tpm_mean" in table_df.columns:
                                                display_cols.append("tpm_mean")
                                            st.dataframe(
                                                table_df[display_cols],
                                                hide_index=True,
                                                width="stretch",
                                                height=table_height,
                                            )
                    else:
                        st.info("No base DEG datasets selected for per-dataset plots.")

                    try:
                        import altair as alt
                    except Exception:
                        st.info("Altair not available for distribution plots.")
                    else:
                        # TPM ridgeline (per dataset)
                        tpm_view = st.radio(
                            "TPM view",
                            ["Log2(TPM + 1)", "Linear (p99 clipped)"],
                            index=0,
                            horizontal=True,
                            key="tpm_ridge_view",
                        )
                        tpm_rows = []
                        tpm_order = []
                        def _extend_tpm_rows(label: str, tpm_map: dict[str, float] | None) -> bool:
                            if not tpm_map:
                                return False
                            values = [float(v) for v in tpm_map.values() if pd.notna(v)]
                            if not values:
                                return False
                            tpm_rows.extend({"dataset": label, "tpm": val} for val in values)
                            tpm_order.append(label)
                            return True
                        # In-house MCD (single dataset)
                        if inhouse_mcd_frames:
                            label = "MCD (in-house)"
                            if not _extend_tpm_rows(label, inhouse_tpm_map):
                                df = next(iter(inhouse_mcd_frames.values()))
                                if "tpm_mean" in df.columns:
                                    tpm_rows.extend(
                                        {"dataset": label, "tpm": float(val)} for val in df["tpm_mean"].dropna().values
                                    )
                                    tpm_order.append(label)
                        # External MCD (GSEs)
                        for label, df in external_mcd_frames.items():
                            short = label.split(" ")[0]
                            if not _extend_tpm_rows(short, external_tpm_maps.get(label)):
                                if "tpm_mean" in df.columns:
                                    tpm_rows.extend(
                                        {"dataset": short, "tpm": float(val)} for val in df["tpm_mean"].dropna().values
                                    )
                                    tpm_order.append(short)
                        # Patient datasets (use NAS high table as representative)
                        for dataset, info in patient_data.items():
                            if info.get("error") or info.get("paths") is None:
                                continue
                            if not _extend_tpm_rows(dataset, patient_tpm_maps.get(dataset)):
                                df = info["nas_high"]
                                if "tpm_mean" in df.columns:
                                    tpm_rows.extend(
                                        {"dataset": dataset, "tpm": float(val)} for val in df["tpm_mean"].dropna().values
                                    )
                                    tpm_order.append(dataset)

                        if tpm_rows:
                            tpm_df = pd.DataFrame(tpm_rows)
                            if tpm_view == "Log2(TPM + 1)":
                                tpm_df["tpm_plot"] = np.log2(tpm_df["tpm"] + 1.0)
                                x_title = "log2(TPM + 1)"
                            else:
                                p99 = tpm_df["tpm"].quantile(0.99)
                                tpm_df["tpm_plot"] = tpm_df["tpm"].clip(upper=p99)
                                x_title = "TPM (clipped at p99)"
                            min_val = tpm_df["tpm_plot"].min()
                            max_val = tpm_df["tpm_plot"].max()
                            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
                                st.info("TPM distribution unavailable (insufficient variation).")
                            else:
                                bins = np.linspace(min_val, max_val, 80)
                                ridge_rows = []
                                for dataset in tpm_order:
                                    vals = tpm_df.loc[tpm_df["dataset"] == dataset, "tpm_plot"].dropna().values
                                    if vals.size < 2:
                                        continue
                                    hist, edges = np.histogram(vals, bins=bins, density=True)
                                    centers = (edges[:-1] + edges[1:]) / 2.0
                                    ridge_rows.extend(
                                        {"dataset": dataset, "tpm_plot": float(c), "density": float(d)}
                                        for c, d in zip(centers, hist)
                                    )
                                if not ridge_rows:
                                    st.info("TPM distribution unavailable (insufficient data).")
                                else:
                                    ridge_df = pd.DataFrame(ridge_rows)
                                    ridge_height = max(280, 28 * len(tpm_order))
                                    density = (
                                        alt.Chart(ridge_df)
                                        .mark_area(interpolate="monotone", fillOpacity=0.6, stroke="white", strokeWidth=0.5)
                                        .encode(
                                            x=alt.X("tpm_plot:Q", title=x_title),
                                            y=alt.Y("density:Q", stack=None, title=None, axis=None),
                                            yOffset=alt.YOffset("dataset:N", sort=tpm_order),
                                            color=alt.Color(
                                                "dataset:N",
                                                legend=alt.Legend(title="Dataset", orient="bottom", columns=2),
                                            ),
                                            tooltip=["dataset:N", "tpm_plot:Q", "density:Q"],
                                        )
                                        .properties(height=ridge_height)
                                    )
                                    st.markdown("**TPM distribution (ridgeline)**")
                                    st.altair_chart(density, use_container_width=True)
                        else:
                            st.info("TPM distribution unavailable (no TPM columns found).")

                        # Log2FC distribution (box/violin grid)
                        log_rows = []
                        log_order = []
                        # In-house MCD
                        for label, df in inhouse_mcd_frames.items():
                            if make_label("MCD (in-house)", "mouse", label) not in selected_labels:
                                continue
                            key = f"inhouse_mcd_{slugify(label)}"
                            padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
                            tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
                            series = log2fc_series(df, padj_eff, tpm_eff, log2fc_cutoff=lfc_eff)
                            if not series.empty:
                                name = f"MCD (in-house) | {label}"
                                log_rows.extend({"analysis": name, "group": "MCD (in-house)", "log2FC": float(v)} for v in series.values)
                                log_order.append(name)
                        # External MCD
                        for label, df in external_mcd_frames.items():
                            if make_label("MCD (external)", "mouse", label) not in selected_labels:
                                continue
                            key = f"external_mcd_{slugify(label)}"
                            padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
                            tpm_eff = get_tpm_cutoff(key, tpm_cutoff)
                            series = log2fc_series(df, padj_eff, tpm_eff, log2fc_cutoff=lfc_eff)
                            if not series.empty:
                                name = f"MCD (external) | {label.split(' ')[0]}"
                                log_rows.extend({"analysis": name, "group": "MCD (external)", "log2FC": float(v)} for v in series.values)
                                log_order.append(name)
                        # Patient datasets
                        for dataset, info in patient_data.items():
                            if info.get("error") or info.get("paths") is None:
                                continue
                            key = f"patient_{slugify(dataset)}"
                            nas_padj, nas_lfc = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
                            top_padj = get_topright_padj(key, default_padj=padj_cutoff)
                            tpm_eff = get_tpm_cutoff(key, tpm_cutoff)

                            if make_label("Patient", dataset, "NAS high (upregulated)") in selected_labels:
                                series = log2fc_series(info["nas_high"], nas_padj, tpm_eff, log2fc_cutoff=nas_lfc)
                                if not series.empty:
                                    name = f"Patient | {dataset} NAS high"
                                    log_rows.extend({"analysis": name, "group": "Patient", "log2FC": float(v)} for v in series.values)
                                    log_order.append(name)

                            if make_label("Patient", dataset, "Fibrosis (upregulated)") in selected_labels:
                                series = log2fc_series(info["fibrosis"], nas_padj, tpm_eff, log2fc_cutoff=nas_lfc)
                                if not series.empty:
                                    name = f"Patient | {dataset} Fibrosis (up)"
                                    log_rows.extend({"analysis": name, "group": "Patient", "log2FC": float(v)} for v in series.values)
                                    log_order.append(name)

                            for label, df in (("NAS low", info["nas_low"]), ("Fibrosis", info["fibrosis"])):
                                compare_label = (
                                    "NAS high vs NAS low (top-right)"
                                    if label == "NAS low"
                                    else "NAS high vs Fibrosis (top-right)"
                                )
                                if make_label("Patient", dataset, compare_label) not in selected_labels:
                                    continue
                                series = log2fc_series(df, top_padj, tpm_eff, log2fc_cutoff=nas_lfc)
                                if not series.empty:
                                    name = f"Patient | {dataset} {label}"
                                    log_rows.extend({"analysis": name, "group": "Patient", "log2FC": float(v)} for v in series.values)
                                    log_order.append(name)

                        if log_rows:
                            log_df = pd.DataFrame(log_rows)
                            log_height = max(280, 18 * len(log_order))
                            log_chart = (
                                alt.Chart(log_df)
                                .mark_boxplot(size=12, extent="min-max")
                                .encode(
                                    x=alt.X("log2FC:Q", title="log2FoldChange (filtered by current cutoffs)"),
                                    y=alt.Y("analysis:N", sort=log_order, title=None),
                                    color=alt.Color("group:N", legend=alt.Legend(title="Group")),
                                    tooltip=["analysis:N", "log2FC:Q"],
                                )
                                .properties(height=log_height)
                            )
                            st.markdown("**Log2FC distribution (box/violin grid)**")
                            st.altair_chart(log_chart, use_container_width=True)
                            st.caption(
                                "All log2FC distributions apply padj+TPM+log2FC cutoffs for consistency."
                            )
                        else:
                            st.info("Log2FC distribution unavailable at current cutoffs.")
                        st.markdown("**DEG biotype breakdown**")
                        if BIOTYPE_PATH is None:
                            st.info("Biotype map not found; biotype breakdown is unavailable.")
                        elif dedup_total == 0:
                            st.info("No genes available for biotype breakdown at current cutoffs.")
                        else:
                            biotype_df = load_biotype_map(BIOTYPE_PATH)
                            human_map = (
                                biotype_df[biotype_df["species"] == "human"]
                                .set_index("ensembl_gene_id")["gene_biotype"]
                                .to_dict()
                            )
                            mouse_map = (
                                biotype_df[biotype_df["species"] == "mouse"]
                                .set_index("ensembl_gene_id")["gene_biotype"]
                                .to_dict()
                            )
                            counts: dict[str, int] = {}
                            for gid in union_genes:
                                if gid.startswith("MOUSE:"):
                                    gid_norm = strip_version(gid.split("MOUSE:", 1)[1])
                                    biotype = mouse_map.get(gid_norm, "unknown_mouse")
                                else:
                                    gid_norm = strip_version(gid)
                                    biotype = human_map.get(gid_norm, "unknown_human")
                                counts[biotype] = counts.get(biotype, 0) + 1

                            breakdown = (
                                pd.DataFrame(
                                    [{"biotype": key, "count": value} for key, value in counts.items()]
                                )
                                .sort_values("count", ascending=False)
                                .reset_index(drop=True)
                            )
                            if not breakdown.empty:
                                breakdown["percent_of_total"] = breakdown["count"].map(
                                    lambda x: f"{(x / dedup_total):.2%}"
                                )
                            st.bar_chart(breakdown.set_index("biotype")["count"], width="stretch")
                            render_table(breakdown)
                with st.expander("Cross-species mapping summary", expanded=False):
                    mapping_rows = []
                    for label in selected_sets:
                        source_entry = summary_sets[label]
                        genes = source_entry["genes"]
                        species = source_entry["species"]
                        if species == "mouse":
                            mapped, unmapped, target_count = mapped_counts(genes, mouse_to_human)
                            target_species = "human"
                        else:
                            mapped, unmapped, target_count = mapped_counts(genes, human_to_mouse)
                            target_species = "mouse"
                        total = mapped + unmapped
                        mapping_rows.append(
                            {
                                "set": label,
                                "from_species": species,
                                "to_species": target_species,
                                "input_genes": total,
                                "mapped_genes": mapped,
                                "unmapped_genes": unmapped,
                                "mapped_%": (mapped / total) if total else 0.0,
                                "unique_targets": target_count,
                            }
                        )
                    mapping_df = pd.DataFrame(mapping_rows)
                    if mapping_df.empty:
                        st.info("No mapping summary available for the current selection.")
                    else:
                        mapping_df["mapped_%"] = mapping_df["mapped_%"].map(lambda x: f"{x:.2%}")
                        render_table(mapping_df)
    else:
        st.info("Ortholog map not found; cross-species de-duplication and overlaps are disabled.")


# ===== Detailed sections (Removed) =====


with st.expander("Sanity check (padj=0.1, log2FC=0)"):
    st.write("MCD counts at padj=0.1, log2FC=0")
    mcd_counts_sc = []
    inhouse_sets_sc = []
    for label, df in inhouse_mcd_frames.items():
        gene_set = upregulated_set(df, 0.1, 0.0, tpm_cutoff)
        inhouse_sets_sc.append(gene_set)
        mcd_counts_sc.append({"analysis": f"{label} (in-house)", "count": len(gene_set)})
    if inhouse_sets_sc:
        mcd_counts_sc.append(
            {"analysis": "MCD Week1/2/3 Intersection (in-house)", "count": intersection_count_from_sets(inhouse_sets_sc)}
        )
    for label, df in external_mcd_frames.items():
        gene_set = upregulated_set(df, 0.1, 0.0, tpm_cutoff)
        mcd_counts_sc.append({"analysis": f"{label} (external)", "count": len(gene_set)})
    if mcd_counts_sc:
        render_table(pd.DataFrame(mcd_counts_sc))

    for dataset, info in patient_data.items():
        st.write(f"{dataset} (padj=0.1, log2FC=0)")
        if info.get("error") or info.get("paths") is None:
            st.write("Unavailable due to missing data.")
            continue
        nas_high_df = info["nas_high"]
        nas_low_df = info["nas_low"]
        fibrosis_df = info["fibrosis"]
        patient_counts_sc = pd.DataFrame(
            [
                {
                    "analysis": "NAS high (upregulated)",
                    "count": len(upregulated_set(nas_high_df, 0.1, 0.0, tpm_cutoff)),
                },
                {
                    "analysis": "Fibrosis (upregulated)",
                    "count": len(upregulated_set(fibrosis_df, 0.1, 0.0, tpm_cutoff)),
                },
                {
                    "analysis": "NAS high vs Fibrosis (top-right)",
                    "count": top_right_count(nas_high_df, fibrosis_df, 0.0, padj_cutoff=0.1, tpm_cutoff=tpm_cutoff),
                },
                {
                    "analysis": "NAS high vs NAS low (top-right)",
                    "count": top_right_count(nas_high_df, nas_low_df, 0.0, padj_cutoff=0.1, tpm_cutoff=tpm_cutoff),
                },
            ]
        )
        render_table(patient_counts_sc)

# ===== Overlap Explorer (bottom) =====
st.subheader("Overlap Explorer")
if summary_rows:
    use_dedup = False
    if ORTHOLOG_PATH is not None and dedup_sets:
        use_dedup = st.checkbox("Use cross-species ortholog-mapped sets", value=True)
    sets_for_overlap = dedup_sets if use_dedup and dedup_sets else raw_sets
    
    # Filter sets based on the active selection from the top
    selected_labels = [label for label in sets_for_overlap.keys() if label in active_labels]

    if len(selected_labels) < 1:
        st.info("Select at least one set above to visualize.")
    else:
        selected_sets = {label: sets_for_overlap[label] for label in selected_labels}
        union_size = len(set().union(*selected_sets.values())) if selected_sets else 0
        if union_size == 0:
            st.info("No elements to plot for the selected sets at current cutoffs.")
        else:
            labels = list(selected_sets.keys())
            overlap = pd.DataFrame(index=labels, columns=labels, dtype=int)
            for a in labels:
                for b in labels:
                    overlap.loc[a, b] = len(selected_sets[a] & selected_sets[b])
            st.caption("Pairwise overlap counts")
            render_table(overlap.reset_index().rename(columns={"index": "set"}))

            try:
                import altair as alt
            except Exception:
                st.info("Altair not available for heatmap rendering.")
            else:
                heat = (
                    overlap.reset_index()
                    .melt(id_vars="index", var_name="set_b", value_name="overlap")
                    .rename(columns={"index": "set_a"})
                )
                heatmap = (
                    alt.Chart(heat)
                    .mark_rect()
                    .encode(
                        x=alt.X("set_b:N", title="Set B"),
                        y=alt.Y("set_a:N", title="Set A"),
                        color=alt.Color("overlap:Q", title="Overlap", scale=alt.Scale(range=["#262730", "#C23B75"])),
                        tooltip=["set_a", "set_b", "overlap"],
                    )
                )
                st.altair_chart(heatmap, use_container_width=True)

st.divider()
st.header("Discordance Analysis (Human vs. Mouse)")
st.caption("Identify genes with contradictory expression profiles between Human and Mouse datasets (e.g., UP in Human, DOWN in Mouse).")

master_path = APP_DIR / "master_ortholog_matrix.csv.gz"
if master_path.exists():
    @st.cache_data(show_spinner=False)
    def load_master_matrix_cached():
        return pd.read_csv(master_path)
        
    master_df = load_master_matrix_cached()
    
    with st.expander("Configuration & Filters for Discordance Analysis", expanded=True):
        col_d1, col_d2, col_d3 = st.columns(3)
        p_cut = col_d1.slider("P-value Cutoff (Adjusted)", 0.001, 0.5, 0.1, 0.001, key="disc_p")
        lfc_cut = col_d2.slider("Log2 FC Magnitude Cutoff", 0.0, 3.0, 0.8, 0.1, key="disc_lfc")
        tpm_cut = col_d3.slider("Min TPM Cutoff", 0.0, 10.0, 1.0, 0.5, key="disc_tpm")
        
        run_btn = st.button("Run Discordance Analysis", type="primary")

    if run_btn:
        with st.spinner("Analyzing Contradictions..."):
            filtered_df = filter_master_matrix(master_df, p_cut, lfc_cut, tpm_cut)
            
            n_genes = len(filtered_df)
            st.write(f"### Found {n_genes} contradictory genes")
            
            if n_genes > 0:
                note = f"Cutoffs: padj < {p_cut}, |log2FC| > {lfc_cut}, TPM > {tpm_cut}"
                
                st.write("#### 1. Tug of War")
                fig1 = plot_tug_of_war(filtered_df, note)
                st.pyplot(fig1)
                
                st.write("#### 2. Discordance Barcode")
                fig2 = plot_barcode_heatmap(filtered_df, note)
                st.pyplot(fig2)
                
                st.write("#### 3. Radial Profiles")
                col_r1, col_r2 = st.columns(2)
                
                cand1 = ["TYMP", "FASN", "TM7SF2", "ACSS2", "PPP1R3C"]
                fig3a = plot_radar(filtered_df, cand1, "Human UP - Mouse DOWN")
                col_r1.pyplot(fig3a)
                
                cand2 = ["BEX1", "ATF3", "CXCR4", "CCL2", "C5AR1"]
                fig3b = plot_radar(filtered_df, cand2, "Human DOWN - Mouse UP")
                col_r2.pyplot(fig3b)
                
                with st.expander("View Underlying Data"):
                    st.dataframe(filtered_df.sort_values("Symbol"))
else:
    st.info("Master Matrix for Discordance Analysis not found. Please run offline generation script.")
