#!/usr/bin/env python3
"""Generate TPM distribution ridgeline plots for bundled RNA-seq datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

ORDERED_DATASETS: list[tuple[str, str]] = [
    ("MCD Week 1", "mcd_week1.tsv.gz"),
    ("MCD Week 2", "mcd_week2.tsv.gz"),
    ("MCD Week 3", "mcd_week3.tsv.gz"),
    ("MCD Week pooled (combined)", "mcd_week_pooled_combined.tsv.gz"),
    ("GSE156918 (external MCD)", "other_mcd_gse156918.tsv.gz"),
    ("GSE205974 (external MCD)", "other_mcd_gse205974.tsv.gz"),
    ("GSE130970 NAS high", "gse130970_nas_high.csv.gz"),
    ("GSE130970 NAS low", "gse130970_nas_low.csv.gz"),
    ("GSE130970 Fibrosis", "gse130970_fibrosis.csv.gz"),
    ("GSE135251 NAS high", "gse135251_nas_high.csv.gz"),
    ("GSE135251 NAS low", "gse135251_nas_low.csv.gz"),
    ("GSE135251 Fibrosis", "gse135251_fibrosis.csv.gz"),
]

ALIAS_LABELS = {
    "MCD Week pooled": "MCD Week pooled (combined)",
    "MCD Week pooled (combined)": "MCD Week pooled (combined)",
    "GSE156918": "GSE156918 (external MCD)",
    "GSE156918 (external MCD)": "GSE156918 (external MCD)",
    "GSE205974": "GSE205974 (external MCD)",
    "GSE205974 (external MCD)": "GSE205974 (external MCD)",
    "GSE130970 NAS high": "GSE130970 NAS high",
    "GSE135251 NAS high": "GSE135251 NAS high",
}

RAW_MCD_COUNTS = ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "counts" / "featurecounts" / "gene_counts.txt"
RAW_MCD_METADATA = ROOT / "RNA-seq" / "in-house_MCD_RNAseq" / "metadata" / "samples.tsv"

RAW_EXTERNAL_MCD = {
    "GSE156918 (external MCD)": {
        "counts": ROOT
        / "RNA-seq"
        / "other_MCD_RNAseq"
        / "GSE156918"
        / "counts"
        / "featurecounts"
        / "gene_counts.txt",
        "metadata": ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE156918" / "metadata" / "samples.tsv",
    },
    "GSE205974 (external MCD)": {
        "counts": ROOT
        / "RNA-seq"
        / "other_MCD_RNAseq"
        / "GSE205974"
        / "counts"
        / "featurecounts"
        / "gene_counts.txt",
        "metadata": ROOT / "RNA-seq" / "other_MCD_RNAseq" / "GSE205974" / "metadata" / "samples.tsv",
    },
}

RAW_PATIENT_DATASETS = {
    "GSE130970 NAS high": "GSE130970",
    "GSE130970 NAS low": "GSE130970",
    "GSE130970 Fibrosis": "GSE130970",
    "GSE135251 NAS high": "GSE135251",
    "GSE135251 NAS low": "GSE135251",
    "GSE135251 Fibrosis": "GSE135251",
}


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


def load_tpm_from_featurecounts(path: Path, sample_ids: Iterable[str] | None) -> pd.Series:
    lengths, counts = read_featurecounts_counts(path)
    counts = _subset_featurecounts_counts(counts, sample_ids)
    return compute_tpm_mean(counts, lengths)


def load_tpm_from_counts_matrix(
    counts_path: Path, length_source: Path, sample_ids: Iterable[str] | None
) -> pd.Series:
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
        raise ValueError(f"Missing Geneid/Length columns in {length_source}")
    lengths = pd.to_numeric(lengths_df[length_col], errors="coerce")
    lengths.index = lengths_df[gene_col].astype(str)
    return compute_tpm_mean(counts, lengths)


def load_raw_tpm(label: str) -> pd.Series:
    if label in RAW_PATIENT_DATASETS:
        dataset = RAW_PATIENT_DATASETS[label]
        counts_dir = ROOT / "RNA-seq" / "patient_RNAseq" / "results" / dataset / "counts"
        counts_path = counts_dir / "gene_counts_matrix.txt"
        length_source = next(counts_dir.glob("individual/*_counts.txt"), None)
        if length_source is None:
            raise FileNotFoundError(f"No length source found for {dataset} in {counts_dir}/individual")
        sample_ids = load_masld_sample_ids(dataset)
        return load_tpm_from_counts_matrix(counts_path, length_source, sample_ids)

    if label in RAW_EXTERNAL_MCD:
        source = RAW_EXTERNAL_MCD[label]
        sample_ids = load_mcd_sample_ids(source["metadata"])
        return load_tpm_from_featurecounts(source["counts"], sample_ids)

    if label.startswith("MCD Week"):
        sample_ids = load_mcd_sample_ids(RAW_MCD_METADATA)
        return load_tpm_from_featurecounts(RAW_MCD_COUNTS, sample_ids)

    raise ValueError(f"No raw TPM source configured for label: {label}")


def read_tpm_mean(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(path)
    sep = "\t" if path.suffixes[-2] == ".tsv" else ","
    cols = pd.read_csv(path, sep=sep, nrows=0).columns
    if "tpm_mean" not in cols:
        raise ValueError(f"tpm_mean column not found in {path}")
    df = pd.read_csv(path, sep=sep, usecols=["tpm_mean"])
    return pd.to_numeric(df["tpm_mean"], errors="coerce")


def make_ridgeline(
    datasets: list[tuple[str, pd.Series]],
    xmax: float,
    output_path: Path,
) -> None:
    labels = [label for label, _ in datasets]
    values = [vals.dropna().astype(float) for _, vals in datasets]

    clipped_counts = []
    clipped_values = []
    for vals in values:
        clipped = (vals > xmax).sum()
        clipped_counts.append(int(clipped))
        clipped_values.append(vals.clip(upper=xmax))

    bins = np.linspace(0.0, xmax, 80)
    histograms = []
    max_density = 0.0
    for vals in clipped_values:
        if vals.empty:
            hist = np.zeros(len(bins) - 1)
        else:
            hist, _ = np.histogram(vals, bins=bins, density=True)
        max_density = max(max_density, float(hist.max()) if hist.size else 0.0)
        histograms.append(hist)

    if max_density <= 0:
        raise RuntimeError("No TPM variation available for plotting.")

    scale = 0.9 / max_density
    centers = (bins[:-1] + bins[1:]) / 2.0

    n = len(labels)
    height = max(8.0, 1.1 * n + 3.0) * 1.5
    fig, ax = plt.subplots(figsize=(11, height))

    cmap = plt.get_cmap("tab20")
    offsets = list(range(n - 1, -1, -1))

    for idx, (label, hist) in enumerate(zip(labels, histograms)):
        offset = offsets[idx]
        color = cmap(idx % cmap.N)
        ridge = hist * scale
        ax.fill_between(centers, offset, ridge + offset, color=color, alpha=0.7)
        ax.plot(centers, ridge + offset, color="white", linewidth=0.6)

    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.5, n - 0.1)
    ax.set_yticks(offsets)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"TPM (values > {xmax:g} clipped)")
    ax.set_ylabel("")
    ax.set_title("TPM distribution (ridgeline)")
    ax.axvline(xmax, color="black", linestyle="--", linewidth=0.8)
    fig.text(
        0.99,
        0.01,
        f"x-axis capped at {xmax:g} TPM for visualization (values > {xmax:g} clipped)",
        ha="right",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.05, 0.03, 0.98, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    summary_path = output_path.with_suffix(".clipped_counts.tsv")
    summary_df = pd.DataFrame(
        {
            "dataset": labels,
            "num_values": [int(v.size) for v in values],
            "num_clipped_over_xmax": clipped_counts,
        }
    )
    summary_df.to_csv(summary_path, sep="\t", index=False)


def _default_violin_output(ridge_output: Path) -> Path:
    stem = ridge_output.stem
    if "ridgeline" in stem:
        stem = stem.replace("ridgeline", "violin_log2")
    else:
        stem = f"{stem}_violin_log2"
    return ridge_output.with_name(stem + ridge_output.suffix)


def make_violin_plot(
    datasets: list[tuple[str, pd.Series]],
    output_path: Path,
) -> None:
    labels = [label for label, _ in datasets]
    values = [vals.dropna().astype(float) for _, vals in datasets]
    log2_values = [np.log2(vals.clip(lower=0) + 1.0) for vals in values]

    n = len(labels)
    height = max(5.0, 0.6 * n + 2.5)
    fig, ax = plt.subplots(figsize=(10, height))

    positions = np.arange(1, n + 1)
    violins = ax.violinplot(
        log2_values,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.9,
    )
    for body in violins["bodies"]:
        body.set_facecolor("#009688")
        body.set_edgecolor("white")
        body.set_alpha(0.45)

    box = ax.boxplot(
        log2_values,
        positions=positions,
        widths=0.25,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#00796B")
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
    for element in ("medians", "whiskers", "caps"):
        for item in box[element]:
            item.set_color("black")

    rng = np.random.default_rng(0)
    for pos, vals in zip(positions, log2_values):
        if vals.empty:
            continue
        jitter = rng.normal(loc=pos, scale=0.07, size=len(vals))
        ax.scatter(jitter, vals, s=6, color="#00796B", alpha=0.35, linewidths=0)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("log2(TPM + 1)")
    ax.set_title("TPM distribution (log2 TPM + 1)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TPM distribution ridgeline plots.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data",
        help="Directory containing bundled DEG tables with tpm_mean.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plots" / "tpm_distribution_ridgeline_xcap5.png",
        help="Output PNG path.",
    )
    parser.add_argument("--xmax", type=float, default=5.0, help="X-axis cap for TPM values.")
    parser.add_argument(
        "--source",
        choices=["bundled", "raw"],
        default="bundled",
        help="TPM source: bundled tpm_mean columns or raw counts (disease-only samples).",
    )
    parser.add_argument(
        "--violin-output",
        type=Path,
        default=None,
        help="Output PNG path for log2(TPM+1) violin plot (defaults to a name based on --output).",
    )
    parser.add_argument(
        "--no-violin",
        action="store_true",
        help="Skip generating the log2(TPM+1) violin plot.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Optional list of dataset labels or aliases to include.",
    )
    args = parser.parse_args()

    allowed_labels = {label for label, _ in ORDERED_DATASETS}
    selected_labels = None
    if args.only:
        normalized = []
        for label in args.only:
            key = ALIAS_LABELS.get(label, label)
            if key not in allowed_labels:
                raise SystemExit(f"Unknown dataset label: {label}")
            normalized.append(key)
        selected_labels = set(normalized)

    datasets: list[tuple[str, pd.Series]] = []
    missing = []
    for label, filename in ORDERED_DATASETS:
        if selected_labels is not None and label not in selected_labels:
            continue
        try:
            if args.source == "raw":
                vals = load_raw_tpm(label)
            else:
                path = args.data_dir / filename
                vals = read_tpm_mean(path)
        except Exception as exc:
            missing.append((label, str(exc)))
            continue
        datasets.append((label, vals))

    if not datasets:
        raise SystemExit("No datasets with TPM values found.")

    make_ridgeline(datasets, args.xmax, args.output)
    if not args.no_violin:
        violin_output = args.violin_output or _default_violin_output(args.output)
        make_violin_plot(datasets, violin_output)

    if missing:
        print("Skipped datasets:")
        for label, reason in missing:
            print(f"- {label}: {reason}")


if __name__ == "__main__":
    main()
