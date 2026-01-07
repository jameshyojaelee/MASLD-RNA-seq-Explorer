"""Streamlit app to count upregulated DEGs by adjustable cutoffs.
Usage: streamlit run streamlit_deg_explorer/app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR
for parent in APP_DIR.parents:
    if (parent / "RNA-seq").exists():
        ROOT = parent
        break

DATA_DIR_ENV = os.environ.get("DEG_DATA_DIR")
if DATA_DIR_ENV:
    DATA_DIR = Path(DATA_DIR_ENV).expanduser().resolve()
elif (APP_DIR / "data").exists():
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


@st.cache_data(show_spinner=False)
def find_latest_run(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    run_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name[:1].isdigit()]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda p: p.name)[-1]


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
    if gene_col is None or lfc_col is None or padj_col is None:
        missing = [k for k, v in {"gene": gene_col, "log2FoldChange": lfc_col, "padj": padj_col}.items() if v is None]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.rename(columns={gene_col: "gene_id", lfc_col: "log2FoldChange", padj_col: "padj"})
    df = df[["gene_id", "log2FoldChange", "padj"]]
    df = df.dropna(subset=["log2FoldChange", "padj", "gene_id"])
    return df


def strip_version(gene_id: str) -> str:
    return gene_id.split(".")[0] if isinstance(gene_id, str) else gene_id


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


def upregulated_set(df: pd.DataFrame, padj_cutoff: float, log2fc_cutoff: float) -> set[str]:
    mask = (df["padj"] < padj_cutoff) & (df["log2FoldChange"] > log2fc_cutoff)
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
) -> set[str]:
    nas_mask = (nas_df["padj"] < padj_cutoff) & (nas_df["log2FoldChange"] > nas_log2fc_cutoff)
    other_mask = other_df["padj"] < padj_cutoff
    nas_set = set(nas_df.loc[nas_mask, "gene_id"].astype(str))
    other_set = set(other_df.loc[other_mask, "gene_id"].astype(str))
    return nas_set & other_set


def cross_dataset_top_right_set(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    padj_cutoff: float,
    log2fc_cutoff: float,
) -> set[str]:
    return upregulated_set(df_a, padj_cutoff, log2fc_cutoff) & upregulated_set(df_b, padj_cutoff, log2fc_cutoff)


def top_right_count(
    nas_df: pd.DataFrame,
    other_df: pd.DataFrame,
    nas_log2fc_cutoff: float,
    padj_cutoff: float,
) -> int:
    return len(top_right_set(nas_df, other_df, nas_log2fc_cutoff, padj_cutoff))


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
    if DATA_DIR is None:
        return INHOUSE_MCD_PATHS
    return {label: DATA_DIR / filename for label, filename in BUNDLED_INHOUSE_MCD_FILES.items()}


def get_external_mcd_paths() -> dict[str, Path]:
    if DATA_DIR is None:
        return EXTERNAL_MCD_PATHS
    return {label: DATA_DIR / filename for label, filename in BUNDLED_EXTERNAL_MCD_FILES.items()}


def patient_paths(dataset: str) -> dict[str, Path] | None:
    if DATA_DIR is not None:
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
    latest = find_latest_run(base_dir)
    if latest is None:
        return None
    return {
        "run": latest,
        "nas_high": latest
        / "nas_high"
        / "deseq2_results"
        / "NAS_4plus_vs_NAS_0"
        / "differential_expression.csv",
        "nas_low": latest
        / "nas_low"
        / "deseq2_results"
        / "NAS_1to3_vs_NAS_0"
        / "differential_expression.csv",
        "fibrosis": latest
        / "fibrosis"
        / "deseq2_results"
        / "F1to4_vs_F0"
        / "differential_expression.csv",
    }


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def make_label(group: str, dataset: str, analysis: str) -> str:
    return f"{group} | {dataset} | {analysis}"


def render_table(df: pd.DataFrame) -> None:
    df_display = df.copy()
    for col in ("padj", "log2FC"):
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


st.set_page_config(page_title="MASLD RNA-seq DEG Explorer", layout="wide")

st.title("MASLD RNA-seq DEG Explorer")
st.markdown(
    """
**Overview**
- **Mouse (in-house MCD)**: Week 1/2/3 MCD vs control contrasts from the in-house MCD diet study.
- **Mouse (external MCD GEO)**: **GSE156918** and **GSE205974** Control vs MCD contrasts.
- **Patient (human)**: GEO datasets **GSE130970** and **GSE135251** (NAFLD/NASH/MASLD cohorts).
- This app reports **upregulated DEGs only** (log2FC > cutoff) and lets you adjust padj/log2FC cutoffs globally or per-dataset.
- **Patient (cross-dataset)**: top-right quadrant overlaps using the global padj/log2FC cutoffs.

**Citations / datasets**: GEO **GSE156918**, **GSE205974**, **GSE130970**, **GSE135251**, and in-house MCD RNA-seq (week 1–3 diet contrasts).
"""
)

padj_cutoff = st.slider("Global padj cutoff (MCD + NAS high)", 0.0, 0.2, 0.1, 0.005)
log2fc_cutoff = st.slider("Global log2FC cutoff (upregulated only)", 0.0, 5.0, 0.0, 0.1)

st.caption(
    "Within-dataset top-right uses a dataset-specific padj cutoff (defaults to the global padj unless overridden), "
    "and log2FC cutoff applies only to NAS high. Cross-dataset top-right uses the global padj/log2FC cutoffs."
)

# Load MCD data (in-house + external)
inhouse_mcd_frames = {}
for label, path in get_inhouse_mcd_paths().items():
    if not path.exists():
        st.warning(f"Missing in-house MCD file: {path}")
        continue
    inhouse_mcd_frames[label] = load_mcd_tsv(path)

external_mcd_frames = {}
for label, path in get_external_mcd_paths().items():
    if not path.exists():
        st.warning(f"Missing external MCD file: {path}")
        continue
    external_mcd_frames[label] = load_mcd_tsv(path)

# Load patient data for both datasets
patient_data = {}
for dataset in PATIENT_DATASETS:
    paths = patient_paths(dataset)
    if paths is None:
        patient_data[dataset] = {"paths": None, "error": "No run folder found."}
        continue
    missing = [k for k, p in paths.items() if k not in ("run", "run_label") and not p.exists()]
    if missing:
        patient_data[dataset] = {"paths": paths, "error": f"Missing files: {', '.join(missing)}"}
        continue
    patient_data[dataset] = {
        "paths": paths,
        "nas_high": load_patient_csv(paths["nas_high"]),
        "nas_low": load_patient_csv(paths["nas_low"]),
        "fibrosis": load_patient_csv(paths["fibrosis"]),
    }

# ===== Top summary =====
st.subheader("Overall summary")

summary_rows = []
summary_sets: dict[str, dict[str, object]] = {}
raw_sets: dict[str, set[str]] = {}
dedup_sets: dict[str, set[str]] = {}
active_labels: list[str] = []
cross_dataset_rows: list[dict[str, object]] = []
cross_dataset_sets: dict[str, set[str]] = {}

inhouse_mcd_week_sets = []
for label, df in inhouse_mcd_frames.items():
    key = f"inhouse_mcd_{slugify(label)}"
    padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    gene_set = upregulated_set(df, padj_eff, lfc_eff)
    if label in INHOUSE_MCD_WEEK_LABELS:
        inhouse_mcd_week_sets.append(gene_set)
    summary_rows.append(
        {
            "group": "MCD (in-house)",
            "dataset": "mouse",
            "analysis": label,
            "padj": padj_eff,
            "log2FC": lfc_eff,
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
    gene_set = upregulated_set(df, padj_eff, lfc_eff)
    summary_rows.append(
        {
            "group": "MCD (external)",
            "dataset": "mouse",
            "analysis": label,
            "padj": padj_eff,
            "log2FC": lfc_eff,
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
                "count": 0,
            }
        )
        continue

    key = f"patient_{slugify(dataset)}"
    nas_padj, nas_lfc = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    top_padj = get_topright_padj(key, default_padj=padj_cutoff)

    nas_high_df = info["nas_high"]
    nas_low_df = info["nas_low"]
    fibrosis_df = info["fibrosis"]

    nas_high_set = upregulated_set(nas_high_df, nas_padj, nas_lfc)
    nas_high_vs_fibrosis_set = top_right_set(nas_high_df, fibrosis_df, nas_lfc, padj_cutoff=top_padj)
    nas_high_vs_nas_low_set = top_right_set(nas_high_df, nas_low_df, nas_lfc, padj_cutoff=top_padj)

    summary_rows.append(
        {
            "group": "Patient",
            "dataset": dataset,
            "analysis": "NAS high (upregulated)",
            "padj": nas_padj,
            "log2FC": nas_lfc,
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
            "analysis": "NAS high vs Fibrosis (top-right)",
            "padj": top_padj,
            "log2FC": nas_lfc,
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
            "count": len(nas_high_vs_nas_low_set),
        }
    )
    summary_sets[make_label("Patient", dataset, "NAS high vs NAS low (top-right)")] = {
        "species": "human",
        "genes": nas_high_vs_nas_low_set,
    }

pair_a, pair_b = PATIENT_CROSS_DATASET_PAIR
info_a = patient_data.get(pair_a)
info_b = patient_data.get(pair_b)
if info_a and info_b and not info_a.get("error") and not info_b.get("error") and info_a.get("paths") and info_b.get("paths"):
    for comp_label, comp_key in PATIENT_CROSS_COMPARISONS.items():
        df_a = info_a.get(comp_key)
        df_b = info_b.get(comp_key)
        if df_a is None or df_b is None:
            continue
        cross_set = cross_dataset_top_right_set(df_a, df_b, padj_cutoff, log2fc_cutoff)
        cross_dataset_rows.append(
            {
                "group": "Patient (cross-dataset)",
                "dataset": f"{pair_a} vs {pair_b}",
                "analysis": f"{comp_label} (top-right)",
                "padj": padj_cutoff,
                "log2FC": f">{log2fc_cutoff}",
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
    all_labels = list(raw_sets.keys())
    st.info("Select the datasets/analyses you want to include in totals, overlaps, and contribution breakdowns.")
    active_labels = st.multiselect(
        "Choose datasets/analyses to include",
        options=all_labels,
        default=[],
    )

    if not active_labels:
        st.info("No datasets selected yet. Pick one or more above to populate the downstream analyses.")

    dedup_sets = {}
    if ORTHOLOG_PATH is not None:
        use_one2one = st.checkbox("Use one-to-one orthologs only", value=True)
        include_unmapped = st.checkbox("Include unmapped mouse genes", value=True)
        ortholog_df = load_ortholog_map(ORTHOLOG_PATH)
        mouse_to_human = build_mouse_to_human_map(ortholog_df, one2one_only=use_one2one)
        for label, entry in summary_sets.items():
            species = entry["species"]
            genes = entry["genes"]
            if species == "human":
                dedup_sets[label] = canonicalize_human_set(genes)
            else:
                dedup_sets[label] = canonicalize_mouse_set(genes, mouse_to_human, include_unmapped)
        if dedup_sets and active_labels:
            selected_labels = [label for label in active_labels if label in dedup_sets]
            combo_counts, selected_sets, union_genes = build_combo_counts(dedup_sets, selected_labels)
            dedup_total = len(union_genes)
            st.metric("Total DEGs (deduplicated across mouse+human)", dedup_total)
            st.caption(
                "Mouse genes are mapped to human Ensembl orthologs for cross-species de-duplication."
            )
            st.markdown("**DEG contribution breakdown**")
            if dedup_total == 0:
                st.info("No genes available for contribution breakdown at current cutoffs.")
            else:
                contrib_df = build_contribution_table(selected_sets, combo_counts, dedup_total)
                st.markdown("**Per-set contribution to the deduplicated union**")
                render_table(contrib_df)

                bar_df = contrib_df[["set", "unique_only", "shared_any"]].copy()
                bar_df = bar_df.set_index("set")
                st.markdown("**Unique vs shared contributions (stacked)**")
                st.bar_chart(bar_df, stack=True, width="stretch")

            if BIOTYPE_PATH is None:
                st.info("Biotype map not found; biotype breakdown is unavailable.")
            else:
                with st.expander("DEG biotype breakdown", expanded=False):
                    if dedup_total == 0:
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
    else:
        st.info("Ortholog map not found; cross-species de-duplication and overlaps are disabled.")


# ===== Detailed sections =====
st.subheader("MCD (in-house) — individual cutoffs")

inhouse_detail_rows = []
inhouse_week_sets = []
for label, df in inhouse_mcd_frames.items():
    key = f"inhouse_mcd_{slugify(label)}"
    with st.expander(f"{label} settings", expanded=False):
        st.checkbox("Override cutoffs", key=f"{key}_override")
        if st.session_state.get(f"{key}_override", False):
            st.slider("padj cutoff", 0.0, 0.2, padj_cutoff, 0.005, key=f"{key}_padj")
            st.slider("log2FC cutoff", 0.0, 5.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
        padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
        st.caption(f"Effective: padj < {padj_eff:.3f}, log2FC > {lfc_eff:.2f}")

    padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    gene_set = upregulated_set(df, padj_eff, lfc_eff)
    if label in INHOUSE_MCD_WEEK_LABELS:
        inhouse_week_sets.append(gene_set)
    inhouse_detail_rows.append({"analysis": label, "padj": padj_eff, "log2FC": lfc_eff, "count": len(gene_set)})

if inhouse_week_sets:
    inhouse_detail_rows.append(
        {
            "analysis": "MCD Week1/2/3 Intersection",
            "padj": "varies",
            "log2FC": "varies",
            "count": intersection_count_from_sets(inhouse_week_sets),
        }
    )

if inhouse_detail_rows:
    render_table(pd.DataFrame(inhouse_detail_rows))

st.subheader("MCD (external GEO) — individual cutoffs")

external_detail_rows = []
for label, df in external_mcd_frames.items():
    key = f"external_mcd_{slugify(label)}"
    with st.expander(f"{label} settings", expanded=False):
        st.checkbox("Override cutoffs", key=f"{key}_override")
        if st.session_state.get(f"{key}_override", False):
            st.slider("padj cutoff", 0.0, 0.2, padj_cutoff, 0.005, key=f"{key}_padj")
            st.slider("log2FC cutoff", 0.0, 5.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
        padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
        st.caption(f"Effective: padj < {padj_eff:.3f}, log2FC > {lfc_eff:.2f}")

    padj_eff, lfc_eff = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
    gene_set = upregulated_set(df, padj_eff, lfc_eff)
    external_detail_rows.append({"analysis": label, "padj": padj_eff, "log2FC": lfc_eff, "count": len(gene_set)})

if external_detail_rows:
    render_table(pd.DataFrame(external_detail_rows))

st.subheader("Patient (Human) — individual cutoffs")

for dataset, info in patient_data.items():
    st.markdown(f"#### {dataset}")
    if info.get("paths") is None:
        st.warning(f"No run folder found for {dataset}.")
        continue
    if info.get("error"):
        st.warning(f"{dataset}: {info['error']}")
        continue

    run_label = info["paths"].get("run_label")
    if run_label is None:
        run_label = info["paths"]["run"].name
        st.caption(f"Latest run: {run_label}")
    else:
        st.caption("Data source: bundled")

    key = f"patient_{slugify(dataset)}"
    with st.expander(f"{dataset} settings", expanded=False):
        st.checkbox("Override cutoffs", key=f"{key}_override")
        if st.session_state.get(f"{key}_override", False):
            st.slider("NAS high padj cutoff", 0.0, 0.2, padj_cutoff, 0.005, key=f"{key}_padj")
            st.slider("NAS high log2FC cutoff", 0.0, 5.0, log2fc_cutoff, 0.1, key=f"{key}_lfc")
            st.slider("Top-right padj cutoff", 0.0, 0.2, 0.1, 0.005, key=f"{key}_top_padj")

        nas_padj, nas_lfc = get_cutoffs(key, padj_cutoff, log2fc_cutoff)
        top_padj = get_topright_padj(key, default_padj=padj_cutoff)
        st.caption(
            f"Effective: NAS high padj < {nas_padj:.3f}, NAS high log2FC > {nas_lfc:.2f}; "
            f"Top-right padj < {top_padj:.3f}"
        )

    nas_high_df = info["nas_high"]
    nas_low_df = info["nas_low"]
    fibrosis_df = info["fibrosis"]

    patient_rows = [
        {
            "analysis": "NAS high (upregulated)",
            "padj": nas_padj,
            "log2FC": nas_lfc,
            "count": len(upregulated_set(nas_high_df, nas_padj, nas_lfc)),
        },
        {
            "analysis": "NAS high vs Fibrosis (top-right)",
            "padj": top_padj,
            "log2FC": nas_lfc,
            "count": top_right_count(nas_high_df, fibrosis_df, nas_lfc, padj_cutoff=top_padj),
        },
        {
            "analysis": "NAS high vs NAS low (top-right)",
            "padj": top_padj,
            "log2FC": nas_lfc,
            "count": top_right_count(nas_high_df, nas_low_df, nas_lfc, padj_cutoff=top_padj),
        },
    ]
    render_table(pd.DataFrame(patient_rows))

st.subheader("Patient (cross-dataset) — top-right quadrant")
if not cross_dataset_rows:
    st.info("Cross-dataset top-right sets are unavailable (missing or incomplete patient runs).")
else:
    st.caption(f"Dataset pair: {pair_a} vs {pair_b}")
    cross_df = pd.DataFrame(cross_dataset_rows)
    render_table(cross_df[["analysis", "log2FC", "count"]])
    with st.expander("Cross-dataset top-right gene lists", expanded=False):
        for label, genes in cross_dataset_sets.items():
            st.markdown(f"**{label}** ({len(genes):,} genes)")
            render_table(pd.DataFrame({"gene_id": sorted(genes)}))

with st.expander("Sanity check (padj=0.1, log2FC=0)"):
    st.write("MCD counts at padj=0.1, log2FC=0")
    mcd_counts_sc = []
    inhouse_sets_sc = []
    for label, df in inhouse_mcd_frames.items():
        gene_set = upregulated_set(df, 0.1, 0.0)
        inhouse_sets_sc.append(gene_set)
        mcd_counts_sc.append({"analysis": f"{label} (in-house)", "count": len(gene_set)})
    if inhouse_sets_sc:
        mcd_counts_sc.append(
            {"analysis": "MCD Week1/2/3 Intersection (in-house)", "count": intersection_count_from_sets(inhouse_sets_sc)}
        )
    for label, df in external_mcd_frames.items():
        gene_set = upregulated_set(df, 0.1, 0.0)
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
                    "count": len(upregulated_set(nas_high_df, 0.1, 0.0)),
                },
                {
                    "analysis": "NAS high vs Fibrosis (top-right)",
                    "count": top_right_count(nas_high_df, fibrosis_df, 0.0, padj_cutoff=0.1),
                },
                {
                    "analysis": "NAS high vs NAS low (top-right)",
                    "count": top_right_count(nas_high_df, nas_low_df, 0.0, padj_cutoff=0.1),
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
    if active_labels:
        options = [label for label in sets_for_overlap.keys() if label in active_labels]
    else:
        options = list(sets_for_overlap.keys())
    default_sel = options[: min(6, len(options))]
    selected = st.multiselect("Select sets to visualize", options, default=default_sel)
    if len(selected) < 2:
        st.info("Select at least two sets to visualize overlaps.")
    else:
        selected_sets = {label: sets_for_overlap[label] for label in selected}
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
                        color=alt.Color("overlap:Q", title="Overlap"),
                        tooltip=["set_a", "set_b", "overlap"],
                    )
                )
                st.altair_chart(heatmap, use_container_width=True)
else:
    st.info("Summary data not available yet.")
