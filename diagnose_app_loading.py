
from pathlib import Path
import pandas as pd
import sys

# Definitions from app.py
ROOT = Path("/gpfs/commons/groups/sanjana_lab/Cas13")
DATA_DIR = ROOT / "streamlit_deg_explorer" / "data"
USE_BUNDLED_RESULTS = True

BUNDLED_PATIENT_FILES = {
    "GSE130970": {
        "nas_1plus": "gse130970_nas_1plus_raw.csv",
        "fibrosis": "gse130970_fibrosis.csv.gz",
    },
    "GSE135251": {
        "nas_1plus": "gse135251_nas_1plus_raw.csv",
        "fibrosis": "gse135251_fibrosis.csv.gz",
    },
}

def patient_paths(dataset: str):
    if USE_BUNDLED_RESULTS:
        files = BUNDLED_PATIENT_FILES.get(dataset)
        if not files:
            return None
        return {
            "run_label": "bundled",
            "nas_1plus": DATA_DIR / files["nas_1plus"],
            "fibrosis": DATA_DIR / files["fibrosis"],
        }
    return None

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    gene_col = cols.get("gene") or cols.get("gene_id")
    lfc_col = cols.get("log2foldchange")
    padj_col = cols.get("padj")
    
    # Debug: Print found columns
    print(f"    Raw columns: {list(df.columns)}")
    print(f"    Mapped: gene={gene_col}, lfc={lfc_col}, padj={padj_col}")
    
    tpm_col = cols.get("tpm_mean") or cols.get("tpm")
    if gene_col is None or lfc_col is None or padj_col is None:
        missing = [k for k, v in {"gene": gene_col, "log2FoldChange": lfc_col, "padj": padj_col}.items() if v is None]
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df

def load_patient_csv(path: Path) -> pd.DataFrame:
    print(f"  Reading {path}...")
    try:
        df = pd.read_csv(path)
        return normalize_df(df)
    except Exception as e:
        print(f"  ERROR reading/normalizing: {e}")
        return None

# Diagnosis Loop
PATIENT_DATASETS = ["GSE130970", "GSE135251"]

print(f"DATA_DIR: {DATA_DIR}")
for dataset in PATIENT_DATASETS:
    print(f"\ndataset: {dataset}")
    paths = patient_paths(dataset)
    if paths is None:
        print("  Paths is None")
        continue
    
    missing = [k for k, p in paths.items() if k not in ("run", "run_label") and not p.exists()]
    if missing:
        print(f"  MISSING FILES CHECK FAILED: {missing}")
        for k, p in paths.items():
            if k not in ("run", "run_label"):
                print(f"    {k}: {p} -> Exists? {p.exists()}")
        continue
    
    # Try loading NAS 1+
    print("  Loading NAS 1+...")
    nas_1plus = load_patient_csv(paths["nas_1plus"])
    if nas_1plus is None:
        print("  FAILED to load NAS 1+")
    else:
        print(f"  SUCCESS. Rows: {len(nas_1plus)}")
