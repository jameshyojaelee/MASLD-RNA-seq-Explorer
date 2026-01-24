
from pathlib import Path
import os

ROOT = Path("/gpfs/commons/groups/sanjana_lab/Cas13")
DATA_DIR = ROOT / "streamlit_deg_explorer" / "data"
USE_BUNDLED_RESULTS = True

BUNDLED_PATIENT_FILES = {
    "GSE130970": {
        "nas_1plus": "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE130970/nas_1_vs_0/results.csv",
        "fibrosis": "gse130970_fibrosis.csv.gz",
    },
    "GSE135251": {
        "nas_1plus": "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE135251/nas_1_vs_0/results.csv",
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

patient_data = {}
PATIENT_DATASETS = ["GSE130970", "GSE135251"]

print(f"DATA_DIR: {DATA_DIR}")
print(f"Exists: {DATA_DIR.exists()}")

for dataset in PATIENT_DATASETS:
    paths = patient_paths(dataset)
    print(f"\nChecking {dataset}...")
    if paths is None:
        print("  Paths is None")
        continue
    
    print(f"  nas_1plus: {paths['nas_1plus']}")
    print(f"  Exists: {paths['nas_1plus'].exists()}")
    
    print(f"  fibrosis: {paths['fibrosis']}")
    print(f"  Exists: {paths['fibrosis'].exists()}")
    
    missing = [k for k, p in paths.items() if k not in ("run", "run_label") and not p.exists()]
    if missing:
        print(f"  MISSING: {missing}")
    else:
        print("  ALL OK")
