import sys
import os
from pathlib import Path
import pandas as pd

# Mock streamlit to avoid import errors
import types
sys.modules["streamlit"] = types.ModuleType("streamlit")
sys.modules["streamlit"].cache_data = lambda **kwargs: lambda f: f

# Now we can import parts of app or just copy logic. 
# Copying logic is safer to avoid unrelated import issues with plotting libs.

ROOT = Path("/gpfs/commons/groups/sanjana_lab/Cas13")
PATIENT_DATASETS = ["GSE130970", "GSE135251"]

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

def check_local_available():
    print("Checking local availability...")
    for dataset in PATIENT_DATASETS:
        base_dir = ROOT / "RNA-seq" / "patient_RNAseq" / "results" / dataset / "deseq2_results"
        required = [
            Path("nas_high/deseq2_results/NAS_4plus_vs_NAS_0/differential_expression.csv"),
            Path("nas_low/deseq2_results/NAS_1to3_vs_NAS_0/differential_expression.csv"),
            Path("fibrosis_strict/deseq2_results/Fibrosis_vs_Healthy/differential_expression.csv"),
        ]
        latest = _latest_run_with_files(base_dir, required)
        print(f"  {dataset}: Latest complete run = {latest}")
        if latest is None:
            return False
    return True

def verify_bundled_loading():
    print("\nVerifying bundled data loading...")
    data_dir = ROOT / "streamlit_deg_explorer/data"
    files = [
        "gse135251_fibrosis.csv.gz",
        "gse135251_nas_high.csv.gz", 
        "gse130970_fibrosis.csv.gz"
    ]
    for f in files:
        path = data_dir / f
        if not path.exists():
            print(f"  [MISSING] {f}")
            continue
        try:
            df = pd.read_csv(path)
            # Check for critical columns
            required = {"gene_id", "log2FoldChange", "padj", "tpm_mean"}
            missing = required - set(df.columns)
            if missing:
                print(f"  [INVALID] {f} missing {missing}")
            else:
                # Check TPM
                zero_tpm = (df["tpm_mean"] == 0).sum()
                total = len(df)
                print(f"  [OK] {f}: {len(df)} rows, {zero_tpm} zero TPMs ({zero_tpm/total:.1%})")
        except Exception as e:
            print(f"  [ERROR] {f}: {e}")

if __name__ == "__main__":
    local_avail = check_local_available()
    print(f"\nLocal results available? {local_avail}")
    
    if not local_avail:
        print("-> App will use BUNDLED mode (EXPECTED)")
        verify_bundled_loading()
    else:
        print("-> App will use LOCAL mode (UNEXPECTED for split runs)")
