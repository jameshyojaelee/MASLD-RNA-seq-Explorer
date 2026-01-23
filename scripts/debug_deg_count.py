
import os
import sys
from pathlib import Path
import pandas as pd

# Reuse logic
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "streamlit_deg_explorer" / "data"
ORTHOLOG_PATH = DATA_DIR / "mouse_human_orthologs.tsv.gz"

INHOUSE_MCD_FILES = {
    "MCD Week 1": "mcd_week1.tsv.gz",
    "MCD Week 2": "mcd_week2.tsv.gz",
    "MCD Week 3": "mcd_week3.tsv.gz",
    "MCD Week pooled (combined)": "mcd_week_pooled_combined.tsv.gz",
}
EXTERNAL_MCD_FILES = {
    "GSE156918 (external MCD)": "other_mcd_gse156918.tsv.gz",
    "GSE205974 (external MCD)": "other_mcd_gse205974.tsv.gz",
}
PATIENT_FILES = {
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

def load_df(path):
    sep = "\t" if path.name.endswith(".tsv.gz") else ","
    df = pd.read_csv(path, sep=sep)
    cols = {c.lower(): c for c in df.columns}
    rename = {}
    if "gene" in cols: rename[cols["gene"]] = "gene_id"
    elif "gene_id" in cols: rename[cols["gene_id"]] = "gene_id"
    if "log2foldchange" in cols: rename[cols["log2foldchange"]] = "log2FoldChange"
    if "padj" in cols: rename[cols["padj"]] = "padj"
    if "tpm_mean" in cols: rename[cols["tpm_mean"]] = "tpm_mean"
    elif "tpm" in cols: rename[cols["tpm"]] = "tpm_mean"
    return df.rename(columns=rename)

def strip_version(gene_id: str) -> str:
    return str(gene_id).split(".")[0]

def load_ortholog_map(path):
    df = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in df.columns}
    mouse_col = cols.get("mouse_ensembl_gene_id") or cols.get("ensembl_gene_id")
    human_col = cols.get("human_ensembl_gene_id") or cols.get("hsapiens_homolog_ensembl_gene")
    df = df.rename(columns={mouse_col: "m", human_col: "h"})
    df["m"] = df["m"].map(strip_version)
    df["h"] = df["h"].map(strip_version)
    return df

def build_maps(df):
    m2h = {}
    h2m = {}
    for row in df.itertuples():
        m2h.setdefault(row.m, set()).add(row.h)
        h2m.setdefault(row.h, set()).add(row.m)
    return m2h, h2m

def main():
    print("Loading all datasets...")
    datasets = {}
    # Load ALL
    for label, fname in INHOUSE_MCD_FILES.items():
        datasets[label] = {"df": load_df(DATA_DIR / fname), "species": "mouse"}
    for label, fname in EXTERNAL_MCD_FILES.items():
        datasets[label] = {"df": load_df(DATA_DIR / fname), "species": "mouse"}
    for ds_name, files in PATIENT_FILES.items():
        for comp, fname in files.items():
            label = f"Patient | {ds_name} | {comp}"
            datasets[label] = {"df": load_df(DATA_DIR / fname), "species": "human"}
            
    print(f"Loaded {len(datasets)} datasets (Total available in app).")
    
    ortho_df = load_ortholog_map(ORTHOLOG_PATH)
    m2h, h2m = build_maps(ortho_df) # one2one_only=False
    
    # Params
    padj = 0.1
    lfc = 0.8
    tpm = 1.0
    
    # Filter
    union_set = set()
    
    print(f"Calculating Union at TPM={tpm}...")
    
    for label, info in datasets.items():
        df = info["df"]
        mask = (df["padj"] < padj) & (df["log2FoldChange"] > lfc)
        if "tpm_mean" in df.columns:
            mask &= (df["tpm_mean"] >= tpm)
        
        genes = set(df.loc[mask, "gene_id"].astype(str).map(strip_version))
        species = info["species"]
        
        mapped = set()
        if species == "human":
            for g in genes:
                if g in h2m: # Exclude unmapped human
                    mapped.add(g)
        else:
            for g in genes:
                targets = m2h.get(g)
                if targets: # Exclude unmapped mouse
                    mapped.update(targets)
        
        union_set.update(mapped)
        
    print(f"Total Union Size (All datasets): {len(union_set)}")

    # Also calculate the subset union (the 5 requested)
    subset_keys = [
        "MCD Week pooled (combined)", 
        "GSE156918 (external MCD)", 
        "GSE205974 (external MCD)",
        "Patient | GSE130970 | nas_high",
        "Patient | GSE135251 | nas_high"
    ]
    subset_union = set()
    for label, info in datasets.items():
        # Match keys loosely
        is_in_subset = False
        for k in subset_keys:
            if k in label:
                is_in_subset = True
                break
        
        if is_in_subset:
            df = info["df"]
            mask = (df["padj"] < padj) & (df["log2FoldChange"] > lfc)
            if "tpm_mean" in df.columns:
                mask &= (df["tpm_mean"] >= tpm)
            genes = set(df.loc[mask, "gene_id"].astype(str).map(strip_version))
            species = info["species"]
            mapped = set()
            if species == "human":
                for g in genes:
                    if g in h2m: mapped.add(g)
            else:
                for g in genes:
                    targets = m2h.get(g)
                    if targets: mapped.update(targets)
            subset_union.update(mapped)
            
    print(f"Subset Union Size (5 datasets): {len(subset_union)}")

if __name__ == "__main__":
    main()
