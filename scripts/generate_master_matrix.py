
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
DATA_DIR = "data"
OUT_FILE = "master_ortholog_matrix.csv.gz"

# Dataset Mappings
DATASETS = {
    "Hoang_et_al": "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE130970/nas_1_vs_0/results.csv",
    "Govaere_et_al": "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE135251/nas_1_vs_0/results.csv",
    "In_house_MCD": "data/mcd_week_pooled_combined.tsv.gz",
    "External_MCD_1": "data/other_mcd_gse156918.tsv.gz",
    "External_MCD_2": "data/other_mcd_gse205974.tsv.gz"
}

# Mouse datasets need ortholog mapping to align with Human IDs
MOUSE_DATASETS = ["In_house_MCD", "External_MCD_1", "External_MCD_2"]
HUMAN_DATASETS = ["Hoang_et_al", "Govaere_et_al"]

ORTHOLOG_FILE = "data/mouse_human_orthologs.tsv.gz"

# --- Functions ---

def strip_version(gene_id):
    return str(gene_id).strip().split(".")[0]

def load_symbol_mapping():
    # Load from pre-extracted mapping
    try:
        df = pd.read_csv("data/gene_symbol_mapping.csv")
        return dict(zip(df['gene_id'].apply(strip_version), df['gene_symbol']))
    except:
        return {}

def load_tpm_mapping(dataset_name):
    # Map dataset name to old bundled file
    path_map = {
        "Hoang_et_al": "data/gse130970_nas_high.csv.gz",
        "Govaere_et_al": "data/gse135251_nas_high.csv.gz"
    }
    path = path_map.get(dataset_name)
    if not path: return {}
    
    try:
        print(f"Loading TPM reference for {dataset_name} from {path}...")
        df = pd.read_csv(path)
        # Check for tpm col
        cols = {c.lower(): c for c in df.columns}
        tpm_col = cols.get("tpm_mean") or cols.get("tpm")
        gene_col = cols.get("gene_id") or cols.get("gene")
        
        if tpm_col and gene_col:
            return dict(zip(df[gene_col].apply(strip_version), df[tpm_col]))
    except Exception as e:
        print(f"Error loading TPMs: {e}")
    return {}

def load_orthologs():
    print("Loading orthologs...")
    df = pd.read_csv(ORTHOLOG_FILE, sep="\t")
    ortho_map = {} # Human -> {Mouse}
    for _, row in df.iterrows():
        h = strip_version(row["human_ensembl_gene_id"])
        m = strip_version(row["mouse_ensembl_gene_id"])
        if pd.isna(h) or pd.isna(m): continue
        if h not in ortho_map: ortho_map[h] = set()
        ortho_map[h].add(m)
    return ortho_map

def main():
    # 1. Load All Datasets
    dfs = {}
    for name, path in DATASETS.items():
        print(f"Loading {name} from {path}...")
        sep = "\t" if path.endswith(".tsv.gz") else ","
        df = pd.read_csv(path, sep=sep)
        
        # Standardize Columns
        cols = {c.lower(): c for c in df.columns}
        gene_col = cols.get("gene_id") or cols.get("gene")
        padj_col = cols.get("padj") or cols.get("fdr") # some might be fdr?
        lfc_col = cols.get("log2foldchange")
        tpm_col = cols.get("tpm_mean") or cols.get("tpm") or cols.get("tpm_avg")
        
        if not padj_col and "PValue" in df.columns: padj_col = "PValue" 
        
        # Rename for global merge
        rename_map = {
            gene_col: "gene_id_raw",
            lfc_col: f"{name}_lfc",
            padj_col: f"{name}_padj"
        }
        if tpm_col:
            rename_map[tpm_col] = f"{name}_tpm"
            
        df = df.rename(columns=rename_map)
        
        # Normalize Index (Strip Version)
        df["common_id"] = df["gene_id_raw"].apply(strip_version)
        
        # Keep minimal cols
        keep = ["common_id", f"{name}_lfc", f"{name}_padj"]
        if f"{name}_tpm" in df.columns: 
            keep.append(f"{name}_tpm")
        else:
            # Try to load TPMs from reference
            tpm_map = load_tpm_mapping(name)
            if tpm_map:
                print(f"Mapping TPMs for {name} ({len(tpm_map)} genes)...")
                df[f"{name}_tpm"] = df["common_id"].map(tpm_map)
                keep.append(f"{name}_tpm")
        
        if "gene_symbol" in cols:
            df = df.rename(columns={cols["gene_symbol"]: f"{name}_symbol"})
            keep.append(f"{name}_symbol")
        else:
            # Map symbols if missing
            print(f"Mapping symbols for {name}...")
            symbol_map = load_symbol_mapping()
            df[f"{name}_symbol"] = df["common_id"].map(symbol_map)
            keep.append(f"{name}_symbol")
            
        dfs[name] = df[keep].copy()

    # 2. Align Mouse Datasets to Human IDs
    ortho_map = load_orthologs() # Human -> Set(Mouse)
    
    # Master ID list = Union of all Human IDs in Human Datasets
    all_human_ids = set(dfs["Hoang_et_al"]["common_id"]).union(set(dfs["Govaere_et_al"]["common_id"]))
    
    # Create Base DataFrame
    master = pd.DataFrame({"human_id": list(all_human_ids)})
    
    # Merge Human Datasets
    master = pd.merge(master, dfs["Hoang_et_al"], left_on="human_id", right_on="common_id", how="left").drop(columns="common_id")
    master = pd.merge(master, dfs["Govaere_et_al"], left_on="human_id", right_on="common_id", how="left").drop(columns="common_id")
    
    # Merge Mouse Datasets (Complex mapping)
    for m_name in MOUSE_DATASETS:
        print(f"Mapping {m_name} (Mouse) to Human IDs...")
        m_df = dfs[m_name].set_index("common_id")
        
        lfc_list = []
        padj_list = []
        tpm_list = []
        has_tpm = f"{m_name}_tpm" in m_df.columns
        
        for hid in master["human_id"]:
            mids = ortho_map.get(hid, set())
            matches = m_df.index.intersection(list(mids))
            
            if len(matches) == 0:
                lfc_list.append(np.nan)
                padj_list.append(np.nan)
                tpm_list.append(np.nan)
            elif len(matches) == 1:
                mid = matches[0]
                lfc_list.append(m_df.loc[mid, f"{m_name}_lfc"])
                padj_list.append(m_df.loc[mid, f"{m_name}_padj"])
                tpm_list.append(m_df.loc[mid, f"{m_name}_tpm"] if has_tpm else np.nan)
            else:
                # 1:Many - Take best padj
                rows = m_df.loc[matches]
                valid_rows = rows.dropna(subset=[f"{m_name}_padj"])
                if len(valid_rows) == 0:
                    lfc_list.append(np.nan)
                    padj_list.append(np.nan)
                    tpm_list.append(np.nan)
                else:
                    best_idx = valid_rows[f"{m_name}_padj"].idxmin()
                    lfc_list.append(m_df.loc[best_idx, f"{m_name}_lfc"])
                    padj_list.append(m_df.loc[best_idx, f"{m_name}_padj"])
                    tpm_list.append(m_df.loc[best_idx, f"{m_name}_tpm"] if has_tpm else np.nan)
        
        master[f"{m_name}_lfc"] = lfc_list
        master[f"{m_name}_padj"] = padj_list
        master[f"{m_name}_tpm"] = tpm_list

    # Add Symbol (Coalesce from Human datasets)
    master["Symbol"] = master["Govaere_et_al_symbol"].combine_first(master["Hoang_et_al_symbol"])

    print(f"Saving Master Matrix with {len(master)} genes to {OUT_FILE}...")
    master.to_csv(OUT_FILE, index=False, compression="gzip")
    
if __name__ == "__main__":
    main()
