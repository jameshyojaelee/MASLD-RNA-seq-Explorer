
import pandas as pd
import numpy as np
from pathlib import Path

# --- Configuration ---
DATA_DIR = "data"
OUT_FILE = "deg_contradictions_5datasets.csv"

# Dataset Mappings
# Dataset Mappings
DATASETS = {
    "Hoang_et_al": "data/gse130970_nas_high.csv.gz",
    "Govaere_et_al": "data/gse135251_nas_high.csv.gz",
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

def load_orthologs():
    print("Loading orthologs...")
    df = pd.read_csv(ORTHOLOG_FILE, sep="\t")
    # We need Mouse ID -> Human ID mapping
    # One mouse gene can map to multiple human genes and vice versa.
    # For simplicity in "Alignment", if we are grounding in Human IDs (Union of all Humans),
    # we map Mouse -> Human.
    # The file has: gene_id_human, gene_id_mouse, ...
    
    # Create dictionary: MouseID -> set(HumanIDs)
    # But for a flat table, we might Explode? 
    # Let's pivot to: HumanID as index.
    # Actually, we want to start with the UNION of all Human IDs found in Human Datasets.
    # Then for Mouse Datasets, we map their rows to "Inferred Human Rows".
    
    ortho_map = {} # Human -> {Mouse}
    for _, row in df.iterrows():
        h = strip_version(row["human_ensembl_gene_id"])
        m = strip_version(row["mouse_ensembl_gene_id"])
        if pd.isna(h) or pd.isna(m): continue
        if h not in ortho_map: ortho_map[h] = set()
        ortho_map[h].add(m)
    return ortho_map

def get_status(row, prefix, padj_col, lfc_col, tpm_col=None, padj_cut=0.1, lfc_cut=0.8, tpm_cut=1.0):
    try:
        padj = row[padj_col]
        lfc = row[lfc_col]
        tpm = row[tpm_col] if tpm_col and tpm_col in row else np.nan
        
        # Check P-value
        if pd.isna(padj) or padj >= padj_cut:
            return "NS"
            
        # Check LFC Magnitude
        if pd.isna(lfc) or abs(lfc) <= lfc_cut:
            return "NS"
            
        # Check TPM (if applicable/available)
        # Assuming if TPM column exists, we must enforce it. If nan, fail? 
        # For this request, we strictly want TPM >= 1.0. 
        if tpm_col and pd.notna(tpm) and tpm < tpm_cut:
            return "NS"
            
        if lfc > 0:
            return "UP"
        if lfc < 0:
            return "DOWN"
        return "NS"
    except KeyError:
        return "NS"
        return "NS"

# --- Main Logic ---

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
        tpm_col = cols.get("tpm_mean") or cols.get("tpm")
        
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
        if f"{name}_tpm" in df.columns: keep.append(f"{name}_tpm")
        if "gene_symbol" in cols:
            df = df.rename(columns={cols["gene_symbol"]: f"{name}_symbol"})
            keep.append(f"{name}_symbol")
            
        dfs[name] = df[keep].copy()

    # 2. Align Mouse Datasets to Human IDs
    # Strategy: 
    # Create a Master Human Table.
    # For GSE130/GSE135, directly merge on EnsemblID.
    # For Mouse, map HumanID -> MouseID -> Lookup in MouseTable.
    
    ortho_map = load_orthologs() # Human -> Set(Mouse)
    
    # Initialize Master with Union of HUMAN available genes
    # (Since we care about library design for Human targets mostly? Or Mouse targets?)
    # "I wanna look at all these contradictory cases..."
    # Usually we start with Human DEGs and verify in Mouse.
    
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
        
        # We need to construct columns for the Master table
        lfc_list = []
        padj_list = []
        tpm_list = []
        has_tpm = f"{m_name}_tpm" in m_df.columns
        
        # Lookup helper
        for hid in master["human_id"]:
            mids = ortho_map.get(hid, set())
            
            # Find matching rows in Mouse DF
            matches = m_df.index.intersection(list(mids))
            
            if len(matches) == 0:
                lfc_list.append(np.nan)
                padj_list.append(np.nan)
                tpm_list.append(np.nan)
            elif len(matches) == 1:
                # 1:1 mapping (simplest)
                mid = matches[0]
                lfc_list.append(m_df.loc[mid, f"{m_name}_lfc"])
                padj_list.append(m_df.loc[mid, f"{m_name}_padj"])
                tpm_list.append(m_df.loc[mid, f"{m_name}_tpm"] if has_tpm else np.nan)
            else:
                # 1:Many (One Human -> Many Mouse)
                # Logic: Is ANY mouse ortholog significant?
                # We want to detect Contradictions.
                # If ANY is UP -> UP. If ANY is DOWN -> DOWN.
                # If Both UP and DOWN exist in paralogs -> Conflicting Mouse Signal?
                
                rows = m_df.loc[matches]
                # Relaxed sig check for projection
                sigs = rows[rows[f"{m_name}_padj"] < 0.1]
                
                if len(sigs) == 0:
                    # None significant -> Take best p-value
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
                else:
                    # Some Significant -> Take best p-value among sigs
                    best_idx = sigs[f"{m_name}_padj"].idxmin()
                    lfc_list.append(m_df.loc[best_idx, f"{m_name}_lfc"])
                    padj_list.append(m_df.loc[best_idx, f"{m_name}_padj"])
                    tpm_list.append(m_df.loc[best_idx, f"{m_name}_tpm"] if has_tpm else np.nan)
        
        master[f"{m_name}_lfc"] = lfc_list
        master[f"{m_name}_padj"] = padj_list
        master[f"{m_name}_tpm"] = tpm_list

    # 3. Determine Global Status
    datasets_ordered = ["Hoang_et_al", "Govaere_et_al", "In_house_MCD", "External_MCD_1", "External_MCD_2"]
    
    for d in datasets_ordered:
        master[f"Status_{d}"] = master.apply(lambda r: get_status(r, d, f"{d}_padj", f"{d}_lfc", f"{d}_tpm", padj_cut=0.1, lfc_cut=0.8, tpm_cut=1.0), axis=1)

    # 4. Identify Contradictions
    # A gene is contradictory if it has at least one "UP" and at least one "DOWN" across ANY of the 5 datasets.
    
    def check_global_contradiction(r):
        statuses = [r[f"Status_{d}"] for d in datasets_ordered]
        has_up = "UP" in statuses
        has_down = "DOWN" in statuses
        
        if has_up and has_down:
            return True
        return False

    master["Is_Contradictory"] = master.apply(check_global_contradiction, axis=1)
    
    # Filter
    contradictory_df = master[master["Is_Contradictory"]].copy()
    
    # Add Symbol (Coalesce from Human datasets)
    contradictory_df["Symbol"] = contradictory_df["Govaere_et_al_symbol"].combine_first(contradictory_df["Hoang_et_al_symbol"])
    
    # Calculate a "Contradiction Score" or Description
    def describe(r):
        ups = [d for d in datasets_ordered if r[f"Status_{d}"] == "UP"]
        downs = [d for d in datasets_ordered if r[f"Status_{d}"] == "DOWN"]
        return f"UP in {len(ups)} ({','.join(ups)}) | DOWN in {len(downs)} ({','.join(downs)})"

    contradictory_df["Pattern"] = contradictory_df.apply(describe, axis=1)
    
    # Cleanup Columns for Export
    # ID, Symbol, Pattern, then the LFC/Padj/Status cols
    base_cols = ["human_id", "Symbol", "Pattern"]
    data_cols = []
    for d in datasets_ordered:
        data_cols.extend([f"Status_{d}", f"{d}_lfc", f"{d}_padj"])
        
    final_cols = base_cols + data_cols
    out_df = contradictory_df[final_cols]
    
    # Sort by number of active datasets involved?
    
    print(f"\nFound {len(out_df)} Globally Contradictory Genes.")
    print("Saving...")
    out_df.to_csv(OUT_FILE, index=False)
    
    # Preview
    print("\nSample Contradictions (Top 5):")
    print(out_df[["Symbol", "Pattern"]].head().to_string(index=False))

if __name__ == "__main__":
    main()
