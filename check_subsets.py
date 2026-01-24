
import pandas as pd
import os

# Paths
hoang_bundled_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/streamlit_deg_explorer/data/gse130970_nas_high.csv.gz"
govaere_bundled_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/streamlit_deg_explorer/data/gse135251_nas_high.csv.gz"
master_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/streamlit_deg_explorer/master_ortholog_matrix.csv.gz"

# NAS results
hoang_nas1_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE130970/nas_1_vs_0/results.csv"
hoang_nas4_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE130970/nas_4_vs_0/results.csv"

def load_set(path, lfc_col, padj_col, filter=True):
    try:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        curr_lfc = cols.get(lfc_col.lower())
        curr_padj = cols.get(padj_col.lower())
        
        if filter and curr_lfc and curr_padj:
            mask = (df[curr_padj] < 0.1) & (df[curr_lfc] > 0.8)
            genes = df.loc[mask]
        else:
            genes = df
        
        # Get gene IDs (first column usually gene_id or human_id)
        if 'human_id' in df.columns:
            gids = df['human_id']
        elif 'gene_id' in df.columns:
            gids = df['gene_id']
        elif 'gene' in df.columns:
            gids = df['gene']
        elif 'GeneID' in df.columns:
            gids = df['GeneID']
        else:
            gids = df.iloc[:, 0]
            
        return set(gids.astype(str).apply(lambda x: x.split('.')[0]))
    except:
        return set()

# Load sets
h_bundled = load_set(hoang_bundled_path, "log2FoldChange", "padj", filter=True)
h_nas1 = load_set(hoang_nas1_path, "log2FoldChange", "padj", filter=True)
h_nas4 = load_set(hoang_nas4_path, "log2FoldChange", "padj", filter=True)

h_master_raw = pd.read_csv(master_path)
h_master = set(h_master_raw[
    (h_master_raw['Hoang_et_al_padj'] < 0.1) & 
    (h_master_raw['Hoang_et_al_lfc'] > 0.8)
]['human_id'].astype(str).apply(lambda x: x.split('.')[0]))


print(f"Bundled Hoang (filtered): {len(h_bundled)}")
print(f"Hoang NAS 1+ (filtered): {len(h_nas1)}")
print(f"Hoang NAS 4+ (filtered): {len(h_nas4)}")
print(f"Master Hoang (filtered): {len(h_master)}")

print("\n--- Hoang Inclusions ---")
print(f"Bundled inside NAS 1+? {h_bundled.issubset(h_nas1)} (Overlap: {len(h_bundled & h_nas1)})")
print(f"Bundled inside NAS 4+? {h_bundled.issubset(h_nas4)} (Overlap: {len(h_bundled & h_nas4)})")
print(f"Bundled inside Master? {h_bundled.issubset(h_master)} (Overlap: {len(h_bundled & h_master)})")
print(f"Master inside NAS 4+? {len(h_master & h_nas4)}/{len(h_master)}")

