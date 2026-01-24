
import pandas as pd
import os

# Paths
# Paths
data_dir = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/streamlit_deg_explorer/data"
app_dir = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/streamlit_deg_explorer"

hoang_raw_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE130970/nas_1_vs_0/results.csv"
govaere_raw_path = "/gpfs/commons/groups/sanjana_lab/Cas13/MASLD_library_design/RNA-seq/patient_RNAseq/analysis/differential_expression/current/nas_threshold_sensitivity/cumulative_nas/GSE135251/nas_1_vs_0/results.csv"

master_path = os.path.join(app_dir, "master_ortholog_matrix.csv.gz")

# Reference values for Hoang
# NAS 1+ vs 0: ~745 DEGs (padj < 0.1, LFC > 0.8)

PADJ_CUT = 0.1
LFC_CUT = 0.8

def check_file(path, name, lfc_col, padj_col):
    print(f"\nChecking {name} at {path}...")
    if not os.path.exists(path):
        print("  File not found.")
        return

    try:
        df = pd.read_csv(path)
        # Normalize columns if needed
        cols = {c.lower(): c for c in df.columns}
        
        # Determine actual column names
        curr_lfc = cols.get(lfc_col.lower())
        curr_padj = cols.get(padj_col.lower())
        
        if not curr_lfc or not curr_padj:
            print(f"  Missing columns. Available: {df.columns.tolist()}")
            return

        # Filter
        df_sig = df[(df[curr_padj] < PADJ_CUT) & (df[curr_lfc] > LFC_CUT)]
        print(f"  Total rows: {len(df)}")
        print(f"  Significant Upregulated (padj < {PADJ_CUT}, LFC > {LFC_CUT}): {len(df_sig)}")
        
        # Check top gene to identify dataset signature
        top_gene = df_sig.sort_values(by=curr_lfc, ascending=False).iloc[0] if not df_sig.empty else None
        if top_gene is not None:
             # Try to get gene id
             gene_col = cols.get('gene_id') or cols.get('human_id') or 'gene_id'
             gid = top_gene.get(gene_col, 'Unknown')
             print(f"  Top Gene: {gid} (LFC={top_gene[curr_lfc]:.2f})")

    except Exception as e:
        print(f"  Error reading file: {e}")

# 1. Raw Hoang
check_file(hoang_raw_path, "Raw Hoang (NAS 1+)", "log2FoldChange", "padj")

# 2. Raw Govaere
check_file(govaere_raw_path, "Raw Govaere (NAS 1+)", "log2FoldChange", "padj")

# 3. Master Matrix Hoang
check_file(master_path, "Master Matrix (Hoang)", "Hoang_et_al_lfc", "Hoang_et_al_padj")

# 4. Master Matrix Govaere
check_file(master_path, "Master Matrix (Govaere)", "Govaere_et_al_lfc", "Govaere_et_al_padj")
