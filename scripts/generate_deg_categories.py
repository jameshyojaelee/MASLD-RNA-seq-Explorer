
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
FILE_130 = "data/gse130970_nas_high.csv.gz"
FILE_135 = "data/gse135251_nas_high.csv.gz"
OUT_FILE = "gene_categories.csv"

def strip_version(gene_id):
    return str(gene_id).strip().split(".")[0]

def get_status(row, prefix, padj_cut=0.05):
    padj = row[f"{prefix}_padj"]
    lfc = row[f"{prefix}_lfc"]
    
    if pd.isna(padj) or padj >= padj_cut:
        return "NS"
    if lfc > 0:
        return "UP"
    if lfc < 0:
        return "DOWN"
    return "NS" # Should not happen if sig

# Load
print("Loading datasets...")
df130 = pd.read_csv(FILE_130)
df135 = pd.read_csv(FILE_135)

# Normalize ID
df130["gene_id_clean"] = df130["gene_id"].apply(strip_version)
df135["gene_id_clean"] = df135["gene_id"].apply(strip_version)

# Select Cols and Rename
cols = ["gene_id_clean", "gene_symbol", "log2FoldChange", "padj"]
d1 = df130[cols].rename(columns={"log2FoldChange": "GSE130_lfc", "padj": "GSE130_padj", "gene_symbol": "symbol_130"})
d2 = df135[cols].rename(columns={"log2FoldChange": "GSE135_lfc", "padj": "GSE135_padj", "gene_symbol": "symbol_135"})

# Merge
print("Merging...")
merged = pd.merge(d1, d2, on="gene_id_clean", how="outer")

# Coalesce symbol
merged["gene_symbol"] = merged["symbol_135"].combine_first(merged["symbol_130"])
merged = merged.drop(columns=["symbol_130", "symbol_135"])

# Determine Status
merged["status_130"] = merged.apply(lambda r: get_status(r, "GSE130"), axis=1)
merged["status_135"] = merged.apply(lambda r: get_status(r, "GSE135"), axis=1)

# Categorize
def categorize(r):
    s130 = r["status_130"]
    s135 = r["status_135"]
    
    if s130 == "UP" and s135 == "UP":
        return "Shared_Up"
    
    if (s130 == "UP" and s135 == "DOWN") or (s130 == "DOWN" and s135 == "UP"):
        return "Contradictory"
    
    if (s130 == "UP" and s135 == "NS") or (s130 == "NS" and s135 == "UP"):
        return "Unique_Up"
        
    if s130 == "DOWN" and s135 == "DOWN":
        return "Shared_Down"
        
    if (s130 == "DOWN" and s135 == "NS") or (s130 == "NS" and s135 == "DOWN"):
        return "Unique_Down"
        
    return "Not_Significant"

merged["Category"] = merged.apply(categorize, axis=1)

# Filter for relevant categories (App Context)
# App includes anything where at least ONE is UP.
relevant_cats = ["Shared_Up", "Unique_Up", "Contradictory"]
final_df = merged[merged["Category"].isin(relevant_cats)].copy()

# Add logic breakdown
def distinct_logic(r):
    cat = r["Category"]
    if cat == "Contradictory":
        if r["status_135"] == "UP": return "UP_in_135_DOWN_in_130"
        return "UP_in_130_DOWN_in_135"
    if cat == "Unique_Up":
        if r["status_135"] == "UP": return "Unique_to_135"
        return "Unique_to_130"
    return "Shared"

final_df["Sub_Category"] = final_df.apply(distinct_logic, axis=1)

# Output Summary
print("\nSummary of Categories:")
summary = final_df["Category"].value_counts()
print(summary)

print("\nDetailed Breakdown:")
print(final_df["Sub_Category"].value_counts())

# Save
print(f"\nSaving {len(final_df)} genes to {OUT_FILE}...")
final_df.to_csv(OUT_FILE, index=False)
