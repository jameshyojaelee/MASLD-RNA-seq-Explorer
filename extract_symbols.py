
import pandas as pd

# Load old bundled file to get symbol mapping
df = pd.read_csv("data/gse130970_nas_high.csv.gz")
print("Columns:", df.columns)
if 'gene_symbol' in df.columns:
    mapping = df[['gene_id', 'gene_symbol']].dropna().drop_duplicates()
    mapping.to_csv("data/gene_symbol_mapping.csv", index=False)
    print(f"Saved mapping with {len(mapping)} genes.")
else:
    print("No gene_symbol column found.")
