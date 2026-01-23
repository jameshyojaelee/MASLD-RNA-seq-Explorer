
import pandas as pd

df = pd.read_csv("master_ortholog_matrix.csv.gz")
print("Columns:", df.columns.tolist())

# Check for TPMs
for col in ["Hoang_et_al_tpm", "Govaere_et_al_tpm"]:
    if col in df.columns:
        print(f"{col} - NaNs: {df[col].isna().sum()}/{len(df)}")
        print(f"{col} - Mean: {df[col].mean()}")
    else:
        print(f"{col} missing!")
