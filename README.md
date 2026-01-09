# MASLD RNA-seq DEG Explorer

Streamlit app to explore **upregulated** DEG counts across in-house MCD (mouse) and
public GEO patient datasets (GSE130970, GSE135251).
Includes an optional **TPM cutoff** (mean TPM per dataset) when TPM columns are present.

## Live app (Streamlit Community Cloud)
This URL should remain stable as long as the app name/workspace are unchanged:
```
https://masld-rna-seq-explorer-f9uhpppgfmcyadgbzx2zrr.streamlit.app/
```

## Run locally
```
streamlit run app.py
```

## Data
Bundled DEG tables live in `data/` (compressed), plus a mouse→human ortholog map
used for cross-species de-duplication. A manifest is in `data/manifest.tsv`.
Bundled tables may include a `tpm_mean` column (mean TPM across all samples in each dataset).

To rebuild the manifest:
```
python scripts/build_data_manifest.py
```

## Config
Optional environment variable:
- `DEG_ORTHOLOG_MAP` — override the ortholog mapping file path.

## Streamlit Community Cloud
- Point Streamlit to `app.py` in this repo.
- Keep only the slim DEG tables in `data/` to avoid large-file bloat.
