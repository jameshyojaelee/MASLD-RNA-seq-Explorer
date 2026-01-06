# MASLD RNA-seq DEG Explorer

Streamlit app to explore **upregulated** DEG counts across in-house MCD (mouse) and
public GEO patient datasets (GSE130970, GSE135251).

## Run locally
```
streamlit run app.py
```

## Data
Bundled DEG tables live in `data/` (compressed). A manifest is in `data/manifest.tsv`.

To rebuild the manifest:
```
python scripts/build_data_manifest.py
```

## Config
Optional environment variable:
- `DEG_DATA_DIR` — override where the app looks for the bundled DEG tables.

## Streamlit Community Cloud
- Point Streamlit to `app.py` in this repo.
- Keep only the slim DEG tables in `data/` to avoid large-file bloat.
