# MASLD RNA-seq DEG Explorer

Streamlit app to explore **upregulated** DEG counts across in-house MCD (mouse) and
public GEO patient datasets (GSE130970, GSE135251), plus cross-dataset comparisons.
Includes an optional **TPM cutoff** (mean TPM per dataset) when TPM columns are present.

## What’s in the app
- Global padj/log2FC/TPM cutoffs with per-dataset overrides.
- Patient top-right intersections and **cross-dataset** top-right comparisons (GSE135251 vs GSE130970).
- Cross-species **de-duplication** using mouse→human orthologs (toggle one-to-one only, include unmapped).
- Contribution breakdown + downloadable deduplicated gene list.
- Overlap Explorer for pairwise and multi-set intersections.
- Optional GWAS closest-gene set (from `GWAS/Closest_genes.csv`) when available.

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
Bundled tables may include a `gene_symbol` column.
If `GWAS/Closest_genes.csv` exists, the GWAS set becomes available as a selectable human set.

To rebuild the manifest:
```
python scripts/build_data_manifest.py
```

## Config
Optional environment variable:
- `DEG_ORTHOLOG_MAP` — override the ortholog mapping file path.
- `DEG_BIOTYPE_MAP` — override the gene biotype mapping file path (columns: ensembl_gene_id, gene_biotype, species).

## Streamlit Community Cloud
- Point Streamlit to `app.py` in this repo.
- Keep only the slim DEG tables in `data/` to avoid large-file bloat.
