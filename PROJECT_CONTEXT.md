# PROJECT_CONTEXT (Codex handoff log)

## Purpose
- Streamlit app to interactively count **upregulated** DEGs for MASLD-related RNA-seq.
- Covers in-house mouse MCD diet RNA-seq (week1/2/3) and public GEO patient datasets (GSE130970, GSE135251).
- Supports adjustable padj/log2FC cutoffs per dataset and global defaults.
- Provides overall summary, de-dup totals across mouse+human via ortholog mapping, and overlap visualization (UpSet).

## Key definitions
- Upregulated = log2FoldChange > cutoff (strict).
- padj cutoff for NAS-high used for counts and for top-right intersections; log2FC cutoff applies ONLY to NAS-high in top-right intersections.
- Top-right quadrant = intersection of (NAS-high padj/log2FC filtered set) with (other contrast padj-filtered set).

## App entry
- File: app.py
- Streamlit Cloud URL: https://masld-rna-seq-explorer-f9uhpppgfmcyadgbzx2zrr.streamlit.app/
- Uses bundled data in ./data by default.
- Env overrides:
  - DEG_DATA_DIR: override data directory
  - DEG_ORTHOLOG_MAP: override ortholog map path

## Data inputs (bundled)
### Mouse (MCD)
- data/mcd_week1.tsv.gz
- data/mcd_week2.tsv.gz
- data/mcd_week3.tsv.gz
Columns: gene_id, log2FoldChange, padj

### Human (GEO)
- data/gse130970_nas_high.csv.gz
- data/gse130970_nas_low.csv.gz
- data/gse130970_fibrosis.csv.gz
- data/gse135251_nas_high.csv.gz
- data/gse135251_nas_low.csv.gz
- data/gse135251_fibrosis.csv.gz
Columns: gene_id, log2FoldChange, padj

### Ortholog map (mouse->human)
- data/mouse_human_orthologs.tsv.gz
Columns: mouse_ensembl_gene_id, human_ensembl_gene_id, orthology_type
Generated via Ensembl BioMart query in scripts/build_mouse_human_orthologs.py

## De-dup across species logic
- Human sets: strip Ensembl version suffix (e.g., ENSG... .12 -> ENSG...)
- Mouse sets: map mouse Ensembl -> human Ensembl via ortholog map
  - default: one-to-one only (toggle in app)
  - optional: include unmapped mouse genes as "MOUSE:<id>" (toggle)
- Dedup total is union across canonicalized sets.

## Overlap visualization
- Uses upsetplot + matplotlib
- Overlap Explorer allows multiselect on summary sets
- Supports ortholog-mapped sets if ortholog file exists
- Guards for empty data/plot errors (common on Streamlit Cloud)

## Scripts
- scripts/build_data_manifest.py
  - builds data/manifest.tsv with sha256 + size + rows/cols
- scripts/build_mouse_human_orthologs.py
  - pulls BioMart (mmusculus_gene_ensembl) with human orthology columns
  - outputs data/mouse_human_orthologs.tsv.gz

## Data provenance (local build)
- Mouse: in-house MCD DESeq2 results
  - RNA-seq/in-house_MCD_RNAseq/analysis_v5_week{1,2,3}MCD_vs_controls/deseq2_results.tsv
- Human: patient_RNAseq DESeq2 results, latest run per dataset
  - RNA-seq/patient_RNAseq/results/GSE130970/deseq2_results/<latest>/...
  - RNA-seq/patient_RNAseq/results/GSE135251/deseq2_results/<latest>/...

## Build steps (how bundle was produced)
- Slim the DESeq2 output to only gene_id/log2FoldChange/padj and gzip.
- Bundle into data/ with fixed filenames expected by app.
- Generate ortholog map via BioMart script and bundle.
- Generate manifest.tsv.

## Dependencies
- streamlit
- pandas
- matplotlib
- upsetplot

## Deployment
- Repo: https://github.com/jameshyojaelee/MASLD-RNA-seq-Explorer
- Streamlit Cloud uses app.py at repo root.

## Known pitfalls
- Streamlit Cloud uses Python 3.13; plotting errors were guarded with try/except.
- UpSet data can be empty or numeric Series; use data.values.sum() for checks.
- Do NOT treat global totals as de-duplicated unless ortholog map is present.

## Last updates
- Added cross-species de-dup, ortholog map + overlap explorer.
- Added Streamlit Cloud URL to README.
