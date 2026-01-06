#!/usr/bin/env python3
"""Download mouse->human Ensembl ortholog mappings from Ensembl BioMart."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import pandas as pd

QUERY = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "1" count = "" datasetConfigVersion = "0.6" >
  <Dataset name = "mmusculus_gene_ensembl" interface = "default" >
    <Attribute name = "ensembl_gene_id" />
    <Attribute name = "hsapiens_homolog_ensembl_gene" />
    <Attribute name = "hsapiens_homolog_orthology_type" />
  </Dataset>
</Query>
"""


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.ensembl.org/biomart/martservice?query=" + quote(QUERY)
    df = pd.read_csv(url, sep="\t")
    df = df.rename(
        columns={
            "Gene stable ID": "mouse_ensembl_gene_id",
            "Human gene stable ID": "human_ensembl_gene_id",
            "Human orthology type": "orthology_type",
            "Human homology type": "orthology_type",
        }
    )
    df = df[["mouse_ensembl_gene_id", "human_ensembl_gene_id", "orthology_type"]]
    df = df.dropna(subset=["mouse_ensembl_gene_id", "human_ensembl_gene_id"])

    out_path = data_dir / "mouse_human_orthologs.tsv.gz"
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"Wrote {out_path} with {len(df)} rows")


if __name__ == "__main__":
    main()
