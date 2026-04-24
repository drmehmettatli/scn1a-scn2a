# Data Directory

This directory contains all data used in the SCN1A/SCN2A epilepsy research pipeline.

## Subdirectories

- **raw/** — Raw input data: protein sequences (FASTA), clinical variant lists (VCF/CSV from ClinVar), experimental PDB structures
- **processed/** — Intermediate processed files: AlphaFold3 outputs, prepared MD systems, variant score tables
- **results/** — Final analysis results: figures, summary tables, consensus predictions, reports

## Data Sources

| Source | URL | Description |
|--------|-----|-------------|
| UniProt | https://www.uniprot.org/ | SCN1A (P35498), SCN2A (Q99250) sequences |
| ClinVar | https://www.ncbi.nlm.nih.gov/clinvar/ | Clinical variant classifications |
| PDB | https://www.rcsb.org/ | Experimental sodium channel structures |
| AlphaMissense | https://alphamissense.hegelab.org/ | Precomputed pathogenicity scores |

## Notes

- Raw data files are **not** committed to git (see .gitignore)
- Scripts in each stage directory automatically download required data
- Large trajectory files (.xtc, .dcd) should be stored on external storage
