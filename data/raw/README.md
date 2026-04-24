# Raw Data

This directory holds unmodified input data files.

## Expected Contents

```
raw/
├── sequences/
│   ├── SCN1A_P35498.fasta      # SCN1A protein sequence (UniProt P35498)
│   └── SCN2A_Q99250.fasta      # SCN2A protein sequence (UniProt Q99250)
├── structures/
│   ├── 6agf.pdb                # Experimental Nav structure (cryo-EM)
│   └── 7d7y.pdb                # Human Nav1.2 structure
├── variants/
│   ├── SCN1A_clinvar.vcf       # ClinVar variants for SCN1A
│   └── SCN2A_clinvar.vcf       # ClinVar variants for SCN2A
└── ligands/
    └── carbamazepine.sdf       # Carbamazepine 3D structure (PubChem CID 2554)
```

## How to Populate

Run from the project root:
```bash
python stage1_alphafold/01_prepare_sequences.py --output_dir data/raw/sequences
```

ClinVar variants can be downloaded from:
https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/
