# Stage 1: AlphaFold3 Protein Structure Analysis

## Overview

This stage predicts the 3D structures of SCN1A and SCN2A sodium channel proteins
using AlphaFold3 and analyzes protein-sodium ion interactions.

## Scripts

| Script | Description |
|--------|-------------|
| `01_prepare_sequences.py` | Fetch SCN1A/SCN2A sequences from UniProt, validate, save FASTA |
| `02_run_alphafold3.py` | Build AF3 JSON inputs (with Na⁺ ions) and run predictions |
| `03_analyze_structures.py` | Compute pLDDT scores, identify disordered regions |
| `04_protein_ion_interaction.py` | Map Na⁺ binding sites, coordination geometry |

## Quick Start

```bash
# 1. Fetch protein sequences
python 01_prepare_sequences.py --output_dir ../data/raw/sequences

# 2. Prepare AlphaFold3 inputs (and optionally run)
python 02_run_alphafold3.py \
    --sequences_dir ../data/raw/sequences \
    --output_dir ../data/processed/alphafold \
    [--local | --server]

# 3. Analyze structures
python 03_analyze_structures.py \
    --structures_dir ../data/processed/alphafold \
    --output_dir ../data/results/structure_analysis

# 4. Protein-ion interaction analysis
python 04_protein_ion_interaction.py \
    --structures_dir ../data/processed/alphafold \
    --output_dir ../data/results/ion_interactions
```

## AlphaFold3 Setup Options

### Option A: Local Installation
```bash
export AF3_BINARY=/path/to/run_alphafold.sh
export AF3_MODEL_DIR=/path/to/af3/models
export AF3_DB_DIR=/path/to/af3/databases
python 02_run_alphafold3.py --local
```

### Option B: AlphaFold3 Server
```bash
export AF3_SESSION_TOKEN=your_token_here
python 02_run_alphafold3.py --server
```

## Key Parameters

- **Na⁺ ions**: 4 sodium ions included (one per transmembrane domain)
- **Model seeds**: [42, 1337] for reproducibility
- **Coordination cutoff**: 3.5 Å for ion-protein interactions
- **pLDDT thresholds**: Very high ≥90, High ≥70, Low ≥50, Very low <50
