# Stage 3: Variant Pathogenicity Analysis

## Overview

Systematic pathogenicity prediction for all SCN1A and SCN2A missense variants
using three complementary tools: AlphaMissense, ESM-1b/ESM-2, and PRESCOTT.

## Scripts

| Script | Description |
|--------|-------------|
| `01_generate_variants.py` | Generate all missense variants, annotate with ClinVar |
| `02_run_alphamissense.py` | AlphaMissense pathogenicity scores (DB or API) |
| `03_run_esm1b.py` | ESM-2 log-likelihood ratio variant scoring |
| `04_run_prescott.py` | PRESCOTT evolutionary conservation scores |
| `05_consensus_analysis.py` | Weighted consensus + VUS reclassification |
| `06_compare_clinical_data.py` | Clinical data comparison, metrics, domain mapping |

## Quick Start

```bash
# 1. Generate all variants
python 01_generate_variants.py \
    --sequences_dir ../data/raw/sequences \
    --clinvar_dir ../data/raw/variants \
    --output_dir ../data/processed/variants

# 2. AlphaMissense (with local DB)
python 02_run_alphamissense.py \
    --variants ../data/processed/variants/scn1a_all_variants.csv \
    --gene SCN1A \
    --db_path /path/to/AlphaMissense_aa_substitutions.tsv.gz

# 2. AlphaMissense (download DB automatically, ~3 GB)
python 02_run_alphamissense.py \
    --variants ../data/processed/variants/scn1a_all_variants.csv \
    --gene SCN1A --download_db

# 3. ESM-1b/ESM-2
python 03_run_esm1b.py \
    --variants ../data/processed/variants/scn1a_all_variants.csv \
    --sequences_dir ../data/raw/sequences \
    --gene SCN1A --model esm2_t33_650M_UR50D

# 4. PRESCOTT (conservation proxy)
python 04_run_prescott.py \
    --variants ../data/processed/variants/scn1a_all_variants.csv \
    --sequences_dir ../data/raw/sequences \
    --gene SCN1A

# 5. Consensus analysis
python 05_consensus_analysis.py \
    --alphamissense_dir ../data/results/alphamissense \
    --esm1b_dir ../data/results/esm1b \
    --prescott_dir ../data/results/prescott \
    --gene ALL

# 6. Clinical comparison
python 06_compare_clinical_data.py \
    --consensus_dir ../data/results/consensus \
    --variants_dir ../data/processed/variants
```

## Tool Weights (Consensus)

| Tool | Weight | Rationale |
|------|--------|-----------|
| AlphaMissense | 0.45 | State-of-the-art, trained on human pathogenic variants |
| ESM-1b/ESM-2 | 0.35 | Evolutionary language model, broad coverage |
| PRESCOTT | 0.20 | Evolutionary constraint, orthogonal signal |

## AlphaMissense Database

Download from Zenodo:
```bash
wget "https://zenodo.org/records/10813168/files/AlphaMissense_aa_substitutions.tsv.gz"
```
