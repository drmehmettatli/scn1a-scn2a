# SCN1A ve SCN2A Epilepsi Araştırması / SCN1A and SCN2A Epilepsy Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🇹🇷 Türkçe

### Proje Özeti

Bu proje, epilepsi araştırmalarında kritik rol oynayan **SCN1A** (Nav1.1) ve **SCN2A** (Nav1.2) voltaj kapılı sodyum kanalı genlerini üç aşamalı kapsamlı bir biyoinformatik analiz pipeline'ı ile incelemektedir.

**Klinik Arka Plan:**
- **SCN1A** mutasyonları: Dravet Sendromu, GEFS+ (Febril Nöbetler Plus ile Genetik Epilepsi)
- **SCN2A** mutasyonları: Benign Neonatal Epilepsi, Otizm Spektrum Bozukluğu, EIMFS

### Analiz Aşamaları

#### Aşama 1: AlphaFold3 Protein Yapı Analizi
- UniProt'tan SCN1A (P35498) ve SCN2A (Q99250) protein dizilerinin alınması
- AlphaFold3 ile 3D protein yapılarının tahmin edilmesi
- pLDDT güven skorlarının hesaplanması
- **Protein-iyon etkileşim analizi**: Sodyum iyonu bağlanma bölgelerinin belirlenmesi, koordinasyon geometrisi, anahtar rezidülerin tespiti

#### Aşama 2: Moleküler Dinamik (MD) Simülasyonları
- Protein sisteminin hazırlanması (hidrojen ekleme, solvasyon kutusu, yük nötralizasyonu)
- **Wild-type simülasyon**: 310K, 1 bar koşullarında üretim MD çalıştırması
- **Karbamazepin ile simülasyon**: İlaç bağlanma dinamiklerinin analizi
- RMSD, RMSF hesaplamaları ve konformasyon değişikliklerinin analizi
- MM-PBSA/GBSA ile bağlanma serbest enerjisi hesabı

#### Aşama 3: Varyant Patojeniklik Analizi
- Tüm olası missense varyantların sistematik üretimi
- **AlphaMissense**: Google DeepMind'ın patojeniklik tahmini
- **ESM-1b/ESM-2**: Log-likelihood oranları ile etki tahmini
- **PRESCOTT**: Evrimsel kısıt temelli patojeniklik skoru
- Konsensüs analizi ve klinik veri karşılaştırması (ClinVar, literatür)

---

## 🇬🇧 English

### Project Overview

This project provides a comprehensive three-stage bioinformatics pipeline for analyzing **SCN1A** (Nav1.1) and **SCN2A** (Nav1.2) voltage-gated sodium channel genes in epilepsy research.

**Clinical Background:**
- **SCN1A** mutations: Dravet Syndrome, GEFS+ (Genetic Epilepsy with Febrile Seizures Plus)
- **SCN2A** mutations: Benign Neonatal Epilepsy, Autism Spectrum Disorder, EIMFS (Epilepsy of Infancy with Migrating Focal Seizures)

### Analysis Stages

#### Stage 1: AlphaFold3 Protein Structure Analysis
- Fetch SCN1A (P35498) and SCN2A (Q99250) sequences from UniProt
- Predict 3D protein structures using AlphaFold3
- Calculate pLDDT confidence scores and identify disordered regions
- **Protein-ion interaction analysis**: Identify sodium ion binding sites, coordination geometry, key residues

#### Stage 2: Molecular Dynamics (MD) Simulations
- Prepare protein system (hydrogens, solvation box, ion neutralization)
- **Wild-type simulation**: Production MD at 310K, 1 bar
- **Carbamazepine-bound simulation**: Drug binding dynamics
- RMSD/RMSF analysis and conformational state identification
- Binding free energy via MM-PBSA/GBSA

#### Stage 3: Variant Pathogenicity Analysis
- Generate all possible missense variants systematically
- **AlphaMissense**: Pathogenicity scores from Google DeepMind model
- **ESM-1b/ESM-2**: Log-likelihood ratio-based effect predictions
- **PRESCOTT**: Evolutionary constraint-based pathogenicity scores
- Consensus analysis and clinical data comparison (ClinVar, literature)

---

## Directory Structure

```
scn1a-scn2a/
├── README.md
├── requirements.txt
├── environment.yml
├── data/
│   ├── raw/          # Raw input data (sequences, structures, clinical variants)
│   ├── processed/    # Processed intermediate data
│   └── results/      # Final analysis results
├── stage1_alphafold/
│   ├── 01_prepare_sequences.py
│   ├── 02_run_alphafold3.py
│   ├── 03_analyze_structures.py
│   ├── 04_protein_ion_interaction.py
│   └── notebooks/
│       └── structure_analysis.ipynb
├── stage2_md_simulations/
│   ├── 01_prepare_system.py
│   ├── 02_run_md_simulation.py
│   ├── 03_analyze_trajectory.py
│   ├── 04_drug_binding_analysis.py
│   ├── configs/
│   │   ├── md_parameters.py
│   │   └── carbamazepine.mol2
│   └── notebooks/
│       └── md_analysis.ipynb
├── stage3_variant_pathogenicity/
│   ├── 01_generate_variants.py
│   ├── 02_run_alphamissense.py
│   ├── 03_run_esm1b.py
│   ├── 04_run_prescott.py
│   ├── 05_consensus_analysis.py
│   ├── 06_compare_clinical_data.py
│   └── notebooks/
│       └── variant_analysis.ipynb
└── utils/
    ├── __init__.py
    ├── sequence_utils.py
    ├── structure_utils.py
    └── visualization.py
```

---

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate scn1a-scn2a
```

### Using pip

```bash
pip install -r requirements.txt
```

---

## Usage

### Stage 1: AlphaFold3 Structure Analysis

```bash
cd stage1_alphafold

# Step 1: Fetch and prepare sequences
python 01_prepare_sequences.py --output_dir ../data/raw/sequences

# Step 2: Prepare AlphaFold3 input files
python 02_run_alphafold3.py --sequences_dir ../data/raw/sequences \
                             --output_dir ../data/processed/alphafold

# Step 3: Analyze predicted structures
python 03_analyze_structures.py --structures_dir ../data/processed/alphafold \
                                 --output_dir ../data/results/structure_analysis

# Step 4: Protein-ion interaction analysis
python 04_protein_ion_interaction.py --structures_dir ../data/processed/alphafold \
                                      --output_dir ../data/results/ion_interactions
```

### Stage 2: MD Simulations

```bash
cd stage2_md_simulations

# Step 1: Prepare MD system
python 01_prepare_system.py --structure ../data/processed/alphafold/scn1a_model.pdb \
                              --output_dir ../data/processed/md_systems

# Step 2: Run MD simulation
python 02_run_md_simulation.py --system_dir ../data/processed/md_systems/scn1a \
                                 --output_dir ../data/processed/md_trajectories

# Step 3: Analyze trajectory
python 03_analyze_trajectory.py --trajectory ../data/processed/md_trajectories/scn1a \
                                  --output_dir ../data/results/md_analysis

# Step 4: Drug binding analysis
python 04_drug_binding_analysis.py --trajectory ../data/processed/md_trajectories/scn1a_cbz \
                                    --output_dir ../data/results/drug_binding
```

### Stage 3: Variant Pathogenicity Analysis

```bash
cd stage3_variant_pathogenicity

# Step 1: Generate variants
python 01_generate_variants.py --output_dir ../data/processed/variants

# Step 2: Run AlphaMissense
python 02_run_alphamissense.py --variants ../data/processed/variants/all_variants.csv \
                                --output_dir ../data/results/alphamissense

# Step 3: Run ESM1b
python 03_run_esm1b.py --variants ../data/processed/variants/all_variants.csv \
                        --output_dir ../data/results/esm1b

# Step 4: Run PRESCOTT
python 04_run_prescott.py --variants ../data/processed/variants/all_variants.csv \
                           --output_dir ../data/results/prescott

# Step 5: Consensus analysis
python 05_consensus_analysis.py --alphamissense ../data/results/alphamissense \
                                  --esm1b ../data/results/esm1b \
                                  --prescott ../data/results/prescott \
                                  --output_dir ../data/results/consensus

# Step 6: Clinical comparison
python 06_compare_clinical_data.py --predictions ../data/results/consensus \
                                    --output_dir ../data/results/clinical_comparison
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24.0 | Numerical computing |
| pandas | ≥2.0.0 | Data manipulation |
| scipy | ≥1.10.0 | Scientific computing |
| matplotlib | ≥3.7.0 | Plotting |
| seaborn | ≥0.12.0 | Statistical visualization |
| plotly | ≥5.14.0 | Interactive plots |
| biopython | ≥1.81 | Bioinformatics utilities |
| requests | ≥2.31.0 | API calls |
| mdtraj | ≥1.9.9 | MD trajectory analysis |
| nglview | ≥3.0.8 | Structure visualization |
| torch | ≥2.0.0 | Deep learning (ESM) |
| transformers | ≥4.30.0 | HuggingFace models |
| fair-esm | ≥2.0.0 | ESM protein models |
| jupyter | ≥1.0.0 | Notebooks |
| h5py | ≥3.9.0 | HDF5 file handling |
| tqdm | ≥4.65.0 | Progress bars |

---

## References

1. Jumper, J. et al. (2021). Highly accurate protein structure prediction with AlphaFold. *Nature*, 596, 583–589.
2. Abramson, J. et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. *Nature*, 630, 493–500.
3. Cheng, J. et al. (2023). Accurate proteome-wide missense variant effect prediction with AlphaMissense. *Science*, 381, eadg7492.
4. Rives, A. et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*, 118(15).
5. Dravet, C. (2011). The core Dravet syndrome phenotype. *Epilepsia*, 52(Suppl 2), 3–9.
6. Bhatt, D.L. et al. (2023). Carbamazepine mechanism of action in sodium channel modulation. *Epilepsy Research*, 190, 107086.

---

## License

MIT License — See [LICENSE](LICENSE) for details.

## Contact

For questions about this research pipeline, please open an issue on GitHub.
epilepsy studies on scn1a and scn2a
