# Stage 2: Molecular Dynamics Simulations

## Overview

MD simulations of SCN1A and SCN2A in wild-type and carbamazepine-bound states.

## Scripts

| Script | Description |
|--------|-------------|
| `01_prepare_system.py` | Add hydrogens, solvate, neutralize, prepare drug topology |
| `02_run_md_simulation.py` | Full MD protocol: EM → NVT → NPT → Production |
| `03_analyze_trajectory.py` | RMSD, RMSF, secondary structure, ion contacts |
| `04_drug_binding_analysis.py` | Drug pose stability, binding residues, MM-GBSA estimate |

## Quick Start

```bash
# Prepare wild-type system
python 01_prepare_system.py \
    --structure ../data/processed/alphafold/scn1a/scn1a_model.pdb \
    --output_dir ../data/processed/md_systems \
    --gene SCN1A

# Prepare drug-bound system
python 01_prepare_system.py \
    --structure ../data/processed/alphafold/scn1a/scn1a_model.pdb \
    --output_dir ../data/processed/md_systems \
    --gene SCN1A --with_drug

# Run simulations
python 02_run_md_simulation.py \
    --system_dir ../data/processed/md_systems \
    --output_dir ../data/processed/md_trajectories \
    --gene SCN1A --steps 5000000

# Analyze trajectory
python 03_analyze_trajectory.py \
    --trajectory_dir ../data/processed/md_trajectories/scn1a_wt \
    --topology ../data/processed/md_systems/scn1a_wt_solvated.pdb \
    --output_dir ../data/results/md_analysis \
    --gene SCN1A
```

## Simulation Parameters (configs/md_parameters.py)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Temperature | 310 K | Body temperature |
| Pressure | 1.0 bar | Atmospheric |
| Time step | 2 fs | With H-bond constraints |
| NVT equilibration | 1 ns | Constant volume |
| NPT equilibration | 1 ns | Constant pressure |
| Production | 10 ns | Configurable via --steps |
| Force field | AMBER14 | Protein + TIP3P water |
| Drug FF | GAFF2 | Via antechamber |

## Carbamazepine Parameterization

See `configs/carbamazepine.mol2` for instructions on generating the
drug force field parameters using GAFF2 (antechamber) or CGenFF.
