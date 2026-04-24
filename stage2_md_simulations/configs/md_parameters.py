"""
MD Simulation Parameters
========================
Central configuration for all MD simulation settings.
"""

# ── Temperature and Pressure ──────────────────────────────────────────────────
TEMPERATURE_K = 310.0      # Human body temperature
PRESSURE_BAR = 1.0         # Atmospheric pressure

# ── Time Parameters ───────────────────────────────────────────────────────────
TIMESTEP_PS = 0.002        # 2 fs integration step
EM_MAX_STEPS = 50_000      # Energy minimization steps
NVT_STEPS = 500_000        # NVT equilibration (1 ns)
NPT_STEPS = 500_000        # NPT equilibration (1 ns)
PRODUCTION_STEPS = 5_000_000  # Production run (10 ns default)

# ── Output Frequency ─────────────────────────────────────────────────────────
REPORT_INTERVAL = 5_000    # Save coordinates every 10 ps
ENERGY_LOG_INTERVAL = 5_000
CHECKPOINT_INTERVAL = 50_000

# ── Non-bonded Parameters ─────────────────────────────────────────────────────
NONBONDED_CUTOFF_NM = 1.0  # PME real-space cutoff
SWITCH_DISTANCE_NM = 0.9   # VDW switching distance
EWALD_ERROR_TOL = 5e-4

# ── Solvent / Ions ────────────────────────────────────────────────────────────
BOX_PADDING_NM = 1.5
IONIC_STRENGTH_M = 0.15
POSITIVE_ION = "Na+"
NEGATIVE_ION = "Cl-"
WATER_MODEL = "tip3p"

# ── Force Fields ──────────────────────────────────────────────────────────────
PROTEIN_FF = "amber14-all.xml"
WATER_FF = "amber14/tip3pfb.xml"
DRUG_FF = "gaff-2.11"      # Via openmmforcefields

# ── Carbamazepine ─────────────────────────────────────────────────────────────
CBZ_PUBCHEM_CID = 2554
CBZ_SMILES = "C1=CC2=CC=CC=C2N3C(=O)N=CC3=C1"
CBZ_RESIDUE_NAME = "CBZ"
CBZ_MOLECULAR_WEIGHT = 236.27  # g/mol

# ── Analysis ──────────────────────────────────────────────────────────────────
BINDING_CONTACT_CUTOFF_NM = 0.5    # 5 Å
ION_COORDINATION_CUTOFF_NM = 0.35  # 3.5 Å
CLUSTER_N_STATES = 3
