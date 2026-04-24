"""
Stage 2 - Step 2: Run MD Simulation
=====================================
Configure and execute a full MD protocol:
  1. Energy minimization
  2. NVT equilibration (constant volume + temperature)
  3. NPT equilibration (constant pressure + temperature)
  4. Production MD

Supports wild-type and carbamazepine-bound systems.
Uses OpenMM when available; otherwise generates GROMACS scripts.

Usage:
    python 02_run_md_simulation.py \
        --system_dir ../data/processed/md_systems/scn1a_wt \
        --output_dir ../data/processed/md_trajectories \
        --gene SCN1A \
        [--with_drug] \
        [--steps 5000000]
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Simulation parameters
TEMPERATURE_K = 310.0       # Body temperature
PRESSURE_BAR = 1.0
TIMESTEP_PS = 0.002         # 2 fs
EM_MAX_STEPS = 50_000
NVT_STEPS = 500_000         # 1 ns
NPT_STEPS = 500_000         # 1 ns
PRODUCTION_STEPS = 5_000_000  # 10 ns default
REPORT_INTERVAL = 5_000     # Save every 10 ps


def run_simulation_openmm(
    system_dir: str,
    output_dir: str,
    gene_name: str,
    with_drug: bool = False,
    production_steps: int = PRODUCTION_STEPS,
) -> dict:
    """Run full MD protocol using OpenMM.

    Phases: energy minimization → NVT equilibration → NPT equilibration → production.

    Args:
        system_dir: Directory containing solvated PDB and system XML.
        output_dir: Directory for trajectory and log outputs.
        gene_name: Gene name for file naming.
        with_drug: Whether this is a drug-bound simulation.
        production_steps: Number of production MD steps.

    Returns:
        Dictionary with paths to trajectory files.

    Raises:
        ImportError: If OpenMM is not installed.
        FileNotFoundError: If required system files are missing.
    """
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM is required. Install with: conda install -c conda-forge openmm"
        ) from exc

    suffix = "_cbz" if with_drug else "_wt"
    base_name = f"{gene_name.lower()}{suffix}"

    # Locate input files
    pdb_path = os.path.join(system_dir, f"{base_name}_solvated.pdb")
    xml_path = os.path.join(system_dir, f"{base_name}_system.xml")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Solvated PDB not found: {pdb_path}")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"System XML not found: {xml_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Load system
    logger.info("Loading system: %s", pdb_path)
    pdb = app.PDBFile(pdb_path)
    with open(xml_path, "r", encoding="utf-8") as fh:
        system = mm.XmlSerializer.deserialize(fh.read())

    # Setup integrator (Langevin for NVT)
    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        1.0 / unit.picosecond,
        TIMESTEP_PS * unit.picoseconds,
    )

    # Select platform (CUDA > OpenCL > CPU)
    platform = _select_platform()
    logger.info("Using OpenMM platform: %s", platform.getName())

    simulation = app.Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)

    # ── Phase 1: Energy Minimisation ─────────────────────────────────────────
    logger.info("Phase 1: Energy minimization (max %d steps)...", EM_MAX_STEPS)
    t0 = time.time()
    simulation.minimizeEnergy(maxIterations=EM_MAX_STEPS)
    state = simulation.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy()
    logger.info("  Minimization done in %.1fs, PE = %s", time.time() - t0, pe)

    em_pdb = os.path.join(output_dir, f"{base_name}_minimized.pdb")
    state = simulation.context.getState(getPositions=True)
    with open(em_pdb, "w", encoding="utf-8") as fh:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), fh)
    logger.info("Saved minimized structure: %s", em_pdb)

    # ── Phase 2: NVT Equilibration ────────────────────────────────────────────
    logger.info("Phase 2: NVT equilibration (%d steps = %.1f ns)...",
                NVT_STEPS, NVT_STEPS * TIMESTEP_PS / 1000)
    nvt_log = os.path.join(output_dir, f"{base_name}_nvt.log")
    simulation.reporters.append(
        app.StateDataReporter(
            nvt_log, REPORT_INTERVAL,
            step=True, temperature=True, potentialEnergy=True,
            kineticEnergy=True, totalEnergy=True, progress=True,
            remainingTime=True, speed=True, totalSteps=NVT_STEPS,
        )
    )
    t0 = time.time()
    simulation.step(NVT_STEPS)
    logger.info("  NVT done in %.1fs", time.time() - t0)
    simulation.reporters.clear()

    # ── Phase 3: NPT Equilibration ────────────────────────────────────────────
    logger.info("Phase 3: NPT equilibration (%d steps = %.1f ns)...",
                NPT_STEPS, NPT_STEPS * TIMESTEP_PS / 1000)
    # Add barostat for constant pressure
    system.addForce(
        mm.MonteCarloBarostat(
            PRESSURE_BAR * unit.bar,
            TEMPERATURE_K * unit.kelvin,
            25,
        )
    )
    simulation.context.reinitialize(preserveState=True)

    npt_log = os.path.join(output_dir, f"{base_name}_npt.log")
    simulation.reporters.append(
        app.StateDataReporter(
            npt_log, REPORT_INTERVAL,
            step=True, temperature=True, potentialEnergy=True,
            volume=True, density=True, progress=True,
            remainingTime=True, speed=True, totalSteps=NPT_STEPS,
        )
    )
    t0 = time.time()
    simulation.step(NPT_STEPS)
    logger.info("  NPT done in %.1fs", time.time() - t0)
    simulation.reporters.clear()

    # ── Phase 4: Production MD ────────────────────────────────────────────────
    logger.info("Phase 4: Production MD (%d steps = %.1f ns)...",
                production_steps, production_steps * TIMESTEP_PS / 1000)
    traj_dcd = os.path.join(output_dir, f"{base_name}_production.dcd")
    prod_log = os.path.join(output_dir, f"{base_name}_production.log")

    simulation.reporters.append(
        app.DCDReporter(traj_dcd, REPORT_INTERVAL)
    )
    simulation.reporters.append(
        app.StateDataReporter(
            prod_log, REPORT_INTERVAL,
            step=True, temperature=True, potentialEnergy=True,
            volume=True, density=True, speed=True, progress=True,
            remainingTime=True, totalSteps=production_steps,
        )
    )

    checkpoint_file = os.path.join(output_dir, f"{base_name}_checkpoint.chk")
    simulation.reporters.append(
        app.CheckpointReporter(checkpoint_file, 50_000)
    )

    t0 = time.time()
    simulation.step(production_steps)
    wall_time = time.time() - t0
    logger.info(
        "  Production done in %.1f min (%.2f ns/day)",
        wall_time / 60,
        production_steps * TIMESTEP_PS / 1000 / wall_time * 86400,
    )

    # Save final structure
    final_pdb = os.path.join(output_dir, f"{base_name}_final.pdb")
    state = simulation.context.getState(getPositions=True)
    with open(final_pdb, "w", encoding="utf-8") as fh:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), fh)
    logger.info("Saved final structure: %s", final_pdb)

    return {
        "minimized_pdb": em_pdb,
        "trajectory_dcd": traj_dcd,
        "production_log": prod_log,
        "nvt_log": nvt_log,
        "npt_log": npt_log,
        "final_pdb": final_pdb,
        "checkpoint": checkpoint_file,
        "total_time_simulated_ns": production_steps * TIMESTEP_PS / 1000,
    }


def _select_platform():
    """Select the best available OpenMM compute platform.

    Returns:
        OpenMM Platform object (CUDA > OpenCL > CPU).
    """
    import openmm as mm

    for name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = mm.Platform.getPlatformByName(name)
            logger.info("Selected platform: %s", name)
            return platform
        except Exception:
            continue
    return mm.Platform.getPlatformByName("CPU")


def write_gromacs_scripts(
    output_dir: str,
    gene_name: str,
    with_drug: bool = False,
    production_steps: int = PRODUCTION_STEPS,
) -> dict:
    """Write GROMACS MDP files for each simulation phase.

    Args:
        output_dir: Directory for MDP files.
        gene_name: Gene name.
        with_drug: Drug-bound simulation flag.
        production_steps: Steps for production run.

    Returns:
        Dictionary with paths to generated MDP files.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_cbz" if with_drug else "_wt"
    base = os.path.join(output_dir, f"{gene_name.lower()}{suffix}")
    production_time_ns = production_steps * TIMESTEP_PS / 1000

    def write_mdp(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        logger.info("Wrote MDP: %s", path)

    # Energy minimization
    em_mdp = base + "_em.mdp"
    write_mdp(em_mdp, (
        "; Energy Minimization\n"
        "integrator  = steep\n"
        f"nsteps      = {EM_MAX_STEPS}\n"
        "emtol       = 1000.0\n"
        "emstep      = 0.01\n"
        "nstxout     = 500\n"
        "cutoff-scheme = Verlet\n"
        "coulombtype   = PME\n"
        "rcoulomb      = 1.0\n"
        "rvdw          = 1.0\n"
        "pbc           = xyz\n"
    ))

    # NVT
    nvt_mdp = base + "_nvt.mdp"
    write_mdp(nvt_mdp, (
        "; NVT Equilibration\n"
        "integrator          = md\n"
        f"nsteps              = {NVT_STEPS}\n"
        f"dt                  = {TIMESTEP_PS}\n"
        f"nstxout-compressed  = {REPORT_INTERVAL}\n"
        f"nstlog              = {REPORT_INTERVAL}\n"
        f"nstcalcenergy       = {REPORT_INTERVAL}\n"
        "cutoff-scheme       = Verlet\n"
        "coulombtype         = PME\n"
        "rcoulomb            = 1.0\n"
        "rvdw                = 1.0\n"
        "tcoupl              = V-rescale\n"
        f"ref_t               = {TEMPERATURE_K}\n"
        "tau_t               = 0.1\n"
        "tc-grps             = Protein Non-Protein\n"
        "pcoupl              = no\n"
        "pbc                 = xyz\n"
        "gen_vel             = yes\n"
        f"gen_temp            = {TEMPERATURE_K}\n"
        "constraints         = h-bonds\n"
    ))

    # NPT
    npt_mdp = base + "_npt.mdp"
    write_mdp(npt_mdp, (
        "; NPT Equilibration\n"
        "integrator          = md\n"
        f"nsteps              = {NPT_STEPS}\n"
        f"dt                  = {TIMESTEP_PS}\n"
        f"nstxout-compressed  = {REPORT_INTERVAL}\n"
        f"nstlog              = {REPORT_INTERVAL}\n"
        "cutoff-scheme       = Verlet\n"
        "coulombtype         = PME\n"
        "rcoulomb            = 1.0\n"
        "rvdw                = 1.0\n"
        "tcoupl              = V-rescale\n"
        f"ref_t               = {TEMPERATURE_K}\n"
        "tau_t               = 0.1\n"
        "tc-grps             = Protein Non-Protein\n"
        "pcoupl              = Parrinello-Rahman\n"
        f"ref_p               = {PRESSURE_BAR}\n"
        "tau_p               = 2.0\n"
        "compressibility     = 4.5e-5\n"
        "constraints         = h-bonds\n"
    ))

    # Production
    prod_mdp = base + "_production.mdp"
    write_mdp(prod_mdp, (
        f"; Production MD — {production_time_ns:.1f} ns\n"
        "integrator          = md\n"
        f"nsteps              = {production_steps}\n"
        f"dt                  = {TIMESTEP_PS}\n"
        f"nstxout-compressed  = {REPORT_INTERVAL}\n"
        f"nstlog              = {REPORT_INTERVAL}\n"
        f"nstcalcenergy       = {REPORT_INTERVAL}\n"
        f"nstcheckpoint       = 50000\n"
        "cutoff-scheme       = Verlet\n"
        "coulombtype         = PME\n"
        "rcoulomb            = 1.0\n"
        "rvdw                = 1.0\n"
        "tcoupl              = V-rescale\n"
        f"ref_t               = {TEMPERATURE_K}\n"
        "tau_t               = 0.1\n"
        "tc-grps             = Protein Non-Protein\n"
        "pcoupl              = Parrinello-Rahman\n"
        f"ref_p               = {PRESSURE_BAR}\n"
        "tau_p               = 2.0\n"
        "compressibility     = 4.5e-5\n"
        "constraints         = h-bonds\n"
        "continuation        = yes\n"
    ))

    # Shell script to run GROMACS pipeline
    run_script = base + "_run_gromacs.sh"
    with open(run_script, "w", encoding="utf-8") as fh:
        fh.write(
            "#!/usr/bin/env bash\n"
            "# GROMACS MD pipeline\n"
            "set -euo pipefail\n\n"
            f"GENE={gene_name.lower()}{suffix}\n"
            "TOPDIR=../data/processed/md_systems\n\n"
            "# Energy minimization\n"
            f"gmx grompp -f {em_mdp} -c ${{TOPDIR}}/${{GENE}}_solvated.gro "
            "-p ${{TOPDIR}}/${{GENE}}.top -o em.tpr\n"
            "gmx mdrun -v -deffnm em\n\n"
            "# NVT\n"
            f"gmx grompp -f {nvt_mdp} -c em.gro -r em.gro "
            "-p ${{TOPDIR}}/${{GENE}}.top -o nvt.tpr\n"
            "gmx mdrun -v -deffnm nvt\n\n"
            "# NPT\n"
            f"gmx grompp -f {npt_mdp} -c nvt.gro -r nvt.gro "
            "-t nvt.cpt -p ${{TOPDIR}}/${{GENE}}.top -o npt.tpr\n"
            "gmx mdrun -v -deffnm npt\n\n"
            "# Production\n"
            f"gmx grompp -f {prod_mdp} -c npt.gro -t npt.cpt "
            "-p ${{TOPDIR}}/${{GENE}}.top -o production.tpr\n"
            "gmx mdrun -v -deffnm production\n"
        )
    os.chmod(run_script, 0o755)
    logger.info("Wrote GROMACS run script: %s", run_script)

    return {
        "em_mdp": em_mdp,
        "nvt_mdp": nvt_mdp,
        "npt_mdp": npt_mdp,
        "production_mdp": prod_mdp,
        "run_script": run_script,
    }


def main(
    system_dir: str,
    output_dir: str,
    gene_name: str,
    with_drug: bool,
    production_steps: int,
) -> None:
    """Main simulation workflow.

    Args:
        system_dir: Directory with prepared system files.
        output_dir: Directory for trajectory outputs.
        gene_name: Gene identifier.
        with_drug: Carbamazepine-bound simulation flag.
        production_steps: Number of production MD steps.
    """
    suffix = "_cbz" if with_drug else "_wt"
    sim_output = os.path.join(output_dir, f"{gene_name.lower()}{suffix}")
    os.makedirs(sim_output, exist_ok=True)

    try:
        import openmm  # noqa: F401
        logger.info("OpenMM available — running simulation.")
        result = run_simulation_openmm(
            system_dir, sim_output, gene_name, with_drug, production_steps
        )
        logger.info("Simulation complete: %s", result)
    except ImportError:
        logger.warning("OpenMM not available — writing GROMACS scripts.")
        result = write_gromacs_scripts(sim_output, gene_name, with_drug, production_steps)
        logger.info(
            "GROMACS scripts written. To run:\n  bash %s",
            result.get("run_script", ""),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run MD simulation for SCN1A/SCN2A."
    )
    parser.add_argument("--system_dir", required=True, help="Prepared system directory")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "processed", "md_trajectories"),
    )
    parser.add_argument("--gene", default="SCN1A", choices=["SCN1A", "SCN2A"])
    parser.add_argument("--with_drug", action="store_true")
    parser.add_argument(
        "--steps",
        type=int,
        default=PRODUCTION_STEPS,
        help=f"Production MD steps (default: {PRODUCTION_STEPS} = 10 ns)",
    )
    args = parser.parse_args()
    main(args.system_dir, args.output_dir, args.gene, args.with_drug, args.steps)
