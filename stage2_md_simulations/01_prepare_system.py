"""
Stage 2 - Step 1: Prepare MD System
=====================================
Add hydrogens to the protein, create solvation box, neutralize with
sodium/chloride ions, prepare carbamazepine ligand parameters, and
write OpenMM or GROMACS input files.

Usage:
    python 01_prepare_system.py \
        --structure ../data/processed/alphafold/scn1a/scn1a_model.pdb \
        --output_dir ../data/processed/md_systems \
        --gene SCN1A \
        [--with_drug]
"""

import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Solvation box padding in nm
BOX_PADDING_NM = 1.5

# Ionic strength for neutralization (mol/L)
IONIC_STRENGTH = 0.15

# Force field choices
FORCE_FIELDS = {
    "amber": ("amber14-all.xml", "amber14/tip3pfb.xml"),
    "charmm": ("charmm36.xml", "charmm36/water.xml"),
}
DEFAULT_FORCE_FIELD = "amber"


def check_openmm_available() -> bool:
    """Check if OpenMM is installed.

    Returns:
        True if openmm can be imported.
    """
    try:
        import openmm  # noqa: F401
        return True
    except ImportError:
        return False


def prepare_system_openmm(
    structure_path: str,
    output_dir: str,
    gene_name: str,
    force_field_name: str = DEFAULT_FORCE_FIELD,
    with_drug: bool = False,
) -> dict:
    """Prepare MD system using OpenMM.

    Adds missing hydrogens, creates solvation box, neutralizes charge,
    and saves topology + positions as XML/PDB files.

    Args:
        structure_path: Path to input PDB file.
        output_dir: Directory for output files.
        gene_name: Gene name for file naming.
        force_field_name: 'amber' or 'charmm'.
        with_drug: Include carbamazepine ligand if True.

    Returns:
        Dictionary with paths to output files.

    Raises:
        ImportError: If OpenMM is not installed.
        FileNotFoundError: If structure file does not exist.
    """
    try:
        import openmm as mm
        import openmm.app as app
        import openmm.unit as unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM is required. Install with: conda install -c conda-forge openmm"
        ) from exc

    if not os.path.exists(structure_path):
        raise FileNotFoundError(f"Structure not found: {structure_path}")

    ff_files = FORCE_FIELDS.get(force_field_name, FORCE_FIELDS[DEFAULT_FORCE_FIELD])
    logger.info("Loading force field: %s", ff_files)
    forcefield = app.ForceField(*ff_files)

    # Load structure
    logger.info("Loading structure: %s", structure_path)
    pdb = app.PDBFile(structure_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Add missing hydrogens
    logger.info("Adding missing hydrogens at pH 7.4...")
    modeller.addHydrogens(forcefield, pH=7.4)

    # Add solvent box
    logger.info("Adding solvent box (padding=%.1f nm, ionic_strength=%.2f M)...",
                BOX_PADDING_NM, IONIC_STRENGTH)
    modeller.addSolvent(
        forcefield,
        padding=BOX_PADDING_NM * unit.nanometers,
        ionicStrength=IONIC_STRENGTH * unit.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
    )

    # Create system
    logger.info("Creating OpenMM system...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
    )

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_cbz" if with_drug else "_wt"
    base = os.path.join(output_dir, f"{gene_name.lower()}{suffix}")

    # Save solvated PDB
    pdb_out = base + "_solvated.pdb"
    with open(pdb_out, "w", encoding="utf-8") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
    logger.info("Saved solvated PDB: %s", pdb_out)

    # Save system XML
    xml_out = base + "_system.xml"
    with open(xml_out, "w", encoding="utf-8") as fh:
        fh.write(mm.XmlSerializer.serialize(system))
    logger.info("Saved system XML: %s", xml_out)

    # Count atoms and ions
    n_atoms = modeller.topology.getNumAtoms()
    na_ions = sum(1 for a in modeller.topology.atoms() if a.residue.name in ("NA", "Na+"))
    cl_ions = sum(1 for a in modeller.topology.atoms() if a.residue.name in ("CL", "Cl-"))
    water = sum(1 for r in modeller.topology.residues() if r.name == "HOH")

    logger.info(
        "System summary: %d atoms, %d Na+, %d Cl-, %d water molecules",
        n_atoms, na_ions, cl_ions, water,
    )

    return {
        "pdb": pdb_out,
        "system_xml": xml_out,
        "n_atoms": n_atoms,
        "n_na": na_ions,
        "n_cl": cl_ions,
        "n_water": water,
    }


def prepare_system_fallback(
    structure_path: str,
    output_dir: str,
    gene_name: str,
    with_drug: bool = False,
) -> dict:
    """Fallback system preparation without OpenMM — writes GROMACS-style config.

    Creates mdp configuration files and a preparation instructions file
    when OpenMM is not available.

    Args:
        structure_path: Path to input PDB file.
        output_dir: Directory for output files.
        gene_name: Gene name.
        with_drug: Whether this is a drug-bound system.

    Returns:
        Dictionary with paths to written configuration files.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_cbz" if with_drug else "_wt"
    base = os.path.join(output_dir, f"{gene_name.lower()}{suffix}")

    # Write GROMACS-style mdp for energy minimization
    em_mdp = base + "_em.mdp"
    with open(em_mdp, "w", encoding="utf-8") as fh:
        fh.write(
            "; Energy minimization MDP\n"
            "integrator  = steep\n"
            "nsteps      = 50000\n"
            "emtol       = 1000.0\n"
            "emstep      = 0.01\n"
            "nstxout     = 500\n"
            "; Non-bonded interactions\n"
            "cutoff-scheme   = Verlet\n"
            "ns_type         = grid\n"
            "coulombtype     = PME\n"
            "rcoulomb        = 1.0\n"
            "rvdw            = 1.0\n"
            "pbc             = xyz\n"
        )
    logger.info("Wrote EM MDP: %s", em_mdp)

    # Write preparation instructions
    instructions = base + "_preparation_instructions.txt"
    drug_note = "\n    gmx editconf -f protein_ligand.pdb ..." if with_drug else ""
    with open(instructions, "w", encoding="utf-8") as fh:
        fh.write(
            f"# MD System Preparation Instructions for {gene_name}"
            f"{'+ Carbamazepine' if with_drug else ' (Wild-Type)'}\n\n"
            "## GROMACS Workflow\n\n"
            f"1. Convert PDB to GROMACS topology:\n"
            f"   gmx pdb2gmx -f {structure_path} -o protein.gro -water tip3p -ff amber14sb\n"
            f"{drug_note}\n"
            "2. Create solvation box:\n"
            "   gmx editconf -f protein.gro -o boxed.gro -bt cubic -d 1.5\n"
            "   gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top\n\n"
            "3. Add ions:\n"
            "   gmx grompp -f ions.mdp -c solvated.gro -p topol.top -o ions.tpr\n"
            "   gmx genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral\n\n"
            "4. Energy minimization:\n"
            f"   gmx grompp -f {em_mdp} -c ions.gro -p topol.top -o em.tpr\n"
            "   gmx mdrun -v -deffnm em\n\n"
            "## OpenMM Alternative\n"
            "Install OpenMM: conda install -c conda-forge openmm\n"
            "Then re-run: python 01_prepare_system.py --structure ... --output_dir ...\n"
        )
    logger.info("Wrote preparation instructions: %s", instructions)

    return {"em_mdp": em_mdp, "instructions": instructions}


def write_drug_topology_note(output_dir: str, gene_name: str) -> str:
    """Write a note about carbamazepine force field parameterization.

    Args:
        output_dir: Output directory.
        gene_name: Gene name.

    Returns:
        Path to the written note file.
    """
    note_path = os.path.join(output_dir, f"{gene_name.lower()}_cbz_parameterization.txt")
    with open(note_path, "w", encoding="utf-8") as fh:
        fh.write(
            "# Carbamazepine Force Field Parameterization\n\n"
            "## Recommended Tools\n"
            "- GAFF2 (General Amber Force Field 2) via antechamber:\n"
            "    antechamber -i carbamazepine.sdf -fi sdf -o cbz.mol2 -fo mol2 "
            "-c bcc -s 2 -rn CBZ\n"
            "    parmchk2 -i cbz.mol2 -f mol2 -o cbz.frcmod\n\n"
            "- CGenFF (CHARMM General Force Field) via ParamChem:\n"
            "    https://cgenff.umaryland.edu/\n\n"
            "- OpenFF (Open Force Field) via openmmforcefields:\n"
            "    from openmmforcefields.generators import GAFFTemplateGenerator\n"
            "    from openff.toolkit.topology import Molecule\n"
            "    cbz = Molecule.from_file('carbamazepine.sdf')\n"
            "    gaff = GAFFTemplateGenerator(molecules=[cbz], forcefield='gaff-2.11')\n\n"
            "## PubChem CID for Carbamazepine\n"
            "PubChem CID: 2554\n"
            "SMILES: C1=CC2=CC=CC=C2N3C(=O)N=CC3=C1\n"
            "InChI: InChI=1S/C15H12N2O/c18-15-16-9-13-7-3-1-5-11(13)12-6-2-4-8-14(12)"
            "17(15)10-16/h1-10H,(H2,16,18)\n"
        )
    logger.info("Wrote drug parameterization note: %s", note_path)
    return note_path


def main(structure_path: str, output_dir: str, gene_name: str, with_drug: bool) -> None:
    """Main system preparation workflow.

    Args:
        structure_path: Path to input PDB.
        output_dir: Output directory.
        gene_name: Gene identifier.
        with_drug: Include carbamazepine.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(
        "Preparing MD system for %s%s",
        gene_name, " + Carbamazepine" if with_drug else " (Wild-Type)",
    )

    if with_drug:
        write_drug_topology_note(output_dir, gene_name)

    if check_openmm_available():
        logger.info("OpenMM detected — using OpenMM for system preparation.")
        try:
            result = prepare_system_openmm(
                structure_path, output_dir, gene_name, with_drug=with_drug
            )
            logger.info("System prepared successfully: %s", result)
        except Exception as exc:
            logger.error("OpenMM preparation failed: %s", exc)
            logger.info("Falling back to GROMACS config generation.")
            prepare_system_fallback(structure_path, output_dir, gene_name, with_drug)
    else:
        logger.warning(
            "OpenMM not found. Generating GROMACS configuration files instead. "
            "Install OpenMM: conda install -c conda-forge openmm"
        )
        prepare_system_fallback(structure_path, output_dir, gene_name, with_drug)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare MD system for SCN1A/SCN2A simulation."
    )
    parser.add_argument(
        "--structure",
        required=True,
        help="Input PDB file path",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "processed", "md_systems"),
        help="Output directory",
    )
    parser.add_argument(
        "--gene",
        default="SCN1A",
        choices=["SCN1A", "SCN2A"],
        help="Gene name",
    )
    parser.add_argument(
        "--with_drug",
        action="store_true",
        help="Prepare carbamazepine-bound system",
    )
    args = parser.parse_args()
    main(args.structure, args.output_dir, args.gene, args.with_drug)
