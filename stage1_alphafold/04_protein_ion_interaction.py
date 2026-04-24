"""
Stage 1 - Step 4: Protein-Ion Interaction Analysis
====================================================
Identify sodium ion binding sites in AlphaFold3-predicted structures,
analyze coordination geometry (distances, angles), identify key residues,
calculate binding site electrostatic properties, and produce
visualization-ready outputs.

Usage:
    python 04_protein_ion_interaction.py \
        --structures_dir ../data/processed/alphafold \
        --output_dir ../data/results/ion_interactions
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

GENES = ["SCN1A", "SCN2A"]

# Typical Na+ coordination distance cutoff (Å)
NA_COORDINATION_CUTOFF = 3.5

# Residues known to coordinate sodium in voltage-gated channels
ION_COORDINATING_RESIDUES = {"ASP", "GLU", "ASN", "GLN", "SER", "THR", "TYR", "HIS"}

# Selectivity filter residues of Nav channels (DEKA motif, domains I–IV)
SELECTIVITY_FILTER_MOTIF = ["ASP", "GLU", "LYS", "ALA"]


class Atom:
    """Simple atom representation parsed from PDB/mmCIF."""

    def __init__(
        self,
        atom_name: str,
        res_name: str,
        chain: str,
        res_seq: int,
        coords: np.ndarray,
        element: str = "",
    ):
        self.atom_name = atom_name
        self.res_name = res_name
        self.chain = chain
        self.res_seq = res_seq
        self.coords = coords
        self.element = element

    def __repr__(self) -> str:
        return (
            f"Atom({self.atom_name}, {self.res_name} {self.chain}{self.res_seq}, "
            f"coords={self.coords.round(2)})"
        )


def parse_pdb_atoms(pdb_path: str) -> list[Atom]:
    """Parse ATOM and HETATM records from a PDB file.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        List of Atom objects.
    """
    atoms = []
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_seq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else ""
                atoms.append(Atom(atom_name, res_name, chain, res_seq, np.array([x, y, z]), element))
            except (ValueError, IndexError):
                continue
    return atoms


def find_sodium_ions(atoms: list[Atom]) -> list[Atom]:
    """Extract sodium ion atoms from the parsed atom list.

    Args:
        atoms: List of all parsed atoms.

    Returns:
        List of sodium ion atoms (residue name 'NA', 'SOD', or element 'NA').
    """
    return [
        a for a in atoms
        if a.res_name in ("NA", "SOD", "NA+") or a.element.upper() == "NA"
    ]


def find_coordinating_atoms(
    ion: Atom,
    atoms: list[Atom],
    cutoff: float = NA_COORDINATION_CUTOFF,
) -> list[tuple[Atom, float]]:
    """Find protein atoms coordinating a sodium ion within cutoff distance.

    Only considers electronegative atoms: O, N, S.

    Args:
        ion: Sodium ion atom.
        atoms: All atoms in the structure.
        cutoff: Distance cutoff in Angstroms.

    Returns:
        Sorted list of (Atom, distance) tuples within cutoff.
    """
    coordinating = []
    for atom in atoms:
        if atom is ion:
            continue
        if atom.res_name in ("NA", "SOD", "HOH", "WAT"):
            continue
        if atom.atom_name[0] not in ("O", "N", "S"):
            continue
        dist = float(np.linalg.norm(ion.coords - atom.coords))
        if dist <= cutoff:
            coordinating.append((atom, dist))
    return sorted(coordinating, key=lambda x: x[1])


def calculate_coordination_geometry(
    ion: Atom,
    coordinating: list[tuple[Atom, float]],
) -> dict:
    """Calculate geometric properties of ion coordination shell.

    Args:
        ion: Sodium ion atom.
        coordinating: List of (Atom, distance) from find_coordinating_atoms().

    Returns:
        Dictionary with coordination number, angles, and geometry description.
    """
    if not coordinating:
        return {"coordination_number": 0, "mean_distance": None, "geometry": "none"}

    coord_atoms = [a for a, _ in coordinating]
    distances = [d for _, d in coordinating]
    coord_number = len(coordinating)

    # Calculate O-Na-O angles
    angles = []
    for i in range(len(coord_atoms)):
        for j in range(i + 1, len(coord_atoms)):
            v1 = coord_atoms[i].coords - ion.coords
            v2 = coord_atoms[j].coords - ion.coords
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = float(np.degrees(np.arccos(cos_angle)))
            angles.append(angle)

    # Classify geometry by coordination number
    geometry_map = {
        4: "tetrahedral",
        5: "square_pyramidal_or_trigonal_bipyramidal",
        6: "octahedral",
        7: "pentagonal_bipyramidal",
    }
    geometry = geometry_map.get(coord_number, f"irregular_{coord_number}")

    return {
        "coordination_number": coord_number,
        "mean_distance": round(float(np.mean(distances)), 3),
        "min_distance": round(float(np.min(distances)), 3),
        "max_distance": round(float(np.max(distances)), 3),
        "mean_angle": round(float(np.mean(angles)), 2) if angles else None,
        "std_angle": round(float(np.std(angles)), 2) if angles else None,
        "geometry": geometry,
    }


def score_binding_site(coordinating: list[tuple[Atom, float]]) -> dict:
    """Calculate descriptors for a sodium ion binding site.

    Args:
        coordinating: List of (Atom, distance) pairs.

    Returns:
        Dictionary with binding site property scores.
    """
    if not coordinating:
        return {"n_oxygen": 0, "n_nitrogen": 0, "n_sulfur": 0, "n_acidic": 0}

    n_oxygen = sum(1 for a, _ in coordinating if a.atom_name.startswith("O"))
    n_nitrogen = sum(1 for a, _ in coordinating if a.atom_name.startswith("N"))
    n_sulfur = sum(1 for a, _ in coordinating if a.atom_name.startswith("S"))
    n_acidic = sum(
        1 for a, _ in coordinating if a.res_name in ("ASP", "GLU")
    )
    n_polar = sum(
        1 for a, _ in coordinating if a.res_name in ION_COORDINATING_RESIDUES
    )

    return {
        "n_oxygen": n_oxygen,
        "n_nitrogen": n_nitrogen,
        "n_sulfur": n_sulfur,
        "n_acidic_residues": n_acidic,
        "n_polar_residues": n_polar,
        "selectivity_filter_residues": [
            f"{a.res_name}{a.res_seq}" for a, _ in coordinating
            if a.res_name in ("ASP", "GLU", "LYS", "ALA")
        ],
    }


def analyze_structure(pdb_path: str, gene_name: str) -> list[dict]:
    """Full protein-ion interaction analysis for one structure.

    Args:
        pdb_path: Path to PDB file.
        gene_name: Gene name for output labelling.

    Returns:
        List of per-ion-site analysis dictionaries.
    """
    atoms = parse_pdb_atoms(pdb_path)
    logger.info("%s: parsed %d atoms", gene_name, len(atoms))

    sodium_ions = find_sodium_ions(atoms)
    logger.info("%s: found %d sodium ion(s)", gene_name, len(sodium_ions))

    if not sodium_ions:
        logger.warning(
            "%s: No sodium ions in structure. "
            "Searching for putative binding sites based on DEKA motif proximity.",
            gene_name,
        )
        return analyze_putative_sites(atoms, gene_name)

    results = []
    for idx, ion in enumerate(sodium_ions):
        coordinating = find_coordinating_atoms(ion, atoms)
        geometry = calculate_coordination_geometry(ion, coordinating)
        site_props = score_binding_site(coordinating)

        row = {
            "gene": gene_name,
            "ion_index": idx + 1,
            "ion_chain": ion.chain,
            "ion_res_seq": ion.res_seq,
            "ion_x": round(ion.coords[0], 3),
            "ion_y": round(ion.coords[1], 3),
            "ion_z": round(ion.coords[2], 3),
            **geometry,
            **site_props,
            "coordinating_residues": ";".join(
                f"{a.res_name}{a.chain}{a.res_seq}({a.atom_name}):{d:.2f}Å"
                for a, d in coordinating
            ),
        }
        results.append(row)
        logger.info(
            "  Ion %d: coord_number=%d, geometry=%s, mean_dist=%.2fÅ",
            idx + 1, geometry["coordination_number"], geometry["geometry"],
            geometry["mean_distance"] or 0,
        )

    return results


def analyze_putative_sites(atoms: list[Atom], gene_name: str) -> list[dict]:
    """Identify putative sodium binding sites without explicit ions.

    Searches for the DEKA selectivity filter motif and electronegative
    clusters likely to coordinate Na+.

    Args:
        atoms: All parsed atoms.
        gene_name: Gene name for labelling.

    Returns:
        List of putative site dictionaries.
    """
    # Collect residue backbone O atoms from acidic residues
    acidic_oxygens = [
        a for a in atoms
        if a.res_name in ("ASP", "GLU") and a.atom_name in ("OD1", "OD2", "OE1", "OE2")
    ]
    logger.info("%s: %d acidic oxygen atoms for putative site search", gene_name, len(acidic_oxygens))
    # Build O(1) lookup from atom object to its index to avoid O(n²) list.index() calls
    ao_idx = {id(a): i for i, a in enumerate(acidic_oxygens)}
    results = []
    used = set()
    for i, oa in enumerate(acidic_oxygens):
        if i in used:
            continue
        cluster = [(oa, 0.0)]
        for j, ob in enumerate(acidic_oxygens):
            if j == i or j in used:
                continue
            d = float(np.linalg.norm(oa.coords - ob.coords))
            if d <= NA_COORDINATION_CUTOFF * 2:
                cluster.append((ob, d))
        if len(cluster) >= 2:
            for _, (a, _) in enumerate(cluster):
                used.add(ao_idx.get(id(a), -1))
            centroid = np.mean([a.coords for a, _ in cluster], axis=0)
            results.append({
                "gene": gene_name,
                "ion_index": len(results) + 1,
                "type": "putative_site",
                "centroid_x": round(centroid[0], 3),
                "centroid_y": round(centroid[1], 3),
                "centroid_z": round(centroid[2], 3),
                "cluster_size": len(cluster),
                "residues": ";".join(f"{a.res_name}{a.chain}{a.res_seq}" for a, _ in cluster),
            })
    return results


def generate_visualization_script(results: list[dict], output_dir: str, gene_name: str) -> None:
    """Generate a PyMOL selection script to visualize binding sites.

    Args:
        results: Ion interaction result rows.
        output_dir: Directory to save the .pml script.
        gene_name: Gene name for file naming.
    """
    lines = [
        f"# PyMOL script for {gene_name} sodium ion binding sites",
        "hide everything",
        "show cartoon",
        "color grey80, all",
    ]
    for row in results:
        if "coordinating_residues" not in row:
            continue
        residues = row["coordinating_residues"].split(";")
        resi_list = set()
        for res_str in residues:
            # Format: RESNAME_CHAIN_RESSEQ(ATOM):dist
            import re
            match = re.search(r"([A-Z]{3})([A-Z])(\d+)", res_str)
            if match:
                resi_list.add(match.group(3))
        if resi_list:
            sele = "+".join(sorted(resi_list))
            lines.append(f"select site_{row['ion_index']}, resi {sele}")
            lines.append(f"show sticks, site_{row['ion_index']}")
            lines.append(f"color red, site_{row['ion_index']}")

    script_path = os.path.join(output_dir, f"{gene_name.lower()}_binding_sites.pml")
    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    logger.info("Saved PyMOL script: %s", script_path)


def plot_coordination_summary(all_results: list[dict], output_dir: str) -> None:
    """Create a bar chart comparing coordination numbers between genes/sites.

    Args:
        all_results: Combined list of result dicts for all genes.
        output_dir: Directory to save figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping coordination plot.")
        return

    df = pd.DataFrame(all_results)
    if "coordination_number" not in df.columns or df["coordination_number"].isna().all():
        logger.warning("No coordination data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    genes = df["gene"].unique()
    colors = {"SCN1A": "#1565C0", "SCN2A": "#E53935"}
    for gene in genes:
        sub = df[df["gene"] == gene]
        ax.bar(
            [f"{gene} Site {i+1}" for i in range(len(sub))],
            sub["coordination_number"].values,
            color=colors.get(gene, "grey"),
            label=gene,
        )
    ax.set_ylabel("Coordination Number", fontsize=12)
    ax.set_title("Na⁺ Coordination Numbers — SCN1A vs SCN2A", fontsize=13, fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 9)
    ax.axhline(6, color="grey", linestyle="--", linewidth=0.8, label="Octahedral (6)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "coordination_geometry.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved coordination plot: %s", out_path)


def main(structures_dir: str, output_dir: str) -> None:
    """Main workflow for protein-ion interaction analysis.

    Args:
        structures_dir: Directory containing AlphaFold3 outputs.
        output_dir: Directory for saving results.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for gene_name in GENES:
        gene_dir = os.path.join(structures_dir, gene_name.lower())
        pdb_path = None
        if os.path.isdir(gene_dir):
            for fname in sorted(os.listdir(gene_dir)):
                if fname.endswith(".pdb"):
                    pdb_path = os.path.join(gene_dir, fname)
                    break

        if pdb_path is None:
            logger.warning(
                "No PDB file found for %s in %s. "
                "Run 02_run_alphafold3.py and ensure PDB output is present.",
                gene_name, structures_dir,
            )
            continue

        logger.info("Analyzing protein-ion interactions for %s: %s", gene_name, pdb_path)
        results = analyze_structure(pdb_path, gene_name)

        if results:
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, f"{gene_name.lower()}_ion_binding_sites.csv")
            df.to_csv(csv_path, index=False)
            logger.info("Saved ion binding sites: %s", csv_path)
            generate_visualization_script(results, output_dir, gene_name)
            all_results.extend(results)
        else:
            logger.warning("No ion binding sites identified for %s.", gene_name)

    if all_results:
        plot_coordination_summary(all_results, output_dir)
        combined_df = pd.DataFrame(all_results)
        combined_path = os.path.join(output_dir, "all_ion_binding_sites.csv")
        combined_df.to_csv(combined_path, index=False)
        logger.info("Saved combined results: %s", combined_path)
    else:
        logger.warning(
            "No ion interaction data generated. Ensure PDB structures are in %s.",
            structures_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze protein-ion interactions in SCN1A/SCN2A structures."
    )
    parser.add_argument(
        "--structures_dir",
        default=os.path.join("..", "data", "processed", "alphafold"),
        help="Directory with AlphaFold3 PDB outputs",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "ion_interactions"),
        help="Directory for ion interaction results",
    )
    args = parser.parse_args()
    main(args.structures_dir, args.output_dir)
