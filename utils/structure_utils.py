"""
utils/structure_utils.py
========================
Utility functions for protein structure analysis:
  - PDB/CIF parsing
  - RMSD calculation (Kabsch algorithm)
  - Contact map generation
  - Binding site identification
"""

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Atom:
    """Representation of a single PDB/mmCIF atom."""
    serial: int
    atom_name: str
    res_name: str
    chain: str
    res_seq: int
    coords: np.ndarray
    element: str = ""
    b_factor: float = 0.0
    occupancy: float = 1.0

    def __repr__(self) -> str:
        return (
            f"Atom({self.atom_name}, {self.res_name} {self.chain}{self.res_seq}, "
            f"coords={self.coords.round(2)})"
        )


def parse_pdb_atoms(pdb_path: str, model: int = 1) -> list[Atom]:
    """Parse ATOM and HETATM records from a PDB file.

    Args:
        pdb_path: Path to PDB file.
        model: MODEL number to extract (for NMR ensembles).

    Returns:
        List of Atom objects.

    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    atoms = []
    current_model = 1
    in_model = False

    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("MODEL"):
                try:
                    current_model = int(line.split()[1])
                except (IndexError, ValueError):
                    current_model += 1
                in_model = True
            elif line.startswith("ENDMDL"):
                if current_model == model:
                    break
                in_model = False
            elif not line.startswith(("ATOM", "HETATM")):
                continue
            elif in_model and current_model != model:
                continue

            try:
                serial = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip() or "A"
                res_seq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                occupancy = float(line[54:60].strip()) if len(line) > 60 else 1.0
                b_factor = float(line[60:66].strip()) if len(line) > 66 else 0.0
                element = line[76:78].strip() if len(line) > 76 else ""
                atoms.append(Atom(
                    serial=serial,
                    atom_name=atom_name,
                    res_name=res_name,
                    chain=chain,
                    res_seq=res_seq,
                    coords=np.array([x, y, z], dtype=np.float64),
                    element=element,
                    b_factor=b_factor,
                    occupancy=occupancy,
                ))
            except (ValueError, IndexError):
                continue

    return atoms


def extract_ca_coords(atoms: list[Atom], chain: str = "A") -> np.ndarray:
    """Extract Cα coordinates from a list of atoms.

    Args:
        atoms: List of Atom objects.
        chain: Chain identifier to extract.

    Returns:
        Nx3 float64 array of Cα coordinates.
    """
    ca_coords = []
    seen_residues: set[int] = set()
    for atom in atoms:
        if atom.chain != chain:
            continue
        if atom.atom_name != "CA":
            continue
        if atom.res_seq in seen_residues:
            continue
        seen_residues.add(atom.res_seq)
        ca_coords.append(atom.coords)
    return np.array(ca_coords, dtype=np.float64)


def calculate_rmsd_kabsch(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD between two coordinate sets using Kabsch superposition.

    Args:
        coords1: Nx3 reference coordinates.
        coords2: Nx3 mobile coordinates.

    Returns:
        RMSD value in the same units as input coordinates.

    Raises:
        ValueError: If coordinate shapes do not match.
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Shape mismatch: {coords1.shape} vs {coords2.shape}. "
            "Ensure same number of equivalent atoms."
        )
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    rot = Vt.T @ D @ U.T
    c1_rot = c1 @ rot.T
    diff = c1_rot - c2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def superpose(
    mobile_coords: np.ndarray,
    reference_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Superpose mobile coordinates onto reference using Kabsch algorithm.

    Args:
        mobile_coords: Nx3 mobile coordinates.
        reference_coords: Nx3 reference coordinates.

    Returns:
        Tuple of (superposed_mobile, rotation_matrix).
    """
    ref_center = reference_coords.mean(axis=0)
    mob_center = mobile_coords.mean(axis=0)
    c_ref = reference_coords - ref_center
    c_mob = mobile_coords - mob_center
    H = c_mob.T @ c_ref
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    rot = Vt.T @ D @ U.T
    superposed = c_mob @ rot.T + ref_center
    return superposed, rot


def compute_contact_map(
    coords: np.ndarray,
    cutoff: float = 8.0,
    sequence_separation: int = 3,
) -> np.ndarray:
    """Compute a binary residue contact map.

    Args:
        coords: Nx3 array of Cα coordinates.
        cutoff: Distance cutoff in Angstroms.
        sequence_separation: Minimum sequence separation for a contact.

    Returns:
        NxN boolean contact matrix.
    """
    n = len(coords)
    contact_map = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + sequence_separation, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist <= cutoff:
                contact_map[i, j] = True
                contact_map[j, i] = True
    return contact_map


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix for a set of coordinates.

    Args:
        coords: Nx3 coordinate array.

    Returns:
        NxN symmetric distance matrix.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def find_binding_site_residues(
    protein_atoms: list[Atom],
    ligand_atoms: list[Atom],
    cutoff: float = 5.0,
) -> list[tuple[int, str, str]]:
    """Identify protein residues within a distance cutoff of a ligand.

    Args:
        protein_atoms: List of protein Atom objects.
        ligand_atoms: List of ligand Atom objects.
        cutoff: Distance cutoff in Angstroms.

    Returns:
        Sorted list of (residue_number, residue_name, chain) tuples.
    """
    if not ligand_atoms:
        return []

    lig_coords = np.array([a.coords for a in ligand_atoms])
    binding_residues: set[tuple[int, str, str]] = set()

    for atom in protein_atoms:
        if atom.atom_name.startswith("H"):
            continue
        dists = np.linalg.norm(lig_coords - atom.coords, axis=1)
        if np.min(dists) <= cutoff:
            binding_residues.add((atom.res_seq, atom.res_name, atom.chain))

    return sorted(binding_residues, key=lambda x: x[0])


def calculate_rmsf(trajectory_coords: list[np.ndarray]) -> np.ndarray:
    """Calculate per-residue RMSF from a list of coordinate frames.

    Args:
        trajectory_coords: List of Nx3 coordinate arrays (one per frame).

    Returns:
        N-length array of RMSF values.
    """
    if not trajectory_coords:
        return np.array([])
    coords_array = np.array(trajectory_coords)  # shape: (n_frames, n_atoms, 3)
    mean_coords = coords_array.mean(axis=0)  # (n_atoms, 3)
    deviations = coords_array - mean_coords[np.newaxis]  # (n_frames, n_atoms, 3)
    return np.sqrt(np.mean(np.sum(deviations ** 2, axis=-1), axis=0))


def get_secondary_structure_counts(dssp_string: str) -> dict:
    """Count secondary structure elements in a DSSP string.

    Args:
        dssp_string: DSSP secondary structure string (H=helix, E=sheet, C=coil).

    Returns:
        Dictionary with counts and fractions.
    """
    n = len(dssp_string)
    helix = dssp_string.count("H")
    sheet = dssp_string.count("E")
    coil = n - helix - sheet
    return {
        "n_helix": helix,
        "n_sheet": sheet,
        "n_coil": coil,
        "frac_helix": helix / n if n > 0 else 0.0,
        "frac_sheet": sheet / n if n > 0 else 0.0,
        "frac_coil": coil / n if n > 0 else 0.0,
    }
