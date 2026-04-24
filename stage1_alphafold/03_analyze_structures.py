"""
Stage 1 - Step 3: Analyze AlphaFold3 Predicted Structures
==========================================================
Load predicted structures, compute pLDDT confidence scores, identify
disordered/loop regions, compare to experimental structures (RMSD),
and save analysis results.

Usage:
    python 03_analyze_structures.py \
        --structures_dir ../data/processed/alphafold \
        --output_dir ../data/results/structure_analysis
"""

import argparse
import json
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

# pLDDT confidence thresholds (AlphaFold convention)
PLDDT_VERY_HIGH = 90.0   # confident, well-structured
PLDDT_HIGH = 70.0        # generally structured
PLDDT_LOW = 50.0         # likely disordered


def find_structure_file(structures_dir: str, gene_name: str) -> str | None:
    """Locate a structure file (CIF or PDB) for a gene.

    Args:
        structures_dir: Base directory for AlphaFold output.
        gene_name: Gene name (e.g., 'SCN1A').

    Returns:
        Path to structure file or None if not found.
    """
    gene_dir = os.path.join(structures_dir, gene_name.lower())
    for ext in (".cif", ".pdb"):
        for prefix in (gene_name.lower(), "model"):
            path = os.path.join(gene_dir, f"{prefix}_model{ext}")
            if os.path.exists(path):
                return path
            path = os.path.join(gene_dir, f"{prefix}{ext}")
            if os.path.exists(path):
                return path
    # Glob fallback
    if os.path.isdir(gene_dir):
        for fname in sorted(os.listdir(gene_dir)):
            if fname.endswith((".cif", ".pdb")):
                return os.path.join(gene_dir, fname)
    return None


def find_confidence_file(structures_dir: str, gene_name: str) -> str | None:
    """Locate the pLDDT confidence JSON for a gene.

    Args:
        structures_dir: Base directory for AlphaFold output.
        gene_name: Gene name.

    Returns:
        Path to confidence JSON or None.
    """
    gene_dir = os.path.join(structures_dir, gene_name.lower())
    candidates = [
        os.path.join(gene_dir, f"{gene_name.lower()}_confidence.json"),
        os.path.join(gene_dir, "confidence.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    if os.path.isdir(gene_dir):
        for fname in sorted(os.listdir(gene_dir)):
            if "confidence" in fname.lower() and fname.endswith(".json"):
                return os.path.join(gene_dir, fname)
    return None


def parse_plddt_from_cif(cif_path: str) -> np.ndarray:
    """Parse per-residue pLDDT scores from an mmCIF file B-factor column.

    AlphaFold3 stores pLDDT in the B-factor field.

    Args:
        cif_path: Path to mmCIF file.

    Returns:
        Array of pLDDT values (float32).
    """
    plddt_values = []
    with open(cif_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                # mmCIF ATOM records: atom_site.B_iso_or_equiv is typically column index 14
                if len(parts) >= 15:
                    try:
                        b_factor = float(parts[14])
                        plddt_values.append(b_factor)
                    except (ValueError, IndexError):
                        continue
    return np.array(plddt_values, dtype=np.float32)


def parse_plddt_from_pdb(pdb_path: str) -> np.ndarray:
    """Parse per-residue pLDDT scores from a PDB file B-factor column.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Array of pLDDT values (float32), one per CA atom.
    """
    plddt_values = []
    seen_residues = set()
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            chain = line[21].strip()
            res_seq = line[22:26].strip()
            key = (chain, res_seq)
            if key in seen_residues:
                continue
            seen_residues.add(key)
            try:
                b_factor = float(line[60:66].strip())
                plddt_values.append(b_factor)
            except ValueError:
                continue
    return np.array(plddt_values, dtype=np.float32)


def parse_plddt_from_json(json_path: str) -> np.ndarray:
    """Parse per-residue pLDDT from an AlphaFold confidence JSON file.

    Args:
        json_path: Path to confidence JSON.

    Returns:
        Array of pLDDT values.
    """
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return np.array(data, dtype=np.float32)
    if "plddt" in data:
        return np.array(data["plddt"], dtype=np.float32)
    if "atom_plddts" in data:
        return np.array(data["atom_plddts"], dtype=np.float32)
    raise ValueError(f"Could not locate pLDDT values in {json_path}")


def load_plddt(structures_dir: str, gene_name: str) -> np.ndarray:
    """Load pLDDT scores from JSON or structure file, whichever is available.

    Args:
        structures_dir: Base directory.
        gene_name: Gene name.

    Returns:
        Array of per-residue pLDDT values.

    Raises:
        FileNotFoundError: If neither JSON nor structure file is found.
    """
    json_path = find_confidence_file(structures_dir, gene_name)
    if json_path:
        logger.info("Loading pLDDT from JSON: %s", json_path)
        return parse_plddt_from_json(json_path)

    struct_path = find_structure_file(structures_dir, gene_name)
    if struct_path is None:
        raise FileNotFoundError(
            f"No structure or confidence file found for {gene_name} in {structures_dir}. "
            "Run 02_run_alphafold3.py first."
        )
    logger.info("Loading pLDDT from structure B-factors: %s", struct_path)
    if struct_path.endswith(".cif"):
        return parse_plddt_from_cif(struct_path)
    return parse_plddt_from_pdb(struct_path)


def compute_plddt_statistics(plddt: np.ndarray) -> dict:
    """Compute summary statistics for a pLDDT array.

    Args:
        plddt: Array of per-residue pLDDT scores (0–100).

    Returns:
        Dictionary of statistics.
    """
    n = len(plddt)
    very_high = np.sum(plddt >= PLDDT_VERY_HIGH)
    high = np.sum((plddt >= PLDDT_HIGH) & (plddt < PLDDT_VERY_HIGH))
    low = np.sum((plddt >= PLDDT_LOW) & (plddt < PLDDT_HIGH))
    very_low = np.sum(plddt < PLDDT_LOW)
    return {
        "n_residues": n,
        "mean_plddt": float(np.mean(plddt)),
        "median_plddt": float(np.median(plddt)),
        "std_plddt": float(np.std(plddt)),
        "min_plddt": float(np.min(plddt)),
        "max_plddt": float(np.max(plddt)),
        "fraction_very_high": round(very_high / n, 4),
        "fraction_high": round(high / n, 4),
        "fraction_low": round(low / n, 4),
        "fraction_very_low": round(very_low / n, 4),
        "n_very_high": int(very_high),
        "n_high": int(high),
        "n_low": int(low),
        "n_very_low": int(very_low),
    }


def identify_disordered_regions(plddt: np.ndarray, threshold: float = PLDDT_LOW) -> list[dict]:
    """Identify contiguous disordered (low-confidence) regions.

    Args:
        plddt: Per-residue pLDDT array.
        threshold: pLDDT value below which residues are considered disordered.

    Returns:
        List of dicts with keys: start, end, length, mean_plddt.
    """
    regions = []
    in_region = False
    start = 0
    for i, score in enumerate(plddt):
        if score < threshold and not in_region:
            in_region = True
            start = i
        elif score >= threshold and in_region:
            in_region = False
            region_plddt = plddt[start:i]
            regions.append({
                "start": start + 1,  # 1-indexed
                "end": i,
                "length": i - start,
                "mean_plddt": float(np.mean(region_plddt)),
            })
    if in_region:
        region_plddt = plddt[start:]
        regions.append({
            "start": start + 1,
            "end": len(plddt),
            "length": len(plddt) - start,
            "mean_plddt": float(np.mean(region_plddt)),
        })
    return regions


def calculate_ca_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD between two sets of Cα coordinates after alignment.

    Uses Kabsch algorithm for optimal superposition.

    Args:
        coords1: Nx3 array of Cα coordinates for structure 1.
        coords2: Nx3 array of Cα coordinates for structure 2.

    Returns:
        RMSD in Angstroms.

    Raises:
        ValueError: If coordinate arrays have incompatible shapes.
    """
    if coords1.shape != coords2.shape:
        raise ValueError(
            f"Coordinate shape mismatch: {coords1.shape} vs {coords2.shape}"
        )
    # Centre both structures
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)
    # Kabsch rotation
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    rot = Vt.T @ D @ U.T
    c1_rot = c1 @ rot.T
    diff = c1_rot - c2
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def extract_ca_coords_from_pdb(pdb_path: str) -> np.ndarray:
    """Extract Cα coordinates from a PDB file.

    Args:
        pdb_path: Path to PDB file.

    Returns:
        Nx3 float64 array of Cα coordinates.
    """
    coords = []
    with open(pdb_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                continue
    return np.array(coords, dtype=np.float64)


def save_plddt_per_residue(plddt: np.ndarray, gene_name: str, output_dir: str) -> str:
    """Save per-residue pLDDT to CSV.

    Args:
        plddt: pLDDT array.
        gene_name: Gene name for filename.
        output_dir: Output directory.

    Returns:
        Path to saved CSV.
    """
    df = pd.DataFrame({
        "residue": np.arange(1, len(plddt) + 1),
        "plddt": plddt,
        "confidence_category": pd.cut(
            plddt,
            bins=[-np.inf, PLDDT_LOW, PLDDT_HIGH, PLDDT_VERY_HIGH, np.inf],
            labels=["very_low", "low", "high", "very_high"],
        ),
    })
    path = os.path.join(output_dir, f"{gene_name.lower()}_plddt_scores.csv")
    df.to_csv(path, index=False)
    logger.info("Saved pLDDT CSV: %s", path)
    return path


def plot_plddt(plddt: np.ndarray, gene_name: str, output_dir: str) -> None:
    """Generate and save pLDDT per-residue plot.

    Args:
        plddt: pLDDT array.
        gene_name: Gene name for title and filename.
        output_dir: Directory to save plot.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — skipping pLDDT plot.")
        return

    fig, ax = plt.subplots(figsize=(18, 4))
    residues = np.arange(1, len(plddt) + 1)
    # Colour by confidence band
    colors = np.where(
        plddt >= PLDDT_VERY_HIGH, "#1565C0",
        np.where(plddt >= PLDDT_HIGH, "#43A047",
                 np.where(plddt >= PLDDT_LOW, "#FFB300", "#E53935")),
    )
    ax.bar(residues, plddt, color=colors, width=1.0, linewidth=0)
    for thresh, color, label in [
        (PLDDT_VERY_HIGH, "#1565C0", f"Very high (≥{PLDDT_VERY_HIGH})"),
        (PLDDT_HIGH, "#43A047", f"High ({PLDDT_HIGH}–{PLDDT_VERY_HIGH})"),
        (PLDDT_LOW, "#FFB300", f"Low ({PLDDT_LOW}–{PLDDT_HIGH})"),
        (0, "#E53935", f"Very low (<{PLDDT_LOW})"),
    ]:
        ax.axhline(thresh, color=color, linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlim(0, len(plddt))
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue position", fontsize=12)
    ax.set_ylabel("pLDDT score", fontsize=12)
    ax.set_title(f"{gene_name} — AlphaFold3 pLDDT Confidence", fontsize=14, fontweight="bold")
    patches = [
        mpatches.Patch(color="#1565C0", label=f"Very high (≥{PLDDT_VERY_HIGH})"),
        mpatches.Patch(color="#43A047", label=f"High ({PLDDT_HIGH}–{PLDDT_VERY_HIGH})"),
        mpatches.Patch(color="#FFB300", label=f"Low ({PLDDT_LOW}–{PLDDT_HIGH})"),
        mpatches.Patch(color="#E53935", label=f"Very low (<{PLDDT_LOW})"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"plddt_plot_{gene_name.lower()}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved pLDDT plot: %s", out_path)


def main(structures_dir: str, output_dir: str) -> None:
    """Main workflow for structure analysis.

    Args:
        structures_dir: AlphaFold3 output directory.
        output_dir: Directory for saving results.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = []

    for gene_name in GENES:
        logger.info("Analyzing structure for %s", gene_name)
        try:
            plddt = load_plddt(structures_dir, gene_name)
        except FileNotFoundError as exc:
            logger.warning("%s — skipping: %s", gene_name, exc)
            continue

        stats = compute_plddt_statistics(plddt)
        stats["gene"] = gene_name
        disordered = identify_disordered_regions(plddt)
        stats["n_disordered_regions"] = len(disordered)
        stats["n_disordered_residues"] = sum(r["length"] for r in disordered)

        save_plddt_per_residue(plddt, gene_name, output_dir)
        plot_plddt(plddt, gene_name, output_dir)

        # Save disordered regions
        if disordered:
            disorder_df = pd.DataFrame(disordered)
            disorder_path = os.path.join(output_dir, f"{gene_name.lower()}_disordered_regions.csv")
            disorder_df.to_csv(disorder_path, index=False)
            logger.info("Saved disordered regions: %s", disorder_path)

        summary_rows.append(stats)
        logger.info(
            "%s: mean pLDDT=%.1f, very_high=%.1f%%, disordered regions=%d",
            gene_name, stats["mean_plddt"],
            stats["fraction_very_high"] * 100, stats["n_disordered_regions"],
        )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "structure_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary: %s", summary_path)
    else:
        logger.warning(
            "No structures were analyzed. Ensure AlphaFold3 outputs are in %s.",
            structures_dir,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze AlphaFold3 structure predictions for SCN1A and SCN2A."
    )
    parser.add_argument(
        "--structures_dir",
        default=os.path.join("..", "data", "processed", "alphafold"),
        help="AlphaFold3 output directory",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "structure_analysis"),
        help="Directory for analysis outputs",
    )
    args = parser.parse_args()
    main(args.structures_dir, args.output_dir)
