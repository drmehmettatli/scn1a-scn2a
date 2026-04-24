"""
Stage 2 - Step 3: Analyze MD Trajectory
=========================================
Load MD trajectory files, calculate RMSD/RMSF, analyze secondary structure
changes, identify conformational states, analyze protein-ion interactions
over time, and generate publication-quality plots.

Usage:
    python 03_analyze_trajectory.py \
        --trajectory_dir ../data/processed/md_trajectories/scn1a_wt \
        --topology ../data/processed/md_systems/scn1a_wt/scn1a_wt_solvated.pdb \
        --output_dir ../data/results/md_analysis \
        --gene SCN1A
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
TIMESTEP_PS = 2.0
REPORT_INTERVAL = 5_000  # frames saved every 10 ps


def find_trajectory_file(traj_dir: str) -> str | None:
    """Find DCD or XTC trajectory file in a directory.

    Args:
        traj_dir: Directory to search.

    Returns:
        Path to trajectory file or None.
    """
    for ext in ("_production.dcd", "_production.xtc", ".dcd", ".xtc"):
        for fname in sorted(os.listdir(traj_dir)):
            if fname.endswith(ext):
                return os.path.join(traj_dir, fname)
    return None


def load_trajectory_mdtraj(trajectory_path: str, topology_path: str):
    """Load MD trajectory using MDTraj.

    Args:
        trajectory_path: Path to DCD/XTC trajectory.
        topology_path: Path to topology PDB.

    Returns:
        MDTraj Trajectory object.

    Raises:
        ImportError: If MDTraj is not installed.
    """
    try:
        import mdtraj as md
    except ImportError as exc:
        raise ImportError(
            "MDTraj required: conda install -c conda-forge mdtraj"
        ) from exc

    logger.info("Loading trajectory: %s", trajectory_path)
    traj = md.load(trajectory_path, top=topology_path)
    logger.info("Loaded %d frames, %d atoms", traj.n_frames, traj.n_atoms)
    return traj


def calculate_rmsd(traj, gene_name: str) -> pd.DataFrame:
    """Calculate per-frame Cα RMSD relative to frame 0.

    Args:
        traj: MDTraj trajectory object.
        gene_name: Gene name for labelling.

    Returns:
        DataFrame with columns: frame, time_ns, rmsd_nm, rmsd_angstrom.
    """
    import mdtraj as md

    ca_indices = traj.topology.select("name CA")
    traj_aligned = traj.superpose(traj, 0, atom_indices=ca_indices)
    rmsd_nm = md.rmsd(traj_aligned, traj_aligned, 0, atom_indices=ca_indices)
    time_ns = np.arange(len(rmsd_nm)) * REPORT_INTERVAL * TIMESTEP_PS / 1e6

    df = pd.DataFrame({
        "gene": gene_name,
        "frame": np.arange(len(rmsd_nm)),
        "time_ns": time_ns,
        "rmsd_nm": rmsd_nm,
        "rmsd_angstrom": rmsd_nm * 10,
    })
    logger.info(
        "%s RMSD: mean=%.3f nm, max=%.3f nm", gene_name,
        df["rmsd_nm"].mean(), df["rmsd_nm"].max(),
    )
    return df


def calculate_rmsf(traj, gene_name: str) -> pd.DataFrame:
    """Calculate per-residue Cα RMSF over the trajectory.

    Args:
        traj: MDTraj trajectory.
        gene_name: Gene name.

    Returns:
        DataFrame with columns: residue, rmsf_nm, rmsf_angstrom.
    """
    import mdtraj as md

    ca_indices = traj.topology.select("name CA")
    traj_aligned = traj.superpose(traj, 0, atom_indices=ca_indices)
    rmsf_nm = md.rmsf(traj_aligned, traj_aligned, atom_indices=ca_indices)

    res_ids = [traj.topology.atom(i).residue.index + 1 for i in ca_indices]
    res_names = [str(traj.topology.atom(i).residue) for i in ca_indices]

    df = pd.DataFrame({
        "gene": gene_name,
        "residue": res_ids,
        "residue_name": res_names,
        "rmsf_nm": rmsf_nm,
        "rmsf_angstrom": rmsf_nm * 10,
    })
    logger.info(
        "%s RMSF: mean=%.3f nm, top-5 flexible residues: %s",
        gene_name, df["rmsf_nm"].mean(),
        df.nlargest(5, "rmsf_nm")["residue_name"].tolist(),
    )
    return df


def analyze_secondary_structure(traj, gene_name: str) -> pd.DataFrame:
    """Compute secondary structure assignment for every frame.

    Uses DSSP algorithm via MDTraj.

    Args:
        traj: MDTraj trajectory.
        gene_name: Gene name.

    Returns:
        DataFrame with SS composition per frame.
    """
    import mdtraj as md

    logger.info("Computing DSSP secondary structure for %s...", gene_name)
    dssp = md.compute_dssp(traj, simplified=True)  # shape: (n_frames, n_residues)
    n_frames = dssp.shape[0]
    n_res = dssp.shape[1]

    ss_codes = {"H": "helix", "E": "sheet", "C": "coil"}
    rows = []
    for frame_idx in range(n_frames):
        frame_ss = dssp[frame_idx]
        row = {
            "gene": gene_name,
            "frame": frame_idx,
            "time_ns": frame_idx * REPORT_INTERVAL * TIMESTEP_PS / 1e6,
        }
        for code, name in ss_codes.items():
            count = np.sum(frame_ss == code)
            row[f"frac_{name}"] = count / n_res
        rows.append(row)

    logger.info(
        "%s mean helix fraction: %.2f%%",
        gene_name, np.mean([r["frac_helix"] for r in rows]) * 100,
    )
    return pd.DataFrame(rows)


def analyze_ion_contacts(traj, gene_name: str, cutoff_nm: float = 0.35) -> pd.DataFrame:
    """Track protein-Na+ contacts over the trajectory.

    Args:
        traj: MDTraj trajectory.
        gene_name: Gene name.
        cutoff_nm: Distance cutoff for contact (default 3.5 Å).

    Returns:
        DataFrame with per-frame sodium ion contact counts.
    """
    import mdtraj as md

    # Find sodium ions
    na_indices = traj.topology.select("resname NA or resname SOD")
    if len(na_indices) == 0:
        logger.warning("%s: No sodium ions found in trajectory.", gene_name)
        return pd.DataFrame()

    protein_indices = traj.topology.select("protein and (name OD1 or name OD2 or "
                                           "name OE1 or name OE2 or name O or name N)")

    rows = []
    for frame_idx in range(traj.n_frames):
        frame = traj[frame_idx]
        positions = frame.xyz[0]  # shape: (n_atoms, 3)
        n_contacts = 0
        for na_idx in na_indices:
            na_pos = positions[na_idx]
            dists = np.linalg.norm(positions[protein_indices] - na_pos, axis=1)
            n_contacts += int(np.sum(dists <= cutoff_nm))
        rows.append({
            "gene": gene_name,
            "frame": frame_idx,
            "time_ns": frame_idx * REPORT_INTERVAL * TIMESTEP_PS / 1e6,
            "n_protein_ion_contacts": n_contacts,
            "n_sodium_ions": len(na_indices),
        })

    df = pd.DataFrame(rows)
    logger.info(
        "%s Na+ contacts: mean=%.1f, max=%d",
        gene_name, df["n_protein_ion_contacts"].mean(),
        df["n_protein_ion_contacts"].max(),
    )
    return df


def identify_conformational_states(rmsd_df: pd.DataFrame, n_states: int = 3) -> pd.DataFrame:
    """Cluster trajectory frames by RMSD into conformational states.

    Uses a simple k-means clustering on RMSD values.

    Args:
        rmsd_df: RMSD DataFrame from calculate_rmsd().
        n_states: Number of conformational states to identify.

    Returns:
        RMSD DataFrame with added 'state' column.
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        logger.warning("scikit-learn not available — skipping state clustering.")
        rmsd_df["state"] = 0
        return rmsd_df

    X = rmsd_df[["rmsd_nm"]].values
    km = KMeans(n_clusters=n_states, random_state=42, n_init=10)
    rmsd_df = rmsd_df.copy()
    rmsd_df["state"] = km.fit_predict(X)

    for state in range(n_states):
        sub = rmsd_df[rmsd_df["state"] == state]
        logger.info(
            "State %d: %d frames (%.1f%%), RMSD=%.3f±%.3f nm",
            state, len(sub), len(sub) / len(rmsd_df) * 100,
            sub["rmsd_nm"].mean(), sub["rmsd_nm"].std(),
        )
    return rmsd_df


def plot_rmsd(rmsd_df: pd.DataFrame, output_dir: str) -> None:
    """Plot RMSD over simulation time.

    Args:
        rmsd_df: DataFrame from calculate_rmsd().
        output_dir: Output directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping RMSD plot.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    for gene, sub in rmsd_df.groupby("gene"):
        ax.plot(sub["time_ns"], sub["rmsd_angstrom"], label=gene, linewidth=0.8)
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel("RMSD (Å)", fontsize=12)
    ax.set_title("Cα RMSD over MD Simulation", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "rmsd_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved RMSD plot: %s", path)


def plot_rmsf(rmsf_df: pd.DataFrame, output_dir: str) -> None:
    """Plot per-residue RMSF.

    Args:
        rmsf_df: DataFrame from calculate_rmsf().
        output_dir: Output directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping RMSF plot.")
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = {"SCN1A": "#1565C0", "SCN2A": "#E53935"}
    for gene, sub in rmsf_df.groupby("gene"):
        ax.plot(
            sub["residue"], sub["rmsf_angstrom"],
            label=gene, color=colors.get(gene), linewidth=0.8,
        )
    ax.set_xlabel("Residue", fontsize=12)
    ax.set_ylabel("RMSF (Å)", fontsize=12)
    ax.set_title("Per-Residue RMSF", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "rmsf_plot.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved RMSF plot: %s", path)


def plot_secondary_structure(ss_df: pd.DataFrame, output_dir: str) -> None:
    """Plot secondary structure composition over time.

    Args:
        ss_df: DataFrame from analyze_secondary_structure().
        output_dir: Output directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    genes = ss_df["gene"].unique()
    fig, axes = plt.subplots(len(genes), 1, figsize=(14, 4 * len(genes)), sharex=True)
    if len(genes) == 1:
        axes = [axes]

    ss_colors = {"frac_helix": "#1565C0", "frac_sheet": "#2E7D32", "frac_coil": "#E53935"}
    for ax, gene in zip(axes, genes):
        sub = ss_df[ss_df["gene"] == gene]
        for col, color in ss_colors.items():
            ax.plot(sub["time_ns"], sub[col] * 100, label=col.replace("frac_", ""),
                    color=color, linewidth=0.8)
        ax.set_ylabel("Fraction (%)", fontsize=11)
        ax.set_title(f"{gene} Secondary Structure", fontsize=12)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (ns)", fontsize=12)
    plt.suptitle("Secondary Structure Composition Over Simulation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "secondary_structure_timeline.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved secondary structure plot: %s", path)


def main(trajectory_dir: str, topology_path: str, output_dir: str, gene_name: str) -> None:
    """Main trajectory analysis workflow.

    Args:
        trajectory_dir: Directory containing trajectory files.
        topology_path: PDB topology file.
        output_dir: Directory for results.
        gene_name: Gene identifier.
    """
    os.makedirs(output_dir, exist_ok=True)

    traj_path = find_trajectory_file(trajectory_dir)
    if traj_path is None:
        logger.error(
            "No trajectory file found in %s. Run 02_run_md_simulation.py first.",
            trajectory_dir,
        )
        sys.exit(1)

    if not os.path.exists(topology_path):
        logger.error("Topology file not found: %s", topology_path)
        sys.exit(1)

    try:
        traj = load_trajectory_mdtraj(traj_path, topology_path)
    except ImportError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # RMSD analysis
    rmsd_df = calculate_rmsd(traj, gene_name)
    rmsd_df = identify_conformational_states(rmsd_df)
    rmsd_df.to_csv(os.path.join(output_dir, f"{gene_name.lower()}_rmsd.csv"), index=False)

    # RMSF analysis
    rmsf_df = calculate_rmsf(traj, gene_name)
    rmsf_df.to_csv(os.path.join(output_dir, f"{gene_name.lower()}_rmsf.csv"), index=False)

    # Secondary structure
    ss_df = analyze_secondary_structure(traj, gene_name)
    ss_df.to_csv(os.path.join(output_dir, f"{gene_name.lower()}_secondary_structure.csv"), index=False)

    # Ion contacts
    ion_df = analyze_ion_contacts(traj, gene_name)
    if not ion_df.empty:
        ion_df.to_csv(os.path.join(output_dir, f"{gene_name.lower()}_ion_contacts.csv"), index=False)

    # Plots
    plot_rmsd(rmsd_df, output_dir)
    plot_rmsf(rmsf_df, output_dir)
    plot_secondary_structure(ss_df, output_dir)

    logger.info("Trajectory analysis complete. Results in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze MD trajectory for SCN1A/SCN2A."
    )
    parser.add_argument("--trajectory_dir", required=True, help="Directory with trajectory files")
    parser.add_argument("--topology", required=True, help="Topology PDB file")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "md_analysis"),
    )
    parser.add_argument("--gene", default="SCN1A", choices=["SCN1A", "SCN2A"])
    args = parser.parse_args()
    main(args.trajectory_dir, args.topology, args.output_dir, args.gene)
