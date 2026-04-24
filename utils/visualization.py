"""
utils/visualization.py
======================
Reusable visualization functions for the SCN1A/SCN2A pipeline:
  - pLDDT per-residue confidence plots
  - Variant pathogenicity heatmaps
  - MD trajectory RMSD/RMSF plots
  - Comparison bar charts
"""

import os

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


def _check_matplotlib() -> None:
    """Raise ImportError if matplotlib is not available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required: pip install matplotlib")


def plot_plddt(
    plddt: np.ndarray,
    gene_name: str,
    output_path: str,
    title: str | None = None,
    figsize: tuple[int, int] = (18, 4),
    dpi: int = 150,
) -> None:
    """Plot per-residue pLDDT confidence scores with colour bands.

    Args:
        plddt: Array of pLDDT values (0–100).
        gene_name: Gene name for axis labels.
        output_path: Path to save the figure.
        title: Optional custom title.
        figsize: Figure dimensions (width, height) in inches.
        dpi: Output resolution.
    """
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    residues = np.arange(1, len(plddt) + 1)

    colors = np.where(
        plddt >= 90, "#1565C0",
        np.where(plddt >= 70, "#43A047",
                 np.where(plddt >= 50, "#FFB300", "#E53935")),
    )
    ax.bar(residues, plddt, color=colors, width=1.0, linewidth=0)
    for thresh, color in [(90, "#1565C0"), (70, "#43A047"), (50, "#FFB300")]:
        ax.axhline(thresh, color=color, linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_xlim(0, len(plddt) + 1)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Residue position", fontsize=12)
    ax.set_ylabel("pLDDT", fontsize=12)
    ax.set_title(title or f"{gene_name} — AlphaFold3 pLDDT Confidence", fontsize=13, fontweight="bold")

    patches = [
        mpatches.Patch(color="#1565C0", label="Very high (≥90)"),
        mpatches.Patch(color="#43A047", label="High (70–90)"),
        mpatches.Patch(color="#FFB300", label="Low (50–70)"),
        mpatches.Patch(color="#E53935", label="Very low (<50)"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_variant_heatmap(
    score_matrix: np.ndarray,
    positions: np.ndarray,
    amino_acids: list[str],
    gene_name: str,
    output_path: str,
    title: str | None = None,
    colorbar_label: str = "Pathogenicity Score",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn_r",
    figsize: tuple[int, int] = (20, 6),
    dpi: int = 150,
) -> None:
    """Plot a variant pathogenicity heatmap (positions × amino acids).

    Args:
        score_matrix: 2D array of shape (len(positions), len(amino_acids)).
        positions: Array of residue positions (x-axis labels).
        amino_acids: List of amino acid labels (y-axis).
        gene_name: Gene name.
        output_path: Output file path.
        title: Optional plot title.
        colorbar_label: Label for the colorbar.
        vmin: Minimum score for colour scale.
        vmax: Maximum score for colour scale.
        cmap: Matplotlib colormap name.
        figsize: Figure size.
        dpi: Output resolution.
    """
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        score_matrix.T,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)

    # X-axis: sample every N positions for readability
    n_pos = len(positions)
    step = max(1, n_pos // 50)
    ax.set_xticks(np.arange(0, n_pos, step))
    ax.set_xticklabels(positions[::step], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(amino_acids)))
    ax.set_yticklabels(amino_acids, fontsize=8)

    ax.set_xlabel("Residue Position", fontsize=11)
    ax.set_ylabel("Mutant Amino Acid", fontsize=11)
    ax.set_title(
        title or f"{gene_name} — Variant Pathogenicity Heatmap",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_rmsd(
    time_ns: np.ndarray,
    rmsd_values: np.ndarray,
    gene_name: str,
    output_path: str,
    label: str = "RMSD",
    color: str = "#1565C0",
    units: str = "Å",
    figsize: tuple[int, int] = (12, 4),
    dpi: int = 150,
) -> None:
    """Plot RMSD over simulation time.

    Args:
        time_ns: Time array in nanoseconds.
        rmsd_values: RMSD values in Å or nm.
        gene_name: Gene name for title.
        output_path: Output file path.
        label: Y-axis legend label.
        color: Line color.
        units: RMSD units for axis label.
        figsize: Figure dimensions.
        dpi: Output resolution.
    """
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_ns, rmsd_values, color=color, linewidth=0.8, label=label)
    # Running mean
    if len(rmsd_values) >= 50:
        window = max(5, len(rmsd_values) // 50)
        running_mean = pd.Series(rmsd_values).rolling(window, center=True).mean().values
        ax.plot(time_ns, running_mean, color="red", linewidth=1.5, label=f"Running mean ({window} frames)")
    ax.set_xlabel("Time (ns)", fontsize=12)
    ax.set_ylabel(f"RMSD ({units})", fontsize=12)
    ax.set_title(f"{gene_name} — Cα RMSD over MD Simulation", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_rmsf(
    residues: np.ndarray,
    rmsf_values: np.ndarray,
    gene_name: str,
    output_path: str,
    color: str = "#1565C0",
    units: str = "Å",
    highlight_regions: list[tuple[int, int, str]] | None = None,
    figsize: tuple[int, int] = (16, 5),
    dpi: int = 150,
) -> None:
    """Plot per-residue RMSF.

    Args:
        residues: Residue number array.
        rmsf_values: RMSF values.
        gene_name: Gene name.
        output_path: Output file path.
        color: Line color.
        units: RMSF units.
        highlight_regions: Optional list of (start, end, label) tuples for shading.
        figsize: Figure dimensions.
        dpi: Output resolution.
    """
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(residues, rmsf_values, color=color, linewidth=0.8)
    ax.fill_between(residues, rmsf_values, alpha=0.2, color=color)

    if highlight_regions:
        region_colors = plt.cm.tab10.colors
        for i, (start, end, label) in enumerate(highlight_regions):
            ax.axvspan(start, end, alpha=0.15, color=region_colors[i % 10], label=label)
        ax.legend(fontsize=8)

    ax.set_xlabel("Residue", fontsize=12)
    ax.set_ylabel(f"RMSF ({units})", fontsize=12)
    ax.set_title(f"{gene_name} — Per-Residue RMSF", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_comparison_bar(
    data: dict[str, float],
    title: str,
    output_path: str,
    ylabel: str = "Value",
    color_map: dict[str, str] | None = None,
    figsize: tuple[int, int] = (10, 5),
    dpi: int = 150,
) -> None:
    """Plot a simple bar chart for comparing values across categories.

    Args:
        data: Dict of {category: value}.
        title: Plot title.
        output_path: Output file path.
        ylabel: Y-axis label.
        color_map: Optional dict of {category: color}.
        figsize: Figure dimensions.
        dpi: Output resolution.
    """
    _check_matplotlib()
    categories = list(data.keys())
    values = [data[k] for k in categories]
    colors = [color_map.get(c, "#1565C0") if color_map else "#1565C0" for c in categories]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", fontsize=9,
        )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


def plot_score_distribution(
    scores: dict[str, np.ndarray],
    output_path: str,
    title: str = "Score Distribution",
    xlabel: str = "Score",
    bins: int = 50,
    figsize: tuple[int, int] = (10, 5),
    dpi: int = 150,
) -> None:
    """Plot overlapping histograms for multiple score arrays.

    Args:
        scores: Dict of {label: score_array}.
        output_path: Output file path.
        title: Plot title.
        xlabel: X-axis label.
        bins: Number of histogram bins.
        figsize: Figure dimensions.
        dpi: Output resolution.
    """
    _check_matplotlib()
    colors = list(mcolors.TABLEAU_COLORS.values())
    fig, ax = plt.subplots(figsize=figsize)
    for i, (label, vals) in enumerate(scores.items()):
        vals_clean = vals[~np.isnan(vals)]
        ax.hist(vals_clean, bins=bins, alpha=0.6, label=label, color=colors[i % len(colors)], density=True)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close()
