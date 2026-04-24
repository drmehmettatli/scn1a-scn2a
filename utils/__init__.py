"""
utils package for SCN1A/SCN2A epilepsy research pipeline.
"""

from .sequence_utils import (
    fetch_uniprot_sequence,
    read_fasta,
    write_fasta,
    generate_all_missense_variants,
    validate_sequence,
    compute_sequence_stats,
)
from .structure_utils import (
    parse_pdb_atoms,
    calculate_rmsd_kabsch,
    compute_contact_map,
    find_binding_site_residues,
    extract_ca_coords,
)
from .visualization import (
    plot_plddt,
    plot_variant_heatmap,
    plot_rmsd,
    plot_rmsf,
    plot_comparison_bar,
)

__all__ = [
    "fetch_uniprot_sequence",
    "read_fasta",
    "write_fasta",
    "generate_all_missense_variants",
    "validate_sequence",
    "compute_sequence_stats",
    "parse_pdb_atoms",
    "calculate_rmsd_kabsch",
    "compute_contact_map",
    "find_binding_site_residues",
    "extract_ca_coords",
    "plot_plddt",
    "plot_variant_heatmap",
    "plot_rmsd",
    "plot_rmsf",
    "plot_comparison_bar",
]

__version__ = "1.0.0"
