"""
Stage 1 - Step 1: Prepare Protein Sequences
============================================
Fetch SCN1A (P35498) and SCN2A (Q99250) protein sequences from UniProt,
validate them, save FASTA files, and print summary statistics.

Usage:
    python 01_prepare_sequences.py --output_dir ../data/raw/sequences
"""

import argparse
import logging
import os
import re
import sys

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

UNIPROT_IDS = {
    "SCN1A": "P35498",
    "SCN2A": "Q99250",
}

STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

UNIPROT_FASTA_URL = "https://www.uniprot.org/uniprot/{uniprot_id}.fasta"


def fetch_uniprot_sequence(uniprot_id: str) -> tuple[str, str]:
    """Fetch protein sequence from UniProt REST API.

    Args:
        uniprot_id: UniProt accession number (e.g., 'P35498').

    Returns:
        Tuple of (header, sequence) strings.

    Raises:
        RuntimeError: If the request fails or sequence is empty.
    """
    url = UNIPROT_FASTA_URL.format(uniprot_id=uniprot_id)
    logger.info("Fetching sequence for %s from %s", uniprot_id, url)
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch {uniprot_id}: HTTP {response.status_code}"
        )
    fasta_text = response.text.strip()
    if not fasta_text:
        raise RuntimeError(f"Empty response for {uniprot_id}")
    lines = fasta_text.splitlines()
    header = lines[0]
    sequence = "".join(lines[1:])
    if not sequence:
        raise RuntimeError(f"Parsed empty sequence for {uniprot_id}")
    return header, sequence


def validate_sequence(gene_name: str, sequence: str) -> dict:
    """Validate a protein sequence for correctness.

    Checks:
    - Non-zero length
    - Contains only standard amino acid characters
    - Plausible length range for a voltage-gated sodium channel (~1800–2200 aa)

    Args:
        gene_name: Gene identifier string (e.g., 'SCN1A').
        sequence: Single-letter amino acid sequence.

    Returns:
        Dictionary with validation results.
    """
    results = {
        "gene": gene_name,
        "length": len(sequence),
        "valid_length": 1500 <= len(sequence) <= 2500,
        "non_standard_residues": [],
        "is_valid": True,
    }
    non_standard = sorted(set(sequence.upper()) - STANDARD_AMINO_ACIDS)
    results["non_standard_residues"] = non_standard
    if non_standard:
        logger.warning(
            "%s contains non-standard residues: %s", gene_name, non_standard
        )
    if not results["valid_length"]:
        logger.warning(
            "%s length %d is outside expected range (1500–2500)",
            gene_name,
            len(sequence),
        )
    results["is_valid"] = results["valid_length"] and not non_standard
    return results


def compute_sequence_stats(sequence: str) -> dict:
    """Compute basic amino acid composition statistics.

    Args:
        sequence: Single-letter amino acid sequence.

    Returns:
        Dictionary with residue counts and percentages.
    """
    sequence = sequence.upper()
    total = len(sequence)
    counts = {aa: sequence.count(aa) for aa in STANDARD_AMINO_ACIDS}
    percentages = {aa: round(count / total * 100, 2) for aa, count in counts.items()}
    charged_pos = sum(sequence.count(aa) for aa in "KRH")
    charged_neg = sum(sequence.count(aa) for aa in "DE")
    hydrophobic = sum(sequence.count(aa) for aa in "AILMFWV")
    return {
        "total_residues": total,
        "residue_counts": counts,
        "residue_percentages": percentages,
        "positively_charged": charged_pos,
        "negatively_charged": charged_neg,
        "net_charge_proxy": charged_pos - charged_neg,
        "hydrophobic_count": hydrophobic,
        "hydrophobic_fraction": round(hydrophobic / total, 4),
    }


def save_fasta(output_path: str, header: str, sequence: str, line_width: int = 60) -> None:
    """Save a protein sequence to a FASTA file.

    Args:
        output_path: Destination file path.
        header: FASTA header line (must start with '>').
        sequence: Amino acid sequence.
        line_width: Characters per line in the sequence block.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not header.startswith(">"):
        header = ">" + header
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for i in range(0, len(sequence), line_width):
            fh.write(sequence[i : i + line_width] + "\n")
    logger.info("Saved FASTA to %s", output_path)


def print_summary(gene_name: str, header: str, validation: dict, stats: dict) -> None:
    """Print a formatted summary of sequence information.

    Args:
        gene_name: Gene name string.
        header: FASTA header.
        validation: Output of validate_sequence().
        stats: Output of compute_sequence_stats().
    """
    print(f"\n{'='*60}")
    print(f"  {gene_name} Sequence Summary")
    print(f"{'='*60}")
    print(f"  Header        : {header[:80]}")
    print(f"  Length        : {stats['total_residues']} aa")
    print(f"  Valid         : {validation['is_valid']}")
    print(f"  + Charged (K,R,H): {stats['positively_charged']}")
    print(f"  - Charged (D,E)  : {stats['negatively_charged']}")
    print(f"  Net charge proxy : {stats['net_charge_proxy']}")
    print(f"  Hydrophobic frac : {stats['hydrophobic_fraction']:.2%}")
    if validation["non_standard_residues"]:
        print(f"  Non-standard aa  : {validation['non_standard_residues']}")
    print(f"{'='*60}\n")


def main(output_dir: str) -> None:
    """Main workflow: fetch, validate, save, and report sequences.

    Args:
        output_dir: Directory where FASTA files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_valid = True

    for gene_name, uniprot_id in UNIPROT_IDS.items():
        try:
            header, sequence = fetch_uniprot_sequence(uniprot_id)
        except RuntimeError as exc:
            logger.error("Could not fetch %s: %s", gene_name, exc)
            all_valid = False
            continue

        validation = validate_sequence(gene_name, sequence)
        stats = compute_sequence_stats(sequence)

        output_path = os.path.join(output_dir, f"{gene_name}_{uniprot_id}.fasta")
        save_fasta(output_path, header, sequence)
        print_summary(gene_name, header, validation, stats)

        if not validation["is_valid"]:
            all_valid = False

    if all_valid:
        logger.info("All sequences fetched and validated successfully.")
    else:
        logger.warning("One or more sequences failed validation — review warnings above.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and validate SCN1A/SCN2A protein sequences from UniProt."
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "raw", "sequences"),
        help="Directory to save FASTA files (default: ../data/raw/sequences)",
    )
    args = parser.parse_args()
    main(args.output_dir)
