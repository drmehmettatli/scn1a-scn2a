"""
utils/sequence_utils.py
=======================
Utility functions for protein sequence handling:
  - Fetching sequences from UniProt
  - FASTA parsing/writing
  - Variant generation
  - Basic sequence statistics
"""

import os
import re
from typing import Generator

import numpy as np
import requests

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_SET = set(AMINO_ACIDS)

ONE_TO_THREE = {
    "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
    "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
    "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
    "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
}

THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}


def fetch_uniprot_sequence(uniprot_id: str, timeout: int = 30) -> tuple[str, str]:
    """Fetch protein sequence from UniProt REST API.

    Args:
        uniprot_id: UniProt accession (e.g., 'P35498').
        timeout: HTTP request timeout in seconds.

    Returns:
        Tuple of (fasta_header, sequence).

    Raises:
        RuntimeError: If fetch fails or sequence is empty.
    """
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for UniProt {uniprot_id}")
    text = response.text.strip()
    lines = text.splitlines()
    header = lines[0].lstrip(">")
    sequence = "".join(l for l in lines[1:] if l.strip())
    if not sequence:
        raise RuntimeError(f"Empty sequence for {uniprot_id}")
    return header, sequence


def read_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Read a (multi-)FASTA file.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        List of (header, sequence) tuples.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file contains no sequences.
    """
    if not os.path.exists(fasta_path):
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    records = []
    header = None
    seq_lines: list[str] = []
    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_lines)))
                header = line.lstrip(">")
                seq_lines = []
            else:
                seq_lines.append(line)
    if header is not None:
        records.append((header, "".join(seq_lines)))
    if not records:
        raise ValueError(f"No sequences found in {fasta_path}")
    return records


def write_fasta(
    records: list[tuple[str, str]],
    output_path: str,
    line_width: int = 60,
) -> None:
    """Write sequences to FASTA format.

    Args:
        records: List of (header, sequence) tuples.
        output_path: Destination file path.
        line_width: Sequence characters per line.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for header, sequence in records:
            if not header.startswith(">"):
                header = ">" + header
            fh.write(header + "\n")
            for i in range(0, len(sequence), line_width):
                fh.write(sequence[i : i + line_width] + "\n")


def validate_sequence(sequence: str, min_len: int = 100, max_len: int = 5000) -> dict:
    """Validate a protein sequence.

    Args:
        sequence: Amino acid sequence string.
        min_len: Minimum acceptable length.
        max_len: Maximum acceptable length.

    Returns:
        Dictionary with 'is_valid', 'length', 'non_standard_residues' keys.
    """
    non_standard = sorted(set(sequence.upper()) - STANDARD_AA_SET)
    return {
        "length": len(sequence),
        "is_valid": min_len <= len(sequence) <= max_len and not non_standard,
        "non_standard_residues": non_standard,
        "valid_length": min_len <= len(sequence) <= max_len,
    }


def compute_sequence_stats(sequence: str) -> dict:
    """Compute basic amino acid composition statistics.

    Args:
        sequence: Protein sequence.

    Returns:
        Dictionary with total_residues, residue_percentages,
        charge_counts, and hydrophobic_fraction.
    """
    sequence = sequence.upper()
    n = len(sequence)
    counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
    charged_pos = sum(sequence.count(aa) for aa in "KRH")
    charged_neg = sum(sequence.count(aa) for aa in "DE")
    hydrophobic = sum(sequence.count(aa) for aa in "AILMFWV")
    return {
        "total_residues": n,
        "residue_counts": counts,
        "residue_percentages": {aa: round(c / n * 100, 2) for aa, c in counts.items()},
        "positively_charged": charged_pos,
        "negatively_charged": charged_neg,
        "net_charge_proxy": charged_pos - charged_neg,
        "hydrophobic_count": hydrophobic,
        "hydrophobic_fraction": round(hydrophobic / n, 4),
    }


def generate_all_missense_variants(
    gene_name: str,
    sequence: str,
) -> Generator[dict, None, None]:
    """Generate all single amino acid substitutions for a sequence.

    Args:
        gene_name: Gene identifier.
        sequence: Wild-type protein sequence.

    Yields:
        Dictionary with gene, position (1-indexed), wt_aa, mut_aa,
        variant_name, and hgvs_p keys.
    """
    for pos_0, wt_aa in enumerate(sequence):
        if wt_aa not in STANDARD_AA_SET:
            continue
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            pos_1 = pos_0 + 1
            yield {
                "gene": gene_name,
                "position": pos_1,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "variant_name": f"{wt_aa}{pos_1}{mut_aa}",
                "hgvs_p": f"p.{ONE_TO_THREE.get(wt_aa, wt_aa)}{pos_1}{ONE_TO_THREE.get(mut_aa, mut_aa)}",
            }


def one_to_three(aa: str) -> str:
    """Convert single-letter to three-letter amino acid code.

    Args:
        aa: Single-letter code.

    Returns:
        Three-letter code or the input if not found.
    """
    return ONE_TO_THREE.get(aa.upper(), aa)


def three_to_one(aa3: str) -> str:
    """Convert three-letter to single-letter amino acid code.

    Args:
        aa3: Three-letter code.

    Returns:
        Single-letter code or '?' if not found.
    """
    return THREE_TO_ONE.get(aa3.capitalize(), "?")


def parse_hgvs_p(hgvs_p: str) -> tuple[str, int, str] | None:
    """Parse HGVS protein notation into (wt_aa, position, mut_aa).

    Args:
        hgvs_p: HGVS notation like 'p.Arg1234His' or 'p.R1234H'.

    Returns:
        Tuple (wt_aa_1letter, position, mut_aa_1letter) or None if unparseable.
    """
    # Try three-letter code: p.Arg1234His
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})", hgvs_p)
    if m:
        wt = three_to_one(m.group(1))
        pos = int(m.group(2))
        mut = three_to_one(m.group(3))
        return wt, pos, mut
    # Try single-letter: p.R1234H or R1234H
    m = re.match(r"p?\.?([A-Z])(\d+)([A-Z])", hgvs_p)
    if m:
        return m.group(1), int(m.group(2)), m.group(3)
    return None


def align_sequences(seq1: str, seq2: str) -> tuple[str, str, float]:
    """Perform simple global sequence alignment using Needleman-Wunsch.

    For production use, prefer Bio.pairwise2 or parasail.

    Args:
        seq1: First protein sequence.
        seq2: Second protein sequence.

    Returns:
        Tuple of (aligned_seq1, aligned_seq2, percent_identity).
    """
    try:
        from Bio import pairwise2
        from Bio.pairwise2 import format_alignment

        alignments = pairwise2.align.globalms(seq1, seq2, 2, -1, -5, -0.5)
        if alignments:
            a = alignments[0]
            aln1, aln2 = a.seqA, a.seqB
            matches = sum(a == b for a, b in zip(aln1, aln2) if a != "-" and b != "-")
            aligned_len = sum(1 for a, b in zip(aln1, aln2) if a != "-" or b != "-")
            pct_id = matches / aligned_len * 100 if aligned_len > 0 else 0.0
            return aln1, aln2, round(pct_id, 2)
    except ImportError:
        pass

    # Fallback: just return sequences and estimate identity from overlapping positions
    min_len = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
    return seq1, seq2, round(matches / min_len * 100, 2)
