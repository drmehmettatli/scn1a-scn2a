"""
Stage 3 - Step 1: Generate Missense Variants
=============================================
Generate all possible missense variants for SCN1A and SCN2A,
load known ClinVar classifications, and output variant lists
in CSV and VCF formats.

Usage:
    python 01_generate_variants.py \
        --sequences_dir ../data/raw/sequences \
        --clinvar_dir ../data/raw/variants \
        --output_dir ../data/processed/variants
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
UNIPROT_IDS = {"SCN1A": "P35498", "SCN2A": "Q99250"}
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ClinVar significance mapping
CLINVAR_SIGNIFICANCE = {
    "Pathogenic": "pathogenic",
    "Likely pathogenic": "likely_pathogenic",
    "Pathogenic/Likely pathogenic": "pathogenic",
    "Benign": "benign",
    "Likely benign": "likely_benign",
    "Benign/Likely benign": "benign",
    "Uncertain significance": "vus",
    "Conflicting interpretations of pathogenicity": "conflicting",
}


def read_fasta(fasta_path: str) -> tuple[str, str]:
    """Read FASTA file and return (gene_name, sequence).

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Tuple (header, sequence).
    """
    with open(fasta_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header = [l.lstrip(">") for l in lines if l.startswith(">")][0]
    sequence = "".join(l for l in lines if not l.startswith(">"))
    return header, sequence


def generate_all_missense_variants(gene_name: str, sequence: str) -> pd.DataFrame:
    """Generate all possible single amino acid substitutions.

    For each position (1-indexed) and each target amino acid (excluding
    the wild-type), creates one variant entry.

    Args:
        gene_name: Gene identifier.
        sequence: Wild-type protein sequence.

    Returns:
        DataFrame with columns: gene, position, wt_aa, mut_aa, variant_name.
    """
    rows = []
    for pos_0, wt_aa in enumerate(sequence):
        if wt_aa not in AMINO_ACIDS:
            continue
        for mut_aa in AMINO_ACIDS:
            if mut_aa == wt_aa:
                continue
            rows.append({
                "gene": gene_name,
                "position": pos_0 + 1,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "variant_name": f"{wt_aa}{pos_0 + 1}{mut_aa}",
                "hgvs_p": f"p.{_one_to_three(wt_aa)}{pos_0 + 1}{_one_to_three(mut_aa)}",
            })
    logger.info(
        "%s: generated %d missense variants (%d positions × 19 substitutions)",
        gene_name, len(rows), len(sequence),
    )
    return pd.DataFrame(rows)


def _one_to_three(aa: str) -> str:
    """Convert single-letter amino acid code to three-letter code.

    Args:
        aa: Single-letter amino acid code.

    Returns:
        Three-letter amino acid code.
    """
    mapping = {
        "A": "Ala", "C": "Cys", "D": "Asp", "E": "Glu", "F": "Phe",
        "G": "Gly", "H": "His", "I": "Ile", "K": "Lys", "L": "Leu",
        "M": "Met", "N": "Asn", "P": "Pro", "Q": "Gln", "R": "Arg",
        "S": "Ser", "T": "Thr", "V": "Val", "W": "Trp", "Y": "Tyr",
    }
    return mapping.get(aa.upper(), aa)


def load_clinvar_variants(clinvar_vcf_path: str, gene_name: str) -> pd.DataFrame:
    """Parse ClinVar VCF file for a gene and extract missense variants.

    Args:
        clinvar_vcf_path: Path to ClinVar VCF file.
        gene_name: Gene name for filtering.

    Returns:
        DataFrame with ClinVar missense variants and classifications.
    """
    if not os.path.exists(clinvar_vcf_path):
        logger.warning("ClinVar VCF not found: %s", clinvar_vcf_path)
        return pd.DataFrame()

    rows = []
    with open(clinvar_vcf_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 8:
                continue
            info = parts[7]
            # Filter by gene
            if f"GENEINFO={gene_name}" not in info and f"GENEINFO={gene_name}:" not in info:
                continue
            # Parse clinical significance
            clnsig = ""
            for field in info.split(";"):
                if field.startswith("CLNSIG="):
                    clnsig = field[7:].replace("_", " ")
                    break
            if not clnsig:
                continue
            # Parse protein change
            hgvs_p = ""
            for field in info.split(";"):
                if field.startswith("CLNHGVS="):
                    hgvs_p = field[8:]
                    break
            rows.append({
                "gene": gene_name,
                "chrom": parts[0],
                "pos": int(parts[1]),
                "ref": parts[3],
                "alt": parts[4],
                "clinvar_significance": CLINVAR_SIGNIFICANCE.get(clnsig, "unknown"),
                "clinvar_significance_raw": clnsig,
                "hgvs_p": hgvs_p,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        logger.info(
            "%s ClinVar: %d variants (%s)",
            gene_name,
            len(df),
            df["clinvar_significance"].value_counts().to_dict(),
        )
    return df


def annotate_with_clinvar(variants_df: pd.DataFrame, clinvar_df: pd.DataFrame) -> pd.DataFrame:
    """Merge variant list with ClinVar classifications.

    Matches on HGVS protein notation.

    Args:
        variants_df: All possible variants DataFrame.
        clinvar_df: ClinVar variants DataFrame.

    Returns:
        Annotated variants DataFrame.
    """
    if clinvar_df.empty:
        variants_df["clinvar_significance"] = "not_in_clinvar"
        return variants_df

    clinvar_lookup = (
        clinvar_df[["hgvs_p", "clinvar_significance"]]
        .drop_duplicates("hgvs_p")
        .set_index("hgvs_p")["clinvar_significance"]
        .to_dict()
    )

    variants_df = variants_df.copy()
    variants_df["clinvar_significance"] = variants_df["hgvs_p"].map(clinvar_lookup).fillna("not_in_clinvar")
    n_annotated = (variants_df["clinvar_significance"] != "not_in_clinvar").sum()
    logger.info(
        "%s: %d / %d variants annotated from ClinVar",
        variants_df["gene"].iloc[0] if not variants_df.empty else "?",
        n_annotated, len(variants_df),
    )
    return variants_df


def save_vcf(variants_df: pd.DataFrame, output_path: str, gene_name: str) -> None:
    """Save variants in VCF format (without genomic coordinates).

    Produces a pseudo-VCF with protein-level information for tool input.

    Args:
        variants_df: Variants DataFrame.
        output_path: Destination file path.
        gene_name: Gene name for header.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write(f"##Gene={gene_name}\n")
        fh.write("##INFO=<ID=AA_POS,Number=1,Type=Integer,Description=\"Amino acid position\">\n")
        fh.write("##INFO=<ID=WT_AA,Number=1,Type=String,Description=\"Wild-type amino acid\">\n")
        fh.write("##INFO=<ID=MUT_AA,Number=1,Type=String,Description=\"Mutant amino acid\">\n")
        fh.write("##INFO=<ID=CLNSIG,Number=1,Type=String,Description=\"ClinVar significance\">\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for _, row in variants_df.iterrows():
            info = (
                f"AA_POS={row['position']};"
                f"WT_AA={row['wt_aa']};"
                f"MUT_AA={row['mut_aa']};"
                f"CLNSIG={row.get('clinvar_significance', '.')}"
            )
            fh.write(
                f"{gene_name}\t{row['position']}\t{row['variant_name']}\t"
                f"{row['wt_aa']}\t{row['mut_aa']}\t.\t.\t{info}\n"
            )
    logger.info("Saved VCF: %s (%d variants)", output_path, len(variants_df))


def print_summary(gene_name: str, df: pd.DataFrame) -> None:
    """Print a summary of generated variants.

    Args:
        gene_name: Gene name.
        df: Variants DataFrame.
    """
    print(f"\n{'='*55}")
    print(f"  {gene_name} Variant Summary")
    print(f"{'='*55}")
    print(f"  Total variants    : {len(df):,}")
    print(f"  Unique positions  : {df['position'].nunique():,}")
    if "clinvar_significance" in df.columns:
        sig_counts = df["clinvar_significance"].value_counts()
        print(f"\n  ClinVar breakdown:")
        for sig, count in sig_counts.items():
            print(f"    {sig:<30}: {count:,}")
    print(f"{'='*55}\n")


def main(sequences_dir: str, clinvar_dir: str, output_dir: str) -> None:
    """Main workflow: generate, annotate, and save all missense variants.

    Args:
        sequences_dir: Directory with FASTA files.
        clinvar_dir: Directory with ClinVar VCF files.
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for gene_name in GENES:
        uniprot_id = UNIPROT_IDS[gene_name]
        # Find FASTA
        fasta_candidates = [
            os.path.join(sequences_dir, f"{gene_name}_{uniprot_id}.fasta"),
            os.path.join(sequences_dir, f"{gene_name}.fasta"),
        ]
        fasta_path = next((p for p in fasta_candidates if os.path.exists(p)), None)
        if fasta_path is None:
            logger.error(
                "FASTA not found for %s. Run stage1/01_prepare_sequences.py first.", gene_name
            )
            continue

        _, sequence = read_fasta(fasta_path)
        variants_df = generate_all_missense_variants(gene_name, sequence)

        # Load ClinVar
        clinvar_path = os.path.join(clinvar_dir, f"{gene_name}_clinvar.vcf")
        clinvar_df = load_clinvar_variants(clinvar_path, gene_name)
        variants_df = annotate_with_clinvar(variants_df, clinvar_df)

        # Save outputs
        csv_path = os.path.join(output_dir, f"{gene_name.lower()}_all_variants.csv")
        variants_df.to_csv(csv_path, index=False)
        logger.info("Saved variants CSV: %s", csv_path)

        vcf_path = os.path.join(output_dir, f"{gene_name.lower()}_variants.vcf")
        save_vcf(variants_df, vcf_path, gene_name)

        # ClinVar-filtered subset
        clinvar_subset = variants_df[
            variants_df["clinvar_significance"].isin(
                ["pathogenic", "likely_pathogenic", "benign", "likely_benign"]
            )
        ]
        if not clinvar_subset.empty:
            filtered_path = os.path.join(output_dir, f"{gene_name.lower()}_clinvar_filtered.csv")
            clinvar_subset.to_csv(filtered_path, index=False)
            logger.info("Saved ClinVar-filtered variants: %s", filtered_path)

        print_summary(gene_name, variants_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate all missense variants for SCN1A and SCN2A."
    )
    parser.add_argument(
        "--sequences_dir",
        default=os.path.join("..", "data", "raw", "sequences"),
    )
    parser.add_argument(
        "--clinvar_dir",
        default=os.path.join("..", "data", "raw", "variants"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "processed", "variants"),
    )
    args = parser.parse_args()
    main(args.sequences_dir, args.clinvar_dir, args.output_dir)
