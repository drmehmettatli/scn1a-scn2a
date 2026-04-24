"""
Stage 3 - Step 3: Run ESM-1b / ESM-2 Variant Effect Prediction
===============================================================
Use Facebook AI's ESM protein language models to compute log-likelihood
ratios (LLR) for each variant, which correlate with pathogenicity.

Usage:
    python 03_run_esm1b.py \
        --variants ../data/processed/variants/scn1a_all_variants.csv \
        --sequences_dir ../data/raw/sequences \
        --output_dir ../data/results/esm1b \
        --gene SCN1A \
        [--model esm2_t33_650M_UR50D]
"""

import argparse
import logging
import math
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

UNIPROT_IDS = {"SCN1A": "P35498", "SCN2A": "Q99250"}

# ESM model options (from smallest to largest)
ESM_MODELS = {
    "esm2_t6_8M_UR50D": "ESM-2 8M (fast, lower accuracy)",
    "esm2_t12_35M_UR50D": "ESM-2 35M",
    "esm2_t30_150M_UR50D": "ESM-2 150M",
    "esm2_t33_650M_UR50D": "ESM-2 650M (recommended)",
    "esm2_t36_3B_UR50D": "ESM-2 3B (highest accuracy, slow)",
    "esm1b_t33_650M_UR50S": "ESM-1b 650M (original ESM1b)",
}
DEFAULT_MODEL = "esm2_t33_650M_UR50D"

# LLR threshold for pathogenicity (calibrated on ClinVar)
LLR_PATHOGENIC_THRESHOLD = -5.0
LLR_BENIGN_THRESHOLD = 2.0


def load_esm_model(model_name: str):
    """Load an ESM model and alphabet.

    Args:
        model_name: ESM model identifier string.

    Returns:
        Tuple of (model, alphabet, batch_converter).

    Raises:
        ImportError: If fair-esm is not installed.
    """
    try:
        import esm
    except ImportError:
        try:
            import esm  # noqa
        except ImportError as exc:
            raise ImportError(
                "fair-esm required: pip install fair-esm\n"
                "Or install ESM-2: pip install git+https://github.com/facebookresearch/esm.git"
            ) from exc

    logger.info("Loading ESM model: %s (%s)", model_name, ESM_MODELS.get(model_name, ""))
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()

    try:
        import torch
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info("Using GPU acceleration.")
        else:
            logger.info("GPU not available — running on CPU (slow for large sequences).")
    except ImportError:
        pass

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def compute_sequence_log_likelihood(
    model,
    alphabet,
    batch_converter,
    sequence: str,
    device: str = "cpu",
) -> np.ndarray:
    """Compute per-position log-probabilities for each amino acid.

    Uses masked marginal inference: masks each position and records
    the model's probability for the wild-type amino acid.

    Args:
        model: ESM model.
        alphabet: ESM alphabet.
        batch_converter: Batch converter from alphabet.
        sequence: Wild-type protein sequence.
        device: Torch device string.

    Returns:
        Array of shape (len(sequence), 20) with log-probabilities
        for each standard amino acid at each position.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch required: pip install torch") from exc

    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    aa_indices = [alphabet.get_idx(aa) for aa in AMINO_ACIDS]
    seq_len = len(sequence)
    log_probs = np.zeros((seq_len, 20), dtype=np.float32)

    data = [("protein", sequence)]
    _, _, tokens = batch_converter(data)
    tokens = tokens.to(device)

    logger.info("Computing ESM log-probabilities for %d positions...", seq_len)
    batch_size = 64  # process in batches to manage memory

    with torch.no_grad():
        for batch_start in range(0, seq_len, batch_size):
            batch_end = min(batch_start + batch_size, seq_len)
            batch_tokens = tokens.repeat(batch_end - batch_start, 1)

            for i, pos in enumerate(range(batch_start, batch_end)):
                # +1 offset for BOS token
                batch_tokens[i, pos + 1] = alphabet.mask_idx

            results = model(batch_tokens, repr_layers=[], return_contacts=False)
            logits = results["logits"]  # (batch, seq+2, vocab)

            for i, pos in enumerate(range(batch_start, batch_end)):
                position_logits = logits[i, pos + 1, aa_indices]
                log_p = torch.log_softmax(position_logits, dim=-1).cpu().numpy()
                log_probs[pos] = log_p

            if batch_start % 256 == 0:
                logger.info("  Progress: %d / %d positions", batch_end, seq_len)

    return log_probs


def compute_llr(
    log_probs: np.ndarray,
    sequence: str,
    variants_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute log-likelihood ratios for all variants.

    LLR = log P(mutant | context) - log P(wildtype | context)
    Negative LLR → variant is less likely than WT → possibly pathogenic.

    Args:
        log_probs: Per-position log-probabilities from compute_sequence_log_likelihood().
        sequence: Wild-type sequence.
        variants_df: Variants DataFrame with wt_aa, position, mut_aa columns.

    Returns:
        DataFrame with added esm_llr and esm_class columns.
    """
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    variants_df = variants_df.copy()
    llr_values = []

    for _, row in variants_df.iterrows():
        pos = row["position"] - 1  # 0-indexed
        wt = row["wt_aa"]
        mut = row["mut_aa"]

        if pos >= len(sequence) or sequence[pos] != wt:
            llr_values.append(float("nan"))
            continue

        wt_idx = aa_to_idx.get(wt)
        mut_idx = aa_to_idx.get(mut)
        if wt_idx is None or mut_idx is None:
            llr_values.append(float("nan"))
            continue

        llr = float(log_probs[pos, mut_idx] - log_probs[pos, wt_idx])
        llr_values.append(llr)

    variants_df["esm_llr"] = llr_values
    variants_df["esm_class"] = variants_df["esm_llr"].apply(
        lambda x: (
            "likely_pathogenic" if pd.notna(x) and x <= LLR_PATHOGENIC_THRESHOLD
            else "likely_benign" if pd.notna(x) and x >= LLR_BENIGN_THRESHOLD
            else "uncertain_significance" if pd.notna(x)
            else "not_scored"
        )
    )

    scored = variants_df["esm_llr"].notna().sum()
    logger.info(
        "LLR computed for %d variants. "
        "Pathogenic: %d, Benign: %d, VUS: %d",
        scored,
        (variants_df["esm_class"] == "likely_pathogenic").sum(),
        (variants_df["esm_class"] == "likely_benign").sum(),
        (variants_df["esm_class"] == "uncertain_significance").sum(),
    )
    return variants_df


def compute_esm_score_from_sequence(
    variants_df: pd.DataFrame,
    sequence: str,
    model_name: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Full ESM scoring pipeline: load model, compute log-probs, get LLRs.

    Args:
        variants_df: Variants DataFrame.
        sequence: Wild-type protein sequence.
        model_name: ESM model identifier.

    Returns:
        Variants DataFrame with esm_llr and esm_class columns.
    """
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    model, alphabet, batch_converter = load_esm_model(model_name)
    log_probs = compute_sequence_log_likelihood(
        model, alphabet, batch_converter, sequence, device
    )
    return compute_llr(log_probs, sequence, variants_df)


def read_fasta(fasta_path: str) -> tuple[str, str]:
    """Read FASTA file.

    Args:
        fasta_path: Path to FASTA file.

    Returns:
        Tuple (header, sequence).
    """
    with open(fasta_path, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    header = next(l.lstrip(">") for l in lines if l.startswith(">"))
    sequence = "".join(l for l in lines if not l.startswith(">"))
    return header, sequence


def find_fasta(sequences_dir: str, gene_name: str, uniprot_id: str) -> str | None:
    """Find FASTA file for gene.

    Args:
        sequences_dir: Directory with FASTA files.
        gene_name: Gene name.
        uniprot_id: UniProt ID.

    Returns:
        Path or None.
    """
    for fname in [f"{gene_name}_{uniprot_id}.fasta", f"{gene_name}.fasta"]:
        path = os.path.join(sequences_dir, fname)
        if os.path.exists(path):
            return path
    return None


def main(
    variants_path: str,
    sequences_dir: str,
    output_dir: str,
    gene_name: str,
    model_name: str,
) -> None:
    """Main ESM variant scoring workflow.

    Args:
        variants_path: Path to variants CSV.
        sequences_dir: Directory with FASTA files.
        output_dir: Output directory.
        gene_name: Gene identifier.
        model_name: ESM model to use.
    """
    os.makedirs(output_dir, exist_ok=True)

    variants_df = pd.read_csv(variants_path)
    logger.info("Loaded %d variants for %s", len(variants_df), gene_name)

    uniprot_id = UNIPROT_IDS[gene_name]
    fasta_path = find_fasta(sequences_dir, gene_name, uniprot_id)
    if fasta_path is None:
        logger.error("FASTA not found for %s in %s", gene_name, sequences_dir)
        sys.exit(1)

    _, sequence = read_fasta(fasta_path)
    logger.info("%s sequence length: %d aa", gene_name, len(sequence))

    try:
        results_df = compute_esm_score_from_sequence(variants_df, sequence, model_name)
    except ImportError as exc:
        logger.error("ESM scoring failed: %s", exc)
        logger.error(
            "Install fair-esm: pip install fair-esm\n"
            "Or: pip install git+https://github.com/facebookresearch/esm.git"
        )
        sys.exit(1)

    out_path = os.path.join(output_dir, f"{gene_name.lower()}_esm1b_scores.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("Saved ESM scores: %s", out_path)

    # Summary
    print(f"\n{'='*55}")
    print(f"  {gene_name} ESM1b/ESM2 Summary ({model_name})")
    print(f"{'='*55}")
    scored = results_df[results_df["esm_llr"].notna()]
    print(f"  Scored variants      : {len(scored):,}")
    print(f"  Mean LLR             : {scored['esm_llr'].mean():.3f}")
    print(f"  Likely pathogenic    : {(scored['esm_class'] == 'likely_pathogenic').sum():,}")
    print(f"  Uncertain (VUS)      : {(scored['esm_class'] == 'uncertain_significance').sum():,}")
    print(f"  Likely benign        : {(scored['esm_class'] == 'likely_benign').sum():,}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ESM-1b/ESM-2 variant effect predictions for SCN1A/SCN2A."
    )
    parser.add_argument("--variants", required=True, help="Variants CSV from step 01")
    parser.add_argument(
        "--sequences_dir",
        default=os.path.join("..", "data", "raw", "sequences"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "esm1b"),
    )
    parser.add_argument("--gene", default="SCN1A", choices=["SCN1A", "SCN2A"])
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(ESM_MODELS.keys()),
        help="ESM model to use",
    )
    args = parser.parse_args()
    main(args.variants, args.sequences_dir, args.output_dir, args.gene, args.model)
