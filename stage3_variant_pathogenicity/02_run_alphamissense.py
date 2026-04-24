"""
Stage 3 - Step 2: Run AlphaMissense
=====================================
Query AlphaMissense pathogenicity predictions for SCN1A and SCN2A
variants via the precomputed database or the AlphaMissense API.

Usage:
    python 02_run_alphamissense.py \
        --variants ../data/processed/variants/scn1a_all_variants.csv \
        --output_dir ../data/results/alphamissense \
        --gene SCN1A \
        [--db_path /path/to/AlphaMissense_aa_substitutions.tsv.gz]
"""

import argparse
import logging
import os
import sys

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

UNIPROT_IDS = {"SCN1A": "P35498", "SCN2A": "Q99250"}

# AlphaMissense precomputed database (Zenodo)
AM_ZENODO_URL = (
    "https://zenodo.org/records/10813168/files/"
    "AlphaMissense_aa_substitutions.tsv.gz?download=1"
)

# Classification thresholds (from Cheng et al. 2023)
AM_PATHOGENIC_THRESHOLD = 0.564
AM_BENIGN_THRESHOLD = 0.34

AM_COLUMNS = ["uniprot_id", "protein_variant", "am_pathogenicity", "am_class"]


def load_alphamissense_db(db_path: str, uniprot_id: str) -> pd.DataFrame:
    """Load AlphaMissense predictions for a specific protein from the TSV database.

    Args:
        db_path: Path to AlphaMissense TSV (gzipped) database file.
        uniprot_id: UniProt accession to filter.

    Returns:
        DataFrame with AlphaMissense scores for the protein.
    """
    logger.info("Loading AlphaMissense DB for %s from %s...", uniprot_id, db_path)
    df = pd.read_csv(db_path, sep="\t", compression="gzip", comment="#", low_memory=False)
    logger.info("DB shape: %s", df.shape)

    # Column name normalization
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    col_map = {}
    for col in df.columns:
        if "uniprot" in col:
            col_map[col] = "uniprot_id"
        elif "variant" in col and "protein" in col:
            col_map[col] = "protein_variant"
        elif "pathogenicity" in col:
            col_map[col] = "am_pathogenicity"
        elif "class" in col:
            col_map[col] = "am_class"
    df = df.rename(columns=col_map)

    filtered = df[df.get("uniprot_id", pd.Series(dtype=str)) == uniprot_id].copy()
    logger.info("%s: %d predictions found in DB", uniprot_id, len(filtered))
    return filtered


def download_alphamissense_db(db_dir: str) -> str:
    """Download the AlphaMissense precomputed database from Zenodo.

    Args:
        db_dir: Directory to save the database file.

    Returns:
        Path to downloaded file.
    """
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "AlphaMissense_aa_substitutions.tsv.gz")
    if os.path.exists(db_path):
        logger.info("AlphaMissense DB already exists: %s", db_path)
        return db_path

    logger.info("Downloading AlphaMissense DB from Zenodo (~3 GB)...")
    logger.info("URL: %s", AM_ZENODO_URL)
    logger.info("This may take several minutes on a slow connection.")

    response = requests.get(AM_ZENODO_URL, stream=True, timeout=3600)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0
    with open(db_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            fh.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                logger.info("  %.1f%% (%d MB)", pct, downloaded // 1024 // 1024)
    logger.info("Downloaded AlphaMissense DB to %s", db_path)
    return db_path


def classify_variant(score: float) -> str:
    """Classify a variant as pathogenic/benign/uncertain based on AlphaMissense score.

    Args:
        score: AlphaMissense pathogenicity score (0–1).

    Returns:
        Classification string.
    """
    if score >= AM_PATHOGENIC_THRESHOLD:
        return "likely_pathogenic"
    if score <= AM_BENIGN_THRESHOLD:
        return "likely_benign"
    return "uncertain_significance"


def query_alphamissense_api(uniprot_id: str, variant: str) -> dict | None:
    """Query the AlphaMissense REST API for a single variant prediction.

    Note: The public AlphaMissense API may require authentication.
    Prefer the precomputed DB for batch processing.

    Args:
        uniprot_id: UniProt accession.
        variant: Protein variant in format 'A123V'.

    Returns:
        Dict with am_pathogenicity and am_class, or None on failure.
    """
    url = f"https://alphamissense.hegelab.org/api/v1/prediction/{uniprot_id}/{variant}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return {
                "am_pathogenicity": data.get("pathogenicity_score"),
                "am_class": data.get("classification"),
            }
        logger.debug("API returned %d for %s %s", response.status_code, uniprot_id, variant)
    except requests.RequestException as exc:
        logger.debug("API error for %s %s: %s", uniprot_id, variant, exc)
    return None


def merge_predictions(variants_df: pd.DataFrame, am_df: pd.DataFrame) -> pd.DataFrame:
    """Merge variant list with AlphaMissense predictions.

    Args:
        variants_df: All variants DataFrame from stage 01.
        am_df: AlphaMissense DB DataFrame.

    Returns:
        Merged DataFrame.
    """
    if am_df.empty:
        logger.warning("Empty AlphaMissense data — no merge performed.")
        variants_df["am_pathogenicity"] = float("nan")
        variants_df["am_class"] = "not_predicted"
        return variants_df

    am_lookup = (
        am_df[["protein_variant", "am_pathogenicity", "am_class"]]
        .drop_duplicates("protein_variant")
        .set_index("protein_variant")
    )

    variants_df = variants_df.copy()
    variants_df["am_pathogenicity"] = variants_df["variant_name"].map(
        am_lookup["am_pathogenicity"]
    )
    variants_df["am_class"] = variants_df["variant_name"].map(
        am_lookup["am_class"]
    ).fillna("not_predicted")

    n_predicted = variants_df["am_pathogenicity"].notna().sum()
    logger.info(
        "%d / %d variants have AlphaMissense predictions",
        n_predicted, len(variants_df),
    )
    return variants_df


def run_api_fallback(
    variants_df: pd.DataFrame, uniprot_id: str, max_queries: int = 100
) -> pd.DataFrame:
    """Query AlphaMissense API for variants missing DB predictions.

    Limits API calls to avoid rate limiting.

    Args:
        variants_df: Variants DataFrame (may have NaN am_pathogenicity).
        uniprot_id: UniProt accession.
        max_queries: Maximum API queries to make.

    Returns:
        Updated DataFrame.
    """
    missing = variants_df[variants_df["am_pathogenicity"].isna()].head(max_queries)
    if missing.empty:
        return variants_df

    logger.info("Querying AlphaMissense API for %d missing variants...", len(missing))
    results = {}
    for _, row in missing.iterrows():
        data = query_alphamissense_api(uniprot_id, row["variant_name"])
        if data:
            results[row["variant_name"]] = data

    if results:
        variants_df = variants_df.copy()
        for var_name, data in results.items():
            mask = variants_df["variant_name"] == var_name
            variants_df.loc[mask, "am_pathogenicity"] = data["am_pathogenicity"]
            variants_df.loc[mask, "am_class"] = data["am_class"]
        logger.info("API filled %d additional predictions", len(results))

    return variants_df


def print_score_summary(gene_name: str, df: pd.DataFrame) -> None:
    """Print AlphaMissense score summary statistics.

    Args:
        gene_name: Gene name.
        df: Variants DataFrame with am_pathogenicity column.
    """
    scored = df[df["am_pathogenicity"].notna()]
    if scored.empty:
        print(f"\n{gene_name}: No AlphaMissense scores available.\n")
        return
    print(f"\n{'='*55}")
    print(f"  {gene_name} AlphaMissense Summary")
    print(f"{'='*55}")
    print(f"  Total variants    : {len(df):,}")
    print(f"  Scored variants   : {len(scored):,}")
    print(f"  Mean score        : {scored['am_pathogenicity'].mean():.3f}")
    print(f"  Likely pathogenic : {(scored['am_class'] == 'likely_pathogenic').sum():,}")
    print(f"  Uncertain (VUS)   : {(scored['am_class'] == 'uncertain_significance').sum():,}")
    print(f"  Likely benign     : {(scored['am_class'] == 'likely_benign').sum():,}")
    print(f"{'='*55}\n")


def main(
    variants_path: str,
    output_dir: str,
    gene_name: str,
    db_path: str | None,
    download_db: bool,
) -> None:
    """Main AlphaMissense analysis workflow.

    Args:
        variants_path: Path to variants CSV.
        output_dir: Output directory.
        gene_name: Gene identifier.
        db_path: Optional path to local AlphaMissense DB.
        download_db: Whether to download DB if not found.
    """
    os.makedirs(output_dir, exist_ok=True)
    uniprot_id = UNIPROT_IDS[gene_name]

    variants_df = pd.read_csv(variants_path)
    logger.info("Loaded %d variants for %s", len(variants_df), gene_name)

    am_df = pd.DataFrame()

    if db_path and os.path.exists(db_path):
        am_df = load_alphamissense_db(db_path, uniprot_id)
    elif download_db:
        db_dir = os.path.join(output_dir, "db")
        try:
            downloaded_path = download_alphamissense_db(db_dir)
            am_df = load_alphamissense_db(downloaded_path, uniprot_id)
        except Exception as exc:
            logger.error("DB download failed: %s — falling back to API.", exc)
    else:
        logger.info("No DB path provided. Using API fallback.")

    variants_df = merge_predictions(variants_df, am_df)

    # Try API for any remaining unscored variants
    if variants_df["am_pathogenicity"].isna().any():
        variants_df = run_api_fallback(variants_df, uniprot_id)

    # Add computed classification where missing
    mask = (variants_df["am_pathogenicity"].notna()) & (variants_df["am_class"] == "not_predicted")
    if mask.any():
        variants_df.loc[mask, "am_class"] = variants_df.loc[mask, "am_pathogenicity"].apply(
            classify_variant
        )

    out_path = os.path.join(output_dir, f"{gene_name.lower()}_alphamissense_scores.csv")
    variants_df.to_csv(out_path, index=False)
    logger.info("Saved AlphaMissense results: %s", out_path)
    print_score_summary(gene_name, variants_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AlphaMissense predictions for SCN1A/SCN2A variants."
    )
    parser.add_argument("--variants", required=True, help="Variants CSV from step 01")
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "alphamissense"),
    )
    parser.add_argument("--gene", default="SCN1A", choices=["SCN1A", "SCN2A"])
    parser.add_argument(
        "--db_path",
        default=None,
        help="Path to AlphaMissense TSV.gz database file",
    )
    parser.add_argument(
        "--download_db",
        action="store_true",
        help="Download AlphaMissense DB from Zenodo (~3 GB)",
    )
    args = parser.parse_args()
    main(args.variants, args.output_dir, args.gene, args.db_path, args.download_db)
