"""
Stage 3 - Step 5: Consensus Pathogenicity Analysis
====================================================
Combine AlphaMissense, ESM1b, and PRESCOTT predictions using
weighted voting to produce consensus scores. Reclassify VUS,
evaluate against known variants, and generate performance metrics.

Usage:
    python 05_consensus_analysis.py \
        --alphamissense_dir ../data/results/alphamissense \
        --esm1b_dir ../data/results/esm1b \
        --prescott_dir ../data/results/prescott \
        --output_dir ../data/results/consensus \
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

# Weights based on published benchmarks (Cheng et al. 2023, Fraternali lab)
TOOL_WEIGHTS = {
    "alphamissense": 0.45,
    "esm1b": 0.35,
    "prescott": 0.20,
}

# Consensus thresholds
CONSENSUS_PATHOGENIC = 0.564  # aligned with AlphaMissense (Cheng et al. 2023)
CONSENSUS_BENIGN = 0.34

CLASS_TO_SCORE = {
    "likely_pathogenic": 1.0,
    "pathogenic": 1.0,
    "uncertain_significance": 0.5,
    "vus": 0.5,
    "likely_benign": 0.0,
    "benign": 0.0,
    "not_predicted": float("nan"),
    "not_scored": float("nan"),
    "not_queried": float("nan"),
}


def load_tool_scores(scores_dir: str, gene_name: str, tool_name: str) -> pd.DataFrame:
    """Load prediction scores for one tool.

    Args:
        scores_dir: Directory containing tool scores CSV.
        gene_name: Gene name.
        tool_name: Tool identifier (alphamissense, esm1b, prescott).

    Returns:
        DataFrame with variant_name and score columns.
    """
    patterns = [
        f"{gene_name.lower()}_{tool_name}_scores.csv",
        f"{gene_name.lower()}_{tool_name}.csv",
    ]
    for pattern in patterns:
        path = os.path.join(scores_dir, pattern)
        if os.path.exists(path):
            df = pd.read_csv(path)
            logger.info("Loaded %s %s scores: %d rows", gene_name, tool_name, len(df))
            return df
    logger.warning("No %s scores file found in %s for %s", tool_name, scores_dir, gene_name)
    return pd.DataFrame()


def normalize_score(df: pd.DataFrame, tool_name: str) -> pd.Series:
    """Extract and normalize tool score to [0, 1] (pathogenicity direction).

    For ESM1b LLR: more negative = more pathogenic → invert and normalize.

    Args:
        df: Tool scores DataFrame.
        tool_name: Tool identifier.

    Returns:
        Series of normalized scores indexed by variant_name.
    """
    score_cols = {
        "alphamissense": "am_pathogenicity",
        "esm1b": "esm_llr",
        "prescott": "prescott_score",
    }
    col = score_cols.get(tool_name)
    if col not in df.columns:
        logger.warning("Column %s not found in %s DataFrame.", col, tool_name)
        return pd.Series(dtype=float)

    df = df.set_index("variant_name")
    scores = df[col].copy()

    if tool_name == "esm1b":
        # LLR: invert and rescale. Typically in range [-20, 5]
        scores = -scores  # higher = more pathogenic
        vmin, vmax = -5.0, 20.0  # typical LLR range (inverted)
        scores = (scores.clip(vmin, vmax) - vmin) / (vmax - vmin)

    scores = scores.clip(0, 1)
    return scores


def compute_consensus(
    am_df: pd.DataFrame,
    esm_df: pd.DataFrame,
    prescott_df: pd.DataFrame,
    gene_name: str,
) -> pd.DataFrame:
    """Compute weighted consensus score across three tools.

    Args:
        am_df: AlphaMissense scores DataFrame.
        esm_df: ESM1b scores DataFrame.
        prescott_df: PRESCOTT scores DataFrame.
        gene_name: Gene name.

    Returns:
        DataFrame with consensus scores and classifications.
    """
    # Start from AlphaMissense as base (most complete coverage)
    if am_df.empty and esm_df.empty and prescott_df.empty:
        logger.error("All three tool DataFrames are empty for %s.", gene_name)
        return pd.DataFrame()

    # Use whichever has variant_name + position + wt_aa + mut_aa columns
    base_df = None
    for df in [am_df, esm_df, prescott_df]:
        if not df.empty and "variant_name" in df.columns:
            base_df = df[["gene", "position", "wt_aa", "mut_aa", "variant_name"]].copy()
            if "clinvar_significance" in df.columns:
                base_df["clinvar_significance"] = df["clinvar_significance"]
            break

    if base_df is None:
        logger.error("Could not find base variant list.")
        return pd.DataFrame()

    base_df = base_df.set_index("variant_name")

    # Normalize each tool's scores
    am_scores = normalize_score(am_df, "alphamissense") if not am_df.empty else pd.Series(dtype=float)
    esm_scores = normalize_score(esm_df, "esm1b") if not esm_df.empty else pd.Series(dtype=float)
    prescott_scores = normalize_score(prescott_df, "prescott") if not prescott_df.empty else pd.Series(dtype=float)

    # Add to base
    base_df["am_score_norm"] = am_scores
    base_df["esm_score_norm"] = esm_scores
    base_df["prescott_score_norm"] = prescott_scores

    # Compute weighted consensus (handle missing tools)
    def weighted_mean(row: pd.Series) -> float:
        tools = {
            "am": (row.get("am_score_norm"), TOOL_WEIGHTS["alphamissense"]),
            "esm": (row.get("esm_score_norm"), TOOL_WEIGHTS["esm1b"]),
            "prescott": (row.get("prescott_score_norm"), TOOL_WEIGHTS["prescott"]),
        }
        total_weight = 0.0
        total_score = 0.0
        for score, weight in tools.values():
            if pd.notna(score):
                total_score += score * weight
                total_weight += weight
        if total_weight == 0:
            return float("nan")
        return total_score / total_weight

    base_df["consensus_score"] = base_df.apply(weighted_mean, axis=1)
    base_df["n_tools_scored"] = base_df[
        ["am_score_norm", "esm_score_norm", "prescott_score_norm"]
    ].notna().sum(axis=1)

    base_df["consensus_class"] = base_df["consensus_score"].apply(
        lambda x: (
            "likely_pathogenic" if pd.notna(x) and x >= CONSENSUS_PATHOGENIC
            else "likely_benign" if pd.notna(x) and x <= CONSENSUS_BENIGN
            else "uncertain_significance" if pd.notna(x)
            else "not_scored"
        )
    )

    logger.info(
        "%s consensus: pathogenic=%d, benign=%d, VUS=%d",
        gene_name,
        (base_df["consensus_class"] == "likely_pathogenic").sum(),
        (base_df["consensus_class"] == "likely_benign").sum(),
        (base_df["consensus_class"] == "uncertain_significance").sum(),
    )

    return base_df.reset_index()


def evaluate_performance(consensus_df: pd.DataFrame, gene_name: str) -> dict:
    """Evaluate consensus predictions against ClinVar labels.

    Computes sensitivity, specificity, MCC, and AUC-ROC.

    Args:
        consensus_df: Consensus DataFrame with clinvar_significance column.
        gene_name: Gene name.

    Returns:
        Dictionary of performance metrics.
    """
    if "clinvar_significance" not in consensus_df.columns:
        logger.warning("No ClinVar labels available for performance evaluation.")
        return {}

    # Create binary labels from ClinVar
    labeled = consensus_df[
        consensus_df["clinvar_significance"].isin(
            ["pathogenic", "likely_pathogenic", "benign", "likely_benign"]
        )
    ].copy()

    if labeled.empty:
        logger.warning("No ClinVar-labeled variants for %s.", gene_name)
        return {}

    labeled["true_label"] = labeled["clinvar_significance"].map(
        {"pathogenic": 1, "likely_pathogenic": 1, "benign": 0, "likely_benign": 0}
    )
    labeled["pred_label"] = labeled["consensus_class"].map(
        {"likely_pathogenic": 1, "uncertain_significance": -1, "likely_benign": 0}
    )

    # Exclude VUS from binary evaluation
    eval_df = labeled[labeled["pred_label"].isin([0, 1])].copy()
    if eval_df.empty:
        return {}

    y_true = eval_df["true_label"].values
    y_pred = eval_df["pred_label"].values

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    mcc_denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    # AUC-ROC
    auc = float("nan")
    if "consensus_score" in eval_df.columns and eval_df["consensus_score"].notna().sum() > 0:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(eval_df["true_label"].values, eval_df["consensus_score"].values)
        except Exception:
            pass

    metrics = {
        "gene": gene_name,
        "n_labeled": len(labeled),
        "n_evaluated": len(eval_df),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "mcc": round(float(mcc), 4),
        "auc_roc": round(auc, 4) if not np.isnan(auc) else None,
    }

    logger.info(
        "%s performance: Sens=%.2f, Spec=%.2f, MCC=%.2f, AUC=%.2f",
        gene_name, sensitivity, specificity, float(mcc), auc if not np.isnan(auc) else 0,
    )
    return metrics


def plot_consensus_scores(consensus_df: pd.DataFrame, output_dir: str, gene_name: str) -> None:
    """Plot consensus score distribution.

    Args:
        consensus_df: Consensus DataFrame.
        output_dir: Output directory.
        gene_name: Gene name.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Score distribution
    scored = consensus_df[consensus_df["consensus_score"].notna()]
    axes[0].hist(
        [
            scored[scored["consensus_class"] == "likely_pathogenic"]["consensus_score"],
            scored[scored["consensus_class"] == "uncertain_significance"]["consensus_score"],
            scored[scored["consensus_class"] == "likely_benign"]["consensus_score"],
        ],
        bins=50, stacked=True,
        label=["Likely Pathogenic", "Uncertain", "Likely Benign"],
        color=["#E53935", "#FFB300", "#43A047"],
        alpha=0.8,
    )
    axes[0].set_xlabel("Consensus Score", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title(f"{gene_name} Consensus Score Distribution", fontsize=12)
    axes[0].legend()
    axes[0].axvline(CONSENSUS_PATHOGENIC, color="#E53935", linestyle="--", linewidth=1)
    axes[0].axvline(CONSENSUS_BENIGN, color="#43A047", linestyle="--", linewidth=1)

    # Class distribution pie chart
    counts = consensus_df["consensus_class"].value_counts()
    labels = counts.index.tolist()
    color_map = {
        "likely_pathogenic": "#E53935",
        "uncertain_significance": "#FFB300",
        "likely_benign": "#43A047",
        "not_scored": "#9E9E9E",
    }
    colors = [color_map.get(l, "grey") for l in labels]
    axes[1].pie(counts.values, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"{gene_name} Classification Breakdown", fontsize=12)

    plt.suptitle(f"Consensus Pathogenicity Analysis — {gene_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{gene_name.lower()}_consensus_scores.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved consensus plot: %s", path)


def main(
    am_dir: str,
    esm_dir: str,
    prescott_dir: str,
    output_dir: str,
    gene_names: list[str],
) -> None:
    """Main consensus analysis workflow.

    Args:
        am_dir: AlphaMissense results directory.
        esm_dir: ESM1b results directory.
        prescott_dir: PRESCOTT results directory.
        output_dir: Output directory.
        gene_names: List of genes to analyze.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []

    for gene_name in gene_names:
        am_df = load_tool_scores(am_dir, gene_name, "alphamissense")
        esm_df = load_tool_scores(esm_dir, gene_name, "esm1b")
        prescott_df = load_tool_scores(prescott_dir, gene_name, "prescott")

        if all(df.empty for df in [am_df, esm_df, prescott_df]):
            logger.error("No scores found for %s. Run steps 02–04 first.", gene_name)
            continue

        consensus_df = compute_consensus(am_df, esm_df, prescott_df, gene_name)
        if consensus_df.empty:
            continue

        # Save
        out_path = os.path.join(output_dir, f"{gene_name.lower()}_consensus_predictions.csv")
        consensus_df.to_csv(out_path, index=False)
        logger.info("Saved consensus: %s", out_path)

        # VUS reclassification
        vus_df = consensus_df[consensus_df.get("clinvar_significance", pd.Series()) == "vus"]
        if not vus_df.empty:
            vus_path = os.path.join(output_dir, f"{gene_name.lower()}_vus_reclassification.csv")
            vus_df.to_csv(vus_path, index=False)

        # Performance
        metrics = evaluate_performance(consensus_df, gene_name)
        if metrics:
            all_metrics.append(metrics)

        plot_consensus_scores(consensus_df, output_dir, gene_name)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
        logger.info("Saved performance metrics.")

        print(f"\n{'='*60}")
        print("  Consensus Performance Summary")
        print(f"{'='*60}")
        for _, row in metrics_df.iterrows():
            print(f"\n  {row['gene']}:")
            print(f"    Sensitivity : {row['sensitivity']:.3f}")
            print(f"    Specificity : {row['specificity']:.3f}")
            print(f"    MCC         : {row['mcc']:.3f}")
            if row['auc_roc']:
                print(f"    AUC-ROC     : {row['auc_roc']:.3f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consensus pathogenicity analysis combining AlphaMissense, ESM1b, PRESCOTT."
    )
    parser.add_argument(
        "--alphamissense_dir",
        default=os.path.join("..", "data", "results", "alphamissense"),
    )
    parser.add_argument(
        "--esm1b_dir",
        default=os.path.join("..", "data", "results", "esm1b"),
    )
    parser.add_argument(
        "--prescott_dir",
        default=os.path.join("..", "data", "results", "prescott"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "consensus"),
    )
    parser.add_argument(
        "--gene",
        default="ALL",
        choices=["SCN1A", "SCN2A", "ALL"],
    )
    args = parser.parse_args()
    genes = GENES if args.gene == "ALL" else [args.gene]
    main(args.alphamissense_dir, args.esm1b_dir, args.prescott_dir, args.output_dir, genes)
