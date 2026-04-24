"""
Stage 3 - Step 6: Compare Predictions to Clinical Data
=======================================================
Load clinical variant data (ClinVar, literature), compare computational
predictions to clinical outcomes, calculate performance metrics,
map variants to protein domains, correlate with disease phenotypes
(Dravet, GEFS+, EIMFS), and generate comprehensive report.

Usage:
    python 06_compare_clinical_data.py \
        --consensus_dir ../data/results/consensus \
        --variants_dir ../data/processed/variants \
        --output_dir ../data/results/clinical_comparison
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

# SCN1A domain boundaries (approximate, based on Nav1.1 topology)
SCN1A_DOMAINS = {
    "N_terminus": (1, 120),
    "DI_S1-S6": (121, 430),
    "DI_DII_linker": (431, 700),
    "DII_S1-S6": (701, 1000),
    "DII_DIII_linker": (1001, 1200),
    "DIII_S1-S6": (1201, 1500),
    "DIII_DIV_linker": (1501, 1610),
    "DIV_S1-S6": (1611, 1900),
    "C_terminus": (1901, 2009),
}

# SCN2A domain boundaries (approximate, Nav1.2)
SCN2A_DOMAINS = {
    "N_terminus": (1, 125),
    "DI_S1-S6": (126, 435),
    "DI_DII_linker": (436, 710),
    "DII_S1-S6": (711, 1010),
    "DII_DIII_linker": (1011, 1205),
    "DIII_S1-S6": (1206, 1505),
    "DIII_DIV_linker": (1506, 1615),
    "DIV_S1-S6": (1616, 1905),
    "C_terminus": (1906, 2005),
}

DOMAIN_BOUNDARIES = {"SCN1A": SCN1A_DOMAINS, "SCN2A": SCN2A_DOMAINS}

# Disease phenotypes associated with SCN1A/SCN2A variants
PHENOTYPE_MAP = {
    "SCN1A": {
        "Dravet Syndrome": ["pathogenic", "likely_pathogenic"],
        "GEFS+": ["likely_pathogenic", "uncertain_significance"],
        "Febrile Seizures": ["uncertain_significance", "likely_benign"],
    },
    "SCN2A": {
        "Benign Neonatal Epilepsy": ["likely_benign", "benign"],
        "Ohtahara Syndrome": ["pathogenic"],
        "EIMFS": ["likely_pathogenic", "pathogenic"],
        "Autism Spectrum Disorder": ["uncertain_significance", "likely_pathogenic"],
    },
}


def assign_domain(position: int, gene_name: str) -> str:
    """Assign protein domain to a residue position.

    Args:
        position: 1-indexed residue position.
        gene_name: Gene name.

    Returns:
        Domain name string.
    """
    domains = DOMAIN_BOUNDARIES.get(gene_name, {})
    for domain, (start, end) in domains.items():
        if start <= position <= end:
            return domain
    return "unknown"


def load_consensus(consensus_dir: str, gene_name: str) -> pd.DataFrame:
    """Load consensus predictions for a gene.

    Args:
        consensus_dir: Directory with consensus CSV files.
        gene_name: Gene name.

    Returns:
        Consensus predictions DataFrame.
    """
    path = os.path.join(consensus_dir, f"{gene_name.lower()}_consensus_predictions.csv")
    if not os.path.exists(path):
        logger.warning("Consensus file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info("Loaded %d consensus predictions for %s", len(df), gene_name)
    return df


def calculate_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> tuple:
    """Calculate ROC curve points without sklearn.

    Args:
        y_true: Binary true labels (0/1).
        y_scores: Predicted scores.

    Returns:
        Tuple of (fpr_array, tpr_array, thresholds).
    """
    thresholds = np.sort(np.unique(y_scores))[::-1]
    fpr_list, tpr_list = [0.0], [0.0]
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fpr = fp / n_neg if n_neg > 0 else 0.0
        tpr = tp / n_pos if n_pos > 0 else 0.0
        fpr_list.append(fpr)
        tpr_list.append(tpr)

    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list), thresholds


def calculate_auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Calculate AUC from FPR/TPR arrays using trapezoidal rule.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.

    Returns:
        AUC value.
    """
    return float(np.trapz(tpr, fpr))


def plot_roc_curves(
    gene_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    output_dir: str,
) -> None:
    """Plot ROC curves for all genes.

    Args:
        gene_data: Dict of gene_name → (fpr, tpr, auc).
        output_dir: Output directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = {"SCN1A": "#1565C0", "SCN2A": "#E53935"}
    for gene, (fpr, tpr, auc) in gene_data.items():
        ax.plot(fpr, tpr, label=f"{gene} (AUC = {auc:.3f})", color=colors.get(gene, "grey"), linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Consensus Pathogenicity Predictions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved ROC curves: %s", path)


def plot_confusion_matrix(
    tp: int, tn: int, fp: int, fn: int,
    gene_name: str, output_dir: str,
) -> None:
    """Plot confusion matrix as a heatmap.

    Args:
        tp, tn, fp, fn: Confusion matrix values.
        gene_name: Gene name for title.
        output_dir: Output directory.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    matrix = np.array([[tp, fn], [fp, tn]])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nPathogenic", "Predicted\nBenign"], fontsize=11)
    ax.set_yticklabels(["True\nPathogenic", "True\nBenign"], fontsize=11)
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, str(val), ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if val > matrix.max() / 2 else "black")
    ax.set_title(f"{gene_name} Confusion Matrix", fontsize=13, fontweight="bold")
    plt.colorbar(im)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{gene_name.lower()}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved confusion matrix: %s", path)


def plot_domain_mapping(consensus_df: pd.DataFrame, output_dir: str, gene_name: str) -> None:
    """Plot variant pathogenicity mapped to protein domains.

    Args:
        consensus_df: Consensus DataFrame with domain column.
        output_dir: Output directory.
        gene_name: Gene name.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if "domain" not in consensus_df.columns:
        return

    domain_order = list(DOMAIN_BOUNDARIES.get(gene_name, {}).keys())
    class_colors = {
        "likely_pathogenic": "#E53935",
        "uncertain_significance": "#FFB300",
        "likely_benign": "#43A047",
        "not_scored": "#9E9E9E",
    }

    fig, ax = plt.subplots(figsize=(16, 6))
    for cls, color in class_colors.items():
        sub = consensus_df[consensus_df["consensus_class"] == cls]
        if sub.empty:
            continue
        ax.scatter(
            sub["position"], sub.get("consensus_score", np.full(len(sub), 0.5)),
            c=color, s=3, alpha=0.5, label=cls, linewidths=0,
        )

    # Domain boundaries
    for domain, (start, end) in DOMAIN_BOUNDARIES.get(gene_name, {}).items():
        ax.axvspan(start, end, alpha=0.07, color="navy")
        ax.text((start + end) / 2, 1.02, domain.replace("_", "\n"), ha="center",
                fontsize=6, transform=ax.get_xaxis_transform())

    ax.set_xlabel("Residue Position", fontsize=12)
    ax.set_ylabel("Consensus Pathogenicity Score", fontsize=12)
    ax.set_title(f"{gene_name} Variant Pathogenicity Map", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, markerscale=4, loc="upper right")
    ax.set_ylim(-0.05, 1.15)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{gene_name.lower()}_domain_mapping.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved domain mapping: %s", path)


def generate_report(
    all_metrics: list[dict],
    domain_summaries: list[pd.DataFrame],
    output_dir: str,
) -> None:
    """Generate a text summary report.

    Args:
        all_metrics: List of performance metrics per gene.
        domain_summaries: Domain-level classification summaries.
        output_dir: Output directory.
    """
    report_path = os.path.join(output_dir, "clinical_comparison_report.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("SCN1A and SCN2A Variant Pathogenicity — Clinical Comparison Report\n")
        fh.write("=" * 70 + "\n\n")

        for metrics in all_metrics:
            gene = metrics.get("gene", "?")
            fh.write(f"Gene: {gene}\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"  Labeled variants evaluated : {metrics.get('n_evaluated', 0)}\n")
            fh.write(f"  Sensitivity                : {metrics.get('sensitivity', 0):.4f}\n")
            fh.write(f"  Specificity                : {metrics.get('specificity', 0):.4f}\n")
            fh.write(f"  Precision                  : {metrics.get('precision', 0):.4f}\n")
            fh.write(f"  F1 Score                   : {metrics.get('f1', 0):.4f}\n")
            fh.write(f"  MCC                        : {metrics.get('mcc', 0):.4f}\n")
            if metrics.get("auc_roc"):
                fh.write(f"  AUC-ROC                    : {metrics['auc_roc']:.4f}\n")
            fh.write("\n")

        fh.write("\nDomain-Level Analysis\n")
        fh.write("-" * 40 + "\n")
        for df in domain_summaries:
            if df.empty:
                continue
            fh.write(df.to_string() + "\n\n")

        fh.write("\nPhenotype-Prediction Correlation Notes\n")
        fh.write("-" * 40 + "\n")
        for gene, phenotypes in PHENOTYPE_MAP.items():
            fh.write(f"\n{gene}:\n")
            for phenotype, expected_classes in phenotypes.items():
                fh.write(f"  {phenotype}: expected {expected_classes}\n")

    logger.info("Saved report: %s", report_path)


def main(consensus_dir: str, variants_dir: str, output_dir: str) -> None:
    """Main clinical comparison workflow.

    Args:
        consensus_dir: Consensus results directory.
        variants_dir: Processed variants directory (for ClinVar annotations).
        output_dir: Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = []
    domain_summaries = []
    roc_data = {}

    for gene_name in GENES:
        logger.info("Clinical comparison for %s", gene_name)

        consensus_df = load_consensus(consensus_dir, gene_name)
        if consensus_df.empty:
            logger.warning("%s: No consensus data. Skipping.", gene_name)
            continue

        # Load ClinVar annotations from variant CSV if not in consensus
        if "clinvar_significance" not in consensus_df.columns:
            var_path = os.path.join(variants_dir, f"{gene_name.lower()}_all_variants.csv")
            if os.path.exists(var_path):
                var_df = pd.read_csv(var_path)[["variant_name", "clinvar_significance"]]
                consensus_df = consensus_df.merge(var_df, on="variant_name", how="left")

        # Assign domains
        consensus_df["domain"] = consensus_df["position"].apply(
            lambda p: assign_domain(int(p), gene_name)
        )

        # Save annotated
        ann_path = os.path.join(output_dir, f"{gene_name.lower()}_annotated_predictions.csv")
        consensus_df.to_csv(ann_path, index=False)

        # Domain-level summary
        domain_summary = (
            consensus_df.groupby(["domain", "consensus_class"])
            .size()
            .reset_index(name="count")
        )
        domain_summary.insert(0, "gene", gene_name)
        domain_summary_path = os.path.join(output_dir, f"{gene_name.lower()}_domain_summary.csv")
        domain_summary.to_csv(domain_summary_path, index=False)
        domain_summaries.append(domain_summary)

        # Performance metrics
        labeled = consensus_df[
            consensus_df.get("clinvar_significance", pd.Series(dtype=str)).isin(
                ["pathogenic", "likely_pathogenic", "benign", "likely_benign"]
            )
        ]
        if not labeled.empty:
            labeled = labeled.copy()
            labeled["true_label"] = labeled["clinvar_significance"].map(
                {"pathogenic": 1, "likely_pathogenic": 1, "benign": 0, "likely_benign": 0}
            )
            eval_df = labeled[labeled["consensus_class"].isin(["likely_pathogenic", "likely_benign"])]
            if not eval_df.empty and "consensus_score" in eval_df.columns:
                y_true = eval_df["true_label"].values
                y_scores = eval_df["consensus_score"].fillna(0.5).values
                fpr, tpr, _ = calculate_roc_curve(y_true, y_scores)
                auc = calculate_auc(fpr, tpr)
                roc_data[gene_name] = (fpr, tpr, auc)

                y_pred = (y_scores >= 0.5).astype(int)
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

                metrics = {
                    "gene": gene_name, "n_labeled": len(labeled), "n_evaluated": len(eval_df),
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "sensitivity": round(sensitivity, 4), "specificity": round(specificity, 4),
                    "precision": round(precision, 4), "f1": round(f1, 4),
                    "mcc": round(float(mcc), 4), "auc_roc": round(auc, 4),
                }
                all_metrics.append(metrics)
                plot_confusion_matrix(tp, tn, fp, fn, gene_name, output_dir)

        plot_domain_mapping(consensus_df, output_dir, gene_name)

    if roc_data:
        plot_roc_curves(roc_data, output_dir)

    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
        logger.info("Saved performance metrics.")

    generate_report(all_metrics, domain_summaries, output_dir)
    logger.info("Clinical comparison complete. Results in %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare consensus predictions to clinical data."
    )
    parser.add_argument(
        "--consensus_dir",
        default=os.path.join("..", "data", "results", "consensus"),
    )
    parser.add_argument(
        "--variants_dir",
        default=os.path.join("..", "data", "processed", "variants"),
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.join("..", "data", "results", "clinical_comparison"),
    )
    args = parser.parse_args()
    main(args.consensus_dir, args.variants_dir, args.output_dir)
