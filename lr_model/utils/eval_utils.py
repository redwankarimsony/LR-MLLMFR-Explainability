# eval_utils.py
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def compute_roc(y_true, y_scores):
    """
    Compute ROC curve arrays and AUC.

    Args:
        y_true (array-like): Ground truth labels (0/1).
        y_scores (array-like): Prediction scores (higher = positive).
    Returns:
        fpr (ndarray), tpr (ndarray), thresholds (ndarray), roc_auc (float)
    """
    fpr, tpr, thr = roc_curve(y_true, y_scores)
    return fpr, tpr, thr, auc(fpr, tpr)


# Compute tmr at specified fmr points and return them in a dict

def compute_tmr_at_fmr(y_true, y_scores, fmr_points=(1, 0.1, 0.01)):
    """
    Compute TMR at specified FMR points (percentages).

    Args:
        y_true (array-like): Ground truth labels (0/1).
        y_scores (array-like): Prediction scores (higher = positive).
        fmr_points (iterable): FMR thresholds in PERCENT (e.g. (1, 0.1, 0.01)).
    Returns:
        dict mapping FMR% -> TMR % value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Convert percentage → decimal
    fmr_points_decimal = [p / 100.0 for p in fmr_points]

    # Compute TMR at requested FMR points
    tmr_at_fmr = {}
    for fmr_perc, fmr in zip(fmr_points, fmr_points_decimal):
        idxs = np.where(fpr <= fmr)[0]
        tmr_at_fmr[fmr_perc] = tpr[idxs[-1]]*100.0 if len(idxs) > 0 else 0.0

    return tmr_at_fmr


def plot_roc_curve(y_true, y_scores, fmr_points=(1, 0.1, 0.01), save_path=None):
    """
    Plot ROC curve with log-scaled FPR axis, thick line,
    and report TMR at given FMR points (percentages).

    Args:
        y_true: array-like of shape (n_samples,), binary labels (0/1).
        y_scores: array-like of shape (n_samples,), scores (higher = positive).
        fmr_points: iterable of FMR thresholds in PERCENT (e.g. (1, 0.1, 0.01)).
    Returns:
        dict mapping FMR% -> TMR value.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Convert percentage → decimal
    fmr_points_decimal = [p / 100.0 for p in fmr_points]

    # Compute TMR at requested FMR points
    tmr_at_fmr = {}
    for fmr_perc, fmr in zip(fmr_points, fmr_points_decimal):
        idxs = np.where(fpr <= fmr)[0]
        tmr_at_fmr[fmr_perc] = tpr[idxs[-1]] if len(idxs) > 0 else 0.0

    # --- Plot ---
    plt.figure(figsize=(10, 6), dpi=200)
    plt.plot(fpr, tpr, color="blue", lw=3)
    plt.plot([1e-6, 1], [0, 1], color="gray", linestyle="--", lw=1)

    plt.xscale("log")
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Match Rate (log scale)")
    plt.ylabel("True Match Rate")
    plt.title("ROC Curve")

    # Mark the TMR points on the curve
    for fmr_perc, fmr in zip(fmr_points, fmr_points_decimal):
        tmr = tmr_at_fmr[fmr_perc]
        plt.scatter([fmr], [tmr], marker="o", color="red", zorder=5)
        plt.text(fmr, tmr + 0.02, f"TMR@FMR={fmr_perc:.3f}%: {tmr:.3f}", fontsize=10, ha="center")

    plt.legend(loc="lower right")
    plt.grid(True, which="both", ls="--", lw=0.5)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return tmr_at_fmr


def plot_llr_density(llr_scores_norm, y_val, save_name,
                     title="Normalized LLR Score Density", dpi=150, show_plot=True):
    """
    Plot density of normalized LLR scores for match vs non-match.

    Args:
        llr_scores_norm (array-like): Normalized LLR scores in [0,1].
        y_val (array-like): Ground truth labels (1=match, 0=non-match).
        save_name (str): Path to save the plot (e.g., "results/plots/llr_density.pdf").
        title (str): Plot title.
        dpi (int): Resolution of saved figure.
    """
    print("\n[Stage] Visualization: Plotting normalized LLR density")

    plt.figure(figsize=(3, 3), dpi=dpi)
    try:
        # sns.kdeplot(llr_scores_norm[y_val == 1], label='Genuine (Match)',
        #             fill=True, common_norm=False, alpha=0.5)
        # sns.kdeplot(llr_scores_norm[y_val == 0], label='Impostor (Non-Match)',
        #             fill=True, common_norm=False, alpha=0.5)

        # use histplot with kde
        sns.histplot(llr_scores_norm[y_val == 1], label='Genuine',
                     stat='density', bins=100, kde=True, color='blue', alpha=0.5)
        sns.histplot(llr_scores_norm[y_val == 0], label='Impostor',
                     stat='density', bins=100, kde=True, color='red', alpha=0.5)

        plt.title(title)
        plt.xlabel('Normalized LLR Score (0–1)')
        plt.ylabel('Density')
        plt.legend()
        # plt.grid(True)
        plt.tight_layout()

        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        plt.savefig(save_name, dpi=dpi)
        print(f"  -> Saved plot: {save_name}")
    finally:
        if show_plot:
            plt.show()
        else:
            plt.close()
