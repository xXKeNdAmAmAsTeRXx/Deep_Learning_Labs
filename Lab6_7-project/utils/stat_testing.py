import numpy as np
from scipy.stats import friedmanchisquare, rankdata
from itertools import combinations
from typing import Optional


# Prompt (Sonnet 4.6)
"""Suggest a statistical test for diffrences between more than two ensembles in multiclass classification task. Write a python function that takes ensembles prediction and return test results."""

# Nemenyi critical values q_alpha for k groups at common alpha levels
# Source: Zar (1999), Table B.13; Demšar (2006)
_Q_ALPHA = {
    0.10: {2: 1.960, 3: 2.344, 4: 2.569, 5: 2.728, 6: 2.850,
           7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164},
    0.05: {2: 2.343, 3: 2.569, 4: 2.728, 5: 2.850, 6: 2.949,
           7: 3.031, 8: 3.102, 9: 3.164, 10: 3.219},
    0.01: {2: 2.576, 3: 2.913, 4: 3.113, 5: 3.255, 6: 3.364,
           7: 3.452, 8: 3.526, 9: 3.590, 10: 3.646},
}


def compare_ensembles(
    ensemble_predictions: list,
    y_true: np.ndarray,
    ensemble_names: Optional[list] = None,
    alpha: float = 0.05,
    metric: str = "accuracy",
) -> dict:
    """
    Compare k ≥ 3 ensembles using the Friedman test with post-hoc Nemenyi correction.

    Follows the benchmark methodology of Demšar (2006):
    "Statistical comparisons of classifiers over multiple data sets."

    Parameters
    ----------
    ensemble_predictions : list of array-like, each shape (n_samples,)
        Predicted class labels from each ensemble.
    y_true : array-like, shape (n_samples,)
        Ground-truth class labels.
    ensemble_names : list of str, optional
        Human-readable names for each ensemble. Defaults to "ensemble_0", …
    alpha : float
        Family-wise significance level (0.10, 0.05, or 0.01).
    metric : str
        "accuracy"  — per-sample 0/1 correctness (default)
        "f1_macro"  — per-sample 0/1 correctness (same signal, macro F1 reported separately)

    Returns
    -------
    dict with keys
        k                  : number of ensembles
        n                  : number of samples
        alpha              : significance level used
        ensemble_scores    : mean score per ensemble
        mean_ranks         : mean Friedman rank per ensemble (lower = better)
        critical_difference: Nemenyi CD threshold
        friedman_stat      : Friedman chi-squared statistic
        friedman_p         : p-value
        significant        : whether H₀ is rejected at alpha
        pairwise           : dict of pair → {rank_diff, cd, significant, better}
    """
    y_true = np.asarray(y_true)
    preds  = [np.asarray(p) for p in ensemble_predictions]
    k      = len(preds)
    n      = len(y_true)

    if k < 3:
        raise ValueError(
            f"At least 3 ensembles required (got {k}). "
            "For 2 ensembles use the Wilcoxon signed-rank test."
        )
    if any(len(p) != n for p in preds):
        raise ValueError("All prediction arrays must have the same length as y_true.")
    if alpha not in _Q_ALPHA:
        raise ValueError(f"alpha must be one of {list(_Q_ALPHA)}.")
    if k not in _Q_ALPHA[alpha]:
        raise ValueError(
            f"Nemenyi q_alpha table only covers k ≤ 10 (got k={k}). "
            "Use scikit-posthocs for larger k."
        )
    if ensemble_names is None:
        ensemble_names = [f"ensemble_{i}" for i in range(k)]
    if len(ensemble_names) != k:
        raise ValueError("ensemble_names length must match number of ensembles.")

    # ── Per-sample correctness scores (one observation per sample per model) ──
    scores = [(p == y_true).astype(float) for p in preds]    # shape: k × (n,)
    mean_scores = {name: float(s.mean()) for name, s in zip(ensemble_names, scores)}

    # ── Friedman test ─────────────────────────────────────────────────────────
    friedman_stat, friedman_p = friedmanchisquare(*scores)
    is_significant = bool(friedman_p < alpha)

    # ── Mean ranks (rank 1 = best within each sample) ─────────────────────────
    data       = np.column_stack(scores)                      # shape: (n, k)
    rank_matrix = np.apply_along_axis(
        lambda row: rankdata(-row, method="average"),         # negate: rank 1 = best
        axis=1, arr=data,
    )
    mean_ranks = {
        name: float(r) for name, r in zip(ensemble_names, rank_matrix.mean(axis=0))
    }

    # ── Nemenyi critical difference ───────────────────────────────────────────
    q   = _Q_ALPHA[alpha][k]
    cd  = q * np.sqrt(k * (k + 1) / (6 * n))

    # ── Pairwise post-hoc comparisons ─────────────────────────────────────────
    pairwise = {}
    if is_significant:
        mr = rank_matrix.mean(axis=0)
        for i, j in combinations(range(k), 2):
            diff        = abs(mr[i] - mr[j])
            pair_sig    = bool(diff > cd)
            better_idx  = i if mr[i] < mr[j] else j
            key = f"{ensemble_names[i]} vs {ensemble_names[j]}"
            pairwise[key] = {
                "rank_diff"            : round(diff, 4),
                "critical_difference"  : round(cd, 4),
                "significant"          : pair_sig,
                "better"               : ensemble_names[better_idx],
            }

    return {
        "k"                   : k,
        "n"                   : n,
        "alpha"               : alpha,
        "ensemble_scores"     : mean_scores,
        "mean_ranks"          : mean_ranks,
        "critical_difference" : round(cd, 4),
        "friedman_stat"       : round(friedman_stat, 4),
        "friedman_p"          : round(friedman_p, 6),
        "significant"         : is_significant,
        "pairwise"            : pairwise,
    }


def print_results(results: dict) -> None:
    """Pretty-print the output of compare_ensembles."""
    k, n = results["k"], results["n"]
    print(f"\n{'═'*54}")
    print(f"  Friedman test  ({k} ensembles, n={n} samples)")
    print(f"{'═'*54}")
    print(f"  χ² = {results['friedman_stat']:.4f}   p = {results['friedman_p']:.6f}   α = {results['alpha']}")
    sig_label = "✓ SIGNIFICANT" if results["significant"] else "✗ not significant"
    print(f"  Result: {sig_label}\n")

    print(f"  {'Ensemble':<20} {'Accuracy':>9} {'Mean rank':>10}")
    print(f"  {'-'*42}")
    for name in results["ensemble_scores"]:
        acc  = results["ensemble_scores"][name]
        rank = results["mean_ranks"][name]
        print(f"  {name:<20} {acc:>9.4f} {rank:>10.4f}")

    if results["pairwise"]:
        print(f"\n  Post-hoc Nemenyi  (CD = {results['critical_difference']:.4f})")
        print(f"  {'-'*52}")
        for pair, info in results["pairwise"].items():
            sig = "✓" if info["significant"] else "✗"
            better = f"→ {info['better']}" if info["significant"] else ""
            print(f"  {pair:<36} diff={info['rank_diff']:.4f}  {sig} {better}")
    elif results["significant"] is False:
        print("\n  Pairwise tests skipped (Friedman not significant).")
    print(f"{'═'*54}\n")

