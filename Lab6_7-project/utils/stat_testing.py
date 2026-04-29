import numpy as np
from scipy.stats import friedmanchisquare
from itertools import combinations

# Prompt (Sonnet 4.6)
"""Suggest a statistical test for diffrences between 3 ensembles in multiclass classification task. Write a python function that takes ensembles prediction and return test results"""

def compare_ensembles(
    ensemble_predictions: list[np.ndarray],
    y_true: np.ndarray,
    alpha: float = 0.05,
    metric: str = "accuracy",
) -> dict:
    """
    Compare 3 ensembles using Friedman test + post-hoc Nemenyi test.

    Parameters
    ----------
    ensemble_predictions : list of arrays, shape [(n_samples,), ...]
        Predicted class labels from each ensemble. Must be 3 arrays.
    y_true : array, shape (n_samples,)
        True class labels.
    alpha : float
        Significance level (default 0.05).
    metric : str
        Scoring metric — "accuracy" or "f1_macro".

    Returns
    -------
    dict with keys:
        friedman_stat, friedman_p, significant,
        pairwise (dict of pair -> (stat, p, significant))
    """
    if len(ensemble_predictions) != 3:
        raise ValueError("Exactly 3 ensemble prediction arrays are required.")

    n = len(y_true)

    # --- Score each prediction per sample (1 = correct, 0 = wrong) ---
    def sample_scores(preds):
        if metric == "accuracy":
            return (preds == y_true).astype(float)
        elif metric == "f1_macro":
            # Per-sample binary correct/incorrect (F1 per sample is 0 or 1)
            return (preds == y_true).astype(float)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

    scores = [sample_scores(np.asarray(p)) for p in ensemble_predictions]
    # scores[i] has shape (n_samples,) — one observation per sample per model

    # --- Friedman test (non-parametric k-related-samples test) ---
    friedman_stat, friedman_p = friedmanchisquare(*scores)
    is_significant = friedman_p < alpha

    results = {
        "friedman_stat": round(friedman_stat, 4),
        "friedman_p": round(friedman_p, 6),
        "significant": is_significant,
        "ensemble_means": {
            f"ensemble_{i}": round(float(s.mean()), 4)
            for i, s in enumerate(scores)
        },
        "pairwise": {},
    }

    # --- Post-hoc Nemenyi test (only if Friedman is significant) ---
    if is_significant:
        results["pairwise"] = _nemenyi_test(scores, n, alpha)

    return results


def _nemenyi_test(scores: list, n: int, alpha: float) -> dict:
    """
    Nemenyi post-hoc test for pairwise comparisons after Friedman.

    Uses the critical difference based on the Studentized range distribution
    approximation (standard in ML benchmarking literature).
    """
    k = len(scores)

    # Rank each sample across the k models (higher score = better rank)
    # Shape: (n_samples, k)
    data = np.column_stack(scores)
    # Rank within each row (sample), ties averaged
    from scipy.stats import rankdata
    ranks = np.apply_along_axis(
        lambda row: rankdata(-row, method="average"),  # negate: rank 1 = best
        axis=1,
        arr=data,
    )
    mean_ranks = ranks.mean(axis=0)  # shape (k,)

    # Critical difference at significance level alpha
    # q_alpha values for k=3 at alpha=0.05: 2.343, alpha=0.10: 1.960
    q_alpha_table = {
        (3, 0.05): 2.343,
        (3, 0.10): 1.960,
        (4, 0.05): 2.569,
        (4, 0.10): 2.291,
    }
    q = q_alpha_table.get((k, alpha), 2.343)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))

    pairwise = {}
    for i, j in combinations(range(k), 2):
        rank_diff = abs(mean_ranks[i] - mean_ranks[j])
        pair_significant = rank_diff > cd
        pairwise[f"ensemble_{i}_vs_ensemble_{j}"] = {
            "mean_rank_diff": round(float(rank_diff), 4),
            "critical_difference": round(float(cd), 4),
            "significant": pair_significant,
            "better": f"ensemble_{i}" if mean_ranks[i] < mean_ranks[j] else f"ensemble_{j}",
        }

    return pairwise