"""
Statistical tests for brain tumor segmentation model comparison.

Tests specifically chosen for medical image segmentation:
    - Paired Wilcoxon signed-rank test (2 models on same patients)
    - Friedman test + Nemenyi post-hoc (>2 models)
    - Bootstrap 95% confidence intervals
    - Cohen's d effect size
    - McNemar's test (segmentation failure rates)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats


def paired_wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str = "Model A",
    name_b: str = "Model B",
) -> Dict:
    """
    Paired Wilcoxon signed-rank test between two models.

    Tests whether there is a statistically significant difference
    between per-patient Dice scores from two models. Non-parametric
    alternative to paired t-test — appropriate because Dice scores
    are bounded [0, 1] and typically non-normal.

    Args:
        scores_a: Per-patient Dice scores from model A
        scores_b: Per-patient Dice scores from model B

    Returns:
        Dict with statistic, p_value, interpretation, means, etc.
    """
    assert len(scores_a) == len(scores_b), "Must have same patients"

    statistic, p_value = stats.wilcoxon(scores_a, scores_b, alternative="two-sided")

    diff = scores_a - scores_b
    mean_diff = np.mean(diff)
    median_diff = np.median(diff)

    # Interpretation
    alpha = 0.05
    significant = p_value < alpha
    if significant:
        better = name_a if mean_diff > 0 else name_b
        interpretation = f"{better} is significantly better (p={p_value:.4f})"
    else:
        interpretation = f"No significant difference (p={p_value:.4f})"

    return {
        "test": "Wilcoxon signed-rank",
        "model_a": name_a,
        "model_b": name_b,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "mean_difference": float(mean_diff),
        "median_difference": float(median_diff),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": significant,
        "interpretation": interpretation,
        "n_patients": len(scores_a),
    }


def friedman_test(
    scores_dict: Dict[str, np.ndarray],
) -> Dict:
    """
    Friedman test for comparing >2 models simultaneously.

    Non-parametric repeated measures ANOVA. Tests whether there is
    a significant difference among 3+ models evaluated on the same patients.

    Args:
        scores_dict: {model_name: per_patient_dice_array}

    Returns:
        Dict with statistic, p_value, model rankings
    """
    model_names = list(scores_dict.keys())
    score_arrays = [scores_dict[name] for name in model_names]

    # All must have same length
    n = len(score_arrays[0])
    assert all(len(s) == n for s in score_arrays), "All models must have same patients"

    statistic, p_value = stats.friedmanchisquare(*score_arrays)

    # Compute average ranks
    scores_matrix = np.column_stack(score_arrays)  # (n_patients, n_models)
    # Rank per patient (higher Dice = lower rank number)
    ranks = np.zeros_like(scores_matrix)
    for i in range(n):
        ranks[i] = stats.rankdata(-scores_matrix[i])  # negative for descending
    avg_ranks = ranks.mean(axis=0)

    ranking = sorted(zip(model_names, avg_ranks), key=lambda x: x[1])

    return {
        "test": "Friedman",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_models": len(model_names),
        "n_patients": n,
        "rankings": [{"model": name, "avg_rank": float(rank)} for name, rank in ranking],
        "model_means": {name: float(np.mean(scores)) for name, scores in scores_dict.items()},
    }


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict:
    """
    Bootstrap 95% confidence interval for mean Dice.

    Non-parametric CI estimation via resampling.

    Args:
        scores: Per-patient Dice scores
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95)

    Returns:
        Dict with mean, ci_lower, ci_upper, std
    """
    rng = np.random.RandomState(seed)
    n = len(scores)

    bootstrap_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(scores, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "n_patients": n,
    }


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict:
    """
    Cohen's d effect size for paired observations.

    Quantifies the practical significance of the difference
    between two models. Supplements p-values with magnitude.

    Interpretation:
        |d| < 0.2: negligible
        0.2 ≤ |d| < 0.5: small
        0.5 ≤ |d| < 0.8: medium
        |d| ≥ 0.8: large

    Returns:
        Dict with d-value and interpretation
    """
    diff = scores_a - scores_b
    d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0.0

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"

    return {
        "cohens_d": float(d),
        "magnitude": magnitude,
        "interpretation": f"{magnitude} effect size (d={d:.3f})",
    }


def mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    McNemar's test for comparing segmentation failure rates.

    Compares whether two models fail on different cases.
    A "failure" is defined as Dice < threshold for a given patient.

    Args:
        preds_a: Per-patient Dice scores from model A
        preds_b: Per-patient Dice scores from model B
        ground_truth: Not used directly — thresholds are on Dice scores
        threshold: Dice threshold for "clinically acceptable"

    Returns:
        Dict with contingency table and p-value
    """
    success_a = preds_a >= threshold
    success_b = preds_b >= threshold

    # Contingency table
    # Both succeed, A succeeds B fails, A fails B succeeds, Both fail
    both_success = np.sum(success_a & success_b)
    a_only = np.sum(success_a & ~success_b)
    b_only = np.sum(~success_a & success_b)
    both_fail = np.sum(~success_a & ~success_b)

    # McNemar's test uses the discordant pairs
    n_discordant = a_only + b_only
    if n_discordant == 0:
        p_value = 1.0
        statistic = 0.0
    else:
        # Use exact binomial test for small samples
        if n_discordant < 25:
            p_value = stats.binom_test(a_only, n_discordant, 0.5)
            statistic = float(a_only)
        else:
            # Chi-squared approximation with continuity correction
            statistic = (abs(a_only - b_only) - 1) ** 2 / (a_only + b_only)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "test": "McNemar",
        "threshold": threshold,
        "contingency": {
            "both_success": int(both_success),
            "a_success_b_fail": int(a_only),
            "a_fail_b_success": int(b_only),
            "both_fail": int(both_fail),
        },
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "failure_rate_a": float(1 - np.mean(success_a)),
        "failure_rate_b": float(1 - np.mean(success_b)),
    }


def run_full_comparison(
    results: Dict[str, Dict],
    class_names: List[str] = None,
) -> Dict:
    """
    Run all statistical tests on a set of experiment results.

    Args:
        results: {experiment_name: {"dice_per_patient": np.array of shape (n_patients, 3)}}
        class_names: Names for the 3 classes (default: WT, TC, ET)

    Returns:
        Comprehensive comparison dict with all test results
    """
    if class_names is None:
        class_names = ["WT", "TC", "ET"]

    model_names = list(results.keys())
    comparison = {
        "models": model_names,
        "pairwise_wilcoxon": [],
        "bootstrap_ci": {},
        "effect_sizes": [],
        "mcnemar": [],
    }

    # Per-model bootstrap CI
    for name in model_names:
        dice = results[name]["dice_per_patient"]  # (n_patients, 3)
        mean_dice = dice.mean(axis=1)  # mean across classes per patient
        comparison["bootstrap_ci"][name] = {
            "overall": bootstrap_confidence_interval(mean_dice),
        }
        for c, cname in enumerate(class_names):
            comparison["bootstrap_ci"][name][cname] = bootstrap_confidence_interval(dice[:, c])

    # Pairwise Wilcoxon + Cohen's d
    for i, name_a in enumerate(model_names):
        for j, name_b in enumerate(model_names):
            if i >= j:
                continue
            dice_a = results[name_a]["dice_per_patient"].mean(axis=1)
            dice_b = results[name_b]["dice_per_patient"].mean(axis=1)

            wilcoxon = paired_wilcoxon_test(dice_a, dice_b, name_a, name_b)
            comparison["pairwise_wilcoxon"].append(wilcoxon)

            effect = cohens_d(dice_a, dice_b)
            effect["model_a"] = name_a
            effect["model_b"] = name_b
            comparison["effect_sizes"].append(effect)

            mcn = mcnemar_test(dice_a, dice_b, None)
            mcn["model_a"] = name_a
            mcn["model_b"] = name_b
            comparison["mcnemar"].append(mcn)

    # Friedman test (if >2 models)
    if len(model_names) > 2:
        scores_dict = {
            name: results[name]["dice_per_patient"].mean(axis=1)
            for name in model_names
        }
        comparison["friedman"] = friedman_test(scores_dict)

    return comparison
