"""Paired statistics and usage accounting for an explicitly preliminary evaluation."""
import math

import numpy as np
from scipy.stats import PermutationMethod, wilcoxon

from .final_score import CONDITIONS


def paired_statistics(differences, *, seed=42, resamples=10000):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Paired differences must be a nonempty finite vector")
    if resamples < 1:
        raise ValueError("Bootstrap resamples must be positive")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    interval = np.quantile(means, [0.025, 0.975])
    # Zeros contribute no rank; average ranks of ties are preserved by permutation.
    nonzero = values[values != 0]
    if not nonzero.size:
        statistic, pvalue, method_name = 0.0, 1.0, "all_differences_zero"
    else:
        exact = nonzero.size <= 13
        method = PermutationMethod(n_resamples=math.inf if exact else 99999,
                                   batch=1000, rng=np.random.default_rng(seed))
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox", method=method)
        statistic, pvalue = float(result.statistic), float(result.pvalue)
        method_name = "exact_sign_permutation" if exact else "99999_sign_permutations"
    return {"n": int(values.size), "mean_difference": float(values.mean()),
            "median_difference": float(np.median(values)), "ci95": interval.tolist(),
            "positive": int((values > 0).sum()), "ties": int((values == 0).sum()),
            "negative": int((values < 0).sum()), "wilcoxon_statistic": statistic,
            "wilcoxon_p_two_sided": pvalue, "wilcoxon_method": method_name,
            "bootstrap_resamples": resamples, "seed": seed}


def aggregate(rows, rubric_keys, *, seed=42, resamples=10000):
    index = {(row["essay_id"], row["condition"]): row for row in rows}
    ids = list(dict.fromkeys(row["essay_id"] for row in rows))
    if len(index) != len(rows) or len(rows) != 3 * len(ids):
        raise ValueError("Need one original and two condition rows per essay")
    expected = {(essay_id, condition) for essay_id in ids for condition in ("original", *CONDITIONS)}
    if set(index) != expected:
        raise ValueError("Unpaired or unknown evaluation conditions")
    comparisons = {}
    for before, after in (("original", "whole_once"), ("original", "incremental"),
                          ("whole_once", "incremental")):
        comparisons[f"{after}_minus_{before}"] = {
            metric: paired_statistics([index[e, after][metric] - index[e, before][metric] for e in ids],
                                      seed=seed, resamples=resamples)
            for metric in (*rubric_keys, "total")}
    return comparisons


def call_costs(calls, essay_id, condition):
    rows = [c for c in calls if c["essay_id"] == essay_id and c["condition"] == condition]
    return {"real_api_calls": sum(bool(c.get("real_api_call")) for c in rows),
            "cache_hits": sum(bool(c["cache_hit"]) for c in rows),
            "input_tokens": sum(c["tokens"].get("input_tokens", 0) for c in rows),
            "output_tokens": sum(c["tokens"].get("output_tokens", 0) for c in rows),
            "total_tokens": sum(c["tokens"].get("total_tokens", 0) for c in rows),
            "llm_latency_s": sum(c["latency_s"] for c in rows),
            "parse_repairs": sum(c["attempt"] > 0 for c in rows)}
