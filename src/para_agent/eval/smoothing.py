"""Read-only statistics for the preregistered stage-1.5 scoring experiment."""
import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr


def _sample_values(samples, method):
    fields = {'integer_mean': 'integers', 'expected_mean': 'expected'}
    if method not in fields:
        raise ValueError('Unknown smoothed estimator')
    values = np.asarray([s[fields[method]] for s in samples], dtype=np.float64)
    if values.shape != (64, 8) or not np.isfinite(values).all():
        raise ValueError('Block diagnostics require all 64 complete eight-rubric samples')
    return values


def _block_summary(values, n):
    if n not in (8, 16, 32, 64):
        raise ValueError('Block size must be 8, 16, 32, or 64')
    means = values.reshape(64 // n, n, 8).mean(axis=1)
    totals = means.sum(axis=1)
    return {'n': n, 'block_count': len(means),
            'sample_ranges_one_based': [[i*n+1, (i+1)*n] for i in range(64//n)],
            'rubric_block_means': means.tolist(), 'total_block_means': totals.tolist(),
            'rubric_block_sd': means.std(axis=0, ddof=1).tolist() if len(means)>1 else None,
            'total_block_sd': float(totals.std(ddof=1)) if len(means)>1 else None,
            'ddof': 1,
            'interpretation': 'SD across disjoint N-sample estimates, not divided by sqrt(block_count); N64 has one block and undefined SD'}


def block_statistics(samples, method, n):
    """Monte Carlo variability of an N-sample score, using all 64 draws."""
    return _block_summary(_sample_values(samples, method), n)


def paired_block_statistics(baseline_samples, variant_samples, method, n):
    """Pair identical accepted-sample block indices; retain covariance."""
    result = _block_summary(_sample_values(variant_samples, method) -
                            _sample_values(baseline_samples, method), n)
    attempts_a = [s.get('attempt') for s in baseline_samples]
    attempts_b = [s.get('attempt') for s in variant_samples]
    known = all(x is not None for x in attempts_a + attempts_b)
    result['accepted_sample_attempt_indices_aligned'] = attempts_a == attempts_b if known else None
    result['attempt_alignment_note'] = ('Common essay seed; pair accepted draw indices. Rejections or different generation lengths can weaken common-random-number coupling; variance reduction is not assumed.')
    return result


def agreement(human, predicted, rubric_keys):
    human = np.asarray(human, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if human.shape != predicted.shape or human.ndim != 2 or human.shape[1] != len(rubric_keys):
        raise ValueError("Human and model scores must be aligned essay-by-rubric matrices")
    if len(human) < 3 or not np.isfinite(human).all() or not np.isfinite(predicted).all():
        raise ValueError("At least three complete finite paired essays are required")
    output = {}
    for index, key in enumerate([*rubric_keys, "total"]):
        a = human[:, index] if key != "total" else human.sum(axis=1)
        b = predicted[:, index] if key != "total" else predicted.sum(axis=1)
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            raise ValueError(f"Correlation undefined for constant {key} scores")
        pearson, spearman = pearsonr(a, b), spearmanr(a, b)
        output[key] = {"n": len(a), "pearson": float(pearson.statistic),
                       "spearman": float(spearman.statistic),
                       "pearson_p_two_sided": float(pearson.pvalue),
                       "spearman_p_two_sided": float(spearman.pvalue)}
    return output


def _row_correlations(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a*a).sum(axis=1) * (b*b).sum(axis=1))
    return np.divide((a*b).sum(axis=1), denom, out=np.full(len(a), np.nan), where=denom>0)


def paired_stratified_spearman_interval(human_total, baseline_total, candidate_total, strata,
                                        *, resamples=10000, seed=15043):
    """Preserve 34/33/33 strata and paired essay IDs; re-rank each resample."""
    arrays = [np.asarray(x, dtype=np.float64) for x in (human_total, baseline_total, candidate_total)]
    n = len(arrays[0])
    if any(x.shape != (n,) or not np.isfinite(x).all() for x in arrays) or len(strata) != n:
        raise ValueError("Invalid paired arrays or strata")
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    indices = np.concatenate([rng.choice(np.flatnonzero(strata == name),
                                         size=(resamples, int(np.sum(strata == name))), replace=True)
                              for name in sorted(set(strata))], axis=1)
    ranks = [rankdata(a[indices], method="average", axis=1) for a in arrays]
    differences = _row_correlations(ranks[0], ranks[2]) - _row_correlations(ranks[0], ranks[1])
    valid = differences[np.isfinite(differences)]
    if not len(valid):
        raise ValueError("No defined bootstrap correlations")
    return {"resamples": resamples, "valid_resamples": len(valid), "seed": seed,
            "stratum_sizes": {str(k): int(np.sum(strata == k)) for k in sorted(set(strata))},
            "ci95_spearman_difference": np.quantile(valid, [.025, .975]).tolist(),
            "method": "paired stratified percentile bootstrap, average ranks recomputed within each resample",
            "scope": "essay sampling uncertainty, conditional on the realized Monte Carlo scores; not used to override the primary point-estimate rule"}


def convergence_summary(case_results, method, n):
    if not case_results:
        raise ValueError("No cases")
    differences, ses, changes, block_sds = [], [], [], []
    for result in case_results:
        summary = result["smoothed"]["summary"]
        value, reference = summary[str(n)][method], summary['64'][method]
        differences.append(abs(value['total'] - reference['total']))
        ses.append(value['total_mc_se'])
        block_sd = block_statistics(result['smoothed']['samples'], method, n)['total_block_sd']
        if block_sd is not None:
            block_sds.append(block_sd)
        if n != 8:
            changes.append(abs(value['total'] - summary[str(n//2)][method]['total']))
    return {"inputs": len(case_results), "method": method, "n": n,
            "blocks_per_input": 64//n,
            "mean_total_block_sd": float(np.mean(block_sds)) if block_sds else None,
            "median_total_block_sd": float(np.median(block_sds)) if block_sds else None,
            "max_total_block_sd": float(np.max(block_sds)) if block_sds else None,
            "mean_abs_total_difference_to_n64": float(np.mean(differences)),
            "median_abs_total_difference_to_n64": float(np.median(differences)),
            "max_abs_total_difference_to_n64": float(np.max(differences)),
            "mean_total_mc_se": float(np.mean(ses)), "max_total_mc_se": float(np.max(ses)),
            "mean_abs_total_change_from_previous_n": float(np.mean(changes)) if changes else None,
            "max_abs_total_change_from_previous_n": float(np.max(changes)) if changes else None,
            "n64_is_a_finite_sample_reference_not_ground_truth": True}
