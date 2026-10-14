import math

import pytest

from para_agent.diagnose.smoothed_scorer import essay_seed, primary_decision, summarize_samples


def test_sampling_average_marginalizes_multiple_score_lines_and_keeps_nested_prefixes():
    samples = [{"integers": [2] * 8, "expected": [2.5] * 8} for _ in range(8)]
    samples += [{"integers": [8] * 8, "expected": [7.5] * 8} for _ in range(8)]
    result = summarize_samples(samples, (8, 16))
    assert result['8']['integer_mean']['total'] == 16
    assert result['16']['integer_mean']['rubrics'] == [5] * 8
    assert result['16']['expected_mean']['rubrics'] == [5] * 8
    assert result['8']['expected_mean']['total'] == 20
    # Perfectly correlated rubric variation must not be treated as independent.
    summary = result['16']['integer_mean']
    assert summary['total_mc_se'] == pytest.approx(8 * summary['rubric_mc_se'][0])
    assert summary['total_mc_se'] > math.sqrt(sum(x*x for x in summary['rubric_mc_se']))


def test_missing_or_invalid_samples_do_not_silently_change_the_estimator():
    with pytest.raises(ValueError, match='enough'):
        summarize_samples([{'integers': [5]*8, 'expected': [5]*8}]*8, (8, 16))
    with pytest.raises(ValueError, match='eight finite'):
        summarize_samples([{'integers': [5]*7, 'expected': [5]*8}]*8, (8,))
    with pytest.raises(ValueError, match='1–9'):
        summarize_samples([{'integers': [0]*8, 'expected': [5]*8}]*8, (8,))


def test_gate_requires_all_three_prespecified_conditions_without_cherry_picking():
    assert primary_decision([1, -1], [.5], .6, .6)['passes']
    assert not primary_decision([1.00001], [.5], .6, .9)['passes']
    assert not primary_decision([.2], [-1.01], .6, .9)['passes']
    assert not primary_decision([.2], [.5], .6, .599999)['passes']
    assert not primary_decision([1.1], [.5], .6, .9)['existing_evaluation_settings_changed']
    with pytest.raises(ValueError):
        primary_decision([], [.5], .6, .9)
    with pytest.raises(ValueError):
        primary_decision([.2], [.5], .6, float('nan'))


def test_seed_is_stable_and_groups_variants_by_essay_not_text():
    assert essay_seed('essay-a') == essay_seed('essay-a')
    assert essay_seed('essay-a') != essay_seed('essay-b')
    assert essay_seed('essay-a', 123) != essay_seed('essay-a', 124)


def test_documented_seed_configuration_matches_actual_unchanged_rule():
    import hashlib
    from pathlib import Path
    import yaml
    settings=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/smoothed_scoring.yaml').read_text())
    seed=settings['seed']
    value=hashlib.sha256(f"{seed['master_seed']}:essay-a".encode(seed['encoding'])).digest()
    expected=int.from_bytes(value[:seed['digest_prefix_bytes']],seed['byte_order'])%seed['modulus']
    assert expected==essay_seed('essay-a')
    assert settings['future_evaluation']['same_seed_for_original_and_all_revisions']
