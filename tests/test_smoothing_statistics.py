import numpy as np
import pytest

from para_agent.eval.smoothing import (agreement, paired_stratified_spearman_interval,
                                      block_statistics, paired_block_statistics)


def test_agreement_uses_per_rubric_and_essay_totals_and_handles_tied_human_scores():
    human = [[1, 1], [2, 1], [2, 2], [4, 4], [5, 5], [6, 7]]
    model = np.asarray(human, dtype=float) * 1.5 + 2
    result = agreement(human, model, ['a', 'b'])
    assert set(result) == {'a', 'b', 'total'}
    assert all(row['pearson'] == pytest.approx(1) and row['spearman'] == pytest.approx(1)
               for row in result.values())
    inverse = agreement(human, 20-model, ['a', 'b'])
    assert inverse['total']['spearman'] == pytest.approx(-1)
    with pytest.raises(ValueError, match='constant'):
        agreement(human, np.ones((6,2)), ['a', 'b'])


def test_bootstrap_preserves_paired_strata_and_does_not_count_rubrics_as_essays():
    human = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    reverse = list(reversed(human))
    strata = ['low']*3 + ['middle']*3 + ['high']*3
    result = paired_stratified_spearman_interval(human, reverse, human, strata, resamples=100, seed=123)
    assert result['stratum_sizes'] == {'high':3, 'low':3, 'middle':3}
    assert result['valid_resamples'] == 100
    assert result['ci95_spearman_difference'] == pytest.approx([2,2])
    assert result == paired_stratified_spearman_interval(human, reverse, human, strata, resamples=100, seed=123)


def test_disjoint_blocks_report_sd_of_n_sample_scores_not_se_of_the_64_sample_average():
    samples = [{'integers':[i//8+1]*8,'expected':[i//8+1.0]*8,'attempt':i+1} for i in range(64)]
    for n, totals in [(8,[8,16,24,32,40,48,56,64]), (16,[12,28,44,60]), (32,[20,52])]:
        result=block_statistics(samples,'integer_mean',n)
        assert result['total_block_means']==totals
        assert result['block_count']==64//n
        assert result['total_block_sd']==pytest.approx(np.std(totals,ddof=1))
        assert result['rubric_block_sd'][0]*8==pytest.approx(result['total_block_sd'])
    assert block_statistics(samples,'integer_mean',64)['total_block_sd'] is None
    assert block_statistics(samples,'expected_mean',32)['sample_ranges_one_based']==[[1,32],[33,64]]
    with pytest.raises(ValueError,match='all 64'):
        block_statistics(samples[:32],'integer_mean',32)


def test_paired_block_sd_preserves_common_random_number_cancellation():
    baseline=[{'integers':[i//8+1]*8,'expected':[i//8+1.0]*8,'attempt':i+1} for i in range(64)]
    variant=[{'integers':[i//8+2]*8,'expected':[i//8+2.0]*8,'attempt':i+1} for i in range(64)]
    assert block_statistics(baseline,'integer_mean',32)['total_block_sd']>0
    paired=paired_block_statistics(baseline,variant,'integer_mean',32)
    assert paired['total_block_means']==[8,8]
    assert paired['total_block_sd']==0
    assert paired['accepted_sample_attempt_indices_aligned']
    variant[40]['attempt']+=1
    assert not paired_block_statistics(baseline,variant,'integer_mean',32)['accepted_sample_attempt_indices_aligned']
