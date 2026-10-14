import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from para_agent.config import PROJECT_ROOT
from para_agent.eval.agreement import (
    bootstrap_metric, kappa, pairwise_agreement, select_judge, semantic_agreement,
)
from para_agent.eval.judge import (
    PairwiseJudgment, judge_pairwise, pairwise_payload, semantic_payload,
)
from para_agent.eval.labeling import FIELDS, read_human_labels, write_human_sheet
from para_agent.eval.stage3_metrics import (
    aggregate_operational, meaning_preservation, operational_metrics, style_consistency,
)
from para_agent.eval.validation_data import (
    controlled_pairs, identifier, sample_human_semantics, select_original_pairs,
)


def test_style_counts_only_classifiable_adjacent_pairs_and_reports_coverage():
    result = style_consistency(['읽는다.', '간다.', '갑니다.', '제목', '합니다.'])
    assert result['dominant_styles'] == ['해라체', '해라체', '하십시오체', '기타', '하십시오체']
    assert (result['numerator'], result['denominator'], result['rate']) == (1, 2, .5)
    assert result['possible_adjacent_pairs'] == 4 and result['coverage'] == .5
    assert style_consistency(['제목', '목차'])['rate'] is None
    assert style_consistency(['읽는다.'])['coverage'] is None
    assert style_consistency(['읽는다. 갑니다.', '간다.'])['denominator'] == 0


def test_operational_rates_include_failed_turns_and_do_not_count_issues_twice():
    turns = [
        {'turn_status': 'failed_anonymization',
         'memory_update_validation': {'accepted_count': 1, 'rejected_count': 2},
         'issue_application_audit': {'issues': [
             {'mismatches': ['reported_applied_but_location_unchanged', 'reported_both_applied_and_skipped']},
             {'mismatches': []}]}},
        {'turn_status': 'ok', 'memory_update_validation': {'accepted_count': 1, 'rejected_count': 0},
         'issue_application_audit': {'issues': [{'mismatches': ['reported_skipped_but_location_changed']}]}},
        {'turn_status': 'ok'},
    ]
    result = operational_metrics(turns, '읽는다.\n\n간다.', ['읽는다.', '갑니다.'])
    assert result['fallback']['rate'] == pytest.approx(1 / 3)
    assert result['memory_note_discard']['rate'] == .5
    assert result['issue_application_mismatch']['rate'] == pytest.approx(2 / 3)
    assert result['missing_audit_turns'] == 1
    assert result['original_style_consistency']['rate'] == 1
    assert result['style_consistency']['rate'] == 0
    second = operational_metrics([], '책.', ['책.'])
    assert second['fallback']['rate'] is None
    grouped = aggregate_operational([
        {'condition': 'incremental', **result}, {'condition': 'incremental', **second},
    ])['incremental']
    assert grouped['issue_application_mismatch']['rate'] == pytest.approx(2 / 3)
    assert grouped['essays'] == 2


def test_meaning_rate_zero_edits_is_undefined_and_nuance_counts_as_change():
    assert meaning_preservation([])['rate'] is None
    result = meaning_preservation(['보존', '경미한 뉘앙스 변화', '의미 변화'])
    assert result['rate'] == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match='Unknown'):
        meaning_preservation(['임의 판정'])


def test_kappa_uses_chance_agreement_and_merges_two_change_labels():
    assert kappa(['a', 'a', 'b', 'b'], ['a', 'a', 'a', 'b']) == .5
    assert kappa(['a', 'b'], ['b', 'a']) == -1
    assert kappa(['a'], ['a']) is None
    result = semantic_agreement(
        ['보존', '보존', '경미한 뉘앙스 변화', '의미 변화'],
        ['보존', '보존', '의미 변화', '경미한 뉘앙스 변화'],
        ['essay1', 'essay1', 'essay2', 'essay2'], resamples=100,
    )
    assert result['estimate'] == 1 and result['passes']
    assert result['three_class_agreement'] == .5
    assert result['confusion']['변화']['변화'] == 2
    assert result['valid_resamples'] < 100  # one-class resamples have undefined kappa


def test_cluster_bootstrap_keeps_both_display_orders_together():
    result = pairwise_agreement(['A', 'A', 'B', 'B'], ['A', 'A', '동률', '동률'],
                                ['p1', 'p1', 'p2', 'p2'], resamples=200)
    assert result['estimate'] == .5 and not result['passes']
    assert result['ci95'] == [0, 1] and result['clusters'] == 2
    assert result == pairwise_agreement(['A', 'A', 'B', 'B'], ['A', 'A', '동률', '동률'],
                                       ['p1', 'p1', 'p2', 'p2'], resamples=200)
    undefined = bootstrap_metric(['a'], ['a'], ['c'], kappa, resamples=10)
    assert undefined['ci95'] is None and undefined['valid_resamples'] == 0


@pytest.mark.parametrize('haiku,sonnet,status,model', [
    ({}, {}, 'pending', None),
    ({'pairwise': {'passes': True}}, {}, 'pending', None),
    ({'pairwise': {'passes': False}}, {}, 'pending', None),
    ({'pairwise': {'passes': False}}, {'pairwise': {'passes': False}}, 'stop', None),
    ({k: {'passes': True} for k in ('pairwise', 'controlled_semantic', 'human_semantic')},
     {}, 'selected', 'claude-haiku-5-5'),
    ({'controlled_semantic': {'passes': False}},
     {k: {'passes': True} for k in ('pairwise', 'controlled_semantic', 'human_semantic')},
     'selected', 'claude-sonnet-5-5'),
])
def test_judge_selection_never_accepts_incomplete_validation(haiku, sonnet, status, model):
    result = select_judge({'claude-haiku-5-5': haiku, 'claude-sonnet-5-5': sonnet})
    assert (result['status'], result['model']) == (status, model)


def test_blind_judge_payloads_and_reverse_winner_mapping():
    item = {'prompt': '문항', 'before': '원문', 'after': '수정', 'context_before': '원문 문맥',
            'context_after': '수정 문맥', 'gold': 'A', 'agent_model': 'secret-model',
            'human_scores': {'A': 30, 'B': 60}, 'condition': 'secret-condition', 'A': '글A', 'B': '글B'}
    assert set(semantic_payload(item)) == {'prompt', 'before', 'after', 'context_before', 'context_after'}
    assert pairwise_payload(item, reverse=True)['A'] == '글B'
    assert pairwise_payload(item)['original'] is None
    class Fake:
        def request(self, agent, payload, schema, **kwargs):
            assert not set(payload) & {'gold', 'agent_model', 'human_scores', 'condition'}
            data = {
                k: {'winner': '동률' if k == 'meaning' else 'A', 'reason': '글에 근거한 이유'}
                for k in ('style', 'correction_accuracy', 'meaning', 'overall')}
            data['content_addition'] = {side: {'status': '판단불가', 'added_spans': [], 'reason': '원문 미제공'}
                                        for side in ('A', 'B')}
            return PairwiseJudgment.model_validate(data)
    result = judge_pairwise(Fake(), item, context={}, reverse=True)
    assert result['displayed']['overall']['winner'] == 'A'
    assert result['canonical']['overall']['winner'] == 'B'
    assert result['canonical']['meaning']['winner'] == '동률'


def test_controlled_samples_are_balanced_with_known_transformations():
    rows = controlled_pairs()
    assert len(rows) == len({r['item_id'] for r in rows}) == 80
    assert sum(r['gold'] == '보존' for r in rows) == 40
    for category in ('spacing', 'spelling', 'contraction', 'particle', 'negation', 'modality', 'tense', 'emotion'):
        assert sum(r['category'] == category for r in rows) == 10
    assert all(r['before'] != r['after'] for r in rows)


def candidate(id, prompt, total, first, second):
    return SimpleNamespace(essay=SimpleNamespace(essay_id=id, prompt=prompt, paragraphs=[id, '본문', '끝']),
                           total=total, provenance={'rater_1': first, 'rater_2': second})


def test_original_pair_selection_respects_prompt_gap_exclusions_and_both_raters():
    rows = [candidate('low', 'same', 10, [5], [5]), candidate('high', 'same', 40, [20], [20]),
            candidate('excluded', 'same', 72, [36], [36]),
            candidate('other', 'different', 60, [30], [30]),
            candidate('badlow', 'disagree', 10, [5], [5]), candidate('badhigh', 'disagree', 40, [2], [38])]
    pairs, stats = select_original_pairs(rows, {'excluded'}, count=1, min_gap=16)
    assert set(pairs[0]['essay_ids']) == {'low', 'high'}
    assert pairs[0]['human_scores'][pairs[0]['gold']] == 40
    assert stats['available_prompt_groups'] == 1
    assert (pairs, stats) == select_original_pairs(list(reversed(rows)), {'excluded'}, count=1, min_gap=16)
    with pytest.raises(ValueError, match='distinct prompts'):
        select_original_pairs(rows, {'excluded', 'high'}, count=1, min_gap=16)


def revision(index, model, before=None, after=None):
    before = before or f'{model}의 원문 {index}이다.'
    after = after or f'{model}의 수정문 {index}이다.'
    return {'item_id': f'{model}{index}', 'prompt': '문항', 'before': before, 'after': after,
            'context_before': before, 'context_after': after, 'essay_id': f'e{index % 5}',
            'condition': ('whole_once', 'incremental')[index % 2], 'agent_model': model,
            'pair_hash': identifier([before, after])}


def test_actual_sampling_includes_luna_known_risks_and_no_duplicate_pairs():
    luna = [revision(i, 'luna') for i in range(30)]
    haiku = [revision(i, 'haiku') for i in range(40)]
    haiku += [revision(40, 'haiku', '두려움을 느낀다.', '부담을 느낀다.'),
              revision(41, 'haiku', '이길 수 있다.', '이기기도 한다.'), dict(haiku[0])]
    selected, stats = sample_human_semantics(luna, haiku)
    assert len(selected) == len({r['pair_hash'] for r in selected}) == 50
    assert stats['models']['luna'] >= 25 and stats['known_failure_candidates'] == 2
    assert {r['blind_id'] for r in selected} == {f'M{i:03d}' for i in range(1, 51)}
    assert (selected, stats) == sample_human_semantics(luna, haiku)
    with pytest.raises(ValueError, match='actual changed Luna'):
        sample_human_semantics([], haiku)


def test_human_sheet_hides_provenance_and_escapes_html_without_losing_text(tmp_path):
    item = {**revision(1, 'hidden-agent', '</script><script>attack()</script>', '수정문'),
            'blind_id': 'M001', 'gold': 'hidden-gold', 'sampling_reason': 'hidden-reason'}
    directory = tmp_path / 'human'
    write_human_sheet(directory, [item])
    page = (directory / 'semantic_labels.html').read_text()
    assert '</script><script>attack()' not in page
    for secret in ('hidden-gold', 'hidden-reason', 'source_run', 'agent_model', 'condition'):
        assert secret not in page
    embedded = page.split('const rows=', 1)[1].split(';const host=', 1)[0]
    assert json.loads(embedded)[0]['before'] == item['before']
    path = directory / 'semantic_labels.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == FIELDS
        rows = list(reader)
    with pytest.raises(ValueError, match='filled'):
        read_human_labels(path, ['M001'])
    rows[0]['human_label'] = '의미 변화'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    assert read_human_labels(path, ['M001']) == {'M001': '의미 변화'}
    with pytest.raises(ValueError, match='exactly'):
        read_human_labels(path, ['M002'])


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / f'scripts/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('model', ['haiku', 'sonnet'])
def test_judge_dry_run_costs_both_orders_without_loading_keys(tmp_path, monkeypatch, model):
    from para_agent.artifacts import file_hash, write_json
    script = load_script('judge_stage3')
    data = tmp_path / 'validation'
    write_json(data / 'private/controlled.json', controlled_pairs())
    write_json(data / 'private/original_pairs.json', [
        {'item_id': f'p{i}', 'prompt': '문항', 'A': '글A', 'B': '글B'} for i in range(20)])
    write_json(data / 'setup.json', {'source_hashes': {}, 'frozen_data_hashes': {
        name: file_hash(data / 'private' / name) for name in ('controlled.json', 'original_pairs.json')}})
    def forbidden(*args, **kwargs):
        raise AssertionError('Costing must not load an API key')
    monkeypatch.setattr('para_agent.llm.anthropic_transport.dotenv_values', forbidden)
    out = tmp_path / 'cost.json'
    script.main(['--data-dir', str(data), '--model', model, '--scope', 'controlled', 'original_pairs', '--output', str(out)])
    plan = json.loads(out.read_text())
    assert plan['logical_calls'] == 120 and plan['max_calls_with_repairs'] == 360
    assert 0 < plan['expected_usd'] < plan['all_initial_requests_reserved_max_usd']
    assert plan['external_api_calls'] == 0
    # A paid baseline job cannot bypass pending model validation.
    with pytest.raises(ValueError, match='validated selected judge'):
        script.main(['--data-dir', str(data), '--model', model, '--scope', 'luna_pairs',
                     '--output', str(tmp_path / 'execute'), '--execute', '--approved-budget-usd', '1'])


def test_baseline_report_keeps_missing_judge_pending_and_accounts_real_cost(tmp_path, monkeypatch):
    from para_agent.artifacts import write_json
    monkeypatch.syspath_prepend(str(PROJECT_ROOT / 'scripts'))
    script = load_script('summarize_stage3')
    source = tmp_path / 'source'
    write_json(source / 'acceptance.json', {'ok': True, 'mock': False, 'real_api_calls': 1,
               'estimated_billed_usd': .00015, 'approved_budget_usd': .5})
    (source / 'llm_calls.jsonl').write_text(json.dumps({
        'condition': 'incremental', 'real_api_call': True,
        'tokens': {'input_tokens': 1000, 'output_tokens': 100}}) + '\n')
    data = tmp_path / 'validation'
    write_json(data / 'luna_local_metrics.json', {'source_run': str(source), 'conditions': {
        'whole_once': {'essays': 5}, 'incremental': {'essays': 5}}})
    result = script.build_report(data)
    assert not result['primary_evaluation_complete'] and not result['baseline_contains_haiku']
    assert result['conditions']['incremental']['estimated_api_usd'] == pytest.approx(.00015)
    assert result['conditions']['whole_once']['estimated_api_usd'] == 0
    assert result['conditions']['incremental']['meaning_preservation']['status'] == 'pending_validated_judge'
    with pytest.raises(ValueError, match='passing validation'):
        script.build_report(data, judge_results=[tmp_path / 'unvalidated'])
