#!/usr/bin/env python
"""Export an already completed Sonnet batch run; never call a model."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.dont_write_bytecode = True

from para_agent.artifacts import file_hash, write_json, json_text
from para_agent.llm.anthropic_batch import batch_cost
import yaml


def read(path):
    return json.loads(path.read_text())


def csv_file(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(directory):
    state = read(directory / 'status.json')
    if state['state'] not in {'completed', 'stopped_failed_gate', 'failed'}:
        raise ValueError('Run is still active; no partial results report')
    setup = read(directory / 'setup.json')
    amendment = directory / 'recovery_01.json'
    if not amendment.exists():
        amendment = directory / 'execution_amendment_01.json'
    source_hashes = read(amendment)['source_hashes'] if amendment.exists() else setup['source_hashes']
    for relative, digest in source_hashes.items():
        if file_hash(PROJECT / relative) != digest:
            raise ValueError('Frozen source hash changed')
    rates = yaml.safe_load((PROJECT / 'configs/api_pricing_2026-10-15.yaml').read_text())['models']['claude-sonnet-5-5']
    totals = Counter()
    by_phase = defaultdict(Counter)
    output_types = Counter()
    responses = []
    for path in sorted(directory.glob('*/batches/*/raw_results.json')):
        responses.extend((path.relative_to(directory).parts[0], row['result'], .5)
                         for row in read(path))
    for path in sorted(directory.glob('*/requests/*/response.json')):
        phase = path.relative_to(directory).parts[0]
        if phase == 'actual_semantics_interrupted':
            phase = 'actual_semantics'
        responses.append((phase, {'type': 'succeeded', 'message': read(path)}, 1))
    for phase, outcome, discount in responses:
        totals['returned_requests'] += 1
        totals[outcome['type']] += 1
        if outcome['type'] != 'succeeded':
            continue
        message = outcome['message']
        usage = message['usage']
        charge = 2 * discount * batch_cost(usage, rates)
        totals['estimated_billed_usd'] += charge
        by_phase[phase]['estimated_billed_usd'] += charge
        by_phase[phase]['succeeded_requests'] += 1
        for key in ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'):
            totals[key] += usage.get(key) or 0
            by_phase[phase][key] += usage.get(key) or 0
        for block in message['content']:
            output_types[block['type']] += 1
    judgment_count = sum(len(read(p)) for p in directory.glob('*/results.json'))
    totals['logical_judgments'] = judgment_count
    totals['extra_successful_requests'] = totals['succeeded'] - judgment_count
    summary = {'state': state, 'usage': dict(totals), 'usage_by_phase': dict(by_phase),
               'output_block_types': dict(output_types), 'human_agreement_calculated': False,
               'limitation': '실제 수정에 대한 judge 판정은 사람으로 검증하지 않았다'}
    if (directory / 'gate_report.json').exists():
        summary['gates'] = read(directory / 'gate_report.json')
    if (directory / 'validation/results.json').exists():
        index = {(r['scope'], r['item_id'], r['reverse']): r['result']
                 for r in read(directory / 'validation/results.json')}
        controlled, correct_by_category, total_by_category = [], Counter(), Counter()
        for item in read(directory / 'private/controlled.json'):
            value = index['controlled', item['item_id'], False]
            gold = '보존' if item['gold'] == '보존' else '변화'
            predicted = '보존' if value['label'] == '보존' else '변화'
            correct_by_category[item['category']] += gold == predicted
            total_by_category[item['category']] += 1
            controlled.append({'item_id': item['item_id'], 'category': item['category'],
                'before': item['before'], 'after': item['after'], 'gold': item['gold'],
                'judge_label': value['label'], 'binary_correct': gold == predicted,
                'reason': value['reason'], 'content_addition': value['content_addition']['status']})
        summary['controlled_by_category'] = {k: {'correct': correct_by_category[k], 'n': n}
                                              for k, n in total_by_category.items()}
        csv_file(directory / 'controlled_results.csv', controlled)
        originals = []
        for item in read(directory / 'private/original_pairs.json'):
            for reverse in (False, True):
                value = index['original_pairs', item['item_id'], reverse]['canonical']
                originals.append({'item_id': item['item_id'], 'reverse': reverse,
                    'essay_A': item['essay_ids'][0], 'essay_B': item['essay_ids'][1],
                    'human_higher': item['gold'], 'human_gap': item['gap'],
                    'judge_winner': value['overall']['winner'],
                    'correct': value['overall']['winner'] == item['gold'],
                    'reason': value['overall']['reason'],
                    **{k: value[k]['winner'] for k in ('correction_accuracy', 'meaning', 'style')}})
        csv_file(directory / 'original_quality_results.csv', originals)
    if (directory / 'actual_semantics/results.json').exists():
        items = {r['blind_id']: r for r in read(directory / 'private/actual_semantics.json')}
        actual, labels, additions, by_model = [], Counter(), Counter(), defaultdict(Counter)
        for row in sorted(read(directory / 'actual_semantics/results.json'), key=lambda r: r['item_id']):
            item, value = items[row['item_id']], row['result']
            labels[value['label']] += 1
            additions[value['content_addition']['status']] += 1
            by_model[item['agent_model']][value['label']] += 1
            actual.append({'item_id': row['item_id'], 'before': item['before'], 'after': item['after'],
                'judge_model': row['model'], 'label': value['label'], 'reason': value['reason'],
                'content_addition': value['content_addition']['status'],
                'added_spans': json_text(value['content_addition']['added_spans']),
                'content_addition_reason': value['content_addition']['reason']})
        csv_file(directory / 'actual_semantic_labels.csv', actual)
        summary['actual_selected_sample'] = {'n': len(actual), 'labels': dict(labels),
            'content_additions': dict(additions), 'labels_by_source_model': dict(by_model),
            'is_overall_agent_preservation_estimate': False}
    if (directory / 'reference_agreement.json').exists():
        summary['claude_reference_auxiliary'] = read(directory / 'reference_agreement.json')
    write_json(directory / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json_text(report(args.data_dir.resolve())))
