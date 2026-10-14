#!/usr/bin/env python
"""Join the Luna baseline's saved metrics; incomplete judge gates stay pending."""
import argparse
import csv
import json
from pathlib import Path
import sys

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.dont_write_bytecode = True

from para_agent.artifacts import file_hash, write_json
from para_agent.eval.costs import token_cost
from para_agent.eval.stage3_metrics import meaning_preservation
from validate_stage3_judges import load_results


def build_report(data_dir, *, kanana=None, selection=None, judge_results=()):
    local = json.loads((data_dir / 'luna_local_metrics.json').read_text())
    source = Path(local['source_run'])
    acceptance = json.loads((source / 'acceptance.json').read_text())
    if not acceptance['ok'] or acceptance['mock']:
        raise ValueError('Need a completed real Luna baseline')
    calls = [json.loads(line) for line in (source / 'llm_calls.jsonl').read_text().splitlines()]
    rates = yaml.safe_load((PROJECT / 'configs/api_pricing_2026-10-15.yaml').read_text())['models']['gpt-6-luna']
    report = {'source_run': str(source), 'agent_model': 'gpt-6-luna', 'baseline_contains_haiku': False,
              'conditions': local['conditions'], 'primary_evaluation_complete': False,
              'judge_status': 'pending_validation_and_api_approval',
              'agent_api_calls': acceptance['real_api_calls'],
              'agent_estimated_billed_usd': acceptance['estimated_billed_usd'],
              'approved_agent_budget_usd': acceptance['approved_budget_usd'],
              'source_hashes': {'local_metrics': file_hash(data_dir / 'luna_local_metrics.json'),
                                'agent_acceptance': file_hash(source / 'acceptance.json')}}
    for condition, metrics in report['conditions'].items():
        selected_calls = [c for c in calls if c['condition'] == condition and c.get('real_api_call')]
        metrics['estimated_api_usd'] = sum(token_cost(c['tokens'], rates) for c in selected_calls)
        metrics['api_calls'] = len(selected_calls)
        metrics['meaning_preservation'] = {'status': 'pending_validated_judge'}
    if kanana:
        score_report = json.loads((kanana / 'acceptance.json').read_text())
        provenance = yaml.safe_load((kanana / 'config.yaml').read_text())['provenance']
        if not score_report['ok'] or Path(provenance['source_run']).resolve() != source.resolve():
            raise ValueError('Kanana evaluation must be complete and belong to this Luna run')
        with (kanana / 'results.csv').open(newline='') as f:
            scores = list(csv.DictReader(f))
        for condition, metrics in report['conditions'].items():
            rows = [r for r in scores if r['condition'] == condition]
            if len(rows) != metrics['essays']:
                raise ValueError('Incomplete Kanana rows')
            metrics['kanana'] = {key: sum(float(r[key]) for r in rows) / len(rows)
                                for key in ('total', 'delta_total', 'greedy_total', 'greedy_delta_total')}
            metrics['kanana'].update(role='auxiliary', integer_mean_n=64)
        report['kanana_dir'] = str(kanana)
        report['source_hashes']['kanana_csv'] = file_hash(kanana / 'results.csv')
    if judge_results:
        if selection is None:
            raise ValueError('Judge results require a passing validation report')
        validation = json.loads(selection.read_text())
        decision = validation['selection']
        if decision['status'] != 'selected':
            raise ValueError('Judge has not passed required validation gates')
        index = load_results(judge_results, decision['model'], data_dir)
        changes = json.loads((data_dir / 'private/luna_semantics.json').read_text())
        pairs = json.loads((data_dir / 'private/luna_pairs.json').read_text())
        if not all(('luna_semantics', r['item_id'], False) in index for r in changes):
            raise ValueError('Need all changed Luna pairs, not the enriched human validation subset')
        if not all(('luna_pairs', r['item_id'], rev) in index for r in pairs for rev in (False, True)):
            raise ValueError('Need both display orders for each Luna essay')
        for condition, metrics in report['conditions'].items():
            labels = [index['luna_semantics', r['item_id'], False]['label']
                      for r in changes if r['condition'] == condition]
            metrics['meaning_preservation'] = meaning_preservation(labels)
        winners = [index['luna_pairs', r['item_id'], rev]['canonical']['overall']['winner']
                   for r in pairs for rev in (False, True)]
        report['pairwise'] = {'ordered_judgments': len(winners),
            'whole_once_wins': winners.count('A'), 'incremental_wins': winners.count('B'),
            'ties': winners.count('동률'),
            'order_disagreement_rate': sum(winners[2*i] != winners[2*i+1] for i in range(len(pairs))) / len(pairs)}
        report['judge_status'] = 'validated_and_evaluated'
        report['primary_evaluation_complete'] = True
        report['judge_model'] = decision['model']
        report['human_revision_validation'] = validation.get('human_labels_status', 'unknown')
        if validation.get('limitation'):
            report['judge_validation_limitation'] = validation['limitation']
        report['source_hashes']['selection'] = file_hash(selection)
        report['baseline_judge_estimated_usd'] = sum(
            json.loads((p / 'acceptance.json').read_text())['estimated_billed_usd'] for p in judge_results)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--kanana', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--judge-results', type=Path, nargs='*', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_report(args.data_dir, kanana=args.kanana, selection=args.selection,
                          judge_results=args.judge_results)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
