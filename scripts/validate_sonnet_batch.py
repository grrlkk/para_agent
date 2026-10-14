#!/usr/bin/env python
"""Freeze/cost, then run two Sonnet gates and conditional 50-pair reference analysis."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.dont_write_bytecode = True

import yaml
from para_agent.artifacts import file_hash, write_json, json_text
from para_agent.config import ModelConfig, load_config, project_path
from para_agent.eval.fixed_validation import validate_fixed_sonnet, compare_claude_reference, LIMITATION
from para_agent.llm.anthropic_batch import BatchRunner, make_jobs
from para_agent.llm.anthropic_transport import AnthropicTransport


def status(directory, state, **fields):
    value = {'state': state, 'time_utc': datetime.now(timezone.utc).isoformat(), **fields}
    tmp = directory / 'status.tmp'
    tmp.write_text(json_text(value) + '\n')
    tmp.replace(directory / 'status.json')


def prepare(directory):
    if (directory / 'setup.json').exists():
        raise ValueError('Preparation is already frozen; use the existing setup rather than overwriting it')
    decision = json.loads((directory / 'decision_before_results.json').read_text())
    assert decision['fixed_judge'] == 'claude-sonnet-5-5'
    assert decision['model_choice_before_validation_results'] and decision['no_human_labels_decision_before_results']
    model = ModelConfig.model_validate(yaml.safe_load((PROJECT / 'configs/judge_sonnet55.yaml').read_text()))
    rates = yaml.safe_load((PROJECT / 'configs/api_pricing_2026-10-15.yaml').read_text())['models'][model.model]
    protocol = yaml.safe_load((PROJECT / 'configs/stage3_protocol.yaml').read_text())
    if protocol['judge_selection'] != 'user_fixed_sonnet_two_gates_else_stop':
        raise ValueError('Expected the approved two-gate fixed-model protocol')
    jobs, hashes = {}, {}
    for scope, expected in (('controlled', 80), ('original_pairs', 20), ('actual_semantics', 50)):
        source = directory / 'private' / f'{scope}.json'
        rows = json.loads(source.read_text())
        if len(rows) != expected:
            raise ValueError('Frozen sample size differs from the predeclared plan')
        hashes[source.name] = file_hash(source)
        jobs[scope] = make_jobs(scope, rows, model, PROJECT / 'prompts', rates)
    reference = Path(decision['existing_reference_path'])
    if file_hash(reference) != decision['existing_reference_sha256']:
        raise ValueError('Reference labels changed after the decision record')
    source_files = [Path(__file__), PROJECT / 'configs/stage3_protocol.yaml',
        PROJECT / 'configs/evaluation_policy.yaml', PROJECT / 'configs/judge_sonnet55.yaml',
        PROJECT / 'configs/api_pricing_2026-10-15.yaml', PROJECT / 'prompts/judge_semantic.md',
        PROJECT / 'prompts/judge_pairwise.md', PROJECT / 'prompts/repair.md',
        *sorted((PROJECT / 'src').rglob('*.py'))]
    write_json(directory / 'setup.json', {'protocol': protocol, 'model': model.model_dump(),
        'source_hashes': {str(p.relative_to(PROJECT)): file_hash(p) for p in source_files},
        'frozen_data_hashes': hashes, 'reference_path': str(reference), 'reference_sha256': file_hash(reference),
        'decision_sha256': file_hash(directory / 'decision_before_results.json'),
        'prepared_utc': datetime.now(timezone.utc).isoformat()})
    write_json(directory / 'private/jobs.json', jobs)
    plan = {'model': model.model, 'model_config': model.model_dump(), 'processing': 'batch',
        'prompt_caching': {'ttl': '1h', 'breakpoint': 'shared_system_prompt', 'hits_assumed': False},
        'validation_calls': 120, 'conditional_actual_calls': 50, 'maximum_calls_with_repairs': 510,
        'approved_budget_usd': decision['approved_budget_usd'], 'phases': {},
        'human_revision_labels_used': False, 'reference_comparison_is_a_gate': False,
        'external_api_calls_during_preparation': 0,
        'estimate_method': 'UTF-8 bytes/2 plus 1024 input overhead, no cache hits, 1h prefix writes, 256/640 output tokens; Batch discount 0.5',
        'reservation_method': 'UTF-8 bytes plus 4096 overhead, full max_tokens, no cache hits; partition batches if needed to reserve within remaining total budget',
        'provider_timing': 'Most batches within 1h; each batch may take up to 24h; two dependent phases plus budget partitions and repairs.'}
    for name, group in [('validation', jobs['controlled'] + jobs['original_pairs']),
                        ('actual_semantics', jobs['actual_semantics'])]:
        plan['phases'][name] = {'logical_calls': len(group),
            'expected_usd_without_cache_hits': sum(j['expected_usd_without_hits'] for j in group),
            'all_initial_requests_reserved_max_usd': sum(j['reserved_max_usd'] for j in group)}
    plan['expected_total_usd_if_gates_pass'] = sum(p['expected_usd_without_cache_hits'] for p in plan['phases'].values())
    write_json(directory / 'plan.json', plan)
    status(directory, 'prepared_waiting_dispatch', actual_api_calls=0)
    return plan


def execute(directory, approved_budget):
    setup = json.loads((directory / 'setup.json').read_text())
    decision = json.loads((directory / 'decision_before_results.json').read_text())
    if approved_budget != decision['approved_budget_usd'] or approved_budget != 3.0:
        raise ValueError('This execution is authorized for a shared $3.00 budget only')
    for name, digest in setup['source_hashes'].items():
        if file_hash(PROJECT / name) != digest:
            raise ValueError('Frozen source changed before dispatch')
    for name, digest in setup['frozen_data_hashes'].items():
        if file_hash(directory / 'private' / name) != digest:
            raise ValueError('Frozen data changed before dispatch')
    if file_hash(directory / 'decision_before_results.json') != setup['decision_sha256']:
        raise ValueError('Pre-result decision record changed')
    if file_hash(Path(setup['reference_path'])) != setup['reference_sha256']:
        raise ValueError('Reference labels changed before dispatch')
    model = ModelConfig.model_validate(setup['model'])
    rates = yaml.safe_load((PROJECT / 'configs/api_pricing_2026-10-15.yaml').read_text())['models'][model.model]
    jobs = json.loads((directory / 'private/jobs.json').read_text())
    if set(jobs) != {'controlled', 'original_pairs', 'actual_semantics'}:
        raise ValueError('Prepared request scopes differ from the approved plan')
    # Verify regenerated requests, so edited cached jobs cannot bypass the blind-input boundary.
    for scope, saved in jobs.items():
        rows = json.loads((directory / 'private' / f'{scope}.json').read_text())
        if saved != make_jobs(scope, rows, model, PROJECT / 'prompts', rates):
            raise ValueError('Prepared request payloads changed')
    cfg = load_config(PROJECT / 'configs/stage3_luna.yaml')
    transport = AnthropicTransport(model, project_path(cfg.llm.env_file))
    runner = BatchRunner(transport.client, rates, approved_budget, directory,
                         (PROJECT / 'prompts/repair.md').read_text())
    try:
        status(directory, 'validation_running', expected_judgments=120)
        records = runner.run_phase('validation', jobs['controlled'] + jobs['original_pairs'],
            {s: setup['frozen_data_hashes'][f'{s}.json'] for s in ('controlled', 'original_pairs')})
        report = validate_fixed_sonnet(json.loads((directory / 'private/controlled.json').read_text()),
            json.loads((directory / 'private/original_pairs.json').read_text()), records)
        write_json(directory / 'gate_report.json', report)
        if not report['passes']:
            status(directory, 'stopped_failed_gate', spent_usd=runner.spent,
                   real_judge_requests=runner.requests, actual_semantics='not_run_by_stop_rule')
            return report
        status(directory, 'actual_semantics_running', expected_judgments=50, spent_usd=runner.spent)
        actual = runner.run_phase('actual_semantics', jobs['actual_semantics'],
            {'actual_semantics': setup['frozen_data_hashes']['actual_semantics.json']})
        reference = compare_claude_reference(json.loads((directory / 'private/actual_semantics.json').read_text()),
                                            actual, Path(setup['reference_path']))
        write_json(directory / 'reference_agreement.json', reference)
        status(directory, 'completed', spent_usd=runner.spent, real_judge_requests=runner.requests,
               human_agreement_calculated=False, limitation=LIMITATION)
        return report
    except BaseException as exc:
        status(directory, 'failed', error_type=type(exc).__name__, spent_usd=runner.spent,
               accepted_judge_requests=runner.requests, batch_create_attempts=runner.http_posts)
        raise
    finally:
        transport.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approved-budget-usd', type=float)
    args = parser.parse_args(argv)
    result = execute(args.data_dir.resolve(), args.approved_budget_usd) if args.execute else prepare(args.data_dir.resolve())
    print(json_text(result))


if __name__ == '__main__':
    main()
