#!/usr/bin/env python
"""Recover the saved two successful judgments after the progress-file write error."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.dont_write_bytecode = True
import yaml
from para_agent.artifacts import file_hash, write_json, json_text
from para_agent.config import ModelConfig, load_config, project_path
from para_agent.eval.fixed_validation import compare_claude_reference, LIMITATION
from para_agent.llm.anthropic_batch import BatchRunner, batch_cost, make_jobs, parse_judgment
from para_agent.llm.anthropic_transport import AnthropicTransport
from validate_sonnet_batch import status


def read(path):
    return json.loads(path.read_text())


def prepare(directory):
    target = directory / 'recovery_01.json'
    if target.exists():
        raise ValueError('Recovery is already frozen')
    setup = read(directory / 'setup.json')
    amendment = read(directory / 'execution_amendment_01.json')
    changed = {p for p, h in amendment['source_hashes'].items() if file_hash(PROJECT / p) != h}
    if changed != {'src/para_agent/llm/anthropic_batch.py', 'scripts/report_sonnet_validation.py'}:
        raise ValueError('Only progress persistence and report-source selection may change')
    for name, digest in setup['frozen_data_hashes'].items():
        assert file_hash(directory / 'private' / name) == digest
    assert file_hash(directory / 'decision_before_results.json') == setup['decision_sha256']
    assert file_hash(Path(setup['reference_path'])) == setup['reference_sha256']
    gate = read(directory / 'gate_report.json')
    assert gate['passes'] and gate['selection']['model'] == 'claude-sonnet-5-5'
    assert read(directory / 'status.json')['error_type'] == 'FileExistsError'
    validation = read(directory / 'validation/acceptance.json')
    prior = read(directory / 'actual_semantics/acceptance.json')
    assert validation['ok'] and validation['completed'] == 120
    assert not prior['ok'] and prior['completed'] == prior['real_api_calls'] == 2
    model = ModelConfig.model_validate(setup['model'])
    rates = yaml.safe_load((PROJECT / 'configs/api_pricing_2026-10-15.yaml').read_text())['models'][model.model]
    jobs = read(directory / 'private/jobs.json')['actual_semantics']
    assert jobs == make_jobs('actual_semantics', read(directory / 'private/actual_semantics.json'),
                             model, PROJECT / 'prompts', rates)
    requests = sorted((directory / 'actual_semantics/requests').iterdir())
    assert len(requests) == 2
    prefix, prior_cost = [], 0.0
    for job, folder in zip(jobs, requests):
        intent, message = read(folder / 'dispatch_intent.json'), read(folder / 'response.json')
        assert intent['params'] == {**job['params'], 'service_tier': 'standard_only'}
        assert intent['item_id'] == job['item_id'] and intent['attempt'] == 0
        assert message['model'] == model.model and message['stop_reason'] == 'end_turn'
        parsed = parse_judgment(job, ''.join(b['text'] for b in message['content'] if b['type'] == 'text'))
        record = {**{k: job[k] for k in ('scope', 'item_id', 'reverse')},
                  'result': parsed, 'model': model.model, 'attempt': 0}
        assert record == read(directory / 'actual_semantics/items' / f"actual_semantics_{job['item_id']}_0.json")
        prefix.append(record)
        prior_cost += 2 * batch_cost(message['usage'], rates)
    assert math.isclose(prior_cost, prior['estimated_billed_usd'])
    validation_cost = sum(batch_cost(r['result']['message']['usage'], rates)
                          for r in read(directory / 'validation/batches/001/raw_results.json'))
    assert math.isclose(validation_cost, validation['estimated_billed_usd'])
    sources = {p: file_hash(PROJECT / p) for p in amendment['source_hashes']}
    sources[str(Path(__file__).resolve().relative_to(PROJECT))] = file_hash(Path(__file__))
    artifacts = [directory / name for name in ('setup.json', 'execution_amendment_01.json',
        'decision_before_results.json', 'gate_report.json', 'private/jobs.json', 'validation/acceptance.json',
        'validation/results.json', 'validation/batches/001/raw_results.json')]
    artifacts += sorted(p for p in (directory / 'actual_semantics').rglob('*') if p.is_file())
    artifacts += sorted((directory / 'private').glob('*.json'))
    artifacts.append(Path(setup['reference_path']))
    plan = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'Repair progress-file persistence after observing FileExistsError; no judge retuning',
        'judge_model_prompts_inputs_and_gates_unchanged': True,
        'approved_total_budget_usd': 3.0, 'completed_prefix': prefix,
        'prior_actual_cost_usd': prior_cost, 'validation_cost_usd': validation_cost,
        'spent_before_recovery_usd': prior_cost + validation_cost, 'remaining_standard_requests': 48,
        'remaining_expected_usd_without_cache_hits': 2 * sum(j['expected_usd_without_hits'] for j in jobs[2:]),
        'source_hashes': sources, 'artifact_hashes': {str(p.resolve()): file_hash(p) for p in artifacts}}
    write_json(target, plan)
    return {k: v for k, v in plan.items() if k not in {'completed_prefix', 'source_hashes', 'artifact_hashes'}}


def execute(directory, budget):
    plan = read(directory / 'recovery_01.json')
    if budget != plan['approved_total_budget_usd'] or budget != 3.0:
        raise ValueError('Shared approved budget must remain $3.00')
    for p, digest in plan['source_hashes'].items():
        assert file_hash(PROJECT / p) == digest
    for p, digest in plan['artifact_hashes'].items():
        assert file_hash(Path(p)) == digest
    setup = read(directory / 'setup.json')
    model = ModelConfig.model_validate(setup['model'])
    rates = yaml.safe_load((PROJECT / 'configs/api_pricing_2026-10-15.yaml').read_text())['models'][model.model]
    cfg = load_config(PROJECT / 'configs/stage3_luna.yaml')
    transport = AnthropicTransport(model, project_path(cfg.llm.env_file))
    runner = BatchRunner(transport.client, rates, budget, directory, (PROJECT / 'prompts/repair.md').read_text())
    runner.spent = plan['spent_before_recovery_usd']
    runner.requests = 122
    interrupted = directory / 'actual_semantics_interrupted'
    if interrupted.exists():
        raise ValueError('Recovery already started; do not resubmit')
    (directory / 'actual_semantics').rename(interrupted)
    try:
        status(directory, 'actual_semantics_running', expected_judgments=50, reused_judgments=2,
               processing='standard', spent_usd=runner.spent)
        rows = runner.run_standard_phase('actual_semantics', read(directory / 'private/jobs.json')['actual_semantics'],
            {'actual_semantics': setup['frozen_data_hashes']['actual_semantics.json']},
            completed_prefix=plan['completed_prefix'], completed_cost_usd=plan['prior_actual_cost_usd'])
        reference = compare_claude_reference(read(directory / 'private/actual_semantics.json'), rows,
                                            Path(setup['reference_path']))
        write_json(directory / 'reference_agreement.json', reference)
        status(directory, 'completed', spent_usd=runner.spent, real_judge_requests=runner.requests,
               reused_judgments=2, recovered_from='progress_file_write_error',
               human_agreement_calculated=False, limitation=LIMITATION)
    except BaseException as exc:
        status(directory, 'failed', error_type=type(exc).__name__, spent_usd=runner.spent,
               accepted_judge_requests=runner.requests)
        raise
    finally:
        transport.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approved-budget-usd', type=float)
    args = parser.parse_args()
    print(json_text(execute(args.data_dir.resolve(), args.approved_budget_usd)
                    if args.execute else prepare(args.data_dir.resolve())))
