"""Budgeted Messages batches with explicit 1-hour system-prefix caching."""
from copy import deepcopy
import json
import time

from para_agent.artifacts import write_json, json_text
from para_agent.eval.judge import (SemanticJudgment, PairwiseJudgment, semantic_payload,
    pairwise_payload, canonical_pairwise, validate_pairwise_result, validate_addition)
from .anthropic_transport import AnthropicTransport


def batch_cost(usage, rates):
    writes = usage.get('cache_creation_input_tokens') or 0
    detail = usage.get('cache_creation') or {}
    # If TTL detail is omitted, assume every write has the more expensive 1h TTL.
    one_hour = detail.get('ephemeral_1h_input_tokens', writes)
    five_minute = writes - one_hour
    if not 0 <= one_hour <= writes:
        raise ValueError('Invalid cache creation usage')
    return .5 * (usage['input_tokens'] * rates['input'] + usage['output_tokens'] * rates['output']
        + (usage.get('cache_read_input_tokens') or 0) * rates['cached_input']
        + five_minute * rates['cache_write'] + one_hour * rates['cache_write_1h']) / 1e6


def request_reservation(params, rates):
    other = {k: v for k, v in params.items() if k != 'system'}
    # UTF-8 bytes plus protocol/schema allowance; reserve full output, no cache hits.
    fresh = len(json_text(other).encode()) + 4096
    prefix = len(json_text(params['system']).encode())
    return .5 * (fresh * rates['input'] + prefix * rates['cache_write_1h']
                 + params['max_tokens'] * rates['output']) / 1e6


def make_jobs(scope, items, model, prompts, rates):
    adapter = AnthropicTransport(model, prompts / '.unused', sdk_client=object())
    pair = scope == 'original_pairs'
    agent = 'judge_pairwise' if pair else 'judge_semantic'
    schema = PairwiseJudgment if pair else SemanticJudgment
    jobs = []
    for item in items:
        key = item['blind_id'] if scope == 'actual_semantics' else item['item_id']
        for reverse in ((False, True) if pair else (False,)):
            payload = pairwise_payload(item, reverse=reverse) if pair else semantic_payload(item)
            params = adapter.parameters(schema.model_json_schema(), agent)
            params.pop('service_tier', None)
            params.update(system=[{'type': 'text', 'text': (prompts / f'{agent}.md').read_text(),
                                   'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}],
                          messages=[{'role': 'user', 'content': json_text(payload)}])
            fresh = len(json_text({k: v for k, v in params.items() if k != 'system'}).encode()) / 2 + 1024
            prefix = len(json_text(params['system']).encode()) / 2
            expected = .5 * (fresh * rates['input'] + prefix * rates['cache_write_1h']
                             + (640 if pair else 256) * rates['output']) / 1e6
            jobs.append({'scope': scope, 'item_id': key, 'reverse': reverse, 'payload': payload,
                         'params': params, 'attempt': 0, 'expected_usd_without_hits': expected,
                         'reserved_max_usd': request_reservation(params, rates)})
    return jobs


def parse_judgment(job, raw):
    if job['scope'] == 'original_pairs':
        value = PairwiseJudgment.model_validate_json(raw)
        validate_pairwise_result(value, job['payload'])
        return canonical_pairwise(value, job['reverse'])
    value = SemanticJudgment.model_validate_json(raw)
    validate_addition(value.content_addition, job['payload']['after'])
    return value.model_dump()


class BatchRunner:
    """One serial budget ledger shared by validation and post-gate actual pairs."""
    def __init__(self, sdk, rates, budget, output, repair_prompt, *, poll_seconds=30, sleep=time.sleep):
        if budget <= 0:
            raise ValueError('Positive approved budget required')
        self.sdk, self.rates, self.budget, self.output = sdk, rates, budget, output
        self.repair_prompt, self.poll_seconds, self.sleep = repair_prompt, poll_seconds, sleep
        self.spent = 0.0
        self.requests = 0
        self.http_posts = 0

    def run_phase(self, name, jobs, source_hashes, *, resume_pending=False):
        directory = self.output / name
        if resume_pending:
            batch_dir = directory / 'batches' / '001'
            allowed = {'dispatch_intent.json', 'batch.json', 'latest_status.json'}
            if (not (batch_dir / 'batch.json').is_file()
                    or any(p.name not in allowed for p in batch_dir.iterdir())
                    or len(list((directory / 'batches').iterdir())) != 1
                    or any(p.name != 'batches' for p in directory.iterdir())):
                raise ValueError('Resume requires one accepted pending batch without consumed results')
        else:
            directory.mkdir(parents=True, exist_ok=False)
        pending, records, batch_index = deepcopy(jobs), [], 0
        phase_start, requests_start = self.spent, self.requests
        ok = False
        try:
            while pending:
                chosen, reserved = [], 0.0
                for job in pending:
                    charge = request_reservation(job['params'], self.rates)
                    if self.spent + reserved + charge > self.budget:
                        break
                    chosen.append(job); reserved += charge
                if not chosen:
                    raise RuntimeError('Approved budget cannot reserve the next request; no dispatch')
                pending = pending[len(chosen):]
                batch_index += 1
                batch_dir = directory / 'batches' / f'{batch_index:03d}'
                wire = [{'custom_id': f'j{i:04d}', 'params': job['params']} for i, job in enumerate(chosen)]
                if resume_pending and batch_index == 1:
                    intent = json.loads((batch_dir / 'dispatch_intent.json').read_text())
                    if intent['requests'] != wire or intent['spent_before_usd'] != self.spent:
                        raise ValueError('Pending batch does not match the frozen requests or budget ledger')
                    saved_batch = json.loads((batch_dir / 'batch.json').read_text())
                    batch = self.sdk.messages.batches.retrieve(saved_batch['id'])
                else:
                    write_json(batch_dir / 'dispatch_intent.json', {'requests': wire,
                        'reserved_usd': reserved, 'spent_before_usd': self.spent,
                        'jobs': [{k: j[k] for k in ('scope', 'item_id', 'reverse', 'attempt')} for j in chosen]})
                    # Never retry an uncertain create request, which might already be charged.
                    self.http_posts += 1
                    batch = self.sdk.messages.batches.create(requests=wire)
                    write_json(batch_dir / 'batch.json', batch.model_dump(mode='json'))
                self.requests += len(wire)
                while batch.processing_status != 'ended':
                    status = batch.model_dump(mode='json')
                    (batch_dir / 'latest_status.json').write_text(json_text(status) + '\n')
                    self.sleep(self.poll_seconds)
                    batch = self.sdk.messages.batches.retrieve(batch.id)
                raw_results = [r.model_dump(mode='json') for r in self.sdk.messages.batches.results(batch.id)]
                write_json(batch_dir / 'raw_results.json', raw_results)
                by_id = {r['custom_id']: r for r in raw_results}
                if len(by_id) != len(raw_results) or set(by_id) != {r['custom_id'] for r in wire}:
                    raise ValueError('Missing, duplicate, or unexpected batch results')
                # Account every billed response before any parse/transport failure can abort the phase.
                self.spent += sum(batch_cost(r['result']['message']['usage'], self.rates)
                                  for r in raw_results if r['result']['type'] == 'succeeded')
                write_json(batch_dir / 'accounting.json', {'spent_total_usd': self.spent,
                    'reserved_usd': reserved, 'submitted_requests': self.requests})
                if self.spent > self.budget:
                    raise RuntimeError('Reported usage exceeded reservation; stop all further dispatch')
                for i, job in enumerate(chosen):
                    outcome = by_id[f'j{i:04d}']['result']
                    if outcome['type'] != 'succeeded':
                        raise RuntimeError('Batch item failed; stop without silently omitting it')
                    message = outcome['message']
                    raw = ''.join(b['text'] for b in message['content'] if b['type'] == 'text')
                    if message['stop_reason'] != 'end_turn':
                        raise RuntimeError('Judge output incomplete; stop without score-based retry')
                    try:
                        parsed = parse_judgment(job, raw)
                    except ValueError as exc:
                        write_json(batch_dir / f'invalid_{i:04d}.json', {'error': str(exc),
                                   'item_id': job['item_id'], 'attempt': job['attempt']})
                        if job['attempt'] >= 2:
                            raise ValueError('Judge output failed three schema/evidence checks') from None
                        repair = deepcopy(job)
                        repair['attempt'] += 1
                        repair['params']['messages'] += [{'role': 'assistant', 'content': raw},
                            {'role': 'user', 'content': self.repair_prompt.replace('{error}', str(exc))}]
                        pending.append(repair)
                        continue
                    record = {k: job[k] for k in ('scope', 'item_id', 'reverse')}
                    record.update(result=parsed, model=message['model'], attempt=job['attempt'])
                    if message['model'] != job['params']['model']:
                        raise ValueError('Provider returned a different model than the fixed judge')
                    records.append(record)
                    write_json(directory / 'items' / f"{job['scope']}_{job['item_id']}_{int(job['reverse'])}.json", record)
            write_json(directory / 'results.json', records)
            ok = True
            return records
        finally:
            write_json(directory / 'acceptance.json', {'ok': ok, 'completed': len(records),
                'expected': len(jobs), 'model': 'claude-sonnet-5-5', 'source_hashes': source_hashes,
                'real_api_calls': self.requests - requests_start,
                'estimated_billed_usd': self.spent - phase_start, 'spent_total_usd': self.spent,
                'approved_budget_usd': self.budget, 'processing': 'batch', 'prompt_cache_ttl': '1h'})

    def run_standard_phase(self, name, jobs, source_hashes, *, completed_prefix=(), completed_cost_usd=0):
        """Synchronous Messages requests sharing the same total budget ledger."""
        if len(completed_prefix) > len(jobs) or not 0 <= completed_cost_usd <= self.spent:
            raise ValueError('Invalid completed-prefix ledger')
        for job, record in zip(jobs, completed_prefix):
            if (any(job[k] != record[k] for k in ('scope', 'item_id', 'reverse'))
                    or record['model'] != job['params']['model'] or record['attempt'] != 0):
                raise ValueError('Completed prefix must match successful initial requests in order')
        directory = self.output / name
        directory.mkdir(parents=True, exist_ok=False)
        records, ok = deepcopy(list(completed_prefix)), False
        phase_start, requests_start = self.spent - completed_cost_usd, self.requests - len(records)
        for record in records:
            write_json(directory / 'items' / f"{record['scope']}_{record['item_id']}_0.json", record)
        try:
            for original in jobs[len(records):]:
                job = deepcopy(original)
                for attempt in range(3):
                    params = {**job['params'], 'service_tier': 'standard_only'}
                    reserved = 2 * request_reservation(params, self.rates)
                    if self.spent + reserved > self.budget:
                        raise RuntimeError('Approved budget cannot reserve the next standard request; no dispatch')
                    request_dir = directory / 'requests' / f'{len(records):03d}_{attempt}'
                    write_json(request_dir / 'dispatch_intent.json', {
                        'params': params, 'item_id': job['item_id'], 'attempt': attempt,
                        'spent_before_usd': self.spent, 'reserved_usd': reserved})
                    self.requests += 1
                    self.http_posts += 1
                    response = self.sdk.messages.create(**params)
                    message = response.model_dump(mode='json')
                    write_json(request_dir / 'response.json', message)
                    self.spent += 2 * batch_cost(message['usage'], self.rates)
                    write_json(request_dir / 'accounting.json', {'spent_total_usd': self.spent})
                    if self.spent > self.budget:
                        raise RuntimeError('Reported usage exceeded reservation; stop all further dispatch')
                    if message['model'] != params['model'] or message['stop_reason'] != 'end_turn':
                        raise RuntimeError('Unexpected model or incomplete standard response')
                    raw = ''.join(b['text'] for b in message['content'] if b['type'] == 'text')
                    try:
                        parsed = parse_judgment(job, raw)
                    except ValueError as exc:
                        write_json(request_dir / 'invalid.json', {'error': str(exc)})
                        if attempt == 2:
                            raise ValueError('Judge output failed three schema/evidence checks') from None
                        job['params']['messages'] += [{'role': 'assistant', 'content': raw},
                            {'role': 'user', 'content': self.repair_prompt.replace('{error}', str(exc))}]
                        continue
                    record = {k: job[k] for k in ('scope', 'item_id', 'reverse')}
                    record.update(result=parsed, model=message['model'], attempt=attempt)
                    records.append(record)
                    write_json(directory / 'items' / f"{job['scope']}_{job['item_id']}_0.json", record)
                    progress = directory / 'progress.tmp'
                    progress.write_text(json_text({'completed': len(records), 'expected': len(jobs),
                        'spent_total_usd': self.spent, 'processing': 'standard'}) + '\n')
                    progress.replace(directory / 'progress.json')
                    break
            write_json(directory / 'results.json', records)
            ok = True
            return records
        finally:
            write_json(directory / 'acceptance.json', {'ok': ok, 'completed': len(records),
                'expected': len(jobs), 'model': 'claude-sonnet-5-5', 'source_hashes': source_hashes,
                'real_api_calls': self.requests - requests_start,
                'estimated_billed_usd': self.spent - phase_start, 'spent_total_usd': self.spent,
                'approved_budget_usd': self.budget, 'processing': 'standard', 'prompt_cache_ttl': '1h'})
