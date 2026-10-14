"""Two predeclared gates for a user-fixed Sonnet judge; no human revision labels."""
import csv

from .agreement import BINARY, bootstrap_metric, kappa, pairwise_agreement, semantic_agreement

MODEL = 'claude-sonnet-5-5'
LIMITATION = '실제 수정에 대한 judge 판정은 사람으로 검증하지 않았다'


def validate_fixed_sonnet(controlled, pairs, rows, *, resamples=10000):
    if len(controlled) != 80 or len(pairs) != 20:
        raise ValueError('Fixed validation requires the frozen 80 controlled and 20 original pairs')
    index = {}
    for row in rows:
        key = (row['scope'], row['item_id'], row['reverse'])
        if key in index or row['model'] != MODEL:
            raise ValueError('Duplicate result or nonfixed judge model')
        index[key] = row['result']
    expected = {('controlled', r['item_id'], False) for r in controlled}
    expected |= {('original_pairs', r['item_id'], reverse) for r in pairs for reverse in (False, True)}
    if set(index) != expected:
        raise ValueError('Incomplete or extraneous validation judgments; do not drop failed items')
    semantic = semantic_agreement(
        [r['gold'] for r in controlled],
        [index['controlled', r['item_id'], False]['label'] for r in controlled],
        [r['cluster'] for r in controlled], resamples=resamples)
    truth = [r['gold'] for r in pairs for _ in (False, True)]
    predicted = [index['original_pairs', r['item_id'], reverse]['canonical']['overall']['winner']
                 for r in pairs for reverse in (False, True)]
    pairwise = pairwise_agreement(truth, predicted,
        [r['cluster'] for r in pairs for _ in (False, True)], resamples=resamples)
    pairwise.update(correct=sum(a == b for a, b in zip(truth, predicted)),
        both_orders_correct_rate=sum(predicted[2*i] == predicted[2*i+1] == r['gold']
                                     for i, r in enumerate(pairs)) / len(pairs),
        order_disagreement_rate=sum(predicted[2*i] != predicted[2*i+1] for i in range(len(pairs))) / len(pairs),
        limitation='Different-original human score ordering does not validate same-essay revision comparisons.')
    passed = semantic['passes'] and pairwise['passes']
    return {'models': {MODEL: {'controlled_semantic': semantic, 'pairwise': pairwise}},
            'selection': {'model': MODEL, 'status': 'selected' if passed else 'stop',
                          'rule': 'user_fixed_sonnet_two_gates_else_stop',
                          'model_choice_before_validation_results': True},
            'passes': passed, 'human_labels_status': 'not_used_by_pre_result_user_decision',
            'human_revision_agreement_calculated': False, 'limitation': LIMITATION,
            'threshold_policy': 'Original quality >=0.70; controlled binary kappa >=0.60; both required; CI reported separately.'}


def compare_claude_reference(actual, rows, path, *, resamples=10000):
    """Auxiliary same-family agreement, never a gate or human agreement."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reference = list(csv.DictReader(handle))
    ref = {r['item_id']: r for r in reference}
    items = {r['blind_id']: r for r in actual}
    index = {}
    for row in rows:
        if row['scope'] != 'actual_semantics' or row['reverse'] or row['model'] != MODEL:
            raise ValueError('Expected actual semantic judgments from the fixed Sonnet model')
        if row['item_id'] in index:
            raise ValueError('Duplicate actual semantic judgment')
        index[row['item_id']] = row['result']['label']
    if len(ref) != len(reference) or len(items) != 50 or set(ref) != set(items) or set(index) != set(items):
        raise ValueError('Reference and predictions must match all 50 frozen blind IDs exactly')
    for key, item in items.items():
        if any(ref[key][field] != item[field] for field in ('before', 'after')):
            raise ValueError('Reference sentence text differs from the frozen actual pair')
        if ref[key]['claude_label_2class'] != BINARY[ref[key]['claude_label']]:
            raise ValueError('Reference binary and three-class labels disagree')
    keys = sorted(items)
    truth = [BINARY[ref[key]['claude_label']] for key in keys]
    predicted = [BINARY[index[key]] for key in keys]
    groups = [items[key]['essay_id'] for key in keys]
    result = bootstrap_metric(truth, predicted, groups, kappa, resamples=resamples)
    result.update(role='auxiliary_same_family_reference_agreement_not_a_gate',
        reference_family='Claude', reference_model_version='not_recorded_in_reference_csv',
        human_revision_agreement_calculated=False, limitation=LIMITATION,
        binary_agreement=sum(a == b for a, b in zip(truth, predicted)) / len(keys),
        three_class_agreement=sum(ref[key]['claude_label'] == index[key] for key in keys) / len(keys),
        confusion={a: {b: sum(x == a and y == b for x, y in zip(truth, predicted))
                       for b in ('보존', '변화')} for a in ('보존', '변화')},
        binary_disagreement_ids=[key for key, a, b in zip(keys, truth, predicted) if a != b])
    return result
