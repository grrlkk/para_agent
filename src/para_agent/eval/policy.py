"""Explicit user-approved evaluation role and sampling policy."""
from pathlib import Path
import yaml


def load_evaluation_policy(path: Path):
    policy=yaml.safe_load(path.read_text(encoding='utf-8'))
    if policy['primary_metrics']!=['cross_family_pairwise_judge','meaning_preservation_rate']:
        raise ValueError('Primary evaluation metrics differ from the approved policy')
    k=policy['kanana']
    expected={'role':'auxiliary','method':'integer_mean','n':64,'temperature':1.0,
              'generation_batch_size':8,'master_seed':15042,'seed_scheme':'sha256_essay_id_v1',
              'record_greedy':True,'conditional_expectation_average':False}
    if k!=expected:
        raise ValueError('Kanana evaluation settings differ from the approved N64 policy')
    if not policy['decision']['post_result_selection']:
        raise ValueError('Must record post-result Kanana selection')
    judge=policy['judge']
    if judge['model_selection']=='user_fixed_sonnet_two_gates_else_stop':
        if (judge['model']!='claude-sonnet-5-5' or
            judge['required_validation_gates']!=['pairwise','controlled_semantic'] or
            not judge['model_choice_before_validation_results'] or
            not judge['no_human_labels_decision_before_results'] or
            judge['human_agreement']['completion_requires_actual_human_labels'] or
            judge['actual_revision_limitation']!='실제 수정에 대한 judge 판정은 사람으로 검증하지 않았다'):
            raise ValueError('Fixed Sonnet policy must record both gates and unvalidated human-revision limitation')
    elif judge['human_agreement']['required_in_stage']!=3:
        raise ValueError('Historical policy requires stage-3 human agreement validation')
    return policy
