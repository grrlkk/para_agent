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
    if not policy['decision']['post_result_selection'] or policy['judge']['human_agreement']['required_in_stage']!=3:
        raise ValueError('Must record post-result selection and required stage-3 human agreement validation')
    return policy
