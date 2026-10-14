import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from para_agent.config import PROJECT_ROOT
from para_agent.data.loader import Essay
from para_agent.diagnose.smoothed_scorer import essay_seed, summarize_samples
from para_agent.eval.final_score import score_essays
from para_agent.eval.kanana import AuxiliaryKananaScorer
from para_agent.eval.policy import load_evaluation_policy


def test_approved_policy_records_post_result_kanana_and_pre_result_fixed_judge():
    policy=load_evaluation_policy(PROJECT_ROOT/'configs/evaluation_policy.yaml')
    assert policy['decision']['original_n32_primary_adoption_rule']=='failed'
    assert policy['decision']['post_result_selection'] is True
    assert policy['decision']['confirmatory_experiment']=='not_requested_and_not_run'
    assert policy['judge']['model']=='claude-sonnet-5-5'
    assert policy['judge']['model_choice_before_validation_results']
    assert policy['judge']['no_human_labels_decision_before_results']
    assert not policy['judge']['human_agreement']['completion_requires_actual_human_labels']
    assert policy['judge']['required_validation_gates']==['pairwise','controlled_semantic']
    assert '사람으로 검증하지 않았다' in policy['judge']['actual_revision_limitation']
    historical=yaml.safe_load((PROJECT_ROOT/'configs/smoothed_scoring.yaml').read_text())
    assert historical['sampling']['primary_n']==32


def test_auxiliary_n64_and_greedy_recorded_with_same_id_seed_for_all_conditions(tmp_path,contracts):
    keys=contracts['rubric_keys']
    policy=load_evaluation_policy(PROJECT_ROOT/'configs/evaluation_policy.yaml')
    calls=[]
    class Native:
        backend=SimpleNamespace(fingerprint='fake')
        def score(self,text,prompt,*,use_cache):
            assert use_cache is False
            return dict.fromkeys(keys,4 if text=='원문' else 5),None
        def _ensure_backend(self): return self.backend
        def close(self): pass
    class Sample:
        def __init__(self,native,**kwargs):
            assert kwargs==dict(batch_size=8,expected_batch_size=1,temperature=1.0,include_expected=False)
        def score(self,prompt,text,*,n,seed):
            calls.append((prompt,text,n,seed))
            samples=[{'integers':[5 if text=='원문' else 6]*8} for _ in range(n)]
            return {'seed':seed,'n':n,'samples':samples,'summary':summarize_samples(samples,methods=('integer_mean',))}
    scorer=AuxiliaryKananaScorer(Native(),keys,policy,sampler_factory=Sample)
    essay=Essay(essay_id='original-id',prompt='문항',paragraphs=['원문'],original_scores={})
    finals={(c,essay.essay_id):{'final_text':c,'Y':[c]} for c in ('whole_once','incremental')}
    rows=score_essays([essay],finals,scorer,keys,tmp_path)
    expected=int.from_bytes(hashlib.sha256(b'15042:original-id').digest()[:4],'big')%(2**31-1)
    assert expected==essay_seed(essay.essay_id)
    assert [c[2:] for c in calls]==[(64,expected)]*3
    assert [r['total'] for r in rows]==[40,48,48]
    assert [r['greedy_total'] for r in rows]==[32,40,40]
    assert [r['delta_total'] for r in rows]==[0,8,8]
    assert [r['greedy_delta_total'] for r in rows]==[0,8,8]
    assert all(r['metric_role']=='auxiliary' for r in rows)
    saved=json.loads((tmp_path/'incremental/original-id/score.json').read_text())
    assert len(saved['smoothed_sampling']['samples'])==64
    assert set(saved['smoothed_sampling']['summary']['64'])=={'integer_mean'}
    for condition in reversed(['original','whole_once','incremental']):
        scorer.score_for_essay(condition,'문항',essay_id=essay.essay_id)
    assert all(c[3]==expected for c in calls)


def test_policy_rejects_unapproved_silent_metric_or_sampling_change(tmp_path):
    policy=load_evaluation_policy(PROJECT_ROOT/'configs/evaluation_policy.yaml')
    policy['kanana']['n']=32
    path=tmp_path/'policy.yaml';path.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError,match='approved N64 policy'):
        load_evaluation_policy(path)


def test_integer_only_sampler_never_runs_conditional_expectation_forward(monkeypatch):
    # A CPU tensor stub exercises the real sample loop without a model or CUDA.
    import torch
    from para_agent.diagnose.smoothed_scorer import SmoothedKananaAdapter
    original_fork = torch.random.fork_rng
    monkeypatch.setattr(torch.random, 'fork_rng', lambda devices: original_fork(devices=[]))
    generate_calls=[]
    class Tokenizer:
        pad_token_id=0
        eos_token_id=1
        def decode(self,tokens,**kwargs): return '5 5 5 5 5 5 5 5\n'
        def encode(self,text,**kwargs): return [5]*8
    class Model:
        device=torch.device('cpu')
        def generate(self,**kwargs):
            generate_calls.append(kwargs)
            batch=kwargs['input_ids']
            return torch.cat([batch, torch.full((len(batch),8),5)],dim=1)
    class Native:
        tokenizer=Tokenizer()
        model=Model()
        max_new_tokens=32
        def prepare_input(self,prompt,text):
            return {'input_ids':torch.ones((1,3),dtype=torch.long)}
    sampler=SmoothedKananaAdapter.__new__(SmoothedKananaAdapter)
    sampler.native=Native()
    sampler.batch_size=8
    sampler.expected_batch_size=1
    sampler.temperature=1.0
    sampler.max_attempt_multiplier=4
    sampler.include_expected=False
    sampler.module=SimpleNamespace(parse_first_line=lambda line:[int(x) for x in line.split()],
        teacher_positions=lambda length,ids:list(range(8)), ScoreParseError=RuntimeError)
    def forbidden(*args,**kwargs):
        raise AssertionError('Integer-mean evaluation must not calculate conditional expectations')
    sampler.expected_for_lines=forbidden
    result=sampler.score('문항','원문',n=64,seed=essay_seed('test'))
    assert len(generate_calls)==8 and result['attempted']==64
    assert not result['conditional_expectations_computed']
    assert result['summary']['64']['integer_mean']['total']==40
    assert all('expected' not in s for s in result['samples'])

