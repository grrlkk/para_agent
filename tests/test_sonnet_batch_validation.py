import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from para_agent.config import PROJECT_ROOT, ModelConfig
from para_agent.artifacts import file_hash, write_json
from para_agent.eval.fixed_validation import validate_fixed_sonnet, compare_claude_reference, MODEL
from para_agent.llm.anthropic_batch import BatchRunner, batch_cost, make_jobs, request_reservation


@pytest.fixture
def rates():
    return yaml.safe_load((PROJECT_ROOT/'configs/api_pricing_2026-10-15.yaml').read_text())['models'][MODEL]


def semantic_item(index=0):
    return {'item_id': f'c{index}', 'blind_id': f'M{index+1:03d}', 'essay_id': f'e{index%5}',
            'prompt': '문항', 'before': '원문이다.', 'after': '수정문이다.',
            'context_before': '원문이다.', 'context_after': '수정문이다.', 'gold': '의미 변화',
            'condition': 'hidden-condition', 'agent_model': 'hidden-agent'}


def model():
    return ModelConfig.model_validate(yaml.safe_load((PROJECT_ROOT/'configs/judge_sonnet55.yaml').read_text()))


def response():
    return {'label': '보존', 'reason': '의미 유지', 'content_addition': {
        'status': '없음', 'added_spans': [], 'reason': '추가 없음'}}


def test_batch_prices_distinguish_one_hour_writes_and_stack_discount(rates):
    usage={'input_tokens':1000,'output_tokens':100,'cache_read_input_tokens':500,
           'cache_creation_input_tokens':200,'cache_creation':{'ephemeral_1h_input_tokens':150}}
    assert batch_cost(usage,rates)==pytest.approx(.5*(.002+.001+.00005+.0006+.000125))
    del usage['cache_creation']
    assert batch_cost(usage,rates)==pytest.approx(.5*(.002+.001+.00005+.0008))


def test_prepared_batch_is_blind_and_caches_only_shared_system_prompt(rates):
    jobs=make_jobs('controlled',[semantic_item()],model(),PROJECT_ROOT/'prompts',rates)
    params=jobs[0]['params']
    payload=json.loads(params['messages'][0]['content'])
    assert set(payload)=={'prompt','before','after','context_before','context_after'}
    assert params['system'][0]['cache_control']=={'type':'ephemeral','ttl':'1h'}
    assert params['thinking']=={'type':'between_tools'}
    assert 'temperature' not in params and 'tools' not in params
    assert 'hidden-agent' not in json.dumps(params)
    assert request_reservation(params,rates)>jobs[0]['expected_usd_without_hits']


class Dump:
    def __init__(self,data): self.data=data
    def model_dump(self,**kwargs): return self.data


class FakeBatches:
    def __init__(self): self.submissions=[]
    def create(self,*,requests):
        self.submissions.append(requests)
        value=Dump({'id':str(len(self.submissions)),'processing_status':'ended'})
        value.id=str(len(self.submissions));value.processing_status='ended'
        return value
    def results(self,batch_id):
        for request in reversed(self.submissions[int(batch_id)-1]):
            yield Dump({'custom_id':request['custom_id'],'result':{'type':'succeeded','message':{
                'model':MODEL,'stop_reason':'end_turn','content':[{'type':'text','text':json.dumps(response())}],
                'usage':{'input_tokens':100,'output_tokens':50}}}})

    def retrieve(self,batch_id):
        value=Dump({'id':batch_id,'processing_status':'ended'})
        value.id=batch_id;value.processing_status='ended'
        return value


def test_runner_reserves_before_dispatch_and_handles_unordered_results(tmp_path,rates):
    jobs=make_jobs('controlled',[semantic_item(i) for i in range(3)],model(),PROJECT_ROOT/'prompts',rates)
    batches=FakeBatches();sdk=SimpleNamespace(messages=SimpleNamespace(batches=batches))
    too_small=BatchRunner(sdk,rates,1e-8,tmp_path/'small','repair')
    with pytest.raises(RuntimeError,match='cannot reserve'):
        too_small.run_phase('gate',jobs,{})
    assert batches.submissions==[]
    budget=sum(j['reserved_max_usd'] for j in jobs[:2])+1e-6
    runner=BatchRunner(sdk,rates,budget,tmp_path/'normal','repair')
    rows=runner.run_phase('gate',jobs,{})
    assert [len(b) for b in batches.submissions]==[2,1]
    assert {r['item_id'] for r in rows}=={'c0','c1','c2'}
    assert runner.requests==3 and runner.spent==pytest.approx(3*.00035)
    assert json.loads((tmp_path/'normal/gate/acceptance.json').read_text())['ok']


def gate_fixture(correct_pairs=28):
    controlled=[{'item_id':f'c{i}','gold':'보존' if i<40 else '의미 변화','cluster':f'family{i%8}'} for i in range(80)]
    pairs=[{'item_id':f'p{i}','gold':'A','cluster':f'prompt{i}'} for i in range(20)]
    rows=[{'scope':'controlled','item_id':r['item_id'],'reverse':False,'model':MODEL,
           'result':{'label':r['gold']}} for r in controlled]
    for i,r in enumerate(pairs):
        for order in (False,True):
            winner='A' if 2*i+int(order)<correct_pairs else '동률'
            rows.append({'scope':'original_pairs','item_id':r['item_id'],'reverse':order,'model':MODEL,
                         'result':{'canonical':{'overall':{'winner':winner}}}})
    return controlled,pairs,rows


def test_two_gate_thresholds_need_no_human_labels_and_either_failure_stops():
    c,p,rows=gate_fixture()
    result=validate_fixed_sonnet(c,p,rows,resamples=100)
    assert result['passes'] and result['selection']['model']==MODEL
    assert result['models'][MODEL]['pairwise']['estimate']==.7
    assert not result['human_revision_agreement_calculated']
    c,p,rows=gate_fixture(27)
    assert validate_fixed_sonnet(c,p,rows,resamples=100)['selection']['status']=='stop'
    c,p,rows=gate_fixture(40)
    for r in rows[:80]:r['result']['label']='의미 변화'
    assert not validate_fixed_sonnet(c,p,rows,resamples=100)['passes']
    with pytest.raises(ValueError,match='Incomplete'):
        validate_fixed_sonnet(c,p,rows[:-1],resamples=10)


def test_claude_reference_is_checked_for_text_identity_and_never_treated_as_human_gate(tmp_path):
    items=[semantic_item(i) for i in range(50)]
    rows=[{'scope':'actual_semantics','item_id':r['blind_id'],'reverse':False,'model':MODEL,
           'result':{'label':'보존' if i<25 else '의미 변화'}} for i,r in enumerate(items)]
    path=tmp_path/'reference.csv'
    reference=[{'item_id':r['blind_id'],'before':r['before'],'after':r['after'],
                'claude_label':rows[i]['result']['label'], 'claude_label_2class':'보존' if i<25 else '변화'}
               for i,r in enumerate(items)]
    def save():
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(reference[0]));writer.writeheader();writer.writerows(reference)
    save()
    result=compare_claude_reference(items,rows,path,resamples=100)
    assert result['estimate']==1 and result['clusters']==5 and result['binary_agreement']==1
    assert 'passes' not in result and 'threshold' not in result
    assert not result['human_revision_agreement_calculated']
    reference[0]['before']='다른 문장';save()
    with pytest.raises(ValueError,match='sentence text differs'):
        compare_claude_reference(items,rows,path,resamples=100)


@pytest.mark.parametrize('passed', [False, True])
def test_orchestrator_only_submits_actual_pairs_after_both_gates_pass(tmp_path,monkeypatch,passed):
    spec=importlib.util.spec_from_file_location('sonnet_script',PROJECT_ROOT/'scripts/validate_sonnet_batch.py')
    script=importlib.util.module_from_spec(spec);spec.loader.exec_module(script)
    ref=tmp_path/'reference.csv';ref.write_text('reference placeholder')
    write_json(tmp_path/'decision_before_results.json',{'fixed_judge':MODEL,
        'model_choice_before_validation_results':True,'no_human_labels_decision_before_results':True,
        'existing_reference_path':str(ref),'existing_reference_sha256':file_hash(ref),'approved_budget_usd':3.0})
    write_json(tmp_path/'private/controlled.json',[semantic_item(i) for i in range(80)])
    write_json(tmp_path/'private/actual_semantics.json',[semantic_item(i) for i in range(50)])
    write_json(tmp_path/'private/original_pairs.json',[{'item_id':f'p{i}','prompt':'문항','A':'글 A','B':'글 B'} for i in range(20)])
    plan=script.prepare(tmp_path)
    assert plan['validation_calls']==120 and plan['conditional_actual_calls']==50
    with pytest.raises(ValueError,match='already frozen'):
        script.prepare(tmp_path)
    phases=[]
    class Runner:
        spent=.4;requests=120;http_posts=1
        def __init__(self,*args,**kwargs): pass
        def run_phase(self,name,jobs,hashes,**kwargs): phases.append(name);return []
        def run_standard_phase(self,name,jobs,hashes): phases.append(name);return []
    monkeypatch.setattr(script,'BatchRunner',Runner)
    monkeypatch.setattr(script,'AnthropicTransport',lambda *a,**k:SimpleNamespace(client=object(),close=lambda:None))
    monkeypatch.setattr(script,'validate_fixed_sonnet',lambda *a,**k:{'passes':passed})
    monkeypatch.setattr(script,'compare_claude_reference',lambda *a,**k:{'role':'auxiliary'})
    script.execute(tmp_path,3.0)
    assert phases==(['validation','actual_semantics'] if passed else ['validation'])
    assert json.loads((tmp_path/'status.json').read_text())['state']==('completed' if passed else 'stopped_failed_gate')


def test_budget_is_shared_across_phases(tmp_path,rates):
    jobs=make_jobs('controlled',[semantic_item()],model(),PROJECT_ROOT/'prompts',rates)
    batches=FakeBatches();sdk=SimpleNamespace(messages=SimpleNamespace(batches=batches))
    runner=BatchRunner(sdk,rates,jobs[0]['reserved_max_usd']+1e-6,tmp_path,'repair')
    runner.run_phase('validation',jobs,{})
    with pytest.raises(RuntimeError,match='cannot reserve'):
        runner.run_phase('actual',jobs,{})
    assert len(batches.submissions)==1 and runner.requests==1


def test_resume_retrieves_accepted_batch_without_second_submission(tmp_path,rates):
    jobs=make_jobs('controlled',[semantic_item()],model(),PROJECT_ROOT/'prompts',rates)
    wire=[{'custom_id':'j0000','params':jobs[0]['params']}]
    batches=FakeBatches();accepted=batches.create(requests=wire)
    folder=tmp_path/'validation/batches/001'
    write_json(folder/'batch.json',accepted.model_dump())
    write_json(folder/'dispatch_intent.json',{'requests':wire,'spent_before_usd':0})
    runner=BatchRunner(SimpleNamespace(messages=SimpleNamespace(batches=batches)),rates,3,tmp_path,'repair')
    rows=runner.run_phase('validation',jobs,{},resume_pending=True)
    assert len(batches.submissions)==1 and runner.http_posts==0
    assert runner.requests==1 and len(rows)==1 and runner.spent==pytest.approx(.00035)


def test_standard_phase_never_uses_batch_and_shares_full_price_budget(tmp_path,rates):
    class Messages:
        def __init__(self): self.calls=[]
        def create(self,**params):
            self.calls.append(params)
            return Dump({'model':MODEL,'stop_reason':'end_turn',
                'content':[{'type':'text','text':json.dumps(response())}],
                'usage':{'input_tokens':100,'output_tokens':50}})
    messages=Messages();runner=BatchRunner(SimpleNamespace(messages=messages),rates,3,tmp_path,'repair')
    runner.spent=.4;runner.requests=120
    jobs=make_jobs('actual_semantics',[semantic_item(i) for i in range(3)],model(),PROJECT_ROOT/'prompts',rates)
    rows=runner.run_standard_phase('actual_semantics',jobs,{})
    assert len(rows)==3 and len(messages.calls)==3
    assert messages.calls[0]['service_tier']=='standard_only'
    assert messages.calls[0]['system'][0]['cache_control']=={'type':'ephemeral','ttl':'1h'}
    assert runner.spent==pytest.approx(.4021) and runner.requests==123
    acceptance=json.loads((tmp_path/'actual_semantics/acceptance.json').read_text())
    assert acceptance['processing']=='standard' and acceptance['real_api_calls']==3
    assert json.loads((tmp_path/'actual_semantics/progress.json').read_text())['completed']==3
    runner.budget=runner.spent+1e-8
    with pytest.raises(RuntimeError,match='cannot reserve'):
        runner.run_standard_phase('no_budget',jobs,{})
    assert len(messages.calls)==3
    # Reuse a completed prefix without calling either already-paid item again.
    runner.budget=3
    runner.spent=.4014;runner.requests=122
    resumed=runner.run_standard_phase('resumed',jobs,{},completed_prefix=rows[:2],completed_cost_usd=.0014)
    assert len(resumed)==3 and len(messages.calls)==4
    assert json.loads(messages.calls[-1]['messages'][0]['content'])==jobs[2]['payload']
    assert runner.spent==pytest.approx(.4021) and runner.requests==123
