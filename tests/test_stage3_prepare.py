from types import SimpleNamespace

import pytest
import yaml

from para_agent.config import PROJECT_ROOT, ModelConfig, load_config
from para_agent.eval.costs import BudgetTransport, token_cost
from para_agent.llm.anthropic_transport import AnthropicTransport


def test_agent_configuration_and_semantics_prompt_requirements():
    cfg=load_config(PROJECT_ROOT/'configs/stage3_luna.yaml')
    assert cfg.llm.agent.model=='gpt-6-luna' and cfg.llm.agent.reasoning_effort=='none'
    assert cfg.llm.agent.temperature==0 and cfg.reviser_input=='prev_plus_x'
    for name in ('summarizer','reviser'):
        prompt=(PROJECT_ROOT/f'prompts/{name}.md').read_text()
        for rule in ('문법이나 문맥상 틀렸을 때만','두려움 → 부담','이길 수 있다 → 이기기도 한다','감정의 강도','수량','주장의 방향'):
            assert rule in prompt
    assert 'issue 0개가 정상' in (PROJECT_ROOT/'prompts/summarizer.md').read_text()


@pytest.mark.parametrize('file,mode',[('judge_haiku55','disabled'),('judge_sonnet55','between_tools')])
def test_fixed_judge_settings_omit_upfront_thinking(file,mode,tmp_path):
    model=ModelConfig.model_validate(yaml.safe_load((PROJECT_ROOT/f'configs/{file}.yaml').read_text()))
    sdk=SimpleNamespace(close=lambda:None)
    params=AnthropicTransport(model,tmp_path/'.env',sdk_client=sdk).parameters({'type':'object','properties':{}},'judge')
    assert params['thinking']=={'type':mode}
    assert params['output_config']['effort']=='low'
    assert 'temperature' not in params and 'extra_body' not in params


def test_budget_guard_prevents_dispatch_and_counts_actual_usage():
    rates=yaml.safe_load((PROJECT_ROOT/'configs/api_pricing_2026-10-15.yaml').read_text())['models']['gpt-6-luna']
    class Fake:
        call_count=0
        last_usage={}
        def complete(self,**kwargs):
            self.call_count+=1
            self.last_usage={'input_tokens':1000,'output_tokens':100}
            return '{}',self.last_usage
    f=Fake();guard=BudgetTransport(f,rates,0.01)
    guard.complete(agent='test',messages=[],params={'max_output_tokens':100})
    assert guard.spent_usd==pytest.approx(0.00015)
    guard.budget_usd=guard.spent_usd
    with pytest.raises(RuntimeError,match='Approved budget'):
        guard.complete(agent='test',messages=[],params={'max_output_tokens':100})
    assert f.call_count==1
    cost=token_cost({'input_tokens':1000,'output_tokens':100,'input_tokens_details':{'cached_tokens':500}},rates)
    assert cost==pytest.approx(0.000105)
