import json

import pytest
import yaml

from para_agent.agents.reviser import ReviserOutput
from para_agent.agents.summarizer import Brief
from para_agent.anonymization import MARKER, PLACEHOLDER
from para_agent.artifacts import json_text
from para_agent.config import PROJECT_ROOT
from para_agent.data.loader import Essay
from para_agent.diagnose.mock import MockDiagnoser
from para_agent.memory.schema import MemoryUpdate
from para_agent.run import incremental, whole_once
from para_agent.run.common import Services


class EditClient:
    is_mock = True

    def __init__(self, damage=False):
        self.calls = []
        self.damage = damage

    def request(self, agent, payload, schema, *, context, **kwargs):
        payload = json.loads(json_text(payload))
        self.calls.append((agent, payload))
        assert not MARKER.search(json_text(payload)), 'A raw marker was exposed to an LLM payload'
        if agent == 'summarizer':
            return Brief(issues=[], keep_style='원문 유지', connect_note=None, learner_feedback='유지')
        if agent == 'reviser':
            x = payload['rev_in']['x'] if isinstance(payload['rev_in'],dict) else payload['rev_in']
            y = x.replace('학교를 간다', '학교에 간다')
            if self.damage and context['turn'] == 0:
                y = PLACEHOLDER.sub('', y, count=1)
            return ReviserOutput(revised_paragraph=y, applied_issue_ids=[], skipped=[])
        return MemoryUpdate(error_notes=[
            {'turn':context['turn'],'type':'조사','example':'학교를','fix':'학교에'},
            {'turn':context['turn'],'type':'조사','example':'존재하지 않는 오류','fix':'수정'}],
            profile_update={'style_notes':'짧은 해라체 문장'}, turn_summary='조사 확인')


@pytest.mark.parametrize('condition', ['incremental','whole_once'])
@pytest.mark.parametrize('damage', [False,True])
def test_both_conditions_protect_restore_log_diff_and_fallback(tmp_path,cfg,contracts,kb,condition,damage):
    cfg.reviser_input = 'prev_plus_x'
    essay = Essay(essay_id='protected',prompt='#@소속#에 관한 글을 써라.',original_scores={},
                  paragraphs=['#@이름#는 학교를 간다. #@이름#는 남는다.',
                              '#@소속# 친구도 학교를 간다.', '#@이름#는 웃는다.'])
    client = EditClient(damage)
    diagnoser = MockDiagnoser(contracts['rubric_keys'],contracts['feature_keys'])
    raw_calls = []
    native_diagnose = diagnoser.diagnose
    def record_raw(x,prefix,unit,prompt):
        raw_calls.append((x,prompt))
        return native_diagnose(x,prefix,unit,prompt)
    diagnoser.diagnose = record_raw
    services = Services(diagnoser, client, kb,
        yaml.safe_load((PROJECT_ROOT/'configs/rubric_diag_map.yaml').read_text()),tmp_path)
    runner = incremental if condition=='incremental' else whole_once
    result = runner.run(essay,cfg,services)
    assert '#@이름#' in result['final_text'] and not PLACEHOLDER.search(result['final_text'])
    assert all('#@소속#' in prompt for _,prompt in raw_calls)
    assert MARKER.search(raw_calls[0][0])  # local diagnosis sees unchanged original markers
    records = [json.loads(p.read_text()) for p in sorted((tmp_path/condition/essay.essay_id).glob('turn_*.json'))]
    first = records[0]
    assert first['anonymization']['fallback_to_original'] == damage
    assert first['turn_status'] == ('failed_anonymization' if damage else 'ok')
    assert bool(first['sentence_diff']) != damage
    assert first['issue_application_audit']['behavior_changed'] is False
    assert first['profile_register_update']['source'] == 'code'
    assert len(client.calls) == (9 if condition=='incremental' else 2)  # no extra retry call
    if condition=='incremental':
        assert result['Y'][0] == (essay.paragraphs[0] if damage else essay.paragraphs[0].replace('학교를','학교에'))
        assert records[1]['prefix'] == [result['Y'][0]]
        assert records[1]['memory_update_validation']['accepted_count'] == 1
        assert records[1]['memory_update_validation']['rejected_count'] == 1
        assert first['memory_update_validation']['accepted_count'] == (0 if damage else 1)
        for turn in range(3):
            payload = client.calls[turn*3+2][1]
            assert 'diff' in payload
        assert result['profile'].register_ == '해라체'
        assert not records[2]['sentence_diff'] and not records[2]['memory_update']['error_notes']
    elif damage:
        assert result['final_text'] == '\n\n'.join(essay.paragraphs)

