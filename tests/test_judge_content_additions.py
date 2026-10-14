import pytest

from para_agent.eval.judge import judge_pairwise, judge_semantic


class ValidatingClient:
    def __init__(self, response):
        self.response = response
        self.payload = None

    def request(self, agent, payload, schema, *, context, validate):
        self.payload = payload
        result = schema.model_validate(self.response)
        validate(result)
        return result


def addition(status='없음', spans=None):
    return {'status': status, 'added_spans': spans or [], 'reason': '제공된 원문과 대조했다.'}


def semantic_item():
    return {'prompt': '우정을 설명하시오.', 'before': '친구를 만났다.',
            'after': '친구를 만나 함께 책을 읽었다.', 'context_before': '친구를 만났다.',
            'context_after': '친구를 만나 함께 책을 읽었다.', 'gold': '의미 변화',
            'condition': 'hidden', 'agent_model': 'hidden'}


def test_semantic_judge_records_literal_content_addition_without_leaking_metadata():
    client = ValidatingClient({'label': '의미 변화', 'reason': '독서 행동이 추가됐다.',
                               'content_addition': addition('있음', ['함께 책을 읽었다.'])})
    result = judge_semantic(client, semantic_item(), context={})
    assert result.content_addition.added_spans == ['함께 책을 읽었다.']
    assert not {'gold', 'condition', 'agent_model'} & client.payload.keys()


@pytest.mark.parametrize('value', [
    addition('있음'), addition('있음', ['함께 영화를 보았다.']),
    addition('있음', [' ']), addition('없음', ['책']), addition('판단불가', ['책']),
])
def test_semantic_judge_rejects_missing_invented_or_inconsistent_evidence(value):
    client = ValidatingClient({'label': '의미 변화', 'reason': '판정', 'content_addition': value})
    with pytest.raises(ValueError):
        judge_semantic(client, semantic_item(), context={})


def pair_response(a, b, meaning='동률'):
    result = {key: {'winner': meaning if key == 'meaning' else 'A', 'reason': '판정 근거'}
              for key in ('style', 'coherence', 'meaning', 'overall')}
    result['content_addition'] = {'A': a, 'B': b}
    return result


def test_reversed_pair_maps_content_evidence_back_to_original_sides():
    item = {'prompt': '문항', 'original': '친구를 만났다.', 'A': '친구를 만났다.',
            'B': '친구를 만나 함께 책을 읽었다.'}
    client = ValidatingClient(pair_response(addition('있음', ['함께 책을 읽었다.']), addition()))
    result = judge_pairwise(client, item, context={}, reverse=True)
    assert client.payload['A'] == item['B']
    assert result['canonical']['overall']['winner'] == 'B'
    assert result['canonical']['content_addition']['A']['status'] == '없음'
    assert result['canonical']['content_addition']['B']['added_spans'] == ['함께 책을 읽었다.']


def test_original_pair_does_not_invent_meaning_or_content_baseline():
    item = {'prompt': '문항', 'A': '글 A.', 'B': '글 B.'}
    unknown = addition('판단불가')
    result = judge_pairwise(ValidatingClient(pair_response(unknown, unknown)), item, context={})
    assert result['canonical']['meaning']['winner'] == '동률'
    with pytest.raises(ValueError, match='content addition'):
        judge_pairwise(ValidatingClient(pair_response(addition(), unknown)), item, context={})
    with pytest.raises(ValueError, match='meaning preservation'):
        judge_pairwise(ValidatingClient(pair_response(unknown, unknown, 'A')), item, context={})
