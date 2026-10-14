import pytest

from para_agent.anonymization import MARKER, PLACEHOLDER, TurnAnonymizer, protected_diff
from para_agent.sentence_diff import sentence_diff


def test_all_observed_marker_types_and_adjacent_repetitions_round_trip():
    names = '계정 계층차별 금융번호 기타차별 비속어 성차별 소속 신원 외모성격차별 이름 인종차별 전화번호 정치성향차별 주소 직업차별 출신차별'.split()
    text = ''.join(f'#@{name}#' for name in names) + '#@이름##@이름#이다.'
    guard = TurnAnonymizer('essay', 'incremental', 0)
    view = guard.protect(text)
    assert not MARKER.search(view.masked)
    tokens = PLACEHOLDER.findall(view.masked)
    assert len(tokens) == len(set(tokens)) == 18
    assert guard.validate_revision(view, view.masked)['ok']
    assert guard.restore(view.masked) == text
    assert guard.restored_view(view.masked).original == text


@pytest.mark.parametrize('damage', ['missing', 'changed', 'duplicate', 'reorder', 'context', 'raw', 'malformed', 'empty'])
def test_placeholder_corruption_requires_fallback(damage):
    guard = TurnAnonymizer('essay', 'incremental', 0)
    view = guard.protect('#@이름#와 #@소속#이다.')
    other = guard.protect('#@이름# 이전 문단이다.').masked
    first, second = PLACEHOLDER.findall(view.masked)
    outputs = {
        'missing': view.masked.replace(first, ''),
        'changed': view.masked.replace(first, first.replace('000000', '900000')),
        'duplicate': view.masked + first,
        'reorder': second + '와 ' + first + '이다.',
        'context': view.masked + other,
        'raw': view.masked + '#@이름#',
        'malformed': view.masked.replace(first, first[:-1]),
        'empty': ' ',
    }
    result = guard.validate_revision(view, outputs[damage])
    assert not result['ok'] and result['fallback_to_original']


def test_diff_reuses_occurrence_identity_even_after_edit_moves_offsets():
    guard = TurnAnonymizer('essay', 'incremental', 0)
    before = guard.protect('#@이름#는 간다. #@이름#는 남는다.')
    after = guard.restored_view(before.masked.replace('는 간다', '가 학교에 간다'))
    changes = sentence_diff(before.original, after.original)
    masked = protected_diff(changes, before, after)
    assert len(masked) == 1
    assert PLACEHOLDER.findall(masked[0]['before']) == PLACEHOLDER.findall(masked[0]['after'])
    assert not MARKER.search(str(masked))
    assert guard.restore(masked[0]['before']) == changes[0]['before']


def test_namespace_stable_and_all_nested_values_masked():
    first, second = [TurnAnonymizer('a', 'whole_once', 0) for _ in range(2)]
    payload = {'profile': {'topic': '#@소속#'}, 'notes': [{'example': '#@이름#'}]}
    assert first.mask_data(payload) == second.mask_data(payload)
    assert first.restore_data(first.mask_data(payload)) == payload
    assert TurnAnonymizer('a', 'whole_once', 1).namespace != first.namespace

