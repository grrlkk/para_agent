import copy

import pytest

from para_agent.agents.reviser import ReviserOutput
from para_agent.agents.summarizer import Brief, Issue
from para_agent.sentence_diff import sentence_diff, issue_application_audit
from para_agent.text import sentence_spans


@pytest.mark.parametrize('before,after,operations', [
    ('나는 간다. 너는 남는다.', '나는 갔다. 너는 남는다.', ['replace']),
    ('나는 간다. 너는 남는다.', '나는 간다. 비가 온다. 너는 남는다.', ['insert']),
    ('나는 간다. 비가 온다. 너는 남는다.', '나는 간다. 너는 남는다.', ['delete']),
    ('나는 학교에 가서 친구를 만났다.', '나는 학교에 갔다. 친구를 만났다.', ['split']),
    ('나는 학교에 갔다. 친구를 만났다.', '나는 학교에 가서 친구를 만났다.', ['merge']),
    ('  나는 간다.\n\n너는 남는다.  ', '나는 간다. 너는 남는다.', []),
    ('', '새 문장이다.', ['insert']),
    ('기존 문장이다.', '', ['delete']),
])
def test_sentence_alignment_and_exact_source_spans(before, after, operations):
    changes = sentence_diff(before, after)
    assert [c['operation'] for c in changes] == operations
    for c in changes:
        for text, key in [(before, 'before'), (after, 'after')]:
            span = c[f'{key}_span']
            assert c[key] == (text[span[0]:span[1]] if span else '')


def test_short_correction_before_added_sentences_is_aligned_to_its_correction():
    changes = sentence_diff('원문 0.', '수정 0. 앞부분의 문체도 참고한다. 다음 문장이다. 끝 문장이다.')
    first = next(c for c in changes if c['before'])
    assert first['before'] == '원문 0.' and first['after'] == '수정 0.'


def test_sentence_boundaries_keep_quotes_and_decimal_numbers():
    text = '  “간다.”  3.14는 수다.\n끝이다!'
    assert [s['text'] for s in sentence_spans(text)] == ['“간다.”', '3.14는 수다.', '끝이다!']


def issue(identifier, location):
    return Issue(id=identifier, priority=1, rubric='task_1', diagnostic_item=None,
                 location=location, evidence='', action='', recurring=False)


def test_application_self_report_audit_logs_mismatches_without_changing_output():
    original, revised = '나는 간다. 너는 남는다.', '나는 갔다. 너는 남는다.'
    brief = Brief(issues=[issue('i1',['s1']), issue('i2',['s2']), issue('i3',['s99']), issue('i4',['s1','s2'])],
                  keep_style='', connect_note=None, learner_feedback='')
    out = ReviserOutput(revised_paragraph=revised, applied_issue_ids=['i2','unknown','i2'],
                        skipped=[{'id':'i1','reason':'유지'}])
    snapshot = copy.deepcopy(out.model_dump())
    audit = issue_application_audit(brief, out, sentence_diff(original,revised), original)
    assert audit['issues'][0]['mismatches'] == ['reported_skipped_but_location_changed']
    assert audit['issues'][1]['mismatches'] == ['reported_applied_but_location_unchanged']
    assert 'invalid_or_empty_location' in audit['issues'][2]['mismatches']
    assert audit['issues'][3]['changed_locations'] == ['s1']
    assert audit['issues'][3]['unchanged_locations'] == ['s2']
    assert audit['unknown_reported_ids'] == ['unknown'] and audit['duplicate_applied_ids'] == ['i2']
    assert out.model_dump() == snapshot and audit['behavior_changed'] is False

