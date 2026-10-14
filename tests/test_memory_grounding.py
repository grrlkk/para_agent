import pytest

from para_agent.memory.schema import MemoryUpdate, Profile, ProfilePatch
from para_agent.memory.validation import filter_error_notes
from para_agent.sentence_diff import sentence_diff


def update(example, fix, turn=0):
    return MemoryUpdate(error_notes=[{'turn':turn,'type':'조사','example':example,'fix':fix}],
                        profile_update={}, turn_summary='')


@pytest.mark.parametrize('example,fix,turn,reason', [
    ('학교를','학교에',0,None),
    ('도서관을','도서관에',0,'example_absent_from_diff_before'),
    ('학교를','공원에',0,'fix_absent_from_matching_diff_after'),
    ('나는','학교에',0,'quoted_fragment_not_actually_changed'),
    ('학교를','간다',0,'quoted_fragment_not_actually_changed'),
    ('학교를','학교를',0,'example_equals_fix'),
    ('학교를','학교에',2,'wrong_turn'),
    ('','학교에',0,'empty_example'),
])
def test_notes_must_quote_actual_source_and_target_edit(example,fix,turn,reason):
    diff = sentence_diff('나는 학교를 간다.', '나는 학교에 간다.')
    filtered, audit = filter_error_notes(update(example,fix,turn), diff, 0)
    assert len(filtered.error_notes) == (1 if reason is None else 0)
    assert audit['rejected_count'] == (0 if reason is None else 1)
    if reason:
        assert audit['rejected'][0]['reason'] == reason


def test_no_diff_admits_no_note_and_deleted_example_can_have_empty_fix():
    filtered,audit = filter_error_notes(update('학교를','학교에'), [], 0)
    assert not filtered.error_notes and audit['rejected_count'] == 1
    diff = sentence_diff('나는 매우 매우 기쁘다.', '나는 매우 기쁘다.')
    filtered,audit = filter_error_notes(update('매우 ',''), diff, 0)
    assert len(filtered.error_notes) == 1 and audit['rejected_count'] == 0


def test_fix_in_different_changed_sentence_does_not_validate_note():
    diff = [{'before':'가나다.', 'after':'가마바.'}, {'before':'사아자.', 'after':'사차카.'}]
    filtered,audit = filter_error_notes(update('나다','차카'), diff, 0)
    assert not filtered.error_notes
    assert audit['rejected'][0]['reason'] == 'fix_absent_from_matching_diff_after'


def test_llm_cannot_write_register_but_keeps_existing_topic_and_thesis_fields():
    with pytest.raises(ValueError):
        ProfilePatch.model_validate({'register':'해요체'})
    profile = Profile(prompt='문항', register='해라체', topic='기존 주제')
    patch = MemoryUpdate(error_notes=[], profile_update={'style_notes':'간결한 문장'}, turn_summary='')
    profile.apply(patch)
    assert profile.register_ == '해라체' and profile.style_notes == '간결한 문장'
    assert profile.topic == '기존 주제'

