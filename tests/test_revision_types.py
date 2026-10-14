from copy import deepcopy

import pytest

from para_agent.eval.revision_types import observed_type, summarize_types


@pytest.mark.parametrize('before,after,operation,expected', [
    ('할수 있다.', '할 수 있다.', 'replace', 'whitespace_only'),
    ('가니?', '가니!', 'replace', 'punctuation_only'),
    ('갈수 있니?', '갈 수 있니!', 'replace', 'whitespace_and_punctuation'),
    ('간다.', '갑니다.', 'replace', 'ending_style_change'),
    ('이길 수 있다.', '이기기도 한다.', 'replace', 'other_text_change'),
    ('못하였더라도 간다.', '못했더라도 간다.', 'replace', 'other_text_change'),
    ('', '새 문장이다.', 'insert', 'sentence_insertion'),
    ('원문이다.', '', 'delete', 'sentence_deletion'),
    ('읽고 쓴다.', '읽는다. 쓴다.', 'split', 'split'),
    ('읽는다. 쓴다.', '읽고 쓴다.', 'merge', 'merge'),
])
def test_observed_types_do_not_guess_semantic_or_error_labels(before, after, operation, expected):
    change = {'before': before, 'after': after, 'operation': operation}
    assert observed_type(change) == expected


def turn(essay_id, before, after, **extra):
    return {'condition': 'incremental', 'essay_id': essay_id, 'turn': 1, 'x': before,
            'reviser_output': {'revised_paragraph': after},
            'brief': {'learner_feedback': '개선 아이디어(학습자가 선택할 사항): 경험을 떠올려 보세요.'},
            **extra}


def test_summary_reports_zero_edits_and_boundary_only_edits_with_separate_denominators():
    turns = [turn('e1', '할수 있다.', '할 수 있다.',
                  memory_update={'error_notes': [{'type': '띄어쓰기'}]}),
             turn('e2', '읽는다.\n쓴다.', '읽는다.\n\n쓴다.'),
             turn('e3', '간다.', '간다.')]
    snapshot = deepcopy(turns)
    originals = {t['essay_id']: t['x'] for t in turns}
    finals = {(t['condition'], t['essay_id']): t['reviser_output']['revised_paragraph'] for t in turns}
    report = summarize_types(turns, originals, finals)
    result = report['conditions']['incremental']
    assert result['turns'] == 3 and result['changed_turns'] == 2
    assert result['unchanged_turns'] == result['boundary_only_change_turns'] == 1
    assert result['changed_sentence_pairs'] == result['counts']['whitespace_only'] == 1
    assert result['fractions']['whitespace_only'] == 1
    assert result['original_sentences'] == result['final_sentences'] == 4
    assert result['added_sentence_ratio'] == 0
    assert result['accepted_memory_note_types'] == {'띄어쓰기': 1}
    assert result['feedback_with_idea_heading'] == 3
    assert report['semantic_classification'] == 'not_performed'
    assert turns == snapshot  # Offline analysis cannot mutate revision or memory outputs.


def test_new_sentence_ratio_uses_original_sentence_count_and_undefined_fractions_are_null():
    unchanged = turn('a', '읽는다.', '읽는다.')
    report = summarize_types([unchanged], {'a': '읽는다.'}, {('incremental', 'a'): '읽는다.'})
    result = report['conditions']['incremental']
    assert result['changed_sentence_pairs'] == 0
    assert all(value is None for value in result['fractions'].values())
    added = turn('a', '읽는다.', '읽는다. 우정은 누구에게나 아주 중요한 가치이다.')
    result = summarize_types([added], {'a': '읽는다.'},
        {('incremental', 'a'): added['reviser_output']['revised_paragraph']})['conditions']['incremental']
    assert result['unaligned_added_sentences'] == 1
    assert result['original_sentences'] == 1 and result['final_sentences'] == 2
    assert result['added_sentence_ratio'] == 1
