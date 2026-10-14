"""Observable edit shapes, separate from unvalidated semantic or error labels."""
from collections import Counter
import unicodedata

from para_agent.register import register_distribution
from para_agent.sentence_diff import sentence_diff
from para_agent.text import sentence_spans

TYPE_NAMES = {
    'sentence_insertion': '대응 없는 문장 추가',
    'sentence_deletion': '문장 삭제',
    'split': '문장 분할',
    'merge': '문장 병합',
    'whitespace_only': '공백만 변경',
    'punctuation_only': '문장부호만 변경',
    'whitespace_and_punctuation': '공백·문장부호 변경',
    'ending_style_change': '종결체 변경 포함',
    'other_text_change': '기타 표현·내용 변경(미분류)',
}


def without_space(text):
    return ''.join(c for c in text if not c.isspace())


def without_punctuation(text):
    return ''.join(c for c in text if not unicodedata.category(c).startswith('P'))


def observed_type(change):
    if change['operation'] in ('insert', 'delete', 'split', 'merge'):
        return {'insert': 'sentence_insertion', 'delete': 'sentence_deletion',
                'split': 'split', 'merge': 'merge'}[change['operation']]
    before, after = change['before'], change['after']
    if without_space(before) == without_space(after):
        return 'whitespace_only'
    if without_punctuation(before) == without_punctuation(after):
        return 'punctuation_only'
    if without_space(without_punctuation(before)) == without_space(without_punctuation(after)):
        return 'whitespace_and_punctuation'
    a, b = (register_distribution(text)['dominant'] for text in (before, after))
    if a != b and '기타' not in (a, b):
        return 'ending_style_change'
    return 'other_text_change'


def summarize_types(turns, originals, finals):
    result = {}
    for condition in sorted({t['condition'] for t in turns}):
        selected = [t for t in turns if t['condition'] == condition]
        counts, note_types, examples = Counter(), Counter(), {k: [] for k in TYPE_NAMES}
        added = changed_turns = boundary_only = 0
        feedback = []
        for turn in selected:
            x, y = turn['x'], turn['reviser_output']['revised_paragraph']
            changes = sentence_diff(x, y)
            changed_turns += x != y
            boundary_only += x != y and not changes
            for change in changes:
                category = observed_type(change)
                counts[category] += 1
                if category == 'sentence_insertion':
                    added += len(change['after_ids'])
                if len(examples[category]) < 5:
                    examples[category].append({'essay_id': turn['essay_id'], 'turn': turn['turn'],
                                              **change})
            # These are accepted LLM notes, not independent ground-truth labels.
            update = turn.get('memory_update') or {}
            note_types.update(n['type'] for n in update.get('error_notes', []))
            feedback.append({'essay_id': turn['essay_id'], 'turn': turn['turn'],
                             'learner_feedback': turn['brief']['learner_feedback']})
        essay_ids = {t['essay_id'] for t in selected}
        original_count = sum(len(sentence_spans(originals[eid])) for eid in essay_ids)
        final_count = sum(len(sentence_spans(finals[condition, eid])) for eid in essay_ids)
        result[condition] = {
            'turns': len(selected), 'changed_turns': changed_turns,
            'unchanged_turns': len(selected) - changed_turns,
            'boundary_only_change_turns': boundary_only,
            'changed_sentence_pairs': sum(counts.values()),
            'counts': {k: counts[k] for k in TYPE_NAMES},
            'fractions': {k: counts[k] / sum(counts.values()) if counts else None for k in TYPE_NAMES},
            'examples_up_to_five_each': examples,
            'accepted_memory_note_types': dict(note_types),
            'accepted_memory_note_label_source': 'LLM self-report, retained only after literal diff validation',
            'original_sentences': original_count, 'final_sentences': final_count,
            'unaligned_added_sentences': added,
            'added_sentence_ratio': added / original_count if original_count else None,
            'learner_feedback': feedback,
            'feedback_with_idea_heading': sum('개선 아이디어' in f['learner_feedback'] for f in feedback),
        }
    return {'method': 'sentence_diff_and_surface_rules_only_no_LLM',
            'semantic_classification': 'not_performed', 'type_names': TYPE_NAMES,
            'limitations': [
                'Punctuation or spacing changes do not establish meaning preservation.',
                'An inserted sentence is not necessarily new information; split/merge cases are separate.',
                'Other edits are not guessed to be correction, polish, level increase, or meaning change.',
                'An improvement-idea heading count is not a count of substantive suggestions.',
            ], 'conditions': result}
