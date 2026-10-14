"""Admit error notes only when their literal example and fix refer to actual edits."""
from difflib import SequenceMatcher
import re


def _touches_edit(fragment, text, intervals):
    for match in re.finditer(re.escape(fragment),text):
        a,b=match.span()
        if any((a<=start<=b if start==end else a<end and b>start) for start,end in intervals):
            return True
    return False


def filter_error_notes(update, changes, turn):
    accepted, rejected = [], []
    for note in update.error_notes:
        reason=None
        if note.turn!=turn:
            reason='wrong_turn'
        elif not note.example.strip():
            reason='empty_example'
        elif note.example==note.fix:
            reason='example_equals_fix'
        matches=[(i,c) for i,c in enumerate(changes) if note.example and note.example in c['before']]
        if reason is None and not matches:
            reason='example_absent_from_diff_before'
        if reason is None:
            reason='fix_absent_from_matching_diff_after'
            for i,change in matches:
                if note.fix not in change['after']:
                    continue
                opcodes=SequenceMatcher(None,change['before'],change['after'],autojunk=False).get_opcodes()
                altered=[(tag,a,b,c,d) for tag,a,b,c,d in opcodes if tag!='equal']
                source=[(a,b) for _,a,b,_,_ in altered]
                target=[(c,d) for _,_,_,c,d in altered]
                source_changed=_touches_edit(note.example,change['before'],source)
                target_changed=(_touches_edit(note.fix,change['after'],target) if note.fix else
                                any(tag=='delete' for tag,*_ in altered))
                if source_changed and target_changed:
                    reason=None
                    break
                reason='quoted_fragment_not_actually_changed'
        if reason:
            rejected.append({'note':note.model_dump(),'reason':reason})
        else:
            accepted.append(note)
    return update.model_copy(update={'error_notes':accepted}), {
        'accepted_count':len(accepted),'rejected_count':len(rejected),'rejected':rejected,
        'policy':'Literal example in diff.before, literal fix in the same diff.after, both touching a changed span; no LLM retry.'}
