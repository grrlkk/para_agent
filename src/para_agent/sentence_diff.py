"""Monotone sentence alignment; change records retain exact source spans."""
from difflib import SequenceMatcher
from functools import lru_cache

from para_agent.text import sentence_spans


def sentence_diff(before, after):
    a, b = sentence_spans(before), sentence_spans(after)

    def fragment(text, spans, i, count):
        return text[spans[i]['start']:spans[i+count-1]['end']] if count else ''

    @lru_cache(None)
    def solve(i, j):
        if i == len(a) and j == len(b):
            return 0.0, ()
        choices = []
        # Keep ties deterministic; prefer ordinary one-to-one alignment.
        for na, nb in ((1,1), (1,2), (2,1), (1,0), (0,1)):
            if i+na > len(a) or j+nb > len(b):
                continue
            x, y = fragment(before,a,i,na), fragment(after,b,j,nb)
            if na and nb:
                ratio = SequenceMatcher(None,x,y,autojunk=False).ratio()
                # Charge split/merge mismatch for both sentences. Otherwise a
                # poor split can beat a good replacement plus one insertion.
                cost = (1-ratio)*max(na,nb) + .15*(na+nb-2)
            else:
                cost = .7
            rest, path = solve(i+na,j+nb)
            choices.append((cost+rest, ((i,j,na,nb),)+path))
        return min(choices,key=lambda item:item[0])

    _, alignment = solve(0,0)
    changes = []
    for i,j,na,nb in alignment:
        x,y = fragment(before,a,i,na),fragment(after,b,j,nb)
        if x==y:
            continue
        changes.append({'before':x,'after':y,
                        'before_ids':[p['id'] for p in a[i:i+na]],
                        'after_ids':[p['id'] for p in b[j:j+nb]],
                        'before_span':[a[i]['start'],a[i+na-1]['end']] if na else None,
                        'after_span':[b[j]['start'],b[j+nb-1]['end']] if nb else None,
                        'operation':'insert' if not na else 'delete' if not nb else
                                    'split' if nb>na else 'merge' if na>nb else 'replace'})
    return changes


def issue_application_audit(brief, output, changes, original):
    valid = {x['id'] for x in sentence_spans(original)}
    changed = {sid for item in changes for sid in item['before_ids']}
    applied = output.applied_issue_ids
    skipped = [item.id for item in output.skipped]
    known = {item.id for item in brief.issues}
    records=[]
    for issue in brief.issues:
        locations=set(issue.location)
        invalid=sorted(locations-valid)
        changed_locations=sorted(locations&changed)
        mismatches=[]
        if invalid or not locations:
            mismatches.append('invalid_or_empty_location')
        if issue.id in applied and not changed_locations:
            mismatches.append('reported_applied_but_location_unchanged')
        if issue.id in skipped and changed_locations:
            mismatches.append('reported_skipped_but_location_changed')
        if issue.id in applied and issue.id in skipped:
            mismatches.append('reported_both_applied_and_skipped')
        if issue.id not in applied and issue.id not in skipped:
            mismatches.append('missing_self_report')
        records.append({'issue_id':issue.id,'location':issue.location,'changed_locations':changed_locations,
                        'unchanged_locations':sorted((locations&valid)-changed),'invalid_locations':invalid,
                        'reported_applied':issue.id in applied,'reported_skipped':issue.id in skipped,
                        'mismatches':mismatches})
    return {'issues':records,'unknown_reported_ids':sorted((set(applied)|set(skipped))-known),
            'duplicate_applied_ids':sorted({x for x in applied if applied.count(x)>1}),
            'duplicate_skipped_ids':sorted({x for x in skipped if skipped.count(x)>1}),
            'behavior_changed':False,
            'scope':'Observed sentence change only; does not establish semantic fulfillment or which issue caused it.'}
