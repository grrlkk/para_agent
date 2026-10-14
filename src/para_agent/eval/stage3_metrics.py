"""Operational metrics with explicit denominators; no LLM calls."""
from collections import Counter

from para_agent.register import register_distribution
from .edit_metrics import edit_metrics


def ratio(numerator,denominator):
    return {'numerator':numerator,'denominator':denominator,
            'rate':numerator/denominator if denominator else None}


def style_consistency(paragraphs):
    distributions=[register_distribution(p) for p in paragraphs]
    labels=[d['dominant'] for d in distributions]
    eligible=[(a,b) for a,b in zip(labels,labels[1:]) if a!='기타' and b!='기타']
    return {**ratio(sum(a==b for a,b in eligible),len(eligible)),
            'possible_adjacent_pairs':max(0,len(labels)-1),
            'coverage':len(eligible)/(len(labels)-1) if len(labels)>1 else None,
            'dominant_styles':labels,'distributions':distributions}


def operational_metrics(turns,original,final_paragraphs):
    rejected=accepted=issues=mismatches=unavailable=0
    for t in turns:
        audit=t.get('memory_update_validation') or {}
        rejected+=audit.get('rejected_count',0);accepted+=audit.get('accepted_count',0)
        application=t.get('issue_application_audit')
        if application is None:
            unavailable+=1
            continue
        for issue in application['issues']:
            issues+=1
            mismatches+=bool(set(issue['mismatches']) & {
                'reported_applied_but_location_unchanged','reported_skipped_but_location_changed',
                'reported_both_applied_and_skipped'})
    return {'fallback':ratio(sum(t.get('turn_status')=='failed_anonymization' for t in turns),len(turns)),
            'memory_note_discard':ratio(rejected,rejected+accepted),
            'issue_application_mismatch':ratio(mismatches,issues),
            'missing_audit_turns':unavailable,
            'style_consistency':style_consistency(final_paragraphs),
            'original_style_consistency':style_consistency(original.split('\n\n')),
            'edit':edit_metrics(original,'\n\n'.join(final_paragraphs))}


def meaning_preservation(labels):
    counts=Counter(labels)
    if set(counts)-{'보존','경미한 뉘앙스 변화','의미 변화'}:
        raise ValueError('Unknown semantic label')
    return {**ratio(counts['보존'],len(labels)),
            'counts':{k:counts[k] for k in ('보존','경미한 뉘앙스 변화','의미 변화')}}


def aggregate_operational(rows):
    output={}
    for condition in sorted({r['condition'] for r in rows}):
        group=[r for r in rows if r['condition']==condition]
        output[condition]={'essays':len(group)}
        for name in ('fallback','memory_note_discard','issue_application_mismatch','style_consistency'):
            output[condition][name]=ratio(sum(r[name]['numerator'] for r in group),sum(r[name]['denominator'] for r in group))
        output[condition]['mean_edit']={k:sum(r['edit'][k] for r in group)/len(group) for k in group[0]['edit']}
    return output
