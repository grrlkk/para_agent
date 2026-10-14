from dataclasses import dataclass
from pathlib import Path

from para_agent.agents.reviser import revise
from para_agent.agents.summarizer import summarize
from para_agent.anonymization import TurnAnonymizer
from para_agent.diagnose.evidence import build_evidence
from para_agent.diagnose.service import DiagnosisService
from para_agent.diagnose.schema import evidence_views
from para_agent.kb.lookup import KnowledgeBase
from para_agent.llm.client import Client
from para_agent.memory.schema import Profile
from para_agent.register import register_distribution, update_profile_register
from para_agent.sentence_diff import sentence_diff, issue_application_audit
from para_agent.text import last_sentences


@dataclass
class Services:
    diagnoser: DiagnosisService
    client: Client
    kb: KnowledgeBase
    rubric_diag_map: dict
    run_dir: Path


def diagnose_and_revise(x, prefix, memory, profile, *, cfg, services, context, k, unit):
    diag = services.diagnoser.diagnose(x, prefix, unit, profile.prompt)
    memory.add_turn_diag(context["turn"], diag)
    evidence = build_evidence(diag, memory, delta_mode=cfg.delta_mode,
                              weak_rubrics=cfg.weak_rubrics,
                              rubric_diag_map=services.rubric_diag_map, kb=services.kb)
    items = {item for view in evidence_views(evidence) for item in view.diagnostic_items}
    entries = services.kb.lookup(items)
    prev = prefix[-1] if prefix else None
    guard = TurnAnonymizer(context['essay_id'], context['condition'], context['turn'])
    original_view = guard.protect(x)
    safe_x = original_view.masked
    safe_prev = guard.protect(prev).masked if prev is not None else None
    safe_profile = Profile.model_validate(guard.mask_data(profile))
    safe_evidence = type(evidence).model_validate(guard.mask_data(evidence))
    safe_brief = summarize(services.client, safe_x, safe_evidence, guard.mask_data(entries),
                      guard.mask_data(memory.notes()), safe_profile, safe_prev,
                      k=k, context=context)
    safe_brief = type(safe_brief).model_validate(guard.mask_data(safe_brief))
    rev_in = safe_x if cfg.reviser_input == "x_only" else {"prev": safe_prev, "x": safe_x}
    candidate = revise(services.client, rev_in, safe_brief, safe_profile,
                       last_sentences(safe_prev, 2), context=context)
    protection = guard.validate_revision(original_view, candidate.revised_paragraph)
    safe_y = candidate.revised_paragraph if protection['ok'] else safe_x
    revised_view = guard.restored_view(safe_y)
    brief = type(safe_brief).model_validate(guard.restore_data(safe_brief))
    raw_out = type(candidate).model_validate(guard.restore_data(candidate))
    out = raw_out.model_copy(update={'revised_paragraph':revised_view.original})
    changes = sentence_diff(x,out.revised_paragraph)
    styles = {'original':[register_distribution(p) for p in x.split('\n\n')],
              'revised':[register_distribution(p) for p in out.revised_paragraph.split('\n\n')]}
    register_update = update_profile_register(profile,styles['revised'])
    # User-requested local fallback; no extra revision call or retrospective editing.
    record = {**context, "mock": services.client.is_mock, "x": x, "prefix": list(prefix), "diag": diag,
              "evidence": evidence, "kb_entry_ids": [entry.get("id", entry["code"]) for entry in entries],
              "brief": brief, "reviser_output": out, "reviser_output_candidate":raw_out,
              "issue_limit": k, "anonymization":protection,
              "turn_status":"ok" if protection['ok'] else 'failed_anonymization',
              "sentence_diff":changes, 'text_changed':x!=out.revised_paragraph,
              'boundary_only_change':x!=out.revised_paragraph and not changes,
              "register_distribution":styles, "profile_register_update":register_update,
              "issue_application_audit":issue_application_audit(brief,out,changes,x)}
    protected = {'guard':guard,'original':original_view,'revised':revised_view,'brief':safe_brief}
    return brief, out, record, protected

