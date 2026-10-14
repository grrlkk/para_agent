from para_agent.agents.memory_update import update_memory
from para_agent.artifacts import write_json
from para_agent.anonymization import protected_diff, PLACEHOLDER
from para_agent.memory.schema import Profile, SessionMemory, MemoryUpdate, ProfilePatch
from para_agent.memory.validation import filter_error_notes
from para_agent.register import register_distribution

from .common import diagnose_and_revise


def run(essay, cfg, services):
    directory = services.run_dir / "incremental" / essay.essay_id
    memory, profile = SessionMemory(), Profile(prompt=essay.prompt)
    revised = []
    for turn, x in enumerate(essay.paragraphs):
        if not cfg.use_memory:
            memory, profile = SessionMemory(), Profile(prompt=essay.prompt)
        context = {"condition": "incremental", "essay_id": essay.essay_id, "turn": turn}
        before = {"memory": memory.model_dump(), "profile": profile.model_dump()}
        brief, out, record, protected = diagnose_and_revise(
            x, revised, memory, profile, cfg=cfg, services=services, context=context,
            k=cfg.k_issues, unit=cfg.diag_unit)
        revised.append(out.revised_paragraph)
        # A separate call boundary leaves room for future changes; no §11 mechanism is installed.
        guard=protected['guard']
        safe_diff=protected_diff(record['sentence_diff'],protected['original'],protected['revised'])
        masked_update = update_memory(services.client, protected['original'].masked,
                               protected['revised'].masked, protected['brief'],
                               guard.mask_data(memory.notes()),
                               Profile.model_validate(guard.mask_data(profile)),
                               diff=safe_diff, context=context)
        raw_update=MemoryUpdate.model_validate(guard.restore_data(masked_update.model_dump(exclude_unset=True)))
        update, validation=filter_error_notes(raw_update,record['sentence_diff'],turn)
        patch=update.profile_update.model_dump(exclude_unset=True)
        unresolved=[key for key,value in patch.items() if isinstance(value,str) and PLACEHOLDER.search(value)]
        if unresolved:
            update=update.model_copy(update={'profile_update':ProfilePatch.model_validate({k:v for k,v in patch.items() if k not in unresolved})})
        validation['unresolved_profile_fields_dropped']=unresolved
        if PLACEHOLDER.search(update.turn_summary):
            update=update.model_copy(update={'turn_summary':''})
            validation['unresolved_turn_summary_dropped']=True
        if cfg.use_memory:
            memory.apply(update)
            profile.apply(update)
        else:
            # Same call schedule, but no notes, diagnosis history or profile survives a turn.
            memory, profile = SessionMemory(), Profile(prompt=essay.prompt)
        write_json(directory / f"turn_{turn}.json", {
            **record, "state_before": before, "memory_update": update,
            "memory_update_candidate":raw_update,"memory_update_validation":validation,
            "memory_update_applied": cfg.use_memory,
            "state_after": {"memory": memory, "profile": profile},
        })
    final = {"essay_id": essay.essay_id, "condition": "incremental", "mock": services.client.is_mock,
             "Y": revised, "final_text": "\n\n".join(revised), "memory": memory, "profile": profile,
             "register_distributions":[register_distribution(p) for p in revised]}
    write_json(directory / "final.json", final)
    return final

