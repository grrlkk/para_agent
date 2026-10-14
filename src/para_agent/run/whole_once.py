from para_agent.artifacts import write_json
from para_agent.memory.schema import Profile, SessionMemory
from para_agent.register import register_distribution

from .common import diagnose_and_revise


def run(essay, cfg, services):
    directory = services.run_dir / "whole_once" / essay.essay_id
    profile = Profile(prompt=essay.prompt)
    context = {"condition": "whole_once", "essay_id": essay.essay_id, "turn": 0}
    # Transient current diagnosis for Evidence; no previous turns or memory update.
    _, out, record, _ = diagnose_and_revise(
        "\n\n".join(essay.paragraphs), [], SessionMemory(), profile,
        cfg=cfg, services=services, context=context,
        k=cfg.k_issues * len(essay.paragraphs), unit="cum")
    write_json(directory / "turn_0.json", {**record, "memory_update": None,
                                           "memory_update_applied": False})
    final = {"essay_id": essay.essay_id, "condition": "whole_once", "mock": services.client.is_mock,
             "Y": out.revised_paragraph.split("\n\n"), "final_text": out.revised_paragraph,
             "memory": SessionMemory(), "profile": profile,
             "register_distributions":[register_distribution(p) for p in out.revised_paragraph.split('\n\n')]}
    write_json(directory / "final.json", final)
    return final

