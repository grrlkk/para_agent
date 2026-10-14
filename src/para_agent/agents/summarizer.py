from pydantic import Field

from para_agent.diagnose.schema import evidence_views
from para_agent.schema import Schema
from para_agent.text import numbered_sentences


class Issue(Schema):
    id: str
    priority: int = Field(ge=1)
    rubric: str
    diagnostic_item: str | None
    location: list[str]
    evidence: str
    action: str
    recurring: bool


class Brief(Schema):
    issues: list[Issue]
    keep_style: str
    connect_note: str | None
    learner_feedback: str


def summarize(client, x, evidence, kb_entries, notes, profile, prev, *, k, context):
    rubric_keys = {key for view in evidence_views(evidence) for key in view.rubric_scores}

    def validate(brief):
        if len(brief.issues) > k:
            raise ValueError(f"Brief exceeds k={k}")
        ids = [issue.id for issue in brief.issues]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate issue IDs")
        priorities = [issue.priority for issue in brief.issues]
        if priorities != sorted(priorities):
            raise ValueError("Issues must be in priority order")
        if any(issue.rubric not in rubric_keys for issue in brief.issues):
            raise ValueError("Unknown rubric key")
    return client.request("summarizer", {
        "x": x, "sentences": numbered_sentences(x), "evidence": evidence,
        "kb_entries": kb_entries, "memory_notes": notes, "profile": profile,
        "prev": prev, "k": k, "condition": context["condition"],
    }, Brief, context=context, k=k, validate=validate)

