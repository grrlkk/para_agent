from para_agent.schema import Schema


class SkippedIssue(Schema):
    id: str
    reason: str


class ReviserOutput(Schema):
    revised_paragraph: str
    applied_issue_ids: list[str]
    skipped: list[SkippedIssue]


def revise(client, rev_in, brief, profile, prev_tail, *, context):
    return client.request("reviser", {
        "rev_in": rev_in, "brief": brief, "profile": profile, "prev_tail": prev_tail,
        "condition": context["condition"],
    }, ReviserOutput, context=context)

