from para_agent.memory.schema import MemoryUpdate


def update_memory(client, x, y, brief, notes, profile, *, diff, context):
    return client.request("memory_update", {
        "x": x, "y": y, "brief": brief, "memory_notes": notes, "profile": profile,
        "diff": diff, "turn": context["turn"],
    }, MemoryUpdate, context=context)

