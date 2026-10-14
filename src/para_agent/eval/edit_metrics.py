"""Character edits on the exact stored text, including whitespace and newlines."""


def levenshtein(original: str, revised: str) -> int:
    if len(original) < len(revised):
        original, revised = revised, original
    previous = list(range(len(revised) + 1))
    for i, a in enumerate(original, 1):
        current = [i]
        for j, b in enumerate(revised, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_metrics(original: str, revised: str) -> dict:
    if not original:
        raise ValueError("Length ratio requires a nonempty original")
    distance = levenshtein(original, revised)
    return {"original_chars": len(original), "revised_chars": len(revised),
            "character_edit_distance": distance,
            "normalized_edit_distance": distance / max(len(original), len(revised)),
            "length_ratio": len(revised) / len(original)}
