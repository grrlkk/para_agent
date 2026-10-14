import re


def sentence_spans(text: str | None) -> list[dict]:
    """Deterministic punctuation/newline boundaries with source character offsets."""
    text = text or ''
    boundaries = [m.end() for m in re.finditer(r'[.!?。！？][\"\'”’」』)\]]*(?=\s|$)|\r?\n', text)]
    spans, start = [], 0
    for end in [*boundaries, len(text)]:
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b-1].isspace():
            b -= 1
        if a < b:
            spans.append({'id': f's{len(spans)+1}', 'text': text[a:b], 'start': a, 'end': b})
        start = end
    return spans


def sentences(text: str | None) -> list[str]:
    return [item['text'] for item in sentence_spans(text)]


def numbered_sentences(text: str) -> dict[str, str]:
    return {f"s{i}": value for i, value in enumerate(sentences(text), 1)}


def last_sentences(text: str | None, count: int = 2) -> str | None:
    return " ".join(sentences(text)[-count:]) if text else None

