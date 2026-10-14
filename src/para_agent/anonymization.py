"""Protect each observed #@...# occurrence across one revision turn."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import re

MARKER = re.compile(r'#@[^#\r\n]+#')
PLACEHOLDER = re.compile(r'__ANON_[A-Za-z0-9]+_[A-Za-z0-9]+__')


@dataclass
class ProtectedText:
    original: str
    masked: str
    occurrences: list[dict]

    def span(self, start, end):
        """Mask an exact original-text span using its occurrence identities."""
        result, position = [], start
        for item in self.occurrences:
            a, b = item['start'], item['end']
            if b <= start or a >= end:
                continue
            if a < start or b > end:
                raise ValueError('A text span splits an anonymization marker')
            result.extend([self.original[position:a], item['placeholder']])
            position = b
        result.append(self.original[position:end])
        return ''.join(result)


class TurnAnonymizer:
    def __init__(self, essay_id, condition, turn):
        self.namespace = hashlib.sha256(f'{essay_id}:{condition}:{turn}'.encode()).hexdigest()[:12]
        self.mapping = {}

    def protect(self, text):
        pieces, occurrences, position = [], [], 0
        for match in MARKER.finditer(text):
            token = f'__ANON_{self.namespace}_{len(self.mapping):06d}__'
            self.mapping[token] = match.group()
            pieces.extend([text[position:match.start()], token])
            occurrences.append({'start': match.start(), 'end': match.end(),
                                'marker': match.group(), 'placeholder': token})
            position = match.end()
        pieces.append(text[position:])
        return ProtectedText(text, ''.join(pieces), occurrences)

    def mask_data(self, value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, str):
            return self.protect(value).masked
        if isinstance(value, dict):
            return {key: self.mask_data(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.mask_data(item) for item in value]
        return value

    def restore(self, text):
        return PLACEHOLDER.sub(lambda m: self.mapping.get(m.group(), m.group()), text)

    def restore_data(self, value):
        if hasattr(value, 'model_dump'):
            value = value.model_dump()
        if isinstance(value, str):
            return self.restore(value)
        if isinstance(value, dict):
            return {key: self.restore_data(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.restore_data(item) for item in value]
        return value

    def restored_view(self, masked):
        pieces, occurrences, position, restored_length = [], [], 0, 0
        for match in PLACEHOLDER.finditer(masked):
            if match.group() not in self.mapping:
                raise ValueError('Unknown placeholder in restored text')
            literal = masked[position:match.start()]
            pieces.append(literal)
            restored_length += len(literal)
            marker = self.mapping[match.group()]
            occurrences.append({'start': restored_length, 'end': restored_length+len(marker),
                                'marker': marker, 'placeholder': match.group()})
            pieces.append(marker)
            restored_length += len(marker)
            position = match.end()
        pieces.append(masked[position:])
        return ProtectedText(''.join(pieces), masked, occurrences)

    def validate_revision(self, original_view, candidate):
        expected = [item['placeholder'] for item in original_view.occurrences]
        actual = PLACEHOLDER.findall(candidate)
        counts = Counter(actual)
        missing = [token for token in expected if counts[token] == 0]
        duplicated = [token for token in expected if counts[token] > 1]
        foreign = sorted(set(actual)-set(expected))
        raw_markers = MARKER.findall(candidate)
        malformed = candidate.count('__ANON_') != len(actual)
        empty = not candidate.strip()
        order_changed = not missing and not duplicated and not foreign and actual != expected
        ok = not (missing or duplicated or foreign or raw_markers or order_changed or malformed or empty)
        return {'ok': ok, 'expected_placeholders': expected, 'observed_placeholders': actual,
                'missing': missing, 'duplicated': duplicated, 'foreign_or_changed': foreign,
                'unprotected_markers': raw_markers, 'order_changed': order_changed,
                'malformed_placeholder':malformed,'empty_output':empty,
                'fallback_to_original': not ok, 'mapping': dict(self.mapping)}


def protected_diff(changes, before_view, after_view):
    return [{**item,
             'before': before_view.span(*item['before_span']) if item['before_span'] else '',
             'after': after_view.span(*item['after_span']) if item['after_span'] else ''}
            for item in changes]
