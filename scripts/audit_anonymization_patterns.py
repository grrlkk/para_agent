#!/usr/bin/env python
"""Survey original corpus marker syntax without model calls or data changes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
from para_agent.artifacts import file_hash, write_json
from para_agent.config import load_config
from para_agent.data.loader import iter_source, source_question_essay

PATTERN = re.compile(r'#@[^#\r\n]+#')


def survey(config, output):
    cfg = load_config(config)
    records = {}
    for split in ('train', 'test'):
        path = Path(getattr(cfg.data.source, f'{split}_jsonl'))
        before = file_hash(path)
        occurrences, documents = Counter(), Counter()
        unmatched, rows, starts = [], 0, 0
        for line, row in iter_source(path):
            rows += 1
            question, essay = source_question_essay(row)
            for field, text in (('prompt', question), ('essay', essay)):
                found = list(PATTERN.finditer(text))
                occurrences.update(m.group() for m in found)
                documents.update(set(m.group() for m in found))
                known = {m.start() for m in found}
                for marker in re.finditer(r'#@', text):
                    starts += 1
                    if marker.start() not in known:
                        unmatched.append({'source_line': line, 'field': field, 'offset': marker.start()})
        assert before == file_hash(path)
        records[split] = {'source': str(path), 'source_sha256': before, 'rows': rows,
                          'marker_occurrences': dict(sorted(occurrences.items())),
                          'text_fields_containing_marker': dict(sorted(documents.items())),
                          'marker_start_count': starts, 'matched_occurrences': sum(occurrences.values()),
                          'unmatched_starts': unmatched}
    result = {'pattern': PATTERN.pattern, 'scope': 'original prompt and essay fields; train and test',
              'sources_unchanged': True, 'real_llm_api_calls': 0, 'splits': records}
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/default.yaml')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    survey(args.config, args.output)
