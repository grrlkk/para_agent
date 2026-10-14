#!/usr/bin/env python
"""Report saved v0 revision shapes and local checks without invoking any LLM."""
import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))
sys.dont_write_bytecode = True

from para_agent.artifacts import write_json, file_hash
from para_agent.eval.final_score import load_completed_run
from para_agent.eval.revision_types import summarize_types
from para_agent.eval.stage3_metrics import operational_metrics, aggregate_operational


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg, essays, finals, provenance, acceptance, hashes = load_completed_run(args.run)
    originals = {e.essay_id: '\n\n'.join(e.paragraphs) for e in essays}
    turn_files = sorted(args.run.glob('*/*/turn_*.json'))
    turns = [json.loads(p.read_text()) for p in turn_files]
    report = summarize_types(turns, originals, {key: value['final_text'] for key, value in finals.items()})
    metrics = []
    for (condition, eid), final in finals.items():
        selected = [t for t in turns if t['condition'] == condition and t['essay_id'] == eid]
        metrics.append({'essay_id': eid, 'condition': condition,
                        **operational_metrics(selected, originals[eid], final['Y'])})
    report.update(agent_model=cfg.llm.agent.model, source_run=str(args.run.resolve()),
                  source_hashes={**hashes, **{str(p): file_hash(p) for p in turn_files}},
                  local_metrics=aggregate_operational(metrics), per_essay_metrics=metrics,
                  source_acceptance=acceptance, analysis_api_calls=0)
    write_json(args.output / 'revision_types.json', report)
    names = report['type_names']
    lines = ['# 수정 유형 분포 — 코드 관측', '',
             '의미 보존이나 교정의 정답 분류가 아니다. 기존 diff의 모양과 표면 변화만 집계했다.', '',
             '| 유형 | 전체 수정 | 문단별 수정 |', '|---|---:|---:|']
    for category, label in names.items():
        counts = [report['conditions'][c]['counts'][category] for c in ('whole_once', 'incremental')]
        lines.append(f'| {label} | {counts[0]} | {counts[1]} |')
    for condition, data in report['conditions'].items():
        lines += ['', f'## {condition}', '',
                  f"변경 문장 쌍 {data['changed_sentence_pairs']}개, 원문 유지 턴 {data['unchanged_turns']}/{data['turns']}개. "
                  f"대응 없는 추가 문장 {data['unaligned_added_sentences']}/{data['original_sentences']}개 "
                  '(원문 문장 수를 분모로 사용).', '',
                  '검증 후 남은 오류 노트 유형(LLM 자기 보고): ' + json.dumps(data['accepted_memory_note_types'], ensure_ascii=False)]
        for category, examples in data['examples_up_to_five_each'].items():
            lines += ['', f'### {names[category]} — {len(examples)}개 예시', '']
            for row in examples:
                lines += [f"- {row['essay_id']} / turn {row['turn']}",
                          '  - 원문: ' + (row['before'].replace('\n', ' ') or '(없음)'),
                          '  - 수정: ' + (row['after'].replace('\n', ' ') or '(없음)')]
    lines += ['', '의미 변화·수준 상향 여부가 불명확한 편집은 미분류로 남겼다. '
              '이 분석은 본문 수정·분배·메모리 갱신에 영향을 주지 않는다.']
    (args.output / 'revision_types.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'output': str(args.output), 'analysis_api_calls': 0,
                      'counts': {c: r['counts'] for c, r in report['conditions'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
