#!/usr/bin/env python
"""Preliminary local evaluation of completed runs, with zero external LLM calls."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.dont_write_bytecode = True

from para_agent.artifacts import create_run, file_hash, write_json
from para_agent.config import project_path
from para_agent.diagnose.contracts import read_legacy_contract
from para_agent.diagnose.scorer_adapter import KananaScorerAdapter
from para_agent.eval.aggregate import aggregate, call_costs
from para_agent.eval.final_score import CONDITIONS, load_completed_run, score_essays
from para_agent.eval.kanana import AuxiliaryKananaScorer
from para_agent.eval.policy import load_evaluation_policy


def write_summary(output, source, essays, rows, statistics, acceptance, rubric_keys, rubric_names):
    index = {(r["essay_id"], r["condition"]): r for r in rows}
    lines = ["기존 파일럿 결과의 중간 평가입니다. 새 수정이나 외부 LLM 호출은 수행하지 않았습니다.",
             f"평가 대상: {len(essays)}편, 두 수정 방식. 원본 실행: `{source}`.",
             "주 지표인 다른 계열 judge의 양방향 쌍대 비교·의미 보존율은 이 로컬 평가에서 측정하지 않습니다. "
             "3단계의 judge–사람 판정 일치율 검증과 5단계 새 표본 반복 평가가 별도로 필요합니다.",
             "보조 지표: 동일한 FEAK-Agent 모델·LoRA·문항 입력의 smoothed 정수 평균 N=64. "
             "원문과 모든 조건에 동일한 글 ID 시드를 사용하며 기존 greedy 점수도 저장했습니다. "
             "N=64 선택은 1.5단계 결과를 본 뒤 승인된 결정입니다. "
             "층화 추출용 사람 원점수는 향상폭 계산에 사용하지 않았습니다.", "",
             "| 글 ID | 원문 합계 | 전체 수정 | 문단별 수정 | 전체 Δ | 문단별 Δ |",
             "|---|---:|---:|---:|---:|---:|"]
    for e in essays:
        a, w, i = [index[e.essay_id, c] for c in ("original", *CONDITIONS)]
        lines.append(f"| {e.essay_id} | {a['total']:.2f} | {w['total']:.2f} | {i['total']:.2f} | "
                     f"{w['delta_total']:+.2f} | {i['delta_total']:+.2f} |")
    lines.extend(["", "기존 greedy 점수(동일 입력; 참고 기록)",
                  "| 글 ID | 원문 | 전체 수정 | 문단별 수정 | 전체 Δ | 문단별 Δ |",
                  "|---|---:|---:|---:|---:|---:|"])
    for e in essays:
        a, w, i = [index[e.essay_id, c] for c in ('original', *CONDITIONS)]
        lines.append(f"| {e.essay_id} | {a['greedy_total']:.2f} | {w['greedy_total']:.2f} | "
                     f"{i['greedy_total']:.2f} | {w['greedy_delta_total']:+.2f} | {i['greedy_delta_total']:+.2f} |")
    lines.extend(["", "| 방식 | 평균 합계 | 평균 향상폭 | 평균 편집 거리 | 평균 길이 비율 | API 호출 | 입력/출력 토큰 | LLM 지연 합계(초) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for c in CONDITIONS:
        selected = [r for r in rows if r["condition"] == c]
        mean = lambda key: sum(r[key] for r in selected) / len(selected)
        lines.append(f"| {c} | {mean('total'):.2f} | {mean('delta_total'):+.2f} | "
                     f"{mean('normalized_edit_distance'):.3f} | {mean('length_ratio'):.3f} | "
                     f"{sum(r['real_api_calls'] for r in selected)} | "
                     f"{sum(r['input_tokens'] for r in selected)}/{sum(r['output_tokens'] for r in selected)} | "
                     f"{sum(r['llm_latency_s'] for r in selected):.1f} |")
    lines.extend(["", f"원본 실험 전체 경과 시간: {acceptance['latency_s']:.1f}초. "
                  "방식별 시간은 호출 로그의 LLM 지연 합계이며, 진단 시간을 포함한 전체 실행 시간이 아닙니다.",
                  "편집 거리 = Unicode 문자 Levenshtein 거리 / max(원문 길이, 수정본 길이). "
                  "길이 비율 = 수정본 길이 / 원문 길이. 공백·개행을 보존하여 계산합니다.", "",
                  "| 루브릭 | 원문→전체 평균 Δ | 원문→문단별 평균 Δ | 문단별−전체 평균 Δ | 95% CI | Wilcoxon p |",
                  "|---|---:|---:|---:|---|---:|"])
    names = rubric_names
    for key in (*rubric_keys, "total"):
        w = statistics["whole_once_minus_original"][key]
        i = statistics["incremental_minus_original"][key]
        d = statistics["incremental_minus_whole_once"][key]
        lines.append(f"| {names.get(key, '합계')} | {w['mean_difference']:+.3f} | "
                     f"{i['mean_difference']:+.3f} | {d['mean_difference']:+.3f} | "
                     f"[{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] | {d['wilcoxon_p_two_sided']:.4f} |")
    lines.extend(["", "각 글의 대응 차이를 재표집한 percentile bootstrap 10,000회, seed=42. "
                  "Wilcoxon은 양측 검정, 0 차이 제외, 동률 평균 순위, 작은 표본에서는 모든 부호 순열을 사용합니다. "
                  "p값은 다중 비교 보정 전 탐색 결과입니다. 모든 비교의 CI/p값은 statistics.json에 저장합니다.",
                  "5편 결과만으로 일반적인 우열을 확정할 수 없습니다. 점수는 의미 보존이나 지시 준수를 보장하지 않습니다.",
                  "LLM judge: 미실행. 현재 저장 결과의 agent는 Claude이며, 다른 계열 judge 설정이 확정되지 않았습니다.",
                  "명세상 미정이었던 편집 거리 분모와 bootstrap 설정은 위와 같이 명시했습니다. "
                  "기존 수정 결과·프롬프트·채점기·자질 코드·KB를 변경하지 않았습니다."])
    with (output / "summary.md").open("x", encoding="utf-8") as handle:
        handle.write("\n\n".join(lines[:4]) + "\n" + "\n".join(lines[4:]) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true", required=True,
                        help="Auxiliary local Kanana only; does not complete primary evaluation")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    source = args.run.resolve()
    cfg, essays, finals, provenance, acceptance, source_hashes = load_completed_run(source)
    policy_path = project_path(cfg.paths.evaluation_policy)
    policy = load_evaluation_policy(policy_path)
    contract = read_legacy_contract(cfg.scorer.package_path)
    if file_hash(project_path(cfg.paths.contracts)) != provenance["contracts_sha256"]:
        raise ValueError("The recorded rubric contract has changed")
    calls = [json.loads(line) for line in (source / "llm_calls.jsonl").read_text().splitlines()]
    original_diagnoses = [json.loads(line) for line in (source / "diagnostic_calls.jsonl").read_text().splitlines()]
    fingerprints = {r["model_fingerprint"] for r in original_diagnoses if r["component"] == "scorer"}
    if len(fingerprints) != 1:
        raise ValueError("Expected one scorer fingerprint in the source run")
    output = create_run(args.output_dir or project_path(cfg.paths.runs), PROJECT_ROOT, cfg.model_dump(), {
        "evaluation_kind": "auxiliary_local_only", "source_run": str(source),
        "evaluation_policy": policy, "evaluation_policy_sha256": file_hash(policy_path),
        "source_sha256": source_hashes, "essay_ids": [e.essay_id for e in essays],
        "evaluation_implementation_sha256": {
            str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in
            [Path(__file__), *sorted((PROJECT_ROOT / "src/para_agent/eval").glob("*.py"))]},
        "new_api_calls": 0, "judge_status": "not_run", "primary_evaluation_complete": False})
    write_json(output / 'evaluation_policy.json', policy)
    print(f"evaluation_dir: {output}", flush=True)
    native = KananaScorerAdapter(cfg.scorer, contract["rubric_keys"], log_path=output / "scorer_calls.jsonl")
    scorer = AuxiliaryKananaScorer(native, contract['rubric_keys'], policy)
    started = time.perf_counter()
    report = {"ok": False, "primary_evaluation_complete": False, "essays": len(essays),
              "evaluation_policy_version": policy['version'], "kanana_role": "auxiliary",
              "kanana_method": "integer_mean", "kanana_n": 64, "greedy_also_recorded": True,
              "judge_status": "not_run", "new_api_calls": 0, "evaluation_dir": str(output)}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        def progress(row):
            if scorer.backend.fingerprint not in fingerprints:
                raise ValueError("Scorer fingerprint differs from original experiment")
            print(f"scored: {row['essay_id']} {row['condition']} total={row['total']:.3f}", flush=True)
        rows = score_essays(essays, finals, scorer, contract["rubric_keys"], output, progress=progress)
        for row in rows:
            row.update(call_costs(calls, row["essay_id"], row["condition"]))
        stats = aggregate(rows, contract["rubric_keys"])
        write_json(output / "statistics.json", stats)
        with (output / "results.csv").open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        if any(file_hash(Path(path)) != digest for path, digest in source_hashes.items()):
            raise ValueError("A source artifact changed during evaluation")
        write_summary(output, source, essays, rows, stats, acceptance,
                      contract["rubric_keys"], contract["rubric_names_ko"])
        report.update(ok=True, scored_texts=len(rows), model_fingerprint=scorer.backend.fingerprint,
                      all_source_hashes_unchanged=True)
    finally:
        try:
            scorer.close()
        finally:
            report["latency_s"] = time.perf_counter() - started
            write_json(output / "acceptance.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

