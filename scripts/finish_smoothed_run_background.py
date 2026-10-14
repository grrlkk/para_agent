#!/usr/bin/env python
"""Wait locally for both stage-1.5 workers, then validate and summarize once.

This supervisor uses no model or API. It never restarts scoring, changes the
preregistered decision, modifies evaluation settings, or advances to stage 2.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_status(run, **values):
    temporary = run / "background_status.json.tmp"
    temporary.write_text(json.dumps({"updated_at": now(), **values}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(run / "background_status.json")
    print(json.dumps({"at": now(), **values}, ensure_ascii=False), flush=True)


def worker_alive(pid, gpu, run_name):
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    decoded = [x.decode(errors="replace") for x in args]
    return ("worker" in decoded and any(x.endswith("check_smoothed_scorer.py") for x in decoded)
            and any(Path(x).name == run_name for x in decoded if x)
            and "--gpu" in decoded and decoded[decoded.index("--gpu") + 1] == str(gpu))


def inspect_workers(run, pids):
    states = []
    for gpu, pid in enumerate(pids):
        directory = run / f"worker_gpu{gpu}"
        acceptance = directory / "acceptance.json"
        state = {"gpu": gpu, "pid": pid, "completed_inputs": len(list((directory / "cases").glob("*.json")))}
        if acceptance.exists():
            record = read(acceptance)
            if not record.get("ok") or record.get("completed") != record.get("assigned"):
                raise RuntimeError(f"GPU {gpu} reported unsuccessful completion; inspect {acceptance}")
            state["state"] = "complete"
        else:
            if not worker_alive(pid, gpu, run.name):
                raise RuntimeError(f"GPU {gpu} worker PID {pid} ended without successful acceptance.json")
            state["state"] = "running"
        states.append(state)
    return states


def write_summary(run):
    analysis = run / "analysis"
    acceptance = read(analysis / "acceptance.json")
    if not acceptance.get("ok"):
        raise RuntimeError("Analysis did not pass its final checks")
    decision = read(analysis / "decision.json")
    mc = read(analysis / "decision_monte_carlo.json")
    audit = read(analysis / "sampling_audit.json")
    recommendation = ("smoothed Kanana를 주 지표로 사용할 조건을 충족했습니다."
                      if decision["passes"] else
                      "조건을 모두 충족하지 못했습니다. 사전 기준에 따라 Kanana는 보조 지표로 두고, 다른 계열 judge의 쌍대 비교와 의미 보존율을 주 지표로 사용하는 쪽을 권고합니다.")
    text = f"""# 1.5단계 자동 집계 결과

완료 시각(UTC): {now()}

{recommendation}

**기존 평가 설정은 변경하지 않았습니다. 설정 변경은 사용자 승인 후에만 하며, 2단계는 시작하지 않았습니다.**

판정 대상은 사전 등록한 N=32 정수 평균, 8개 루브릭 합계입니다.

| 조건 | 결과 | 기준 |
|---|---:|---|
| LF 변형 최대 절대 변화 | {decision['lf_max_abs_total_delta']:.6f} | 1 이하 |
| 축약형 단독 변경 최대 절대 변화 | {decision['contraction_max_abs_total_delta']:.6f} | 1 이하 |
| 기존 합계 Spearman | {decision['greedy_total_spearman']:.6f} | 비교 기준 |
| smoothed 합계 Spearman | {decision['smoothed_total_spearman']:.6f} | 기존 이상 |

위 판정값의 N=32 묶음 간 SD는 LF 최대 절대차 {mc['lf_max_abs_delta_block_sd']:.6f}, 축약형 최대 절대차 {mc['contraction_max_abs_delta_block_sd']:.6f}, 합계 Spearman {mc['canonical_total_spearman_block_sd']:.6f}입니다. 각 텍스트 점수 SD와 짝지은 변형 차이 SD는 전체 수치 표에 별도로 표시했습니다. 두 묶음으로 계산한 SD이며 신뢰구간은 아닙니다. 이 오차 진단으로 판정 기준을 바꾸지 않았습니다.

- 채점 입력: {audit['unique_inputs']}개, 유효 샘플 줄: {audit['accepted_lines']}개, 형식 오류로 제외한 줄: {audit['rejected_lines']}개.
- 실제 LLM API 호출과 API 비용: 0.
- 100편은 원문 문단 구분 정보가 있는 3~6문단 글이며, 사람 원점수로 층화 추출했습니다. 주 분석은 두 줄바꿈 조립 입력, 원자료 입력은 참고 분석입니다.
- 파일럿 10편과 새 100편을 이후 평가에서 제외하도록 ID와 내용 해시를 저장했습니다.
- 점 추정치는 첫 N개를 사용합니다. 수렴 오차는 전체 64개를 8개×8묶음, 16개×4묶음, 32개×2묶음으로 나누어 계산했습니다. N=64는 한 묶음이므로 묶음 간 SD는 산출 불가입니다.
- 다른 N이나 기대값 평균이 더 좋아도 사후에 주 판정 방식을 바꾸지 않았습니다.
- 100편의 원자료·저장 문단·문항·사람 점수·테스트 소속이 모두 일치했습니다. 문단 수는 3문단 20편, 4문단 35편, 5문단 26편, 6문단 19편이고, 원점수 층화는 하·중·상 34·33·33편입니다.

시드는 기존 글 ID 규칙을 유지했습니다. `seed = int.from_bytes(SHA256(UTF8("15042:" + essay_id))[:4], "big") % 2147483647`입니다. 같은 원래 글 ID를 원문과 모든 조건의 수정본에 사용합니다. 이후 본 평가에도 [실행 설정](sampling_settings.yaml)의 같은 규칙을 적용합니다. 본문·실행 순서로 시드를 변경하지 않았고, 이미 완료된 179개 결과도 보존했습니다. 재현에는 같은 모델·LoRA·수치 연산 설정·배치 크기가 필요합니다.

[전체 수치 표](analysis/tables.md) · [변형별 루브릭 변화 CSV](analysis/invariance.csv) · [루브릭별 상관 CSV](analysis/correlations.csv) · [수렴 그림](analysis/convergence.png) · [최종 무결성 검사](analysis/acceptance.json) · [새 100편 ID](calibration_100_ids.txt) · [원자료 재검증](cohort_revalidation.json)

실행: 두 GPU의 `check_smoothed_scorer.py worker` 완료를 확인한 뒤 `scripts/summarize_smoothed_scorer.py --run {run.relative_to(ROOT)}`를 한 번 실행했습니다. 상세 출력은 `background_analysis.log`, 진행 및 오류 기록은 `background_status.json`과 `background_supervisor.log`에 있습니다.

이 파일은 저장된 결과에 대한 자동 집계입니다. 해석을 포함한 대화 보고는 다음 결과 확인 요청에서 이어갑니다.
"""
    (run / "BACKGROUND_SUMMARY.md").write_text(text, encoding="utf-8")


def main(args):
    run = args.run.resolve()
    started = time.monotonic()
    try:
        while True:
            states = inspect_workers(run, args.worker_pids)
            save_status(run, state="waiting_for_workers", workers=states, llm_api_calls=0)
            if all(x["state"] == "complete" for x in states):
                break
            if time.monotonic() - started > args.max_hours * 3600:
                raise TimeoutError("Supervisor time limit reached; workers were left untouched")
            time.sleep(args.poll_seconds)
        save_status(run, state="analyzing", workers=states, llm_api_calls=0)
        with (run / "background_analysis.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/summarize_smoothed_scorer.py"),
                                     "--run", str(run)], cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Analysis exited {result.returncode}; see background_analysis.log")
        write_summary(run)
        save_status(run, state="complete", workers=states, llm_api_calls=0,
                    summary="BACKGROUND_SUMMARY.md", evaluation_settings_changed=False, next_stage_started=False)
    except Exception as exc:
        save_status(run, state="failed", error=str(exc), llm_api_calls=0,
                    evaluation_settings_changed=False, next_stage_started=False)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--worker-pids", type=int, nargs=2, required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=8)
    sys.exit(main(parser.parse_args()))
