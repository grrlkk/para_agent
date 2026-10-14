# 후속 3단계 — Luna 기준선과 Claude judge 검증 준비

이 문서는 최초 Luna 실행의 기록이다. 이후 요청으로 judge 실행과 사람 라벨링을 보류했고, v0 프롬프트를 보완해 승인 후 재실행했다. 최신 결과는 [재실행 보고서](V0_RERUN_RESULTS_2026-10-15.md), 보완 내용과 견적은 [준비 보고서](V0_PROMPT_REVISION_2026-10-15.md)를 따른다. 아래에서 진행 중으로 기록했던 최초 Luna의 Kanana 보조 평가는 15/15 입력 모두 완료됐다.

이 문서는 `PROGRESS_REPORT_2026-10-14.md` 이후의 후속 단계다. 원래 구현 명세의 단계 번호와 구분한다. **3단계 진행 중이며, judge와 사람 라벨 검증 전에는 완료로 판정하지 않는다.** 4단계 동결·5단계 표본 추출은 시작하지 않았다.

## Git 저장

2단계 상태를 초기 커밋 `88ace29fdec738ad08e0f0e27d4c0815fe21d40a`로 기록했다. 작성자는 사용자 계정의 `Chanwoo Jang <grrlkk@inha.edu>`이며 공개 저장소는 [grrlkk/para_agent](https://github.com/grrlkk/para_agent)다. `.env`, `runs/`, 캐시·모델 메타데이터는 제외한다. 초기 커밋은 113개 파일이다.

## 이번 변경

- 에이전트 요약·수정·메모리 갱신을 GPT-6 Luna로 설정했다. temperature 0, reasoning none, 기존 `prev_plus_x`와 두 단위 진단을 유지한다.
- summarizer/reviser에 어휘·표현 issue의 근거 조건, 감정·양태·시제·부정·수량·주장 방향 보존, 금지 치환 예시를 추가했다. 근거가 약하면 issue 0개가 정상임을 명시했다.
- 의미 보존 판정과 양방향 쌍대 비교의 구조화된 출력·검증·비용 산정 경로를 구현했다. API 전송 내용에서 모델·조건·사람 점수·정답을 제외한다.
- 코드 종결체 분포의 인접 문단 일치율, 원문 대체율·메모리 노트 폐기율·지시 적용 불일치율·편집량을 계산한다. 비율은 분자·분모와 함께 저장하고 분모 0은 null로 기록한다.
- 가격표와 요청 전 예산 검사, 실제 응답 사용량에 따른 비용 추정을 추가했다. 비용은 표준 단가에 근거한 추정이며 청구서 확인값은 아니다.
- 사람 라벨링용 CSV·HTML 생성과 판정 지침, Cohen's kappa·클러스터 bootstrap CI, 세 검증 기준에 따른 모델 선택을 준비했다.

원본 Kanana·LoRA·자질 코드·KB는 수정하지 않는다. 에이전트 진단은 기존 greedy 경로이며 최종 보조 평가는 승인된 정수 평균 N=64와 기존 greedy를 함께 기록한다. 원래 글 ID 시드 규칙과 결과를 본 뒤 선택한 N=64의 근거는 [활성 정책](configs/evaluation_policy.yaml)에 유지했다.

## judge 설정과 통과 기준

| 후보 | 답변 전 추론 설정 | effort | 입력/출력 100만 토큰 표준 단가 |
|---|---|---|---:|
| Claude Haiku 5.5 | disabled | low | $0.10 / $0.50 |
| Claude Sonnet 5.5 | between_tools | low | $2.00 / $10.00 |

Sonnet 5.5는 disabled를 거부한다. 도구를 제공하지 않는 between_tools 요청으로 답변 전 추론을 끈다. 두 후보의 설정은 검증 전에 고정했고, 결과를 본 뒤 추론을 켜서 재시도하지 않는다. [추론 문서](https://platform.claude.com/docs/en/build-with-claude/thinking#turning-thinking-off), [가격 문서](https://platform.claude.com/docs/en/about-claude/pricing)

가장 싼 모델부터 **세 기준을 모두 통과해야 선택**한다. 둘 다 실패하면 멈춘다. 점추정치로 통과를 판정하고 95% CI를 함께 보고한다.

| 검증 | 표본 | 정답 | 기준 |
|---|---:|---|---|
| 서로 다른 원문 품질 | 같은 문항 20쌍 × 두 순서 = 40판정 | 더 높은 사람 합계 점수 | 선택률 ≥70% |
| 규칙 변형 의미 보존 | 80문장 쌍 | 변형 규칙의 라벨 | 이분류 kappa ≥0.6 |
| 실제 수정 의미 보존 | 50문장 쌍 | 사용자의 맹검 라벨 | 이분류 kappa ≥0.6 |

의미 라벨은 `보존 / 경미한 뉘앙스 변화 / 의미 변화`이고, 뒤 두 라벨을 `변화`로 합친다. 원문 품질 검증에서 동률은 오답이며 두 순서 모두 정답인 비율·순서 불일치율도 따로 저장한다. 원문 품질 검증은 같은 글의 수정본 비교 능력을 보장하지 않는다.

원문 비교는 문항 20개, 실제 수정은 글 ID 5개, 규칙 변형은 종류 8개로 묶어 bootstrap 10,000회를 사용한다. 실제 수정은 항목 단위 CI도 보조로 보고한다. 5개 글 클러스터만으로 계산한 CI와 일반화에는 한계가 있다. 규칙 변형은 문장 틀 8개에서 이름만 바꾼 사례이므로 독립적인 현상 80개로 해석하지 않는다.

## 데이터와 제외 목록

원문 쌍의 출처는 `/home/chanwoo/FEAK-Agent/data/data_jsonl/test.jsonl`이다. 원자료 JSON의 문항·원문·두 사람 채점값을 대조하고 기존 학습 분할과 겹치는 글을 제외했다. 원래 문단 경계가 있는 3~6문단 글만 사용한다.

- 20쌍은 서로 다른 문항 20개, 중복 없는 글 40편이다.
- 사람 점수 차이 사전 하한은 16점이고 실제 선택 범위는 18~45점이다. 두 채점자 모두 고득점 방향에 동의한다.
- 40편의 문단 수: 3문단 10편, 4문단 13편, 5문단 11편, 6문단 6편.
- 파일럿 10편·1.5단계 100편과 ID 및 공백 무시 내용 해시의 중복은 모두 0개다.
- 아직 5단계 표본이 없으므로 이번 40편을 [추가 제외 목록](data/splits/evaluation_exclusions_stage3.json)에 등록했다. 5단계에서 기존 110편 제외 목록과 합친다.
- 원문 점수·정답·출처·조건은 `validation/private/`에 두며 사람 시트나 judge 요청에 포함하지 않는다.

규칙 변형 80쌍은 유지 40개(띄어쓰기·맞춤법·축약·조사)와 변화 40개(부정·양태·시제·감정)다. 실제 수정 50쌍은 Luna 약 25개와 과거 Haiku를 섞고, 알려진 의미 변형 후보를 우선 포함한다. 위험 후보 선정은 사람 정답을 대신하지 않는다. 해당 표본의 보존율은 전체 Luna 기준선의 의미 보존율로 쓰지 않는다.

## 실행과 승인

작업 경로: `runs/20261014T154518.185434Z_7f79d5e0d9`.

로컬 테스트·자료 준비·견적 산정은 외부 API를 호출하지 않았다.

```bash
PYTHONDONTWRITEBYTECODE=1 /home/chanwoo/anaconda3/envs/feak_agent/bin/python -W error -m pytest -q
python scripts/prepare_stage3_validation.py --output runs/20261014T154518.185434Z_7f79d5e0d9/validation
python scripts/judge_stage3.py --data-dir runs/20261014T154518.185434Z_7f79d5e0d9/validation --model haiku --scope controlled original_pairs --output runs/20261014T154518.185434Z_7f79d5e0d9/haiku_static_cost.json
python scripts/judge_stage3.py --data-dir runs/20261014T154518.185434Z_7f79d5e0d9/validation --model sonnet --scope controlled original_pairs --output runs/20261014T154518.185434Z_7f79d5e0d9/sonnet_static_cost.json
python scripts/validate_stage3_judges.py --data-dir runs/20261014T154518.185434Z_7f79d5e0d9/validation --output runs/20261014T154518.185434Z_7f79d5e0d9/validation/pending_selection.json
```

177개 테스트가 통과한 뒤 Luna를 시작했고, 이후 저장 결과 집계 테스트까지 포함해 **178개가 통과**했다. 위 Python 명령은 모두 `feak_agent` 환경에서 실행한다. 규칙 의미 80회 + 원문 두 순서 40회 = 120회에 대한 예상 비용은 Haiku $0.0602, Sonnet $1.2051이다. 2048 출력 토큰과 보수적인 입력 추정을 모두 예약하면 각각 $0.2465, $4.9327이다. 형식 재요청까지 허용한 최대 호출 수는 360회다. 이 견적은 아직 만들어지지 않은 실제 수정 50쌍과 전체 Luna 기준선 judge 비용을 포함하지 않는다.

Luna 설정에서 공급자만 mock으로 바꾼 두 조건 실행은 `mock_runs/20261014T161143.165282Z_41fc40e080`에 저장했다. 합성 글 2편·25 mock 호출·9턴·최종 글 4개, 실제 API 0회로 완료했다. 원본 채점기·자질 코드·KB 관련 122개 파일은 작업 전 해시와 모두 같았다.

사용자가 **“Luna 실행 승인 — 예산 $0.50 이내”**로 승인했다. 예상 76회·$0.10~$0.20, 형식 재요청 포함 최대 228회다. 이 승인에 judge API는 포함하지 않는다.

```bash
python scripts/run_experiment.py --stage 3 --config configs/stage3_luna.yaml \
  --split-manifest runs/20261014T091043.217356Z_1f33a9eca6/split_manifest.json \
  --condition both \
  --output-dir runs/20261014T154518.185434Z_7f79d5e0d9/luna_runs \
  --cache-dir runs/20261014T154518.185434Z_7f79d5e0d9/luna_cache \
  --approved-budget-usd 0.50
```

새 Luna 실행 경로는 `luna_runs/20261014T160557.467829Z_60621641f5`다. 과거 Haiku의 응답 캐시와 분리한다. 로컬 Kanana와 자질 계산의 장치 설정은 기존 설정의 GPU 1을 유지했다. 실행 중 에이전트 코드·프롬프트·설정을 바꾸지 않았다.

## Luna 실행 결과와 라벨링 시트

Luna는 **76회 실제 API 호출·604.42초·표준 단가 추정 $0.04153118**로 완료했다. 입력 315,677·출력 32,896토큰이며 캐시 재사용·형식 재요청은 모두 0회다. 5편·22문단에서 최종 글 10개와 턴 로그 27개를 저장했다.

| 로컬 지표 | 전체 수정 | 문단별 수정 |
|---|---:|---:|
| 원문 대체율 | 0/5 (0%) | 0/22 (0%) |
| 메모리 노트 폐기율 | 해당 없음 (0/0) | 0/46 (0%) |
| 지시 적용 불일치율 | 9/50 (18.0%) | 5/39 (12.82%) |
| 인접 문단 종결체 일치율 | 16/16 (100%) | 16/16 (100%) |
| 평균 정규화 문자 편집 거리 | 0.03051 | 0.04197 |
| 평균 길이 비율 | 0.99533 | 1.00970 |
| API 호출 / 비용 추정 | 10 / $0.012495 | 66 / $0.029037 |

각 조건의 원문 문단 경계는 총 17개다. 이 중 분류 가능한 16개 경계만 문체 일치율의 분모에 들어간다. 최빈 종결체 일치율 100%가 모든 문장의 문체 일관성을 뜻하지 않는다. 의미 보존율과 양방향 judge 평가는 아직 미측정이다.

관찰된 실제 변경 예시:

- 자기소개 문단별 수정은 `친구가 마음 상해 할 수 있다`를 `친구가 마음 상할 수 있다`로 고쳤다. 이전 Haiku에서 문제가 됐던 같은 문단의 `두려움`, `이길 수 있다`는 이번 문단별 결과에서 유지됐다.
- 우정 글 전체 수정에서는 `많은 대화를 나눈 사이이다`를 `많은 대화를 나누는 사이이다`로 바꿨다. 시제 변화 가능성을 검증해야 하므로 프롬프트만으로 의미 보존이 해결됐다고 판단하지 않는다.
- 법·도덕 글 전체 수정에서는 문장 diff가 0개였다. 근거가 약하면 수정하지 않아도 정상이라는 규칙에 맞는 동작이며, 이 글의 의미 보존율 분모는 0으로 기록한다.

Luna의 전체 변경 쌍은 83개다. 사람 시트는 Luna 26쌍·과거 Haiku 24쌍, 총 50개이며 알려진 의미 변형 후보 2개를 포함한다. 중복된 before/after 쌍은 제외했다. 빈 라벨링 시트와 [지침](docs/HUMAN_SEMANTIC_LABELING.md)은 다음 경로에 있다.

- `runs/20261014T154518.185434Z_7f79d5e0d9/validation/human/semantic_labels.html`
- `runs/20261014T154518.185434Z_7f79d5e0d9/validation/human/semantic_labels.csv`
- `runs/20261014T154518.185434Z_7f79d5e0d9/validation/human/GUIDELINES.md`

규칙 의미 80회·원문 두 순서 40회·실제 수정 의미 50회, **모델당 170회** 검증 요청을 확정했다. 예상 비용은 Haiku $0.084861, Sonnet $1.698237이다. 형식 재요청 포함 모델당 최대 510회이며, 전체 Luna 기준선 judge 비용은 별도다. Haiku $0.50 예산, 실패 시 Sonnet $3.00 예산의 선택지를 사용자에게 제시했으며 **judge API는 승인 대기 중**이다. 사람 라벨 없이 실제 수정 kappa를 만들지 않는다.

```bash
python scripts/prepare_stage3_validation.py \
  --output runs/20261014T154518.185434Z_7f79d5e0d9/validation \
  --luna-run runs/20261014T154518.185434Z_7f79d5e0d9/luna_runs/20261014T160557.467829Z_60621641f5
python scripts/judge_stage3.py \
  --data-dir runs/20261014T154518.185434Z_7f79d5e0d9/validation \
  --model haiku --scope controlled original_pairs actual_semantics \
  --output runs/20261014T154518.185434Z_7f79d5e0d9/haiku_validation_cost.json
# Sonnet 견적도 --model sonnet, 별도 output으로 생성했다. --execute는 아직 사용하지 않았다.
python scripts/evaluate.py \
  --run runs/20261014T154518.185434Z_7f79d5e0d9/luna_runs/20261014T160557.467829Z_60621641f5 \
  --local-only --output-dir runs/20261014T154518.185434Z_7f79d5e0d9/kanana_evaluations
```

Kanana 보조 평가(원문 5개+수정본 10개, 각 N=64와 greedy)는 백그라운드에서 진행 중이다. 평가 경로는 `kanana_evaluations/20261014T161748.146207Z_f1ed7ab144`이며 추가 외부 API는 0회다. 완료 전 점수를 최종 수치로 인용하지 않는다.

백그라운드 완료 처리를 `finish_local.py`로 연결했다. 최신 상태는 작업 경로의 `local_completion_status.json`에 기록하며, 성공하면 `luna_baseline_complete_local.json`과 `LOCAL_BASELINE.md`를 자동 생성한다. 이 후속 처리는 LLM API나 Git 명령을 실행하지 않는다. 본 보고서의 실행 상태는 작성 시점의 기록이고 이후 완료 여부는 해당 상태 파일로 확인한다.

## 완료 기준과 남은 작업

코드·프롬프트·judge 고정 자료·비용 산정·통과 기준·Luna 실험·실제 수정 50쌍의 빈 시트·조건별 로컬 지표는 완료했다. Kanana N=64 보조 평가는 진행 중이다. judge API는 추가 승인 후 실행한다. 사용자 라벨을 받기 전에는 judge의 실제 수정 의미 보존 검증과 모델 선택을 완료로 기록하지 않는다. 따라서 **3단계 전체 완료 기준은 아직 미충족**이다.

애매했던 항목은 실행 전에 다음처럼 명시했다: 큰 사람 점수 차이=16점 이상, 쌍대 통과율 분모=양방향 총 40판정, 의미 보존률에서 경미한 변화도 변화, 문체 지표=인접 문단 최빈 종결체 일치, 분류 불가 경계 제외 및 비율 보고, 신뢰구간은 점추정치 통과 기준과 별도로 제시. 5단계가 아직 추출되지 않은 문제는 이번 검증 글의 선제 제외 등록으로 처리했다.

지시 적용 불일치율은 위치 문장의 변경 여부와 applied/skipped 자기 보고를 대조한 값이다. 같은 문장을 가리키는 여러 issue 중 하나만 적용돼도 다른 skipped issue에서 불일치가 기록될 수 있으므로, 이 수치를 그대로 실제 지시 위반율로 해석하지 않는다. 문체 일치율은 문단의 최빈 종결체를 비교하는 제한된 지표이며 문장 수준의 모든 문체 혼용을 나타내지는 않는다.
