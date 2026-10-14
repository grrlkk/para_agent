# 2단계 완료 보고 — 코드 검증과 평가 정책 반영

**요청한 2단계 완료 기준을 충족했다. 여기서 멈춘다.** 기능별 테스트 155개와 두 조건의 mock 실행이 통과했다. 실제 LLM API 호출 0회, 실제 Kanana 추론 0회다. 1.5단계 추가 확인 실험과 3단계 실행은 하지 않았다.

실행 디렉터리: [20261014T145546.474849Z_a8df13a5b3](runs/20261014T145546.474849Z_a8df13a5b3). 디렉터리 이름은 UTC이며 보고서 날짜는 한국 시간이다. [자동 완료 검사](runs/20261014T145546.474849Z_a8df13a5b3/acceptance.json), [테스트 결과](runs/20261014T145546.474849Z_a8df13a5b3/pytest_results.xml)에 검증 근거를 저장했다.

## 사용한 데이터와 호출 범위

| 작업 | 사용 데이터 | 수행 범위 |
|---|---|---|
| 비식별 패턴 조사 | FEAK-Agent `data/data_jsonl/train.jsonl` 64,017편과 `test.jsonl` 8,004편 | 문항·원문에서 표시 종류·빈도만 읽음. 모델 학습·추론·LLM 전달 없음 |
| 기능별 테스트 | 짧은 합성 문장·문단, 고의로 손상시킨 mock 응답 | 네트워크 연결을 테스트 설정에서 차단. API SDK 테스트도 가짜 응답 사용 |
| 두 조건 전체 실행 | `tests/fixtures/mock_essays.jsonl` 합성 글 2편(3·4문단) | 요약·수정·메모리와 진단 모두 mock |
| 평가 정책 연결 검증 | 가짜 Kanana 반환값과 CPU 텐서 기반 가짜 생성기 | N=64·글 ID 시드·greedy 병행 기록 확인. 실제 모델 미적재 |

개발용 5편과 1.5단계의 100편은 이번에 다시 수정하거나 채점하지 않았다. 1.5단계의 100편은 FEAK-Agent의 기존 테스트 분할에 있는 **사람이 작성하고 사람 채점값이 붙은 한국어 작문 원문**이다. 원래 문단 구분이 있는 3~6문단 글에서 사람 원점수 하·중·상층 34·33·33편을 추출했다. 원자료 JSON은 `/home/chanwoo/FEAK-Agent/data/raw_datasets/test/`에 있으며, [100편 ID](runs/20261014T124844.658852Z_fee9ddb861/calibration_100_ids.txt)와 [선정 기록](runs/20261014T124844.658852Z_fee9ddb861/calibration_manifest.json)을 유지한다. 원문 100편의 문단 수는 3/4/5/6문단 각각 20/35/26/19편이다. 이 글들과 기존 파일럿 10편은 이후 5단계 표본에서 제외한다.

## 완료 기준

| 요구 기능 | 구현·검증 결과 | 충족 |
|---|---|---|
| 실제 비식별 패턴 조사 | 72,021편의 문항·원문에서 16종, 42,915회 확인. `#@` 시작 중 패턴과 불일치한 경우 0개 | 예 |
| 고유 자리표시자와 복원 | 같은 종류가 반복돼도 각 출현에 별도 토큰. 원문·이전 수정본·문항·프로필·메모리·evidence/KB 값·memory diff를 전달 전에 보호 | 예 |
| 손상 시 원문 사용·실패 기록 | 누락·철자 변경·중복·다른 문맥 토큰 복사·순서 변경·보호 안 된 표시·빈 수정본을 감지. 해당 턴 원문 사용, `failed_anonymization` 저장 | 예 |
| 문장 diff | 코드에서 원문/수정본 문장을 정렬해 `{before, after}`와 문장 ID·문자 범위·변경 종류 저장. 추가·삭제·분할·병합 테스트 | 예 |
| diff 기반 메모리 | memory_update 입력에 diff 전달. 예시가 before에 없거나 같은 변경의 after에 수정 표현이 없으면 폐기. 실제로 바뀐 구간인지 추가 확인 | 예 |
| 코드 종결체 분포 | 하십시오체/해요체/해라체/해체/기타의 문단별 개수·비율·문장별 근거 저장. Profile.register는 코드만 작성 | 예 |
| 지시 반영 대조 | issue.location의 변경 여부와 applied/skipped 대조. 불일치·없는 위치/ID·중복 보고를 로그에 기록하며 수정 결과를 바꾸지 않음 | 예 |
| 테스트와 두 조건 mock | 155 passed. 기본 `x_only`와 활성 파일럿의 `prev_plus_x` 각각 whole_once·incremental 완료 | 예 |
| 원본 보호 | 원본 채점기·자질 코드·KB 관련 122개 파일 SHA-256 모두 동일. 조사한 train/test 파일도 동일 | 예 |

주요 구현은 [비식별 보호](src/para_agent/anonymization.py), [문장 정렬과 issue 감사](src/para_agent/sentence_diff.py), [메모리 검증](src/para_agent/memory/validation.py), [종결체 분류](src/para_agent/register.py)에 있다. 프롬프트 수정은 `prompts/summarizer.md`, `reviser.md`, `memory_update.md` 파일에만 작성했다. 3단계에서 요청한 의미 변형 금지 예시 추가는 아직 진행하지 않았다.

비식별 표시의 실제 종류는 계정·계층차별·금융번호·기타차별·비속어·성차별·소속·신원·외모성격차별·이름·인종차별·전화번호·정치성향차별·주소·직업차별·출신차별이다. train에서 38,255회, test에서 4,660회였다. [조사 결과](runs/20261014T145546.474849Z_a8df13a5b3/anonymization_survey.json)에 종류별 개수·원본 파일 해시를 저장했다. 조사는 assistant 피드백 문자열을 포함하지 않는다.

## 실제 코드 동작 예시 — 합성 입력

아래는 실제 LLM의 첨삭 품질 예시가 아니라 검증용 mock 응답을 처리한 로그다.

정상 수정:

```text
원문: #@이름#는 학교를 간다. #@이름#는 남는다.
수정: #@이름#는 학교에 간다. #@이름#는 남는다.
```

두 `#@이름#`는 서로 다른 토큰으로 보호된다. 복원 뒤 diff는 첫 문장만 변경으로 기록한다. `학교를 → 학교에` 노트는 보존하고 `존재하지 않는 오류 → 수정` 노트는 버린다. 문체는 코드가 해라체로 분류한다. [정상 턴 로그](runs/20261014T145546.474849Z_a8df13a5b3/pytest_artifacts/test_both_conditions_protect_r0/incremental/protected/turn_0.json)에 저장했다.

첫 자리표시자를 삭제한 응답:

```text
mock 후보: 는 학교에 간다. #@이름#는 남는다.
최종 사용: #@이름#는 학교를 간다. #@이름#는 남는다.
상태: failed_anonymization
diff: []
```

후보를 원문으로 대체하고 다음 문단에 원문을 앞 문단으로 전달했다. 재수정 LLM 요청은 추가하지 않았다. 원문으로 되돌린 턴에서 수정했다고 주장한 오류 노트는 diff가 없으므로 저장하지 않았다. [실패 턴 로그](runs/20261014T145546.474849Z_a8df13a5b3/pytest_artifacts/test_both_conditions_protect_r2/incremental/protected/turn_0.json), [다음 문단 로그](runs/20261014T145546.474849Z_a8df13a5b3/pytest_artifacts/test_both_conditions_protect_r2/incremental/protected/turn_1.json)에서 확인할 수 있다.

## 승인된 평가 정책 변경

[evaluation_policy.yaml](configs/evaluation_policy.yaml)에 주 지표를 다른 계열 judge의 양방향 쌍대 비교·의미 보존율로 기록했다. judge와 사람 판정 표본의 일치율 확인을 **3단계 필수 작업**으로 추가했다. 실제 사람 라벨·표본 크기·선정법·판정 지침·통과 기준과 judge 모델은 3단계에서 정한다. 일치율·혼동행렬·불일치 예시를 보고하도록 계획했다. 현재는 미실행이며 검증됐다고 표시하지 않는다.

최종 평가 CLI는 별도 어댑터 [AuxiliaryKananaScorer](src/para_agent/eval/kanana.py)를 사용하도록 연결했다. `score.json`과 CSV에 N=64 정수 평균, 루브릭별 기존 greedy 점수·합계·각 향상폭, 글 ID 시드와 정책 버전을 함께 저장한다. 전체 64개 샘플과 N별 평균·표본 내부 MC SE도 보존한다. 기존 채점기가 만든 greedy 줄에 조건을 둔 기대값을 기존 greedy 점수로 기록하며, 이를 greedy 정수 합계와 혼동하지 않는다. 정수 평균 평가에는 추가 teacher-forced 기대값 계산을 하지 않는다.

**N=64 보조 지표 선택은 결과를 본 뒤 승인된 결정이다.** N=32 정수 평균은 LF·축약형 민감도를 줄였지만 합계 Spearman 비하락 조건을 통과하지 못했다(0.62683 대 0.62954). 100편 조립 원문의 정수 평균 묶음 간 SD 평균은 N=8/16/32에서 2.353/1.583/1.016이었다. 이 개선과 남은 MC 오차를 근거로 보조 지표 N=64를 선택했다. N=64의 묶음은 하나이므로 묶음 간 SD를 알 수 없고 0으로 간주하지 않는다. 원래 판정 기준을 고쳐 통과시킨 것이 아니며 추가 확인 실험도 하지 않았다.

시드 규칙은 그대로다.

```python
int.from_bytes(hashlib.sha256(f"15042:{essay_id}".encode("utf-8")).digest()[:4], "big") % (2**31 - 1)
```

원문·모든 조건·반복에 원래 글 ID를 넘긴다. 동일한 모델·수치 연산 환경·샘플링 설정을 전제로 한다. N=64는 **최종 평가의 보조 점수 경로**에 적용했으며 에이전트 내부 진단에 쓰는 기존 호출은 유지했다. 원래 1.5단계 프로토콜의 N=32와 저장 결과는 보존했다. [이후 단계 계획](EVALUATION_PLAN_POST_STAGE1_5.md)에 동결·새 표본 평가까지의 순서를 정리했다.

## 실행한 주요 명령과 결과

작업 디렉터리는 `/home/chanwoo/Feak_next/para_agent`, Python은 `/home/chanwoo/anaconda3/envs/feak_agent/bin/python`이다. 아래 `R`은 작업용 변수이며 기존 환경 변수를 변경하지 않는다.

```bash
R=runs/20261014T145546.474849Z_a8df13a5b3
PY=/home/chanwoo/anaconda3/envs/feak_agent/bin/python

PYTHONDONTWRITEBYTECODE=1 "$PY" scripts/audit_anonymization_patterns.py --output "$R/anonymization_survey.json"

PYTHONDONTWRITEBYTECODE=1 "$PY" -W error -m pytest -q \
  --basetemp "$R/pytest_artifacts" --junitxml "$R/pytest_results.xml"

CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" scripts/run_experiment.py --mock --condition both \
  --output-dir "$R/mock_default_runs" --cache-dir "$R/mock_default_cache"

CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 \
  "$PY" scripts/run_experiment.py --mock --condition both --config "$R/mock_prev_plus_x.yaml" \
  --output-dir "$R/mock_prev_runs" --cache-dir "$R/mock_prev_cache"

PYTHONDONTWRITEBYTECODE=1 "$PY" "$R/audit_stage2_artifacts.py"
```

| 명령 | 결과 |
|---|---|
| 비식별 조사 | 72,021편, 16종, 42,915회, 미일치 시작 패턴 0개, 데이터 해시 보존 |
| 최종 pytest | **155 passed in 5.32s**, 실패·오류·건너뜀 0개 |
| 기본 설정 mock | 합성 2편·두 조건·9턴·최종 글 4개, mock 호출 25회, API 0회, 약 0.067초 |
| prev_plus_x mock | 동일한 실행 수, mock 호출 25회, API 0회, 약 0.067초 |
| 아티팩트 검사 | 통과. 원본 파일 122개·train/test 해시 동일. 정상/손상 응답 통합 테스트 8턴 중 의도한 실패 2턴의 원문 대체 확인 |

`mock_prev_plus_x.yaml`은 활성 `pilot_haiku.yaml`의 입력 방식·진단 설정을 유지하고 agent/judge 공급자를 mock으로 바꿔 만들었다. 실제 `.env`는 읽지 않았다. mock 두 실행은 [기본 실행](runs/20261014T145546.474849Z_a8df13a5b3/mock_default_runs/20261014T152201.854547Z_6863ccde7c/acceptance.json), [prev_plus_x 실행](runs/20261014T145546.474849Z_a8df13a5b3/mock_prev_runs/20261014T152201.857189Z_ba4da380e4/acceptance.json)에 있다.

초기 회귀 검사에서는 89개 통과·2개 실패였다. 빈 수정본을 로컬 원문 대체로 처리하도록 바뀐 스키마에 맞춰 기존 테스트를 갱신했고, 짧은 원문 수정 뒤 문장 여러 개를 추가하면 잘못된 문장에 정렬되는 문제를 고쳤다. 추가·분할 비용을 구분해 수정 위치를 보존하는 회귀 테스트를 넣은 뒤 전체 155개가 통과했다. 최종 아티팩트 검사에서는 시스템 지시문에 있는 일반 패턴 예시 `#@...#`까지 원자료 누출로 오인한 검사 조건을 데이터 payload 대상으로 바로잡았다. 실제 원자료 표시는 보호돼 있었다.

출력 파일은 기존 결과를 덮어쓰지 않도록 생성 전용으로 저장한다. 이 명령들은 실행 기록이며 재실행 시에는 새 출력 디렉터리를 사용해야 한다. 특히 pytest의 `--basetemp`에는 기존 보존 디렉터리를 재사용하지 않는다.

## 애매했던 점과 구현 범위

- **문장 경계·정렬:** 종결 부호와 줄바꿈을 사용한 결정적 규칙이며 형태소 분석 기반 완전한 문장 분석은 아니다. 문장 추가·삭제와 1:2/2:1 정렬을 지원한다. 반복 문장·대규모 재구성에는 정렬이 애매할 수 있다. 문장 앞뒤의 구분 공백만 바뀐 경우 diff는 비고 `boundary_only_change`로 기록한다.
- **종결체 판정:** 접미형 규칙으로 분류하고 각 문장의 적용 규칙을 로그에 남긴다. 인용문만 있는 문장·호칭·분류 불가는 기타, 대표 종결체가 동률이면 기타로 둔다. 처음 확인된 수정 문단의 우세 종결체를 Profile.register에 넣고 이후 문단의 일시적 문체 변경으로 덮어쓰지 않는다. 형태가 모호한 단어와 문장 내부 직접 인용까지 완전히 해석하는 분류기는 아니다.
- **LLM 프로필 쓰기:** 문체 분류 필드 register는 LLM 응답 스키마에서 제거했다. 문체에 관한 LLM 작성 필드는 style_notes만 남기고, 기존 내용 필드 topic·thesis는 유지했다.
- **원문 대체의 범위:** incremental에서는 해당 문단만 원문으로 대체한다. whole_once는 글 전체가 한 수정 턴이므로 전체 원문으로 대체한다. 후보·실패 이유를 모두 남긴다. 빈 응답도 같은 대체 경로를 사용하며 추가 재수정 호출은 없다.
- **자리표시자 순서:** 누락·변경 외에 중복·다른 문맥의 토큰 복사·순서 뒤바뀜도 보존 실패로 처리했다. 모든 동일 종류 표시를 출현별로 구별한다. 보호된 내용을 재배치해 인물 등의 대응이 달라지는 경우를 허용하지 않는 보수적인 처리다.
- **메모리 검증:** example이 diff.before에 있는지 확인하는 요구에 더해, fix가 같은 diff.after에 있는지와 인용 구간이 실제 편집 구간에 닿는지도 검사한다. 이는 문자열 근거 검사이며 해당 첨삭의 문법적 정당성이나 의미 보존까지 보증하지 않는다.
- **지시 적용 감사:** 문장이 바뀌었다는 사실만 확인한다. 그 issue 때문에 바뀌었는지, 실제로 지시를 충족했는지는 이 로그로 확정하지 않는다. 같은 문장을 가리키는 다른 issue 때문에 skipped 문장이 변경될 수도 있다. 결과를 거절하거나 추가 호출하지 않는다.
- **이후 확인 사항:** judge 모델과 사람 판정 표본의 구체 조건은 3단계에서 확인해야 한다. 의미 보존율·문체 일치율·사람 일치율의 실제 수치와 첨삭 품질 개선은 아직 측정하지 않았다. 현재 Git 저장소가 아니므로 4단계 동결 시 커밋 기록을 위한 상태 정리가 필요하다. 이번에는 저장소를 만들거나 커밋하지 않았다.

2단계 코드는 검증됐지만 실제 수정 품질이 개선됐다는 결론은 내리지 않는다. 3단계는 실제 API 호출 수·비용을 먼저 보고하고 사용자 승인을 받은 뒤 진행한다.
