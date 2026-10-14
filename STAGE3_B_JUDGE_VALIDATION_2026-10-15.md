# 3단계 B — Sonnet judge 검증

이 문서는 2026-10-15 승인된 $3.00 검증 실행을 기록한다. 실행 디렉터리는 `runs/sonnet_validation_20261014T183046Z`다.

## 결과 확인 전 고정한 결정

- judge는 `claude-sonnet-5-5`로 고정했다. 모델 비교·선택을 위한 실행이 아니다.
- 실제 수정의 사람 라벨링을 수행하지 않는다. **모델 고정과 사람 라벨을 쓰지 않는 결정은 이번 검증 결과 전에 정했다.** 최초 기록 시각은 2026-10-14 18:30:46 UTC, 한국 시각 2026-10-15 03:30:46이다.
- 규칙 변형 80쌍의 보존 대 변화 kappa ≥0.6, 서로 다른 원문 20쌍을 양방향으로 비교한 40판정의 사람 고득점 글 선택률 ≥70%를 모두 요구한다. 동률은 오답이고 CI는 별도 보고한다. 하나라도 실패하면 중단한다.
- 두 기준 통과 시에만 실제 수정 50쌍의 judge 판정을 저장한다. Claude 참고 라벨과의 비교는 같은 계열 간 보조 참고이며 통과 기준이 아니다. 사람 일치율은 계산하지 않는다.
- **실제 수정에 대한 judge 판정은 사람으로 검증하지 않았다.** 정책과 상세 한계는 [평가 정책](docs/EVALUATION_POLICY.md)에 기록했다.

`decision_before_results.json`은 위 결정·승인 예산·참고 라벨 해시를 기록한다. `setup.json`은 코드·프롬프트·설정·자료·결정 파일 해시를 기록하며 제출 직전에 일치를 검사한다. 이번 검증용 고정이며 4단계 v0.1 동결은 아직 하지 않는다.

## 제출 전 자료와 비용

기존 준비 자료를 재추출하지 않고 복사했다. 규칙 변형은 유지 40쌍·변화 40쌍이며 8개 규칙 종류로 군집화한다. 원문 품질은 같은 문항의 글 20쌍이며 20개 문항으로 군집화한다. 파일럿 10편과 1.5단계 100편은 제외했고 원문 40편은 향후 5단계에서도 제외한다.

실제 수정 50쌍은 초기 Luna 26쌍·과거 Haiku 24쌍이다. 최근 Luna 재실행 결과로 다시 뽑지 않았다. 위험 사례를 포함한 선택 표본이므로 전체 의미 보존율로 해석하지 않는다. 참고 파일은 사용자가 지정한 `runs/20261014T154518.185434Z_7f79d5e0d9/validation/reference/claude_reference_labels.csv`이며 50개 ID와 before/after 본문 일치를 확인했다. 참고 Claude의 정확한 버전은 CSV에 기록돼 있지 않다.

| 단계 | 정상 요청 수 | 캐시 적중 없는 예상 비용 |
|---|---:|---:|
| 규칙 80 + 원문 양방향 40 | 120 | $0.863121 |
| 통과 후 실제 수정 — 일반 API | 50 | $0.644755 |
| 통과 시 합계 | 170 | **$1.507876** |

검증 120건은 Batch 50% 할인, 실제 수정 50쌍은 일반 API이며 두 경로 모두 1시간 시스템 프롬프트 캐시를 적용한다. 최초 계획은 모두 Batch로 $1.1854985였으나 검증 결과를 보기 전 사용자의 추가 지시로 실제 수정 50쌍만 일반 API로 바꿨다. 원래 `setup.json`·`plan.json`은 보존하고 변경 요청·새 코드 해시·수정 비용은 `execution_amendment_01.json`에 별도로 기록했다. 기존 Batch를 취소하거나 중복 제출하지 않고 같은 ID의 결과를 이어받는다. 입력은 UTF-8 바이트/2와 프로토콜 여유분, 출력은 의미 256·쌍대 640토큰으로 예상했다. 제출 전에는 더 보수적인 입력 상한과 최대 출력 2,048토큰으로 요청 금액을 확보한다. 형식 재요청 최대 2회/항목을 허용하지만 전체 예산은 항상 $3.00이다. 초기 검증 120개의 상한 확보액은 $2.740002이고, 실제 지출이 확인된 후 남은 예산으로 다음 요청을 제출한다. 캐시 적중은 보장하지 않는다. [공식 가격](https://platform.claude.com/docs/en/about-claude/pricing)

## 실행 명령과 준비 결과

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/chanwoo/anaconda3/envs/feak_agent/bin/python -m pytest -q -W error
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/chanwoo/anaconda3/envs/feak_agent/bin/python scripts/validate_sonnet_batch.py --data-dir runs/sonnet_validation_20261014T183046Z
git diff --check
```

최초 테스트 206개, 일반 API 경로와 Batch 이어받기 추가 후 **208개 통과**. API 없는 준비에서 검증 120회·조건부 실제 수정 50회와 예상 비용을 산출했다. diff 공백 검사 통과. 예산의 단계 간 공유, 제출 전 예산 확보, 비순차 Batch 결과 대응, 정답·조건 숨김, 두 기준 중 실패 시 후속 호출 차단을 테스트했다. 채점기·자질 코드·KB에는 수정하지 않았다.

승인된 실행 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/chanwoo/anaconda3/envs/feak_agent/bin/python scripts/validate_sonnet_batch.py --data-dir runs/sonnet_validation_20261014T183046Z --execute --approved-budget-usd 3.00
```

사용자 지시에 따른 실행 방식 변경 후 이어받기 명령:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/chanwoo/anaconda3/envs/feak_agent/bin/python scripts/validate_sonnet_batch.py --data-dir runs/sonnet_validation_20261014T183046Z --execute --approved-budget-usd 3.00 --resume-validation
```

## 해석이 필요했던 점

원문 품질 비교에는 수정 전 원문이 없다. 따라서 의미 보존 기준은 동률, 내용 추가 여부는 판단불가로 기록하도록 명시했다. 이때 전체 품질은 문항 충실성·내용·조직·표현으로 비교한다. **다른 원문에서 사람 점수 순서를 맞히는 성능은 같은 글의 수정본 비교를 보장하지 않는다.**

“하나라도 실패하면 멈춤”을 실제 수정 50쌍 호출보다 우선 적용한다. 현재 고정된 Sonnet의 결과가 나쁘더라도 Haiku나 다른 추론 설정으로 바꾸지 않는다. 초기 사람 라벨링 시트는 변경하지 않으며 이번 실행에 사용하지 않는다.

## 실행 결과

제출 전 기록이다. 실행 완료 또는 사전 중단 규칙 적용 후 실제 사용량·검증 수치·CI·후속 50쌍 처리 여부를 이 절에 추가한다.
