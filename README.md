# para_agent — v0 / 3단계

기준은 [IMPLEMENTATION_SPEC.md](../imple/IMPLEMENTATION_SPEC.md)다. **0단계 mock 실행, 1단계 실제 진단 어댑터, 2단계 테스트 파일럿 추출, 3단계 실제 LLM 수정 경로**를 구현했다. 3단계는 지정된 파일럿 5편에서만 실행하며 평가는 4단계다.

진행 상황, 실제 수정 전후 사례, 평가 결과와 남은 문제는 [2026-10-14 진행 보고서](PROGRESS_REPORT_2026-10-14.md)에 정리했다.

후속 1단계의 점수 하락·표기 민감도 분석은 [Kanana 민감도 보고서](SCORER_SENSITIVITY_STAGE1_2026-10-14.md)에 정리했다. 기존 5편은 개발용이며, 향후 새 표본 평가에서는 기존 파일럿 10편 전체를 제외한다.

## 실행

```bash
cd /home/chanwoo/Feak_next/para_agent
python scripts/run_experiment.py --mock
python scripts/validate_kb.py
python scripts/prepare_split.py
python -W error -m pytest -q
# 실제 로컬 채점기와 자질 추출기 사용; LLM API 호출 없음
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/check_diagnosis.py
```

현재 환경에는 의존성이 설치되어 있다. 다른 환경은 `python -m pip install -e '.[test]'`로 설치할 수 있다. Python 3.10 이상을 사용한다.

`--mock`은 기본으로 `whole_once`, `incremental` 두 조건을 실행한다. `--condition incremental` 또는 `--condition whole_once`로 한 조건만 실행할 수 있다. config의 `condition` 기본값은 명세대로 `incremental`로 보존하고, CLI가 실행한 조건 목록은 스냅샷의 `provenance.conditions`에 기록한다.

입력은 `tests/fixtures/mock_essays.jsonl`의 직접 작성한 합성 글 2편(3·4문단)이다. 실제 테스트 분할이나 파일럿 10편으로 간주하지 않는다. mock은 issue 없는 고정 Brief, 원문을 그대로 돌려주는 Reviser, 고정 메모리 갱신 응답을 사용한다. 8개 점수는 모두 고정값 5.0, 실제 자질값은 모두 `None`이다. 품질·성능 측정값이 아니다.

`--mock`은 기존 합성 입력과 가짜 진단·LLM만 사용한다. 실제 실행에는 `--stage 3`, 실제 모델 설정, 검증된 2단계 split manifest가 필요하다. 파일럿 10편의 순서가 유지된 앞 5편만 읽으며, 원자료·split 해시와 원문 문단을 확인한다. judge는 이 단계에서 호출하지 않는다. 11절의 검증·재수정 루프 등은 포함하지 않는다. 구조화 출력의 파싱 재요청은 4.1절에 따른 별도 처리다.

## 설정과 결과

- `configs/default.yaml`: `para`·`cum`·`both`, `prev`·`cum`, `x_only`·`prev_plus_x`, 메모리 켜기/끄기를 mock으로 실행할 수 있다. 사용자의 후속 지시로 진단 기본값은 `both`다.
- `configs/contracts.yaml`: 기존 코드에서 읽은 실제 루브릭 8개·자질 29개와 출처 해시.
- `configs/feature_key_map.yaml`: 원본 KB 키 → 실제 자질 키의 명시적 대응. KB 내용을 수정하거나 출력 키를 바꾸지 않는다.
- `configs/rubric_diag_map.yaml`: 실제 루브릭 기준 초안. 전문가 확인은 아직 하지 않았다.
- `configs/rubric_unit_map.yaml`: 현재 사용하지 않는다. 두 단위 값을 확인한 뒤 선택형 hybrid가 필요할 때만 결정한다.
- `prompts/`: 명세 13절의 네 프롬프트와 파싱 오류 재요청 프롬프트. 두 조건이 같은 파일을 읽는다.

매 실행은 새로운 `runs/{run_id}`를 만들고 기존 결과를 덮어쓰지 않는다. config 스냅샷, 프롬프트·입력·KB·계약·매핑 해시, `llm_calls.jsonl`, 조건/글별 `turn_{t}.json`·`final.json`을 저장한다. 작업 폴더는 Git 저장소가 아니므로 `git_commit: null`, `git_status: not_a_git_repository`를 명시한다. 저장소를 초기화하거나 임의의 커밋 해시를 만들지 않는다.

`--output-dir`·`--cache-dir`로 결과·캐시 경로를 바꿀 수 있다. 캐시는 모델·메시지·파라미터(스키마·mock 응답 해시 포함)로 구분하며, 캐시에서 읽은 호출도 로그를 남긴다. mock 토큰 수와 실제 API 호출 수는 0이다.

메모리 끄기는 매 턴 새 메모리와 `Profile(prompt=문항)`을 사용한다. 메모리 갱신 호출·응답은 로그에 남기되 다음 턴과 최종 상태에 반영하지 않는다. 앞 수정본과 마지막 두 문장은 메모리 스위치와 별도로 전달한다. `whole_once`는 글 전체 진단·요약·수정을 한 번 실행하고, issue 상한은 `k × 문단 수`, 메모리 갱신 호출은 없다.

사용자의 “일단 다 진단하고 나중에 값 보고 판단” 지시에 따라 `both`에서는 매 턴 이번 원문 단독(`para`)과 앞 수정본들 + 이번 원문(`cum`)을 모두 진단한다. `turn_*.json`의 `diag.para`·`diag.cum`과 메모리에 두 결과를 그대로 저장하고, 요약 에이전트에는 `evidence.para`·`evidence.cum`을 전달한다. 루브릭별 점수를 미리 선택하거나 평균 내지 않는다. 약한 루브릭과 Δ도 단위별로 따로 계산하므로, 문단 점수에서 누적 점수를 빼는 일이 없다. 개별 `Diagnosis.unit`은 명세의 `para | cum`을 유지한다. 이는 명세 4.2절의 사전 선택형 hybrid를 당장 구현하는 대신, 후속 사용자 지시를 반영한 변경이다. 기존 자질 함수는 29개를 일괄 반환하므로 어댑터에서 누적 전용 8개를 문단·단일 문단 누적 진단의 경우 `None`으로 제외한다. 두 문단 이상 누적할 때만 그 값을 사용한다. 계산 불가·평균 대치·비유한 값도 `None`으로 둔다.

실제 LLM 키는 `llm.env_file`이 가리키는 `.env`에서 읽으며 키를 config나 로그에 복사하지 않는다. 실제 파일럿 설정은 기존 `/home/chanwoo/FEAK-Agent/.env`를 읽기만 한다. OpenAI는 `OPENAI_API_KEY`, Claude는 `ANTHROPIC_API_KEY`를 사용한다. 다른 환경에서는 프로젝트 `.env`에 키를 넣고 경로를 바꿀 수 있다. `.gitignore`가 `.env`, 실행 결과, 캐시를 제외한다. mock은 `.env`를 읽지 않고 키가 필요 없다.

## 기존 코드 확인 및 후속 결정

아래 파일들은 읽기만 했으며 수정하지 않았다.

| 용도 | 확인한 기존 파일 / 실행 진입점 |
|---|---|
| 실제 키·이름·순서 | `/home/chanwoo/essay_scoring_llm/essay_scoring_llm/rubrics.py` |
| 29자질 호출 | `essay_scoring_llm.feak.extract_feak_features(text)` → 실제 키의 `dict[str, float]` |
| 자질 출처·계산 불가 확인 | `essay_scoring_llm/feak_feature/feak_feature.py`: `extract_features(text, explain=True)`에 `imputed`, `source` 포함 |
| 자질 순서 | `essay_scoring_llm/feak_feature/korcat_core/apps/cohesion/essay_scoring/features/scaler_notlabel.csv` |
| 채점 CLI | 해당 저장소의 `python -m essay_scoring_llm.cli single --essay ...` |
| 채점기 원시 출력 | `essay_scoring_llm/soft_sc.py`: `raw_scores`(8개 점수 배열의 목록), `feedbacks`(문자열 목록) |
| 기존 보정 출력 | `essay_scoring_llm/schema.py`: `correction.rf_corrected_score`, `correction.final_score`; 8개 배열이며 루브릭 순서는 위 상수와 동일 |
| 기존 통합 진단 | `/home/chanwoo/FEAK-Agent/feak_tc/diagnose/kanana.py`: `rubrics`, `features`, `metadata.feedback` |
| 기존 결정적 채점 경로 | `/home/chanwoo/FEAK-Agent/verak/v3/score/kanana.py`: `KananaScorer.score(question, text)` → `ScoreResult(expected, integers, mean, genre, cache_hit, input_tokens, score_line, cache_key)` |

KB는 실제로 `../imple/features.jsonl`에 제공되어 있어 지정 경로로 바이트 그대로 복사했다. 원본 SHA-256은 `7338b9216b8dd3246d6491abd343cc264556bd51474755cfe4c28040031622cc`다. 25개 code가 실제 출력 키와 달랐고, 사용자가 출력에서 섹션 접두어를 제거했다고 확인했다. 별도 매핑으로 29개 모두 일대일 대응시켰다. `validate_kb.py`는 실제 `rubrics.py`와 스케일러 CSV의 자질 키를 읽어 29개 대응을 확인하며, 원문 직접 일치는 4개임을 별도로 보고한다. 임의의 접두어 제거·KB 수정은 하지 않는다.

채점 경로와 진단 범위는 사용자 지시로 확정했고, 다음과 같이 구현했다.

1. **채점 출력 확정**: 사용자의 “FEAK-Agent에서 사용한 것을 그대로 사용” 지시에 따라 `verak.v3.score.kanana.KananaScorer.score(question, text)`의 `ScoreResult.expected`를 사용한다. 실제 호출은 FEAK-Agent v5의 `ReferenceScores`에서 확인했다(`FEAK-Agent/verak/v5/outputs/orchestrator1/code_snapshot/verak/v5/common.py`, 최근 recovery1·selection1 작업 폴더의 같은 파일도 동일). 모델·LoRA·greedy 생성·기대점수 계산을 유지하고 어댑터로만 감싼다. 해당 경로는 루브릭별 근거를 반환하지 않으므로 `rubric_rationales`는 `None`이다. 새 캐시는 프로젝트의 `.cache/scorer/score_cache.sqlite`에 저장한다. 장르 목록은 기존 165개 항목을 보존하고 파일럿에 필요한 항목만 기존 규칙으로 보완한 프로젝트 사본을 사용한다. 원본 장르 파일과 채점 설정 파일은 수정하지 않는다. 보완되지 않은 미등록 문항은 원래 채점기처럼 오류로 처리하며 장르 생성을 위한 API를 호출하지 않는다.
2. **진단 범위 확정**: 먼저 문단·누적을 모두 진단하고 값을 보겠다는 사용자 지시를 `diag_unit: both`로 반영했다. 두 결과를 분리 저장하므로 합쳐진 진단의 unit을 정하거나 루브릭별 단위표를 미리 확정할 필요가 없다. 실제 진단에서도 두 결과를 그대로 반환한다.
3. **계산 불가 처리**: 기존 29자질 추출기는 결측값을 스케일러 평균으로 채운다. 어댑터가 기존 `explain=True`의 `imputed` 정보를 확인하여 대치된 값은 `None`으로 되돌린다. 누락 사유·원래 출처를 진단 호출 로그에 남긴다. 원시 추출기의 오류는 실패로 전달하며 임의의 수치로 대체하지 않는다.
4. **초안 검토**: 실제 루브릭에 맞춘 진단 항목 연결은 초안이며, 의미상 적절성에 대한 전문가 확인이 필요하다. `cum` Δ에서 결측 이력은 평균에서 제외하고, 이전 유효값이 없으면 `None`으로 처리한다.

## 1단계 실제 진단 검증

`scripts/check_diagnosis.py`는 `tests/fixtures/diagnosis_essay.json`의 합성 3문단 글로 문단·단일 문단 누적·두 문단 누적·전체 글을 각각 두 번 진단한다. 반복 실행은 채점 캐시를 끄고 자질도 다시 계산하며 전체 `Diagnosis`가 정확히 같은지 검사한다. 이어서 `both`의 두 결과, 기존 채점기 캐시 재사용, 길이 초과 거부, 기존 소스·KB 해시 불변을 검사한다. 단위 테스트는 GPU와 네트워크 없이 어댑터의 출력 순서·결측·범위·오류 처리를 검사한다.

채점기는 기존 설정의 GPU 1번을 사용한다. 자질 모델은 별도 프로세스에서 GPU 1번을 선택하며(`features.gpu`), 원래 추출 코드는 수정하지 않는다. 모델 파일은 로컬 캐시에서만 읽고, 형태소 분석은 기존 Bareun 서버를 사용한다. 키는 `features.env_file`이 가리키는 기존 `.env`에서 읽는다. 자질 프로세스는 재사용하고 종료 시 정리하며, 기존 추출기의 네이티브 라이브러리 동시 실행을 피하도록 동일한 파일 잠금을 사용한다.

채점 입력 상한은 기존 설정의 3,072토큰이다. 원래 채점기는 `truncation=False`로 토큰화하고 초과하면 `InputTooLong`으로 거부한다. 어댑터는 원문을 자르지 않고 실제 토큰 수·상한·`truncated: false`를 로그에 남긴다. 원래 토크나이저에는 별도로 2,560이라는 경고 기준이 있으나, 기존 채점 경로의 설정을 변경하지 않는다.

결과는 매번 새로운 `runs/{run_id}` 아래에 config·입력·기존 소스 해시·`diagnostic_calls.jsonl`·단위별 `diagnosis.json`·`kb_validation.json`·`acceptance.json`으로 저장한다. LLM 호출 로그는 빈 파일이다. 이 합성 글 검증은 모델 연결과 반복 가능성을 확인하며 실제 파일럿의 품질 평가가 아니다.

## 2단계 실제 데이터 준비

`python scripts/prepare_split.py`는 `configs/default.yaml`의 `data.source`에 명시된 테스트 분할에서 파일럿 10편을 추출한다. `--n-essays 200`으로 명세의 본 실험 크기도 지정할 수 있지만, 현재는 10편만 생성했다. 추출 과정에서 모델 추론·LLM API는 호출하지 않는다.

- 소스는 `/home/chanwoo/FEAK-Agent/data/data_jsonl/test.jsonl`의 8,004편이다. 채점기 학습용 `train.jsonl` 64,017편과의 구분은 기존 운영 결정 기록 및 `verak/v3/data/audit_index.json`에서 확인하고, 실제 파일 해시·행 수·학습 글 해시를 다시 검증한다.
- 원본은 `data/raw_datasets/test/*.json`의 `essay_question.prompt`, `essay_answer.text`, `score.personal.analytic`이다. 테스트 JSONL의 문항·글·두 채점자의 점수가 원본과 정확히 대응해야 후보로 사용한다. 원본 파일명이 `essay_id`다.
- 원문 CRLF/CR/LF 줄바꿈을 문단 경계로 사용한다. 빈 줄은 구분자로만 처리하고 각 문단 내부의 글자·공백은 보존한다. 빈 줄 없는 단일 줄바꿈도 원문에 존재하는 경계로 사용한다. 문장 수·길이에 따른 재분할은 하지 않으며, 문단 경계가 없거나 3~6문단 조건을 벗어나면 제외한다.
- 학습 글과의 정확한 중복뿐 아니라 공백·줄바꿈을 제외했을 때 같은 글도 제외한다. 테스트 후보 내부의 중복 글도 한 번만 사용한다. 모든 제외 행과 사유는 `excluded.jsonl`에 남긴다.
- `original_scores`는 기존 `essay_scoring_llm.scaling.label_from_raters`와 같은 **채점자 1 + 채점자 2 − 1**의 1~9점 원점수다. 루브릭 8개 합계의 1/3·2/3 분위수를 적격 후보 풀에서 계산한다. 이는 표본 층화를 위한 사람 점수이며 Kanana 원문 예측값이 아니다. 4단계 향상폭 계산에는 원문도 Kanana로 별도 채점해야 한다.
- 분위수는 선형 보간(`statistics.quantiles(method="inclusive")`)으로 계산하고 동점은 같은 층에 둔다. 10편의 할당은 하위 4·중위 3·상위 3, 시드는 42다. 각 층에서 중복 없이 뽑고 하·중·상 순으로 교차 배치하여 앞 5편은 2·2·1이 된다. 입력 순서를 바꿔도 같은 표본이 나오는지 검사한다.

현재 실행에서 원문 3~6문단 후보는 1,897편, 중복 제외 후 적격 후보는 1,844편이다. 전체 테스트 분할의 학습 중복 61편을 제외했고 적격 후보 내 테스트 중복은 1편이었다. 원점수 경계는 합계 45·50점, 층별 후보는 693·544·607편이다. 파일럿 문단 수는 **3문단 4편, 4문단 4편, 5문단 1편, 6문단 1편**이다.

각 실행은 `data/splits/pilot_{n}_{run_id}.jsonl`과 `.manifest.json`, `runs/{run_id}/config.yaml`, `split_manifest.json`, `excluded.jsonl`을 새로 만든다. 레코드에는 명세의 `essay_id`, `prompt`, `paragraphs`, `original_scores`만 넣는다. 별도 manifest에는 소스·결과 해시, 원자료 위치·행 번호·점수, 문단 정책, 층화 기준·시드·분포를 기록한다. 원본 데이터는 수정하지 않는다.

최초 파일럿: `data/splits/pilot_10_20261014T091043.217356Z_1f33a9eca6.jsonl`. 10편 모두 원문 문단 내용이 정확히 보존됨을 별도로 대조했다. 단위 테스트 52개가 통과했다.

선택한 10편 중 5편의 문항이 기존 채점기의 장르 목록에 없어, 사용자 승인에 따라 프로젝트 전용 사본을 보완했다. 기존 `verak.v3.data_policy.feedback_headings`와 `HEADING_SETS`를 그대로 호출하여 해당 문항들의 테스트 원자료 전체에서 루브릭 이름 조합이 일치하는지 확인했다. 누락 5개는 설명 3개·정서 2개이며, 기존 165개 항목은 값까지 그대로 보존한다. 판별이 안 되거나 기존 장르와 충돌하면 임의 추정 없이 실패한다. 문항 문구를 고치거나 표본을 다시 뽑지 않았다.

장르 목록을 재생성하고 연결을 검증하는 명령:

```bash
python scripts/prepare_scorer_metadata.py --split-manifest runs/20261014T091043.217356Z_1f33a9eca6/split_manifest.json
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/check_scorer_split.py --split-manifest runs/20261014T091043.217356Z_1f33a9eca6/split_manifest.json
```

현재 사본은 `data/scorer_metadata/d905597c30a3f943/genres.json`이며 원문·규칙·입력 해시와 판별 근거는 같은 디렉터리의 `provenance.json`에 기록했다. `scorer.metadata_dir`가 이 사본을 가리킨다. `null`이면 기존 FEAK-Agent 장르 디렉터리를 그대로 사용한다. 어댑터는 사본의 원본 해시와 기존 항목 보존 여부를 확인하고 실제 사용한 장르 파일 경로·해시를 진단 호출 로그에 남긴다. 모델·LoRA·추론 설정은 그대로 유지된다.

`check_scorer_split.py`는 준비한 10편의 원문을 기존 Kanana로 채점하여 연결만 검증한다. 수정·요약·메모리 갱신 및 외부 LLM API는 호출하지 않는다. 출력 점수는 연결 검증 기록이며 두 실험 조건의 수정 효과 평가가 아니다. 단위 테스트는 56개가 통과했다.

연결 검증 결과는 `runs/20261014T092642.488417Z_4b45e84b8d/acceptance.json`에 저장했다. GPU 1번에서 10편 모두 루브릭 점수 8개를 반환했고 입력은 328~1,533토큰이었다. 기존 코드·KB 122개 파일과 원본 장르 파일의 해시가 유지됨을 확인했다. 2단계 완료 상태이며 실제 LLM API 호출은 0회다.

## 3단계 실제 LLM 파일럿

현재 선택 모델은 **Claude Haiku 5.5 (`claude-haiku-5-5`)**다. 사용자가 OpenAI 비용 때문에 Claude API로 변경을 요청했고, Anthropic API 키·잔액이 있음을 확인했다. 요약·수정·메모리 갱신 모두 같은 Claude 모델을 사용한다. `configs/pilot_haiku.yaml`은 `temperature: null`(전송 생략), `reasoning_effort: none`(Claude의 `thinking: {type: disabled}`), `max_output_tokens: 8192`, 표준 처리(`service_tier: standard_only`)를 사용한다.

**승인된 명세 차이:** 명세 9절의 에이전트 `temperature: 0`을 Haiku 5.5 요청에서는 생략한다. 해당 모델은 0을 지원하지 않아 400 오류를 반환한다는 [공식 호환성 문서](https://platform.claude.com/docs/en/models/haiku-5-5/migration-guide)를 확인한 뒤, 변경 전에 사용자에게 물었고 사용자가 승인했다. Kanana의 결정적 채점 설정은 그대로다. 모델 ID·가격은 [공식 모델 페이지](https://platform.claude.com/docs/en/models/haiku-5-5/overview), 계정 접근·thinking disabled·구조화 출력 지원은 Models API 조회로 확인했다. 모델 조회는 생성 호출이 아니다.

```bash
# 다른 환경에서만 필요: python -m pip install -e '.[test,live]'
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/run_experiment.py --stage 3 --config configs/pilot_haiku.yaml --split-manifest runs/20261014T091043.217356Z_1f33a9eca6/split_manifest.json
```

Claude Messages API의 `output_config.format`에 JSON Schema를 전달하고 Pydantic으로 검증한다. Anthropic SDK의 `transform_schema`가 서버에서 지원하지 않는 길이·숫자 제약을 필드 설명으로 옮기지만, 로컬 검증에는 원래 제약이 모두 남는다. 메모리 갱신 프롬프트는 바뀌지 않은 프로필 값을 유지하도록 한다. 파싱·스키마 실패에만 오류 메시지를 붙여 최대 2회 재요청한다. API 거절·출력 잘림·통신 오류는 기록 후 실패로 처리하며 자동으로 모델이나 파라미터를 바꾸지 않는다. SDK의 숨은 재시도도 끈다. 입력을 자르거나 도구를 호출하지 않는다.

OpenAI Responses 어댑터와 `configs/pilot_luna.yaml`도 남아 있다. Luna 설정은 `temperature: 0`, `reasoning_effort: none`, 출력 상한 8,192토큰, `store: false`다. 중단된 Luna 실행 `runs/20261014T094110.078816Z_7d304b2bcb`에는 54회 실제 API 요청(정상 52·형식 오류 2), 최종 글 8개, 턴 로그 19개가 남아 있다. 사용자의 변경 요청 직후 SIGINT로 실행을 종료했고, GPU 프로세스 정리를 확인했다. `acceptance.json`은 `ok: false`이며 `interruption.json`에 이유를 기록했다. 중단된 실행을 3단계 완료 결과로 취급하지 않는다. Claude 결과는 같은 파일럿 5편의 두 조건을 새 run에서 실행하여 모델이 섞이지 않게 한다.

다섯 편은 원래 6·4·3·4·5문단, **5편의 합계가 22문단**이다. 두 조건 실행은 `whole_once` 10회와 `incremental` 66회, 총 76회의 논리 호출이다. 모두 세 번씩 스키마 재요청하더라도 한 실행의 발송 상한은 228회다. 각 호출은 공급자·모델·메시지·실제 전송 파라미터로 캐시하며, 캐시를 재사용하면 토큰 비용을 0으로 기록하고 최초 응답 메타데이터를 별도 보존한다. 각 API 호출의 응답 모델·ID·상태·원문 출력·토큰 상세·지연 시간과 프롬프트 해시를 저장한다. 4단계 비용 집계에서는 `tokens`를 합하고, 재사용 응답의 `provider_response.usage`를 다시 더하지 않는다. Claude 원시 `input_tokens`는 캐시 읽기·쓰기 토큰을 제외하므로 공통 `tokens.input_tokens`는 이들을 포함한 총 입력으로 정규화한다. 원시 사용량은 `provider_response.usage`에 그대로 남는다.

실제 진단은 기존 Kanana와 29자질 추출기를 GPU 1번에서 사용한다. 매 문단의 `para`와 `cum`을 분리 저장하고, 앞 수정본을 다음 문단의 prefix로 사용한다. 두 조건 모두 동일한 프롬프트 파일과 같은 모델·생성 설정을 쓴다. `whole_once`는 전체 글을 한 번 수정하고 메모리를 갱신하지 않는다. 각 실행의 `acceptance.json`에는 완료 여부, 호출·재요청·캐시·토큰 수, 턴 로그와 최종 글 수를 남긴다. 중간에 실패해도 기존 실행을 덮어쓰지 않으며, 같은 명령을 다시 실행하면 새 run과 기존 성공 응답 캐시를 사용한다.

단위 테스트는 외부 연결을 차단한 상태에서 실제 전송 파라미터, JSON 스키마, 캐시와 비용 기록, 잘린 응답·거절·API 오류, 5편 제한과 자료 변조 거부, 두 조건 실행을 검사한다. 현재 `eval/`과 `scripts/evaluate.py`에는 아래의 로컬 중간 평가가 구현되어 있으며, 다른 계열 LLM judge는 아직 골격이다.

3단계 완료 실행은 `runs/20261014T095651.042158Z_3931618173`이다. 파일럿 5편·두 조건의 최종 글 10개와 턴 로그 27개가 저장됐다. Claude 실제 요청은 77회(정상 논리 호출 76회 + 재요청 1회)이며, issue 상한 4개를 초과한 Brief 한 건을 명세 4.1절의 재요청으로 복구했다. 입력 419,142토큰·출력 76,334토큰, 실행 시간 약 10분 44초다. 공식 표준 단가로 계산한 Claude 비용 추정은 $0.0800812이며 실제 청구서를 확인한 금액은 아니다.

`artifact_audit.json`은 원문과 prefix, 메모리 인계, 정확한 모델·설정, issue 참조, 8개 루브릭·29개 자질, 프롬프트·코드·데이터 해시를 검사한 결과다. 실제 진단 49회(채점·자질 호출 각각 49회), 기존 코드·KB 122개 파일과 원본 장르 파일의 보존을 확인했다. `stage3_report.json`에는 명령·결과, 사전 승인된 명세 차이, 중단된 Luna 사용량과 비용을 함께 기록했다. 단위 테스트 77개가 통과했고, GPU를 숨긴 mock 실행도 두 조건을 끝까지 실행했다. 실행 종료 후 GPU 작업 프로세스가 남아 있지 않음을 확인했다.

이는 **실행 완료 기준 충족**이며 첨삭 품질 검증 완료를 뜻하지 않는다. 예를 들어 `whole_once/14-2-M1-N-0A-E-0215`의 출력에는 원문에 없던 쓰레기 처리와 온실가스의 인과 설명이 추가됐다. 내용 추가 금지 프롬프트에도 이런 지시 위반이 나타날 수 있으며, 해당 출력은 그대로 보존했다. 검증·재수정 루프는 넣지 않았다. 이후 사용자의 품질 평가 요청에 따라 아래 중간 평가를 수행했다.

현재 5편의 중간 평가는 다음 명령으로 재현한다. 다른 환경에서는 평가 의존성 `.[eval]`을 설치한다. 기존 실행의 결과·설정·입력 해시를 읽고, 원문 및 두 방식의 최종 글을 동일한 Kanana 어댑터로 채점한다. 수정 에이전트와 외부 LLM API를 호출하지 않으며 자질을 다시 계산하지 않는다. 매번 새 실행 폴더에 `results.csv`, `statistics.json`, `summary.md`, 원시 채점 로그를 저장한다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/evaluate.py --run runs/20261014T095651.042158Z_3931618173 --local-only
```

평가 실행은 `runs/20261014T101750.514892Z_2cfcc1fbeb`이다. 원문 5개와 최종 글 10개를 채점했고 원문 5개는 기존 캐시를 사용했다. 동일한 채점기 fingerprint를 확인했으며, 입력 잘림 없이 67.5초에 종료했다. 추가 API 호출은 0회다. 원문 평균 합계 53.21, 전체 수정 55.83(+2.62), 문단별 수정 57.53(+4.32)였다. 문단별−전체의 평균 차이 +1.70, 글 단위 bootstrap 95% CI [-1.26, +4.48], 대응 Wilcoxon p=0.3125로 우열을 확정하지 않는다.

`review.md`에는 저장된 10개 수정본을 직접 대조한 정성 검토를 담았다. 이는 독립적인 양방향 LLM judge 결과가 아니다. 문단별 수정의 내용 창작·메모리 오염과 전체 수정의 후반부 누락이 확인됐으므로 점수 상승만으로 품질이 검증됐다고 볼 수 없다. `evaluation_report.json`에 실행 명령·완료 기준·해석상 미정 사항을 기록했다. 평가 관련 테스트를 포함하여 총 82개가 통과했다.

**명세 4단계는 미완료**다. 이번 요청은 저장된 5편의 중간 평가로 수행했으며, 10편 전체 결과와 에이전트와 다른 계열 모델의 두 순서 judge가 남아 있다. 5편 제한을 확대하거나 같은 계열 judge로 대체하지 않았다. 편집 거리의 미정인 분모는 `max(원문 문자 수, 수정본 문자 수)`, bootstrap은 글 단위 대응 차이 10,000회·seed 42·percentile 95%로 명시했다. 실패한 수정본도 제외하지 않고 모든 글을 동일하게 집계했다.

## 파일럿에서 발견한 문제의 프롬프트 보완

사용자의 수정 요청에 따라 `prompts/summarizer.md`, `prompts/reviser.md`, `prompts/memory_update.md`를 보완했다. 문단을 완성된 글로 오인해 내용을 채우지 않도록 하고, 확인된 문제가 없으면 빈 issue 목록을 허용했다. 전체 수정에서는 모든 문단을 반환하도록 명시했으며, 원문에 없는 경험·결론·인과관계를 추가하거나 메모리에 저장하지 않도록 했다. 오류 노트는 실제 원문과 실제 수정의 대응 구간을 기록하도록 했다. 실행 코드·모델 설정·스키마·KB·기존 채점기와 자질 코드는 변경하지 않았다.

같은 파일럿 5편을 두 차례 실행했다. 첫 시도는 `runs/20261014T103217.773696Z_5f8ab97267`, 당시 보완안의 두 번째 시도는 `runs/20261014T104152.627997Z_ecfbfbe8db`다. 각 시도에 프롬프트 변경 전후와 해시를 `prompt_change.json`으로 저장했다. 두 번째 시도의 수정본 10개 모두 원문 문단 수를 유지했고, 편지의 수신인 한 줄을 늘리거나 자기소개에 가족과의 대화 경험을 만들어 넣던 사례는 재발하지 않았다. 아래 결과는 그 시점의 프롬프트와 `x_only` 설정에 대한 기록이며, 이후 변경의 결과로 재사용하지 않는다.

**품질 개선 완료로 판정하지 않는다.** 최종 시도에도 법·도덕 글의 뜻을 바꾼 수정, 원문·수정본과 정확히 대응하지 않는 오류 노트 2개, 기존 논지를 `null`로 지운 프로필 갱신 1개, 실제 수정과 applied/skipped 보고의 불일치가 남았다. 현재 프롬프트는 실험 중인 후보이며 검증된 최종 개선판이 아니다. 검증·재수정 루프나 프로필 적용 정책 변경은 추가하지 않았다.

최종 결과의 로컬 평가 명령:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python scripts/evaluate.py --run runs/20261014T104152.627997Z_ecfbfbe8db --local-only
```

평가는 `runs/20261014T105112.372756Z_de56d1f507`에 저장했다. 동일한 Kanana로 계산한 평균 합계는 원문 53.21, 전체 수정 52.39, 문단별 수정 51.83으로 두 조건 모두 원문보다 낮았다. 기존 프롬프트의 전체 55.83·문단별 57.53보다도 낮다. 수정량과 문단 누락이 줄어든 것을 첨삭 품질 향상과 동일시하지 않는다. 같은 5편을 보완과 평가에 반복 사용했으므로 독립적인 성능 검증도 아니다.

최종 프롬프트에서 `python -W error -m pytest -q`는 82개 통과했고, GPU를 숨긴 mock은 두 조건을 외부 API 호출 없이 완료했다. 두 실제 재실행은 각각 API 76회·최종 글 10개·턴 기록 27개를 저장했고 형식 재요청은 없었다. 추가 API 호출 총 152회, 표준 단가 기준 비용 추정 합계는 $0.1467478이다. 최종 로컬 평가는 외부 API를 호출하지 않았다. 기존 코드·KB 보호 대상 122개 파일의 해시 보존과 GPU 작업 종료를 확인했다.

[상세 비교 보고서](runs/20261014T105112.372756Z_de56d1f507/comparison_review.md)에는 개선·실패 사례와 글별 점수를, [실행 보고서](runs/20261014T105112.372756Z_de56d1f507/prompt_fix_report.json)에는 실행 명령·결과·완료 기준을 기록했다. 3단계 실행 기준은 충족하지만 의미 보존과 메모리 품질 문제는 미해결이며, 4단계의 10편 평가와 다른 계열 judge도 아직 미완료다.

## 앞 수정본의 문체와 문단 연결을 기준으로 지시 정정

사용자가 강조한 문단별 수정의 목표는 첫 수정본을 확정한 뒤, 다음 원문 문단의 내용과 의도를 보존하면서 앞 수정본의 문체와 흐름에 맞추는 것이다. 누적 수정본을 유지하는 실행 흐름은 이미 구현돼 있었지만, 이전 프롬프트 보완의 “issue가 없으면 원문 그대로”와 새 원문의 종결체 보존 지시가 `keep_style`·`connect_note`의 역할을 제한했다. 또한 `x_only`에서는 요약기에 직전 수정본 전체를 주면서 수정기에는 마지막 두 문장과 프로필·Brief만 직접 전달했다. 이는 확인된 입력 구성과 지시 충돌이며, 과거 점수 하락의 원인을 단독으로 증명한 것은 아니다.

현재 `configs/pilot_haiku.yaml`은 명세 4.7절과 5절에 있는 `reviser_input: prev_plus_x`를 선택한다. 직전 수정본 전체를 참고하되 출력은 새 문단만 생성한다. 더 이전 수정본은 누적 진단과 프로필에 반영되며, 모든 이전 문단의 원문을 수정기에 새로 추가하는 방식으로 범위를 넓히지 않았다. `x_only` 옵션과 mock 기본 설정은 비교용으로 유지한다.

세 프롬프트에서 다음 기준을 명시했다.

- 문체 기준은 직전 수정본과 기존 프로필이다. 새 문단의 종결어미가 다르면 내용·주장·감정·불확실성은 유지하며 기존 문체에 맞출 수 있다. 직접 인용·대화 속 화자의 말투는 구분한다.
- `connect_note`는 앞 수정본 끝부분과 새 문단 첫 문장의 연결을 다룬다. 문단 중간의 문제는 해당 위치의 issue로 다룬다. 기존 내용에서 확인되는 관계만 표현하며 새 근거나 인과를 만들지 않는다.
- issue 목록이 비어 있어도 `keep_style`·`connect_note`에 근거한 최소 조정은 가능하다. 세 지침에서 모두 수정 필요가 없으면 원문을 유지한다.
- 메모리의 문체 정보는 확정된 수정본에서 이어받을 기준이다. 새 원문의 말투나 수정본의 일시적 지시 위반으로 기준을 덮어쓰지 않도록 했다. 사실·논지의 근거는 계속 원문에서 확인한다.

`tests/test_turn_flow.py`의 직전 수정본을 네 문장으로 바꾸어, 전체 수정본과 마지막 두 문장 힌트가 각각 정확히 전달되고 앞 출력은 그대로 누적되는지 확인했다. `python -W error -m pytest -q`는 82개 통과했다. 활성 파일럿 설정의 입력 방식을 유지하고 공급자만 mock으로 바꾼 실행 `runs/20261014T110712.226927Z_a085e6b454`는 두 조건을 완료했다(실제 API 0회). 이 변경 검증에서는 실제 LLM을 호출하지 않았으며, 이후 사용자 지시에 따라 수행한 실제 재실행 결과는 아래에 기록했다. 원래 명세의 호출 순서와 수정 횟수, 기존 채점기·자질 코드·KB는 그대로다.

[이번 변경과 검증 보고서](runs/20261014T110505.911063Z_83f37f54a6/review.md)에 변경 전후, 검증 명령과 남은 판단 사항을 기록했다.

## 문체·연결 지시 정정 후 실제 5편 재실행

사용자의 “그럼 그렇게 해봐” 지시에 따라 같은 5편으로 두 조건을 다시 실행했다. `runs/20261014T111644.364558Z_61faf37a02`는 현재 `prev_plus_x` 설정과 정정한 세 프롬프트를 사용한 결과다. 실행 중 코드·프롬프트·설정을 바꾸지 않았다. 실제 API 호출 76회, 형식 재요청 0회, 입력 528,580·출력 62,290토큰, 약 8분 36초에 최종 글 10개와 턴 로그 27개를 저장했다. 공식 표준 단가를 이용한 이번 API 비용 추정은 $0.084003이다.

실제 요약·수정 요청을 각각 22건 확인했고, 첫 문단 이후 17개 경계 모두 직전 수정본 전체가 수정기에 전달됐다. 22턴에서 앞 수정본이 그대로 유지됐고, 최종 글 10개 모두 원문 문단 수를 보존했다. 본문의 주된 문체는 대체로 이어졌다. 법·도덕 글의 수사 의문과 '그러나' 전환은 유지됐으며, 지구온난화 글은 생활 습관 다음 문단을 '개인 차원의 방안'으로 시작했다. 이러한 관찰을 모든 연결의 품질 향상으로 일반화하지 않는다.

의미 보존에는 여전히 실패 사례가 있다. 자기소개에서 '두려움'을 '부담'으로, '이길 수 있다'를 '이기기도 한다'로 바꿨다. 해당 치환은 요약기의 피드백·지시에서 시작해 수정본과 메모리로 이어졌다. 첫 문장 연결 지시의 적용 위치가 어긋난 사례, 중복 연결 표현, 반말을 해요체 혼용으로 기록한 프로필 오류, 비식별 표시 삭제도 남았다. 입력 전달이 맞는 것과 모델이 올바른 첨삭을 하는 것은 각각 점검해야 한다.

로컬 재채점은 `runs/20261014T112554.540206Z_b6294aab18`에 저장했다. 원문 평균 53.21, 전체 수정 50.33(-2.87), 문단별 수정 54.53(+1.32)다. 직전 보완안의 문단별 평균 51.83보다는 높지만, 의미 변형이 있어 품질 문제 해결로 판정하지 않는다. 문단별−전체의 차이 +4.19, bootstrap 95% CI [-0.29, +9.25], Wilcoxon p=0.3125이며, 같은 5편을 반복 사용한 탐색 결과다. 원문 점수와 채점기 fingerprint는 이전 평가와 동일하고 평가의 추가 외부 API 호출은 0회다.

[문체·연결 중심의 검토 보고서](runs/20261014T112554.540206Z_b6294aab18/alignment_review.md), [수치 요약](runs/20261014T112554.540206Z_b6294aab18/summary.md), [실행 명령·완료 기준](runs/20261014T111644.364558Z_61faf37a02/stage3_report.json)에 결과를 기록했다. 3단계 실행 기준은 충족했고, 내용 보존 품질 및 4단계의 10편·다른 계열 judge 평가는 미완료다. 기존 코드·KB 122개 파일의 해시 보존과 GPU 작업 종료를 확인했다.


## 추가 1.5단계: Kanana 점수 평균화 검증

[최종 보고서](SMOOTHED_SCORER_STAGE1_5_2026-10-14.md)에 개발용 변형, 새 원문 100편, 두 종류의 묶음 표준편차, 실행 명령과 제한을 기록했다. 230개 입력·14,720개 유효 샘플을 두 GPU에서 처리했다. N=32 정수 평균의 LF·축약형 민감도는 줄었지만 합계 Spearman 비하락 조건을 충족하지 못해, 사전 기준상 Kanana 보조 지표를 권고했다. 기존 평가 설정은 변경하지 않았고 다음 단계는 시작하지 않았다. 시드 규칙은 [설정](configs/smoothed_scoring.yaml)과 [실행 규칙](SMOOTHED_SCORER_STAGE1_5_PROTOCOL_2026-10-14.md)에 명시했다.

## 1.5단계 판정 승인 후 2단계 완료

사용자 승인에 따라 주 지표를 다른 계열 judge의 쌍대 비교·의미 보존율로, 보조 Kanana를 정수 평균 N=64와 기존 greedy 병행 기록으로 정했다. 이는 1.5단계 결과를 본 뒤 내린 결정이며 원래 N=32 판정 규칙은 유지한다. 글 ID 시드도 유지한다. 3단계에 judge와 사람 판정의 일치율 확인을 추가했다. [활성 평가 정책](configs/evaluation_policy.yaml), [이후 단계 계획](EVALUATION_PLAN_POST_STAGE1_5.md)에 근거와 미정 사항을 기록했다.

비식별 표시 보호·실패 시 원문 사용, 문장 diff, 실제 변경에 근거한 메모리 검사, 코드 종결체 분포, issue 위치 변경과 자기 보고의 대조를 구현했다. 테스트 155개와 두 입력 방식의 두 조건 mock 실행이 통과했다. 실제 API·Kanana 추론은 각각 0회이며 원본 채점기·자질 코드·KB 관련 122개 파일과 원자료 해시는 동일하다. [2단계 보고서](CODE_HARDENING_STAGE2_2026-10-15.md)에 완료 기준·명령·예시·애매했던 점을 정리했다. 2단계에서 멈췄으며 3단계 실행은 아직 하지 않았다.
