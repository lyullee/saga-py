# Digital Twin HAZOP/RAG API

## 처리 순서

1. `POST /api/hazop/rules/import`으로 Java HAZOP 행을 SQLite에 upsert합니다.
2. `GET /api/hazop/rules/monitoring-readiness`로 수치·단위·비교방향·근거·검토상태를 점검합니다.
3. 디지털 트윈이 `POST /api/digital-twin/hazop/evaluate`로 한 시점의 설비·태그 스냅샷을 보냅니다.
4. Python 계층이 `STATIC`, `DELTA_START`, `DELTA_PREV` 조건과 데이터 품질·신선도를 결정론적으로 평가합니다.
5. 감지된 HAZOP 행을 우선으로 SOP 초안을 만들고, 미등록 태그나 설명이 필요한 경우에만 로컬 PDF RAG를 보완합니다.
6. Service Hub LLM은 사실을 새로 판정하는 심판이 아니라, 위 결과를 운영자가 이해할 수 있는 순서로 설명하는 역할을 합니다.

## 직답 HAZOP 경로 복구 안내

최근 디지털 트윈 연계 시험에서 추가했던 `POST /api/digital-twin/hazop/direct`
경로는 SAGA 복구 과정에서 외부 라우트에서 제거했습니다. 상태 평가는
`POST /api/digital-twin/hazop/evaluate`를 사용하며, 기존의 HAZOP 판정·RAG 보완·LLM
설명 순서를 유지합니다. 디지털 트윈의 빠른 일반 LLM 질의가 필요한 경우에는
`/api/digital-twin/chat/direct` 또는 `/api/digital-twin/chat/direct/stream`을
사용합니다. 이 두 LLM 계약과 Service Hub/Groq 연결 설정은 복구 대상에서 제외했습니다.

원문에서 임계값 후보를 찾을 때는 `POST /api/hazop/rules/propose-threshold`를 사용할 수
있습니다. 이 API는 색인 문서에 숫자와 비교방향이 실제로 함께 있는 경우에만 후보를
반환하고, 인용문 안에 제안 숫자가 존재하는지 서버에서 다시 검증합니다. 반환된 후보는
항상 `thresholdConfidence=pending`, `requires_human_approval=true` 상태이며, 승인 전에는
`/api/hazop/rules/import`로 운영 테이블에 넣지 않습니다.

## 상태 값

- `NORMAL`: 요청된 모든 태그를 평가했고 HAZOP 조건이 감지되지 않음
- `WARNING`: 하나 이상의 HAZOP 조건 감지
- `PARTIAL`: 미등록 태그, 품질 불량, 오래된 타임스탬프, 기준값 누락 등으로 일부만 평가
- `UNKNOWN`: 측정값이 전달되지 않음

## 모니터링 기준 승인

`High`, `High-High`, `Low` 같은 guide word는 화면 분류용으로만 보존하며 판정에 사용하지 않습니다. 모니터링 판정에는 반드시 숫자 `thresholdValue`, `compareDir`(이상/초과/이하/미만 또는 부등호), 단위, 그리고 `thresholdBasis`·`thresholdSource`·`standardRef` 중 하나 이상의 근거가 필요합니다. `thresholdConfidence`가 `verified` 또는 `derived`가 아니면 `monitoring_ready=false`로 반환합니다. 근거가 없는 행은 입력 오류가 아니라 검토 대기 상태로 표시되며 자동 경보 기준으로 승인되지 않습니다.

`WARNING`이면서 `data_quality`에도 항목이 있으면 경보 자체와 센서 신뢰성 문제를 모두 확인해야 합니다. `hits`에는 HAZOP 행의 `risk_scenario`, `consequence`, `emergency_action`, `future_measure`, `standard_ref`가 보존됩니다.

`sop`에는 화면 표시용 `answer`와 별도로 디지털 트윈이 사용할 수 있는 구조화된 단계가 함께 반환됩니다. `priority`는 `normal/attention/warning/emergency` 중 하나이며, Java HAZOP 등급(주의=1, 경보=2, 긴급=3)에 맞춰 긴급 등급을 표시합니다. `immediate_actions`, `isolation_evacuation`, `verification_steps`, `restart_requirements`, `records_to_capture`, `escalation`이 각각 즉시조치, 통제·대피 원칙, 확인절차, 재가동 조건, 기록항목, 보고·에스컬레이션을 나타냅니다. 이 단계들은 LLM이 임의로 만든 명령이 아니라 HAZOP 감지 사실과 보수적인 안전 게이트에서 생성됩니다.

## 임계값 규칙

`compareDir`는 `이상/초과/이하/미만`, `>=`, `>`, `<=`, `<`, `HIGH`, `LOW`를 지원합니다. `DELTA_START`와 `DELTA_PREV`는 각 기준값에서 변화량을 계산하며, 기준값이 없으면 정상으로 간주하지 않고 `PARTIAL`로 반환합니다. 센서 단위가 HAZOP 단위와 다르면 자동 변환하지 않고 평가를 보류합니다.

## 근거 정책

- `references[].status=hazop_table`: 입력된 HAZOP 행 자체
- `references[].status=indexed`: SAGA PDF/법령/RAG 색인에서 실제 검색된 발췌
- `references[].status=catalog_only`: 보완 후보 기준의 목록일 뿐 원문 근거가 아님

따라서 `catalog_only` 항목을 법적 의무·수치·거리의 출처로 사용하면 안 됩니다. 국내 법령·KGS Code·해외 ISO/NFPA/API/IEC 문서를 근거로 사용하려면 먼저 PDF로 색인해야 합니다.

## 안전 경계

이 API는 자동 밸브 조작, ESD 실행, 대피 명령을 직접 수행하지 않습니다. 응답은 설계된 인터록·현장 비상대응계획·작업허가·안전관리자 판단을 우선하도록 작성되며, 누출·화재 의심 시 사람을 위험구역으로 보내는 지시는 생성하지 않습니다.
