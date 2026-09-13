# 발표용 실측 팩트 (검증된 값만 사용)

## KB 규모
- 문서: 7개 / 청크: 11개 / 리트리버: tfidf
- 문서: README_console.md, expense_policy.md, info_sec_policy.md, it_helpdesk_sop.md, leave_hr_guide.md, product_faq.md, sales_report_2026h1.md

## 리트리버 평가 (LLM 서버 없이 오프라인 검증, 실측)
- 평가 문항: 20문항 (gold 문서 라벨 포함)
- recall@5 평균: 1.0
- hit-rate: 1.0
- LLM-as-Judge(충실성/관련성): Ollama/vLLM 연결 시 자동 반영 — 실행 절차: python scripts/run_eval.py

## 에이전트 도구 4종
search_docs(RAG·출처 표기) / calculate(안전 수식) / get_date(날짜·요일) / lookup_db(사내 DB)

## 샘플 DB 검증
- sales: 2025-11~2026-08, 6지역×6제품, 월별 규모와 영업리포트(상반기 397억) 정합
  - 상반기(1~6월) DB 합계: 390.9억
  - '2026년 3월 전체 매출 합계' 질의 실측 정상
- leave_balances: 사원 6명 연차 잔여

## 데모 시나리오 (오프라인 데모 모드로 전부 검증 완료)
1) 연차/휴가 절차 → 문서 검색+출처 2) USB 금지 규정 → 보안규정 인용
3) 2026년 2월 서울 냉장고 매출 → DB 조회 4) 3월 전체 매출 합계 → DB 집계
5) (5+37)*12 → 계산 도구 6) 지금 날짜/요일 → get_date (+ 후속 질문 = 세션 기억)

## 구현 스택
FastAPI + SSE 스트리밍, 단일 HTML UI(외부 CDN 없음), char n-gram TF-IDF(오프라인)
LLM 플러그인: OpenAI 호환 API → Ollama / vLLM / llama.cpp (ONQ_LLM_BASE_URL, ONQ_LLM_MODEL)
Docker + docker-compose 배포, 문서 업로드 API(md/txt/pdf/docx)

## 파인튜닝 (GPU 운영환경 적용 사항, 스크립트 제공)
- scripts/finetune_toolcalling.py (export/train) — Unsloth 4bit LoRA
- 도구 호출 대화 ChatML 데이터셋, 운영측정: PPL·tok/s·VRAM·지표 변화

## KPI 구성 (AS-IS → TO-BE)
- AS-IS(가상 가정치): 문서 탐색 평균 15~30분/건, 담당자 반복 문의 3~5시간/주
- TO-BE: 1회 질의 스트리밍 응답, 출처 표기 답변, 24h 무중단 온프레미스
- 핵심 지표: recall@k · hit-rate · LLM-as-Judge(충실성·관련성) · 응답시간 · 출처 표기율
