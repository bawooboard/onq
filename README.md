# OnQ — 온프레미스 경량 사내 지식 에이전트

사내 문서(규정·SOP·FAQ)와 사내 DB를 온프레미스 LLM만으로 조회·답변하는
경량 에이전트 서비스. LLM 서버가 없어도 동작하는 **오프라인 데모 모드**를 내장한다.

## 빠른 시작 (3줄)
```bash
pip install -r requirements.txt
python scripts/gen_sample_data.py && python scripts/build_kb.py
python run.py          # → http://localhost:8000
```

- 서버가 없으면 **데모 모드**로 동작(채팅 UI에서 확인 가능).
- Ollama/vLLM 연결 후 자동으로 **LLM 에이전트 모드**로 전환된다.

## LLM 연결 (Ollama)
```bash
bash scripts/install_ollama.sh          # Ollama 설치 + qwen2.5:7b
ollama serve
python run.py                            # 자동 감지 → LLM 모드 전환
```
기본값 외 환경변수:
`ONQ_LLM_BASE_URL`(예: http://localhost:11434/v1), `ONQ_LLM_MODEL`
(vLLM의 경우 `--served-model-name`과 일치), `ONQ_RETRIEVER=tfidf|dense`.

## Docker
```bash
docker compose up -d --build             # http://localhost:8000
# GPU 서버에 Ollama 연결: docker compose --profile llm up -d
```

## 발표용 데모 시나리오
1. "연차 휴가 신청 절차가 어떻게 되나요?" → 문서 검색 + 출처
2. "USB 사용이 금지인가요?" → 보안 규정 인용
3. "2026년 2월 서울 지역 냉장고 매출 알려줘" → DB 조회
4. "2026년 3월 전체 매출 합계 알려줘" → DB 집계
5. "(5+37)*12 계산해줘" → 계산 도구
6. "지금 날짜와 요일이 궁금해" / 후속 질문 → 날짜 도구 + 세션 기억

## 검증 (Eval)
```bash
python scripts/run_eval.py               # recall@k·hit-rate (LLM 연결 시 충실성/관련성까지)
```
평가셋: `data/eval/questions.json` (20문항, gold 문서 라벨 포함)

## 파인튜닝 (GPU 환경)
```bash
python scripts/finetune_toolcalling.py --mode export   # 학습 데이터셋 생성
python scripts/finetune_toolcalling.py --mode train    # Unsloth LoRA (GPU)
```

## API
| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/health` | 상태(모드·LLM·KB) |
| POST | `/api/chat` | SSE 채팅 (`message`, `session_id`) |
| POST | `/api/kb/upload` | 문서 업로드(md/txt/pdf/docx) → 즉시 인덱싱 |
| POST | `/api/kb/rebuild` | KB 재구축 |

## 구조
```
onq/
├─ app/
│  ├─ agent/   llm.py(LLM 클라이언트) agent.py(도구호출 루프) memory.py demo.py
│  ├─ rag/     kb.py(문서→청크→TF-IDF/dense 검색)
│  ├─ tools/   registry.py impls.py(검색·계산·날짜·DB)
│  ├─ eval/    evaluate.py(recall@k·hit-rate·LLM-as-Judge)
│  ├─ web/     index.html(단일 파일 UI, 외부 CDN 없음)
│  └─ main.py  FastAPI
├─ data/       documents/(샘플 7종) db/ eval/ finetune/
└─ scripts/    build_kb run_eval demo_chat install_ollama finetune…
```

## 스코프 아웃 (향후 과제)
계정·권한 제어, 문서 버저닝, GPU 벤치마크 실측(교육 환경), 멀티모달,
dense 검색 임베딩 스케일링.
