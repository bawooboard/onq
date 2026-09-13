"""검증 시나리오 실행: KB_구축 → 회귀 케이스 → 리트리버/생성 평가 리포트.

python scripts/run_eval.py [--n-judge 8]
"""
import argparse
import json
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from app.rag.kb import KnowledgeBase  # noqa: E402
from app.agent.llm import LLMClient  # noqa: E402
from app.eval.evaluate import run_eval, print_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n-judge", type=int, default=None, help="LLM 판정 문항 수 (기본 전체)")
ap.add_argument("--k", type=int, default=5)
args = ap.parse_args()

kb = KnowledgeBase(config.DOC_DIR, config.KB_DIR, retriever=config.RETRIEVER,
                   chunk_size=config.CHUNK_SIZE, dense_model=config.DENSE_MODEL)
kb.load()
questions = json.loads(config.EVAL_FILE.read_text(encoding="utf-8"))
llm = LLMClient(config.LLM_BASE_URL, config.LLM_MODEL, config.LLM_API_KEY,
                config.LLM_TEMPERATURE, config.LLM_TIMEOUT)

print(f"[검증 시작] 문항 {len(questions)}개 | k={args.k} | 리트리버={kb.retriever}")
if llm.is_available():
    print(f"[LLM 연결됨] {llm.base_url} / {llm.model}")
else:
    print(f"[LLM 미연결] {llm.base_url} → 리트리버 평가만 수행합니다.")
report = run_eval(kb, llm, questions, top_k=args.k, n_judge=args.n_judge)
print_report(report)
config.EVAL_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
print(f"\n보고서 저장: {config.EVAL_REPORT}")
