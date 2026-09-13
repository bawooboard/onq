"""터미널 채팅 데모 — curl 없이 CLI에서 바로 에이전트와 대화.

python scripts/demo_chat.py        ← LLM 연결 시 에이전트 루프
python scripts/demo_chat.py --demo ← 오프라인 데모 모드 강제
"""
import argparse
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from app.agent.llm import LLMClient  # noqa: E402
from app.agent.demo import demo_answer  # noqa: E402
from app.agent.agent import run_agent_task  # noqa: E402
from app.rag.kb import KnowledgeBase  # noqa: E402
from app.tools import impls  # noqa: F401 E402

ap = argparse.ArgumentParser()
ap.add_argument("--demo", action="store_true")
args = ap.parse_args()

kb = KnowledgeBase(config.DOC_DIR, config.KB_DIR, retriever=config.RETRIEVER,
                   chunk_size=config.CHUNK_SIZE, dense_model=config.DENSE_MODEL)
kb.load()
if not kb.chunks:
    kb.rebuild()
llm = LLMClient(config.LLM_BASE_URL, config.LLM_MODEL, config.LLM_API_KEY,
                config.LLM_TEMPERATURE, config.LLM_TIMEOUT)
use_demo = args.demo or not llm.is_available()
print(f"[OnQ] 모드: {'데모(오프라인)' if use_demo else 'LLM 에이전트'} | "
      f"KB: 문서 {len(kb.doc_names())}개, 청크 {len(kb.chunks)}개")
print("종료: exit / 질문 입력: ")
while True:
    try:
        q = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not q or q in ("exit", "quit", "종료"):
        break
    if use_demo:
        events = demo_answer(q, kb, top_k=config.TOP_K)
        for ev in events:
            if ev.get("type") == "tool":
                print(f"  [도구] {ev['name']} {ev.get('detail', '')}")
            if ev.get("type") == "message":
                print("\n" + ev["text"] + "\n")
    else:
        final, events = run_agent_task(llm, kb, impls, q, [], top_k=config.TOP_K,
                                       max_steps=config.MAX_STEPS)
        for ev in events:
            if ev.get("type") == "tool":
                print(f"  [도구] {ev['name']} {ev.get('detail', '')}")
        print("\n" + final + "\n")
