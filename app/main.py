"""OnQ FastAPI 서버 — 정적 웹 + SSE 채팅 + KB 관리 + 상태 확인."""
import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

import config
from app.agent.agent import run_agent_task
from app.agent.demo import demo_answer
from app.agent.llm import LLMClient
from app.agent.memory import SessionMemory
from app.rag.kb import KnowledgeBase
from app.tools import impls  # noqa: F401  (도구 등록)
from app.tools import registry

app = FastAPI(title="OnQ", version="1.0.0")

kb = KnowledgeBase(config.DOC_DIR, config.KB_DIR,
                   retriever=config.RETRIEVER, chunk_size=config.CHUNK_SIZE,
                   dense_model=config.DENSE_MODEL)
impls.set_kb(kb)
mem = SessionMemory()
llm = LLMClient(config.LLM_BASE_URL, config.LLM_MODEL, config.LLM_API_KEY,
                config.LLM_TEMPERATURE, config.LLM_TIMEOUT)


def ensure_kb():
    try:
        kb.load()
    except Exception:
        kb.chunks = []
    if not kb.chunks and config.DOC_DIR.exists():
        try:
            kb.rebuild()
        except Exception as e:
            print(f"[warn] KB 구축 실패: {e}")


ensure_kb()


@app.on_event("startup")
def _startup():
    ensure_kb()


def mode():
    return "llm" if llm.is_available() else "demo"


@app.get("/")
def index():
    return FileResponse(config.WEB_FILE, media_type="text/html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "OnQ",
        "mode": mode(),
        "llm": {"model": llm.model, "base_url": llm.base_url,
                "available": llm.is_available()},
        "retriever": kb.retriever,
        "documents": len(kb.doc_names()),
        "chunks": len(kb.chunks),
        "tools": [t.name for t in registry.all_tools()],
        "session_exchanges": mem.max_exchanges,
    }


@app.post("/api/kb/rebuild")
def kb_rebuild():
    result = kb.rebuild()
    return {"ok": True, **result}


@app.post("/api/kb/upload")
def kb_upload(file: UploadFile = File(...)):
    suffix = file.filename[file.filename.rfind("."):].lower()
    path = config.DOC_DIR / file.filename
    path.write_bytes(file.file.read())
    text = kb.load_file(path)
    kb.add_text(text, file.filename)
    return {"ok": True, "file": file.filename, "chunks_total": len(kb.chunks)}


@app.post("/api/chat")
async def chat(message: str = Form(...), session_id: str = Form("default")):
    query = message.strip()
    if not query:
        return StreamingResponse(iter([_sse({"type": "error", "text": "질문을 입력하세요."})]),
                                 media_type="text/event-stream")

    def gen():
        yield _sse({"type": "meta", "mode": mode(), "llm_model": llm.model,
                    "retriever": kb.retriever})
        m = mode()
        if m == "llm":
            final, events = run_agent_task(llm, kb, registry, query,
                                           mem.history(session_id),
                                           top_k=config.TOP_K,
                                           max_steps=config.MAX_STEPS)
        else:
            events = demo_answer(query, kb, top_k=config.TOP_K)
            final = ""
        for ev in events:
            if ev.get("type") == "tool":
                yield _sse({"type": "tool", "name": ev.get("name", ""),
                            "status": ev.get("status", ""),
                            "detail": ev.get("detail", "")})
            elif ev.get("type") == "message":
                final = ev.get("text", "")
        if final:
            mem.add(session_id, "user", query)
            mem.add(session_id, "assistant", final)
        yield _sse({"type": "message", "text": final})
        yield _sse({"type": "done", "mode": m,
                    "documents": len(kb.doc_names()), "chunks": len(kb.chunks)})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(obj):
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"
