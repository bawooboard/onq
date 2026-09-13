"""OnQ 전역 설정. 환경변수로 모두 오버라이드 가능 (README 참고)."""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data"
DOC_DIR = DATA / "documents"
KB_DIR = DATA / "kb"
DB_DIR = DATA / "db"
EVAL_FILE = DATA / "eval" / "questions.json"
WEB_FILE = ROOT / "app" / "web" / "index.html"
EVAL_REPORT = KB_DIR / "eval_report.json"

# ── LLM 백엔드 (OpenAI 호환 API: vLLM / llama.cpp / Ollama / LM Studio)
LLM_BASE_URL = os.getenv("ONQ_LLM_BASE_URL", "http://localhost:11434/v1")  # Ollama 기본값
LLM_MODEL = os.getenv("ONQ_LLM_MODEL", "qwen2.5:7b")
LLM_API_KEY = os.getenv("ONQ_LLM_API_KEY", "ollama")  # 로컬 서버는 값 무시
LLM_TEMPERATURE = float(os.getenv("ONQ_LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = int(os.getenv("ONQ_LLM_TIMEOUT", "120"))

# ── RAG
RETRIEVER = os.getenv("ONQ_RETRIEVER", "tfidf")          # tfidf(기본/오프라인) | dense
DENSE_MODEL = os.getenv("ONQ_DENSE_MODEL", "intfloat/multilingual-e5-small")
CHUNK_SIZE = int(os.getenv("ONQ_CHUNK_SIZE", "800"))      # 문단 누적 최대 글자 수
MAX_STEPS = int(os.getenv("ONQ_MAX_STEPS", "4"))          # 에이전트 도구 호출 최대 반복
TOP_K = int(os.getenv("ONQ_TOP_K", "3"))

HOST = os.getenv("ONQ_HOST", "0.0.0.0")
PORT = int(os.getenv("ONQ_PORT", "8000"))
