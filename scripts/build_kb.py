"""KB(지식 기반) 구축: data/documents/ 의 파일을 청크→색인한다.

python scripts/build_kb.py
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from app.rag.kb import KnowledgeBase  # noqa: E402

kb = KnowledgeBase(config.DOC_DIR, config.KB_DIR, retriever=config.RETRIEVER,
                   chunk_size=config.CHUNK_SIZE, dense_model=config.DENSE_MODEL)
result = kb.rebuild()
print(f"[KB 구축 완료] 문서 {result['documents']}개, 청크 {result['chunks']}개"
      f" (리트리버: {kb.retriever})")
