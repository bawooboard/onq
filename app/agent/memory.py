"""세션별 대화 기억 저장소 (인메모리, 최근 대화만 유지)."""
from collections import deque


class SessionMemory:
    def __init__(self, max_exchanges=6):
        self._store = {}
        self.max_exchanges = max_exchanges

    def add(self, session_id, role, content):
        mem = self._store.setdefault(session_id, deque(maxlen=self.max_exchanges * 2))
        mem.append({"role": role, "content": content})

    def history(self, session_id):
        return list(self._store.get(session_id, []))

    def reset(self, session_id=None):
        if session_id is None:
            self._store.clear()
        else:
            self._store.pop(session_id, None)
