"""LLM 클라이언트 — OpenAI 호환 /chat/completions 를 사용.

vLLM, llama.cpp(server), Ollama(/v1), LM Studio 등 온프레미스 서버는
전부 이 인터페이스로 연결된다. 미연결 시 is_available()=False 로
오프라인 데모 모드로 동작한다.
"""
import requests


class LLMClient:
    def __init__(self, base_url, model, api_key="ollama",
                 temperature=0.3, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self._available = None  # 미확정 상태

    def probe(self):
        """서버 존재 확인 (GET /models)"""
        try:
            r = requests.get(f"{self.base_url}/models", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def is_available(self):
        if self._available is None:
            self._available = self.probe()
        return self._available

    def chat(self, messages, temperature=None):
        """비스트리밍 채팅. 응답 텍스트 반환."""
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "stream": False,
        }
        r = requests.post(
            f"{self.base_url}/chat/completions",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError):
            return ""
