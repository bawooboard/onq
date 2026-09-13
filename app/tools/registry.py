"""도구 레지스트리 — 에이전트가 호출 가능한 도구의 등록/조회/기술."""
import json


class ToolDef:
    def __init__(self, name, description, args_schema, fn):
        self.name = name
        self.description = description
        self.args_schema = args_schema  # 예: {"query": "string", "k": "int(3)"}
        self.fn = fn

    def call(self, args):
        try:
            result = self.fn(args)
            result = result if isinstance(result, dict) else {"value": result}
            result = {"ok": True, **result}
        except Exception as e:
            result = {"ok": False, "error": str(e)}
        return result

    def __repr__(self):
        return f"{self.name}({self.args_schema})"


_registry = {}


def register(tool):
    _registry[tool.name] = tool


def get(name):
    return _registry.get(name)


def all_tools():
    return list(_registry.values())


def describe():
    lines = []
    for t in _registry.values():
        lines.append(f"- {t.name}({json.dumps({'args': t.args_schema}, ensure_ascii=False)}): {t.description}")
    return "\n".join(lines)
