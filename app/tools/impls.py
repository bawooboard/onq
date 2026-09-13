"""OnQ 도구 구현 4종:
- search_docs : 사내 문서 검색 (RAG)
- calculate   : 안전한 수식 계산 (ast 파서)
- get_date    : 현재 날짜/시간 (Asia/Seoul)
- lookup_db   : 사내 DB 시뮬레이터 조회 (data/db/*.json)
"""
import ast
import json
import operator
import pathlib
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from .registry import ToolDef, register

_kb = {}          # set_kb() 로 주입
_db_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "db"
_db_cache = {}


def set_kb(kb):
    _kb["kb"] = kb


# ── search_docs ────────────────────────────────────────────────
def _search_docs(args):
    kb = _kb.get("kb")
    query = str(args.get("query", "")).strip()
    k = int(args.get("k", 3))
    if not query:
        return {"error": "검색어가 비어 있습니다."}
    hits = kb.search(query, k=k)
    if not hits:
        return {"error": "관련 문서가 없습니다.", "top": []}
    return {"top": hits, "count": len(hits)}


# ── calculate ──────────────────────────────────────────────────
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return +_eval_node(node.operand)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError("허용되지 않는 수식")


def _calculate(args):
    expr = str(args.get("expression", "")).replace("^", "**").strip()
    if not expr:
        return {"error": "수식이 비어 있습니다."}
    tree = ast.parse(expr, mode="eval")
    val = _eval_node(tree.body)
    if isinstance(val, float) and val.is_integer():
        val = int(val)
    if isinstance(val, int):
        text = f"{val:,}"
    else:
        text = f"{val:,.4f}".rstrip("0").rstrip(".")
    return {"expression": expr, "result": val, "text": text}


# ── get_date ───────────────────────────────────────────────────
def _get_date(args):
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    week = "월화수목금토일"[now.weekday()]
    return {"datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "weekday": f"{week}요일",
            "timezone": "Asia/Seoul"}


# ── lookup_db (사내 DB 시뮬레이터) ────────────────────────────
def _load_table(name):
    if name in _db_cache:
        return _db_cache[name]
    p = _db_dir / f"{name}.json"
    if not p.exists():
        raise ValueError(f"테이블이 없습니다: {name}")
    rows = json.loads(p.read_text(encoding="utf-8"))
    for r in rows:  # 검색용 별칭 칼럼
        aliases = [str(r.get("month", "")), str(r.get("employee_id", ""))]
        for key in ("month", "product", "region", "department", "name"):
            v = r.get(key)
            if v:
                aliases.append(str(v))
        if r.get("month"):
            m = r["month"]
            aliases.append(f"{int(m.split('-')[1])}월")
            aliases.append(f"{m.split('-')[0]}년")
        r["_search"] = " ".join(aliases)
    _db_cache[name] = rows
    return rows


def _lookup_db(args):
    table = str(args.get("table", "")).strip()
    query = str(args.get("query", "")).strip()
    rows = _load_table(table)
    if table == "sales":
        amount_key = "amount"
    elif table == "leave_balances":
        amount_key = "leave_balance"
    else:
        amount_key = None

    toks = [t for t in re.split(r"[\s,]+", query) if t]
    if not toks:
        picked = rows[:8]
        return {"rows": picked, "count": len(rows), "note": "전체 조회 (최대 8건)"}
    scored = []
    for r in rows:
        s = r["_search"]
        score = sum(1 for t in toks if t in s)
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    picked = [r for _, r in scored[:8]]
    result = {"rows": picked, "count": len(picked)}
    # 합계는 '가장 많이 맞은 조건(=최고 스코어)의 행'만 합산 (월 필터 유지)
    if amount_key and any(k in query for k in ("합계", "총", "전체")) and scored:
        best = scored[0][0]
        total = sum(float(r.get(amount_key, 0)) for s, r in scored if s == best)
        result["total"] = total
        result["total_label"] = "합계"
    return result


# ── 등록 ───────────────────────────────────────────────────────
register(ToolDef(
    name="search_docs",
    description="사내 문서(규정·SOP·매뉴얼·FAQ)를 검색하고 관련 문단을 반환한다.",
    args_schema={"query": "string(필수)", "k": "int(기본 3)"},
    fn=_search_docs))

register(ToolDef(
    name="calculate",
    description="산술 수식을 계산한다. + - * / ** % 와 괄호 사용 가능.",
    args_schema={"expression": "string(필수, 예: '(5+37)*12')"},
    fn=_calculate))

register(ToolDef(
    name="get_date",
    description="현재 날짜, 시간, 요일을 반환한다. 시간 관련 질문에 사용.",
    args_schema={},
    fn=_get_date))

register(ToolDef(
    name="lookup_db",
    description="사내 DB를 조회한다. table: sales(월별 제품군/지역 매출, amount), "
                "leave_balances(사원별 연차 잔여, leave_balance). "
                "합계가 필요하면 query에 '합계' 또는 '총'을 포함한다.",
    args_schema={"table": "string(필수: sales|leave_balances)",
                 "query": "string(필수, 예: '2026년 2월 서울 냉장고', '2026년 2월 합계')"},
    fn=_lookup_db))
