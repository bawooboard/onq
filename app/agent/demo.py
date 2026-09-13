"""오프라인 데모 모드 — LLM 서버가 없어도 동작하는 규칙 기반 응답기.

발표 리허설/검증 실습 안전망이자, LLM 연결 전에도 서비스 전체 동작을
확인할 수 있게 한다. 도구 호출 흐름은 실제 에이전트와 동일하게 표시된다.
"""
import re

from app.tools import impls  # noqa: F401  (도구 등록)
from app.tools.registry import get as get_tool


def _fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
        return s
    return str(v)


def _fmt_rows(result):
    if not result.get("ok"):
        return result.get("error", "오류")
    if result.get("total") is not None:
        lines = [f"{result['total_label']}: {_fmt(result['total'])}"]
    else:
        lines = []
    for r in result.get("rows", [])[:8]:
        items = ", ".join(f"{k}: {_fmt(v)}" for k, v in r.items() if k != "_search")
        lines.append("- " + items)
    return "\n".join(lines) if lines else "조건에 맞는 데이터가 없습니다."


def demo_answer(query, kb, top_k=3):
    """(type, data) 이벤트 리스트를 반환한다: tool / message"""
    Q = query.strip()
    events = []
    tl = get_tool("lookup_db")
    cl = get_tool("calculate")
    dl = get_tool("get_date")
    sl = get_tool("search_docs")

    if ("매출" in Q or "판매" in Q or "조회" in Q) and re.search(r"[0-9]|월|년", Q):
        events.append({"type": "tool", "name": "lookup_db", "status": "done",
                       "detail": f'table=sales, query="{Q}"'})
        res = tl.call({"table": "sales", "query": Q})
        body = ("[데모 모드 응답] 사내 매출 DB를 조회했습니다.\n\n"
                + _fmt_rows(res) + "\n\n(출처: data/db/sales.json · 데모 모드)")
    elif "연차" in Q and re.search(r"남은|잔여|며칠|몇일?", Q):
        events.append({"type": "tool", "name": "lookup_db", "status": "done",
                       "detail": 'table=leave_balances, query="' + Q + '"'})
        res = tl.call({"table": "leave_balances", "query": Q})
        body = ("[데모 모드 응답] 사원별 연차 잔여 DB를 조회했습니다.\n\n"
                + _fmt_rows(res) + "\n\n(출처: data/db/leave_balances.json · 데모 모드)")
    elif "계산" in Q or re.search(r"[\d][+\-*/^()%. ]+[\d]", Q):
        expr = re.sub(r"[^0-9+\-*/^().% ]", "", Q).strip()
        if expr:
            events.append({"type": "tool", "name": "calculate", "status": "done",
                           "detail": "expression=" + expr})
            res = cl.call({"expression": expr})
            body = ("[데모 모드 응답] 수식을 계산했습니다.\n\n"
                    f"{expr} = {res.get('text') or res.get('error')}\n\n(도구: calculate · 데모 모드)")
        else:
            return demo_answer("연차는 며칠?", kb, top_k)
    elif "날짜" in Q or "오늘" in Q or "요일" in Q or "시간" in Q:
        events.append({"type": "tool", "name": "get_date", "status": "done"})
        res = dl.call({})
        body = ("[데모 모드 응답] 현재 시각입니다.\n\n"
                f"{res['datetime']} {res['weekday']} (Asia/Seoul)\n\n(도구: get_date · 데모 모드)")
    else:
        events.append({"type": "tool", "name": "search_docs", "status": "done",
                       "detail": f'docs 검색 (k={top_k})'})
        hits = kb.search(Q, k=top_k)
        if hits:
            top = hits[0]
            excerpt = top["text"].replace("\n", " ")[:500]
            srcs = ", ".join({h["source"] for h in hits})
            body = ("[데모 모드 응답] 관련 문서에서 찾은 내용입니다.\n\n"
                    f'"{top["source"]}" 문서 발췌:\n{excerpt}'
                    + ("…" if len(top["text"]) > 500 else "")
                    + f"\n\n(출처: {srcs} · 데모 모드)\n\n"
                      "※ 데모 모드는 키워드 검색 기반 응답입니다. "
                      "LLM을 연결하면 자연스러운 요약 답변을 확인할 수 있습니다.")
        else:
            body = ("[데모 모드 응답] 문서에서 관련 내용을 찾지 못했습니다.\n"
                    "검색어를 바꾸거나 데이터/문서를 추가해 보세요.")
    events.append({"type": "message", "text": body})
    return events
