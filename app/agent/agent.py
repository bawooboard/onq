"""핵심: 경량 LLM 에이전트 루프.

소형(7~9B급) 온프레미스 모델에서도 안정적으로 동작하도록
도구 호출을 간단한 마커 형식으로 지시하고, 한 번에 한 도구만 실행한다.

<<<TOOL:{"name": "...", "args": {...}}}>>>
   → 실행 후 결과를 <<<TOOL_RESULT: {...}>>> 로 돌려주고 최종 답변을 받는다.
"""
import json
import re

TOOL_PAT = re.compile(r"<<<TOOL:(.*?)>>>", re.S)


def build_system_prompt(tool_desc, top_k):
    return (
        "당신은 사내 문서 지식 에이전트 'OnQ'입니다. 회사 내부 문서와 사내 DB를 "
        "바탕으로 직원 질문에 정확하고 간결하게 답변합니다.\n\n"
        "규칙:\n"
        "1. 답변에 도움이 되는 도구가 있으면 반드시 먼저 사용합니다.\n"
        "   도구 호출은 정확히 이 형식으로, 한 번에 하나씩만 합니다:\n"
        '   <<<TOOL:{"name":"...","args":{...}}}>>>\n'
        "2. 도구 결과가 <<<TOOL_RESULT: ... >>> 로 전달되면 이를 참고해 한국어로 "
        "최종 답변을 작성합니다.\n"
        "3. 답변 끝에 참고한 문서 출처를 '(출처: 파일명)' 형태로 붙입니다.\n"
        "4. 문서/DB에 근거가 없으면 추측하지 말고 '관련 내용을 찾지 못했습니다'라고 "
        "답합니다.\n"
        "5. 숫자·금액은 천 단위 콤마로 표기하고, 업무에서 바로 쓸 수 있는 톤으로 "
        "답합니다.\n\n"
        f"사용 가능한 도구:\n{tool_desc}\n"
        f"문서 검색은 최대 {top_k}건입니다."
    )


def _parse_tool_call(text):
    m = TOOL_PAT.search(text)
    if not m:
        return None
    inner = m.group(1).strip()
    if inner.startswith("```"):
        inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", inner, flags=re.S).strip()
    try:
        return json.loads(inner)
    except Exception:
        return None


def run_agent_task(llm, kb, tools, query, history, top_k=3, max_steps=4):
    """에이전트 루프 실행. (최종답변, 이벤트목록) 반환."""
    events = []
    messages = [{"role": "system",
                 "content": build_system_prompt(tools.describe(), top_k)}]
    messages += history[-8:]
    messages.append({"role": "user", "content": query})

    final, last_resp = "", ""
    for _ in range(max_steps):
        last_resp = llm.chat(messages)
        call = _parse_tool_call(last_resp)
        if not call:
            final = last_resp.strip()
            break
        name, args = call.get("name"), call.get("args") or {}
        tool = tools.get(name)
        if tool is None:
            messages.append({"role": "user",
                             "content": f"'{name}'는 존재하지 않는 도구입니다. "
                                        "정의된 도구 이름만 사용하세요."})
            continue
        events.append({"type": "tool", "name": name, "status": "running"})
        result = tool.call(args) if isinstance(args, dict) else tool.call({"query": str(args)})
        events.append({"type": "tool", "name": name, "status": "done"})
        snippet = json.dumps(result, ensure_ascii=False)[:2000]
        messages.append({"role": "user",
                         "content": f"<<<TOOL_RESULT: {snippet}>>>\n"
                                    "도구 결과를 참고해 최종 답변을 한국어로 작성하세요."})
    else:
        final = last_resp.strip() if last_resp else ""

    if not final:
        final = "응답을 생성하지 못했습니다. 질문을 다르게 표현해 보세요."
    if not events:
        events.append({"type": "tool", "name": "none", "status": "skipped",
                       "detail": "도구 호출 없이 바로 답변"})
    return final, events
