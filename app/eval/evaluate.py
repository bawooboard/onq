"""검증(평가) 모듈.

1단계(리트리버 평가): QA 평가셋 → recall@k, hit-rate  (LLM 불필요)
2단계(생성 평가):   LLM-as-Judge 로 충실성(faithfulness)·관련성(relevance) 1~5점
LLM 미연결 시 1단계만 수행하고 그 사실을 보고서에 명시한다.
"""
import json
import re

JUDGE_PROMPT = (
    "당신은 RAG 시스템 평가자입니다. 아래 질문-문서 쌍을 평가합니다.\n\n"
    "질문: {question}\n\n"
    "검색된 문서 내용:\n{context}\n\n"
    "평가 기준:\n"
    "- faithfulness(충실성): 이 문서로부터 질문에 대한 답을 도출했을 때 "
    "답변이 문서 내용과 모순되지 않고 잘 근거하는 정도 (1~5)\n"
    "- relevance(관련성): 이 문서가 질문에 실제로 관련된 정도 (1~5)\n\n"
    "숫자만 포함된 JSON으로 답하세요 (설명 금지):\n"
    '{"faithfulness": <1-5>, "relevance": <1-5>}'
)


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _recall_for(hits, expected):
    hit_docs = set(h["source"] for h in hits)
    exp = set(expected)
    if not exp:
        return None
    return len(exp & hit_docs) / len(exp)


def run_eval(kb, llm, questions, top_k=5, n_judge=None):
    per_q = []
    for q in questions:
        hits = kb.search(q["question"], k=top_k)
        per_q.append({
            "id": q["id"],
            "question": q["question"],
            "expected_docs": q.get("expected_docs", []),
            "recall_at_k": _recall_for(hits, q.get("expected_docs", [])),
            "hit_docs": sorted({h["source"] for h in hits}),
        })

    recalls = [r["recall_at_k"] for r in per_q if r["recall_at_k"] is not None]
    retrieval = {
        "retriever": kb.retriever,
        "k": top_k,
        "questions_with_gold": len(recalls),
        "recall_at_k_avg": _avg(recalls),
        "hit_rate": round(sum(1 for r in recalls if r > 0) / len(recalls), 4) if recalls else None,
    }

    judged = []
    if llm is not None and llm.is_available():
        todo = questions if n_judge is None else questions[:n_judge]
        for q in todo:
            hits = kb.search(q["question"], k=1)
            ctx = hits[0]["text"] if hits else "(검색 결과 없음)"
            prompt = JUDGE_PROMPT.format(question=q["question"], context=ctx)
            try:
                text = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
                nums = re.findall(r"\d+", text)
                if len(nums) >= 2:
                    judged.append({"id": q["id"],
                                   "faithfulness": int(nums[-2]),
                                   "relevance": int(nums[-1])})
            except Exception:
                continue

    generation = {
        "llm_as_judge_available": bool(judged),
        "judged_samples": len(judged),
        "avg_faithfulness": _avg([j["faithfulness"] for j in judged]),
        "avg_relevance": _avg([j["relevance"] for j in judged]),
    }

    report = {
        "dataset_size": len(questions),
        "retrieval": retrieval,
        "generation": generation,
        "per_question": per_q,
        "judge_scores": judged,
    }
    return report


def print_report(report):
    R = report
    print("=" * 58)
    print("OnQ 검증 보고서")
    print("=" * 58)
    r = R["retrieval"]
    print(f"[리트리버] {r['retriever']} / k={r['k']}")
    print(f"  평가 문항(gold 포함) : {r['questions_with_gold']} / {R['dataset_size']}")
    print(f"  recall@k 평균        : {r['recall_at_k_avg']}")
    print(f"  hit-rate             : {r['hit_rate']}")
    g = R["generation"]
    if g["llm_as_judge_available"]:
        print(f"[LLM 판정] {g['judged_samples']}문항")
        print(f"  충실성(faithfulness) 평균: {g['avg_faithfulness']} / 5")
        print(f"  관련성(relevance) 평균   : {g['avg_relevance']} / 5")
    else:
        print("[LLM 판정] LLM 서버 미연결 → 건너뜀 (연결 후 재실행하면 반영)")
    print("=" * 58)
    for p in R["per_question"]:
        rec = "%.2f" % p["recall_at_k"] if p["recall_at_k"] is not None else "  -"
        print(f"  {rec}  {p['question'][:42]}  -> {', '.join(p['hit_docs'][:2])}")
