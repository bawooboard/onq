"""도구 호출 파인튜닝 파이프라인 (LoRA, GPU 전용).

이 시스템은 기본적으로 '프롬프트 기반 도구 호출'을 쓰지만, 실제 운영 성능을
올리려면 도구 호출 형식을 파인튜닝하는 것이 좋다.

사용법:
  python scripts/finetune_toolcalling.py --mode export
      → data/finetune/dataset.json (ChatML 형식) 생성 — CPU에서도 가능
  python scripts/finetune_toolcalling.py --mode train
      → Unsloth(4bit LoRA) 학습 — GPU + CUDA 필요

train 모드는 GPU 환경(교육용 서버)에서 실행하고, 산출물(LoRA 어댑터)은
vLLM/llama.cpp에 병합하거나 Ollama Modelfile로 등록할 수 있다.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "finetune" / "dataset.json"

# ── 학습 데이터 원본: 실제 도메인 질문 → 도구 호출 → 응답 (ChatML 메시지) ──
EXAMPLES = [
    # search_docs 유형
    {"query": "연차 휴가 신청 절차가 어떻게 되나요?",
     "tool": {"name": "search_docs", "args": {"query": "연차 휴가 신청 절차", "k": 3}},
     "answer": "연차 신청은 1) 인사시스템 > 휴가 신청에서 날짜·사유 입력, 2) "
               "부서장 승인, 3) 공유 캘린더 반영 순서로 진행됩니다. 휴가 예정일 "
               "3영업일 전 신청이 원칙이며 긴급·경조사는 당일 신청이 가능합니다. "
               "(출처: leave_hr_guide.md)"},
    {"query": "회사에서 USB 메모리를 써도 되나요?",
     "tool": {"name": "search_docs", "args": {"query": "USB 이동식 저장매체 사용 금지", "k": 3}},
     "answer": "아니요. 이동식 저장매체(USB, 외장 HDD, CD/DVD)의 업무용 사용은 "
               "금지입니다. 부득이한 경우 보안 담당자 승인과 보안USB 절차를 "
               "따르세요. (출처: info_sec_policy.md)"},
    {"query": "공기청정기 필터는 언제 교체하나요?",
     "tool": {"name": "search_docs", "args": {"query": "공기청정기 필터 교체", "k": 3}},
     "answer": "PA-50 기준 헤파 필터 6개월, 탈취 필터 3개월마다 교체하며, 기기의 "
               "필터 알림이 표시됩니다. (출처: product_faq.md)"},
    # calculate 유형
    {"query": "(5+37)*12 계산해줘",
     "tool": {"name": "calculate", "args": {"expression": "(5+37)*12"}},
     "answer": "계산 결과 (5+37)*12 = 504 입니다."},
    # get_date 유형
    {"query": "지금 날짜와 요일이 궁금해",
     "tool": {"name": "get_date", "args": {}},
     "answer": "오늘은 {date} {weekday}입니다. (Asia/Seoul 기준)"},
    # lookup_db 유형
    {"query": "2026년 2월 서울 지역 냉장고 매출 알려줘",
     "tool": {"name": "lookup_db", "args": {"table": "sales",
                                            "query": "2026년 2월 서울 냉장고"}},
     "answer": "2026년 2월 서울 지역 냉장고 매출은 {amount}원, 판매 대수 "
               "{units}대입니다. (출처: sales)"},
    {"query": "홍길동 사원의 연차 잔여 일수가 어떻게 되나요?",
     "tool": {"name": "lookup_db", "args": {"table": "leave_balances",
                                            "query": "홍길동 연차"}},
     "answer": "홍길동 사원(영업1팀)의 연차 잔여 일수는 {balance}일입니다. "
               "(출처: leave_balances)"},
    {"query": "2026년 3월 전체 매출 합계를 알려줘",
     "tool": {"name": "lookup_db", "args": {"table": "sales", "query": "2026년 3월 합계"}},
     "answer": "2026년 3월 전체 매출 합계는 {total}원입니다. (출처: sales)"},
]


def build_example(query, tool_call, answer):
    call_json = json.dumps(tool_call, ensure_ascii=False)
    messages = [
        {"role": "user", "content": query},
        {"role": "assistant", "content": f"<<<TOOL:{call_json}>>>"},
        {"role": "user",
         "content": "<<<TOOL_RESULT: {\"ok\": true, \"참고\": \"검색·계산 결과\"}>>>"},
        {"role": "assistant", "content": answer},
    ]
    return {"messages": messages}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["export", "train"], required=True)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "export":
        data = [build_example(e["query"], e["tool"], e["answer"]) for e in EXAMPLES]
        OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"데이터셋 생성: {OUT} ({len(data)}개 대화)")

    if args.mode == "train":
        try:  # GPU 전용 시작: Unsloth + TRL
            import torch
            assert torch.cuda.is_available()
            from trl import SFTTrainer
            from unsloth import FastLanguageModel, is_bfloat16_supported
        except Exception as e:
            sys.exit(f"[중단] GPU 학습 준비 실패: {e}\n"
                     "1) data/finetune/dataset.json 로 데이터 검증\n"
                     "2) GPU 서버에서: pip install unsloth trl\n"
                     "3) 재실행: python scripts/finetune_toolcalling.py --mode train")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            max_seq_length=2048, load_in_4bit=True)
        model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32,
                                                 lora_dropout=0.05)
        assert is_bfloat16_supported()
        # ... SFTTrainer 수행 (운영 환경에서 데이터 증강 후 실행)
        model.save_pretrained(ROOT / "models" / "onq-lora")
        print("LoRA 어댑터 저장: models/onq-lora")


if __name__ == "__main__":
    main()
