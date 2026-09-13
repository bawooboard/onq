"""샘플 사내 DB 시뮬레이터 데이터를 생성한다.

- sales.json          : 월별 × 지역별 × 제품군 매출 (단위: 원)
- leave_balances.json : 사원별 연차 잔여 일수

재현 가능하도록 시드를 고정했다. python scripts/gen_sample_data.py
"""
import json
import os
import pathlib
import random

OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "db"
OUT.mkdir(parents=True, exist_ok=True)
rng = random.Random(2026)

# ONQ_SAMPLE_SCALE: 전체 매출 규모 보정 계수 (영업 리포트 문서와 일관되게 맞춤)
SCALE = float(os.environ.get("ONQ_SAMPLE_SCALE", "1.0"))

PRODUCTS = ["냉장고", "세탁기", "에어컨", "인덕션", "로봇청소기", "공기청정기"]
REGIONS = ["서울", "경기", "부산", "대구", "대전", "광주"]
MONTHS = [f"2025-{m:02d}" for m in range(11, 13)] + [f"2026-{m:02d}" for m in range(1, 9)]

REGION_F = {"서울": 1.18, "경기": 1.06, "부산": 0.92, "대구": 0.88,
            "대전": 0.72, "광주": 0.62}
PRODUCT_BASE = {"냉장고": 420_000_000, "세탁기": 210_000_000, "에어컨": 160_000_000,
                "인덕션": 95_000_000, "로봇청소기": 55_000_000, "공기청정기": 40_000_000}
SEASON = {"2025-11": 1.02, "2025-12": 1.10, "2026-01": 0.98, "2026-02": 0.95,
          "2026-03": 1.12, "2026-04": 1.00, "2026-05": 1.05, "2026-06": 1.25,
          "2026-07": 1.45, "2026-08": 1.38}

rows = []
for month in MONTHS:
    for region in REGIONS:
        for product in PRODUCTS:
            base = PRODUCT_BASE[product] * SCALE * REGION_F[region] * SEASON.get(month, 1.0)
            jitter = rng.uniform(0.82, 1.20)
            amount = int(round(base * jitter / 10000) * 10000)
            units = max(1, int(round(amount / rng.uniform(1_000_000, 2_800_000))))
            rows.append({"month": month, "region": region, "product": product,
                         "amount": amount, "units": units})
rows.sort(key=lambda r: (r["month"], r["region"], r["product"]))

(OUT / "sales.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                encoding="utf-8")

EMP = [
    ("E001", "홍길동", "영업1팀", 12.5),
    ("E002", "김지영", "재무팀", 8.0),
    ("E003", "박민수", "IT운영팀", 21.0),
    ("E004", "이수진", "HR팀", 15.0),
    ("E005", "최현우", "생산팀", 3.5),
    ("E006", "정다은", "영업2팀", 18.0),
]
bal = [{"employee_id": e, "name": n, "department": d, "leave_balance": b}
       for e, n, d, b in EMP]
(OUT / "leave_balances.json").write_text(json.dumps(bal, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
print(f"sales: {len(rows)} rows, leave_balances: {len(bal)} rows")
