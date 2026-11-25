"""raw_*.txt (KOSIS 화면에서 옮긴 원자료) 5개를 앱이 읽는 정돈된 CSV 4개로 바꾼다.

실행: python build_data.py   → annual.csv, regional_srb.csv, order_srb.csv, monthly.csv (같은 폴더)

집계 기준
- 띠는 '양력 연도'로 붙인다: (연도 - 4) % 12 → 0 쥐 … 11 돼지. (실제 띠 경계는 설날/입춘이라 1~2월생 일부는 앞 띠 → 한계로 명시)
- 천간 색: 갑을=청, 병정=적, 무기=황, 경신=백, 임계=흑 (예: 1990 경오=백말, 2007 정해, 2019 기해=황금돼지, 2026 병오=붉은말)
- 추세 제거 편차: 그 해 값 ÷ (앞뒤 2년씩 4개 해의 평균) - 1  (출생성비는 %가 아니라 '포인트' 차이)
  처음·마지막 2개 해는 앞뒤 값이 모자라 빈칸으로 둔다.
- 연간 혼인 건수는 월별 표(DT_1B8000G)를 연도별로 더한 값이다 (1981~2025).
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE
OUT = HERE

ZODIAC = ["쥐", "소", "범", "토끼", "용", "뱀", "말", "양", "원숭이", "닭", "개", "돼지"]
STEM = ["갑", "을", "병", "정", "무", "기", "경", "신", "임", "계"]
BRANCH = ["자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해"]
STEM_COLOR = ["청", "청", "적", "적", "황", "황", "백", "백", "흑", "흑"]


def zodiac_cols(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    y = df[year_col]
    df["zodiac_no"] = (y - 4) % 12
    df["zodiac"] = df["zodiac_no"].map(dict(enumerate(ZODIAC)))
    df["ganji"] = [STEM[(v - 4) % 10] + BRANCH[(v - 4) % 12] for v in y]
    df["stem_color"] = [STEM_COLOR[(v - 4) % 10] for v in y]
    return df


def neighbor_mean(s: pd.Series) -> pd.Series:
    """앞뒤 2년씩(자기 자신 제외) 4개 해의 평균. 하나라도 비면 NaN."""
    parts = [s.shift(k) for k in (2, 1, -1, -2)]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def num(x: str) -> float:
    return float("nan") if x == "-" else float(x.replace(",", ""))


# 1) 연간 표: 출생아 수·합계출산율·출생성비 (1970~2025) ---------------------------------
lines = [l.split() for l in (RAW / "raw_inh_1b8000f_01.txt").read_text(encoding="utf-8").splitlines()
         if l and not l.startswith("#")]
wide = {row[0]: row[1:] for row in lines}
annual = pd.DataFrame({
    "year": [int(v) for v in wide["years"]],
    "births": [int(num(v)) for v in wide["births"]],
    "tfr": [num(v) for v in wide["tfr"]],
    "srb": [num(v) for v in wide["srb"]],
})

# 2) 월별 표: 출생아 수·혼인 건수 (1981.01~2026.08, 2026 = 잠정) ---------------------------
mrows = [l.split() for l in (RAW / "raw_dt_1b8000g_monthly.txt").read_text(encoding="utf-8").splitlines()
         if l and l[0].isdigit()]
monthly = pd.DataFrame({
    "year": [int(m[:4]) for m, _, _ in mrows],
    "month": [int(m[5:7]) for m, _, _ in mrows],
    "births": [int(num(b)) for _, b, _ in mrows],
    "marriages": [int(num(c)) for _, _, c in mrows],
    "provisional": [m.endswith("p") for m, _, _ in mrows],
})

full_years = monthly.groupby("year")["month"].count().pipe(lambda c: c[c == 12].index)
mar_year = (monthly[monthly["year"].isin(full_years)]
            .groupby("year", as_index=False)["marriages"].sum())
annual = annual.merge(mar_year, on="year", how="left")
annual["marriages"] = annual["marriages"].astype("Int64")

# 검산: 월별 출생아 합 = 연간 출생아 수
chk = (monthly[monthly["year"].isin(full_years)].groupby("year")["births"].sum()
       .rename("births_m").reset_index().merge(annual[["year", "births"]], on="year"))
assert (chk["births_m"] == chk["births"]).all(), chk[chk["births_m"] != chk["births"]]

# 추세 제거 편차
for col in ["births", "tfr", "marriages"]:
    base = neighbor_mean(annual[col].astype("float"))
    annual[f"{col}_base"] = base.round(3)
    annual[f"{col}_dev_pct"] = ((annual[col].astype("float") / base - 1) * 100).round(2)
annual["srb_base"] = neighbor_mean(annual["srb"]).round(2)
annual["srb_dev_pt"] = (annual["srb"] - annual["srb_base"]).round(2)
annual = zodiac_cols(annual)

# 3) 시도별 총출생성비 (1990~2025) -> 긴 형식 ------------------------------------------------
rtxt = [l.split() for l in (RAW / "raw_dt_1b81a19_total.txt").read_text(encoding="utf-8").splitlines()
        if l and not l.startswith("#")]
cols = rtxt[0][1:]
regional = pd.DataFrame([[int(r[0])] + [num(v) for v in r[1:]] for r in rtxt[1:]],
                        columns=["year"] + cols)
assert regional.shape == (36, 19), regional.shape
# 검산: 전국 열 = 연간 표 출생성비
m = regional[["year", "전국"]].merge(annual[["year", "srb"]], on="year")
assert ((m["전국"] - m["srb"]).abs() < 0.05).all()
regional = (regional.melt(id_vars="year", var_name="sido", value_name="srb")
            .dropna(subset=["srb"]).sort_values(["year", "sido"]).reset_index(drop=True))
regional = zodiac_cols(regional)

# 4) 전국 출산순위별 출생성비 (1981~2025) -------------------------------------------------
#    DT_1B80A01 의 남·여 출생아 수로 직접 계산: 성비 = 남아 수 / 여아 수 × 100 (출산순위 '미상'은 제외)
#    셋째아 이상 = 3아 + 4아 + … + 8아이상
stxt = [l.split() for l in (RAW / "raw_dt_1b80a01_sex_order.txt").read_text(encoding="utf-8").splitlines()
        if l and not l.startswith("#")]
s_years = [int(v) for v in stxt[0][1:]]
cnt = {(r[0], r[1]): [int(num(v)) for v in r[2:]] for r in stxt[1:]}
ORDER_GROUPS = {"첫째아": ["1아"], "둘째아": ["2아"],
                "셋째아 이상": ["3아", "4아", "5아", "6아", "7아", "8아이상"]}
rows = []
for i, y in enumerate(s_years):
    for name, parts in ORDER_GROUPS.items():
        boys = sum(cnt[("남자", p)][i] for p in parts)
        girls = sum(cnt[("여자", p)][i] for p in parts)
        rows.append({"year": y, "order": name, "boys": boys, "girls": girls,
                     "srb": round(boys / girls * 100, 1)})
order = pd.DataFrame(rows)
# 검산 1: 남+여 총계 = 연간 출생아 수
tot = pd.DataFrame({"year": s_years,
                    "b": [m + f for m, f in zip(cnt[("남자", "총계")], cnt[("여자", "총계")])]})
assert (tot.merge(annual[["year", "births"]], on="year").eval("b == births")).all()
# 검산 2: 계산한 순위별 성비 = KOSIS 「시도/출산순위별 출생성비」(DT_1B81A19) 전국 값 (1990~2025)
otxt = [l.split() for l in (RAW / "raw_dt_1b81a19_order_national.txt").read_text(encoding="utf-8").splitlines()
        if l and l[0].isdigit()]
ref = pd.DataFrame([[int(r[0])] + [num(v) for v in r[1:]] for r in otxt],
                   columns=["year", "첫째아", "둘째아", "셋째아 이상"]).melt(
                       id_vars="year", var_name="order", value_name="srb_ref")
m2 = order.merge(ref, on=["year", "order"])
assert len(m2) == 108 and ((m2["srb"] - m2["srb_ref"]).abs() < 0.05).all()
order["srb_base"] = order.groupby("order")["srb"].transform(neighbor_mean).round(2)
order["srb_dev_pt"] = (order["srb"] - order["srb_base"]).round(2)
order = zodiac_cols(order.sort_values(["order", "year"]).reset_index(drop=True))

monthly = zodiac_cols(monthly)

annual.to_csv(OUT / "annual.csv", index=False, encoding="utf-8-sig")
regional.to_csv(OUT / "regional_srb.csv", index=False, encoding="utf-8-sig")
order.to_csv(OUT / "order_srb.csv", index=False, encoding="utf-8-sig")
monthly.to_csv(OUT / "monthly.csv", index=False, encoding="utf-8-sig")

for name, df in [("annual", annual), ("regional_srb", regional), ("order_srb", order), ("monthly", monthly)]:
    print(f"{name:13s} {df.shape[0]:4d}행 × {df.shape[1]:2d}열  {df['year'].min()}~{df['year'].max()}")
print("검산 통과: 월별 출생아 합 = 연간 출생아 수(1981~2025), 시도표 전국 = 연간표 출생성비(1990~2025),")
print("          남+여 = 연간 출생아 수(1981~2025), 계산한 순위별 성비 = KOSIS 순위별 성비(1990~2025)")
