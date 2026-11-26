"""KOSIS에서 내려받은 원본 CSV 4개(kosis_*.csv, 내려받은 그대로)를 앱이 읽는 정돈된 CSV 4개로 바꾼다.

원본: 국가데이터처 「인구동향조사」, KOSIS 국가통계포털에서 2026-11-26 CSV로 다운로드
- kosis_INH_1B8000F_01.csv  출생아수, 합계출산율, 자연증가 등 (1970~2025)
- kosis_DT_1B8000G.csv      월.분기.연간 인구동향 — 월 (1981.01~2026.09, 2026년은 잠정치)
- kosis_DT_1B80A01.csv      성/모의 연령/출산순위별 출생 (1981~2025)
- kosis_DT_1B81A19.csv      시도/출산순위별 출생성비 (1990~2025)
KOSIS 다운로드 파일은 한글 윈도우 인코딩(cp949)이라 그 인코딩으로 읽는다.

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


def num(x) -> float:
    x = str(x).strip().replace(",", "")
    return float("nan") if x in ("-", "", "nan") else float(x)


def read_kosis(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RAW / name, encoding="cp949", dtype=str, **kw)


# 1) 연간 표: 출생아 수·합계출산율·출생성비 (1970~2025) ---------------------------------
a = read_kosis("kosis_INH_1B8000F_01.csv").set_index("기본항목별")
annual = pd.DataFrame({
    "year": [int(c) for c in a.columns],
    "births": [int(num(v)) for v in a.loc["출생아수(명)"]],
    "tfr": [num(v) for v in a.loc["합계출산율(명)"]],
    "srb": [num(v) for v in a.loc["출생성비(명)"]],
})

# 2) 월별 표: 전국 출생아 수·혼인 건수 (1981.01~2026.09, 2026 = 잠정) ----------------------
mt = read_kosis("kosis_DT_1B8000G.csv")
mt = mt[mt["행정구역별"] == "전국"].set_index("종류별")
mcols = [c for c in mt.columns if c[:4].isdigit()]          # "1981.01 월" ...
monthly = pd.DataFrame({
    "year": [int(c[:4]) for c in mcols],
    "month": [int(c[5:7]) for c in mcols],
    "births": [int(num(v)) for v in mt.loc["출생아수(명)", mcols]],
    "marriages": [int(num(v)) for v in mt.loc["혼인건수(건)", mcols]],
})
monthly["provisional"] = monthly["year"] >= 2026             # KOSIS 표기: 2026년 = 잠정치

full_years = monthly.groupby("year")["month"].count().pipe(lambda c: c[c == 12].index)
mar_year = (monthly[monthly["year"].isin(full_years)]
            .groupby("year", as_index=False)["marriages"].sum())
annual = annual.merge(mar_year, on="year", how="left")
annual["marriages"] = annual["marriages"].astype("Int64")

# 자동 확인: 월별 출생아 합 = 연간 출생아 수
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

# 3) 시도별 출생성비 (1990~2025): 첫 줄 = 연도, 둘째 줄 = 항목(총출생성비, 첫째아 …) ------------
sr = pd.read_csv(RAW / "kosis_DT_1B81A19.csv", encoding="cp949", header=None, dtype=str)
years, items = sr.iloc[0, 1:].astype(int).tolist(), sr.iloc[1, 1:].tolist()
long = pd.DataFrame([(sr.iloc[i, 0], y, it, num(v))
                     for i in range(2, len(sr)) if isinstance(sr.iloc[i, 0], str)
                     for y, it, v in zip(years, items, sr.iloc[i, 1:])],
                    columns=["sido", "year", "item", "srb"])
regional = (long[long["item"] == "총출생성비"].drop(columns="item")
            .dropna(subset=["srb"]).sort_values(["year", "sido"]).reset_index(drop=True))[["year", "sido", "srb"]]
# 자동 확인: 시도표 '전국' = 연간 표 출생성비
m = regional[regional["sido"] == "전국"].merge(annual[["year", "srb"]], on="year")
assert len(m) == 36 and ((m["srb_x"] - m["srb_y"]).abs() < 0.05).all()
regional = zodiac_cols(regional)

# 4) 전국 출산순위별 출생성비 (1981~2025) -------------------------------------------------
#    DT_1B80A01 의 남·여 출생아 수로 계산: 성비 = 남아 수 / 여아 수 × 100 (출산순위 '미상'은 제외)
#    셋째아 이상 = 3아 + 4아 + … + 8아 이상.  KOSIS 순위별 성비 표는 1990년부터라 1981~1989년을 위해 직접 계산한다.
so = read_kosis("kosis_DT_1B80A01.csv")
so = so[so["모의 연령(5세계급)별"] == "계"].drop(columns="모의 연령(5세계급)별").set_index(["성별", "출산순위별"])
s_years = [int(c) for c in so.columns]
cnt = {k: [int(num(v)) for v in row] for k, row in so.iterrows()}
ORDER_GROUPS = {"첫째아": ["1아"], "둘째아": ["2아"],
                "셋째아 이상": ["3아", "4아", "5아", "6아", "7아", "8아 이상"]}
rows = []
for i, y in enumerate(s_years):
    for name, parts in ORDER_GROUPS.items():
        boys = sum(cnt[("남자", p)][i] for p in parts)
        girls = sum(cnt[("여자", p)][i] for p in parts)
        rows.append({"year": y, "order": name, "boys": boys, "girls": girls,
                     "srb": round(boys / girls * 100, 1)})
order = pd.DataFrame(rows)
# 자동 확인: 남+여 = 연간 출생아 수, 계산한 순위별 성비 = KOSIS 발표 순위별 성비(1990~2025)
tot = pd.DataFrame({"year": s_years,
                    "b": [m_ + f for m_, f in zip(cnt[("남자", "총계")], cnt[("여자", "총계")])]})
assert (tot.merge(annual[["year", "births"]], on="year").eval("b == births")).all()
ref = (long[(long["sido"] == "전국") & long["item"].isin(["첫째아", "둘째아", "셋째아이상"])]
       .replace({"item": {"셋째아이상": "셋째아 이상"}})
       .rename(columns={"item": "order", "srb": "srb_ref"})[["year", "order", "srb_ref"]])
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
print("자동 확인 통과 (KOSIS 표끼리 합계가 서로 맞음)")
