from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).parent
DATA = HERE   # CSV는 app.py와 같은 위치(저장소 맨 위)에 둔다

ZODIAC = ["쥐", "소", "범", "토끼", "용", "뱀", "말", "양", "원숭이", "닭", "개", "돼지"]
# 색 — .streamlit/config.toml 의 어두운 남색 배경(#0f1320)에 맞춰 대비·색각이상 구별을 검증한 값
BG = "#0f1320"          # 페이지 배경 (점 테두리·Matplotlib 배경에도 사용)
HIGHLIGHT = "#3987e5"   # 선택한 띠 (파랑)
MUTED = "#5f6779"       # 나머지 해 (배경 대비 3:1 이상, 파랑과 구별됨)
INK = "#c3c2b7"         # 기본 선
INK_STRONG = "#f0efec"  # 강조 글자
PREV, NEXT = "#d95926", "#199e70"   # 월별 비교: 앞 해(주황) / 다음 해(청록)
BAND = "#232a3b"        # 성비 정상 범위(103~107) 띠
GRID = "#1f2535"
TILE_MID = "#2e3445"    # 타일 지도 가운데(105 = 자연 상태)
TILE_HI = "#e66767"     # 타일 지도 높은 쪽(남아 많음)
NORMAL_LO, NORMAL_HI = 103, 107

METRICS = {
    "출생아 수": {"col": "births", "unit": "명", "kind": "pct", "fmt": ",.0f"},
    "합계출산율": {"col": "tfr", "unit": "명", "kind": "pct", "fmt": ".3f"},
    "출생성비": {"col": "srb", "unit": "여아 100명당 남아", "kind": "pt", "fmt": ".1f"},
    "혼인 건수": {"col": "marriages", "unit": "건", "kind": "pct", "fmt": ",.0f"},
}

# 시도 타일 지도 배치 (열, 행) — 모든 시도를 같은 크기 칸으로 그려 면적 착시를 없앤다
TILES = {
    "서울특별시": (1, 0, "서울"), "경기도": (2, 0, "경기"), "강원특별자치도": (3, 0, "강원"),
    "인천광역시": (0, 1, "인천"), "세종특별자치시": (1, 1, "세종"), "충청북도": (2, 1, "충북"),
    "경상북도": (3, 1, "경북"), "충청남도": (0, 2, "충남"), "대전광역시": (1, 2, "대전"),
    "대구광역시": (2, 2, "대구"), "울산광역시": (3, 2, "울산"), "전북특별자치도": (0, 3, "전북"),
    "광주광역시": (1, 3, "광주"), "경상남도": (2, 3, "경남"), "부산광역시": (3, 3, "부산"),
    "전라남도": (1, 4, "전남"), "제주특별자치도": (1, 5, "제주"),
}

st.set_page_config(
    page_title="띠가 아기를 골랐다? | 띠 미신과 한국 출생 통계",
    page_icon="🐎",
    layout="wide",
)

# 작은 꾸밈: 핵심 숫자 카드와 탭을 어두운 배경에서 또렷하게 (색은 config.toml 테마를 따름)
st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem;}
[data-testid="stMetric"] {background: #161b29;}
[data-testid="stMetricValue"] {color: #f0efec;}
.stTabs [data-baseweb="tab"] {font-weight: 600;}
.hero-q {border-left: 4px solid #3987e5; padding: .6rem 1rem; background: #161b29;
         border-radius: 6px; margin: .2rem 0 .6rem 0; line-height: 1.6;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    read = lambda name: pd.read_csv(DATA / name, encoding="utf-8-sig")
    annual = read("annual.csv").sort_values("year").reset_index(drop=True)
    order = read("order_srb.csv").sort_values(["order", "year"]).reset_index(drop=True)
    regional = read("regional_srb.csv")
    monthly = read("monthly.csv").sort_values(["year", "month"]).reset_index(drop=True)
    return annual, order, regional, monthly


def neighbor_base(s: pd.Series, k: int) -> pd.Series:
    """그 해를 뺀 앞뒤 k년씩(총 2k개 해)의 평균. 하나라도 비면 NaN → 추세를 뺀 '기준값'."""
    parts = [s.shift(j) for j in range(1, k + 1)] + [s.shift(-j) for j in range(1, k + 1)]
    return pd.concat(parts, axis=1).mean(axis=1, skipna=False)


def deviation(value: pd.Series, base: pd.Series, kind: str) -> pd.Series:
    return (value / base - 1) * 100 if kind == "pct" else value - base


@st.cache_data
def chance_prob(pool: tuple, k: int, observed: float, n_sim: int = 20000) -> float:
    """아무 해나 k개를 뽑았을 때 평균 편차가 관측값만큼(같은 방향으로) 치우칠 확률 (순열 검사)."""
    arr = np.asarray(pool, dtype=float)
    rng = np.random.default_rng(42)
    idx = np.argsort(rng.random((n_sim, len(arr))), axis=1)[:, :k]
    sims = arr[idx].mean(axis=1)
    return float(np.mean(sims <= observed) if observed < 0 else np.mean(sims >= observed))


def korean_font():
    nanum = Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")   # packages.txt 로 설치되는 경로
    if nanum.exists():
        fm.fontManager.addfont(str(nanum))
    names = {f.name for f in fm.fontManager.ttflist}
    return next(
        (f for f in ["NanumGothic", "Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"]
         if f in names),
        None,
    )


def zodiac_years(z: str, lo: int = 1970, hi: int = 2026) -> list[int]:
    no = ZODIAC.index(z)
    return [y for y in range(lo, hi + 1) if (y - 4) % 12 == no]


st.title("🐎 띠가 아기를 골랐다? — 미신이 한국 출생 통계에 남긴 흔적")
st.markdown(
    '<div class="hero-q"><b>질문:</b> 띠 미신(백말띠·황금돼지띠 등)은 정말 출생아 수와 남녀 성비를 바꿨을까? '
    "언제, 어느 띠에서, 누구(몇째 아이)와 어디서 나타났고, 말띠 해인 2026년은 어떨까?</div>",
    unsafe_allow_html=True,
)
st.caption(
    "출처: 국가데이터처 「인구동향조사」 KOSIS 통계표 INH_1B8000F_01 · DT_1B8000G · DT_1B80A01 · DT_1B81A19 "
    "(열람 2026-11-24) · 띠는 양력 연도 기준 (연도−4) mod 12 · "
    "편차 = 그 해 값 ÷ 앞뒤 해 평균 − 1 (출생성비는 포인트 차이)"
)

try:
    annual, order, regional, monthly = load_data()
except FileNotFoundError:
    st.error("app.py와 같은 위치에 annual.csv, order_srb.csv, regional_srb.csv, monthly.csv를 올려 주세요.")
    st.stop()

# ── 사이드바 조회 조건 ────────────────────────────────────────────────────────────
st.sidebar.header("🔎 분석 조건")
zodiac = st.sidebar.selectbox(
    "어느 띠를 볼까요?",
    ZODIAC,
    index=ZODIAC.index("말"),
    format_func=lambda z: f"{z}띠  ({', '.join(str(y)[2:] for y in zodiac_years(z))})",
)
metric = st.sidebar.radio("지표", list(METRICS), index=list(METRICS).index("출생성비"), horizontal=False,
                          help="처음 화면은 흔적이 가장 뚜렷한 출생성비입니다. 출생아 수로 바꿔 비교해 보세요.")
m = METRICS[metric]
col, kind, fmt = m["col"], m["kind"], m["fmt"]
dev_unit = "%" if kind == "pct" else "pt"
min_y, max_y = int(annual["year"].min()), int(annual["year"].max())
years = st.sidebar.slider("연도 범위", min_y, max_y, (min_y, max_y))
k = st.sidebar.slider(
    "비교 기준: 앞뒤 몇 년의 평균과 비교할까요?", 1, 3, 2,
    help="출생아 수는 해마다 줄어드는 추세가 있어 단순 비교는 불공정합니다. "
         "그 해를 뺀 앞뒤 k년씩의 평균을 '기준값'으로 두고 얼마나 벗어났는지(편차)를 봅니다.",
)
st.sidebar.caption(
    f"선택한 비교 기준: 그 해를 뺀 앞뒤 {k}년씩, 총 {2 * k}개 해의 평균. "
    "기준을 바꿔도 결론이 유지되는지 확인해 보세요."
)

# ── 계산: 전체 기간에서 편차를 구한 뒤 연도 범위로 자른다 (범위 끝의 편차가 잘리지 않도록) ──────────
a = annual.copy()
a["base"] = neighbor_base(a[col].astype(float), k)
a["dev"] = deviation(a[col].astype(float), a["base"], kind)
a["prev1"], a["next1"] = a[col].shift(1), a[col].shift(-1)   # 모양 검사용: 바로 앞 해·다음 해 값
view = a.loc[a["year"].between(years[0], years[1]) & a[col].notna()].copy()

if view.empty:
    st.warning("선택한 연도 범위에 자료가 없습니다. 혼인 건수는 1981년부터 있습니다. 연도 범위를 넓혀 주세요.")
    st.stop()

valid = view.dropna(subset=["dev"])
sel = view[view["zodiac"] == zodiac]
sel_valid = valid[valid["zodiac"] == zodiac]
if sel_valid.empty:
    st.warning(
        f"선택한 기간에 앞뒤 {k}년 자료까지 갖춘 {zodiac}띠 해가 없습니다. 연도 범위를 넓히거나 비교 기준을 줄여 주세요."
    )
    st.stop()

observed = float(sel_valid["dev"].mean())
by_zodiac = valid.groupby("zodiac")["dev"].mean()
lower = observed < 0
rank = int((by_zodiac < observed).sum() + 1) if lower else int((by_zodiac > observed).sum() + 1)
direction = "낮은" if lower else "높은"
p_chance = chance_prob(tuple(valid["dev"].round(6)), len(sel_valid), round(observed, 6))
sel_years = ", ".join(f"{y}({g})" for y, g in zip(sel_valid["year"], sel_valid["ganji"]))

# ── 자동 핵심 인사이트 ───────────────────────────────────────────────────────────────
st.info(
    f"💡 **핵심 인사이트 ({years[0]}\\~{years[1]}년 · {metric}):** {zodiac}띠 해 {len(sel_valid)}개"
    f"[{sel_years}]의 {metric}는 앞뒤 {k}년 평균보다 **평균 {observed:+.1f}{dev_unit}** "
    f"{'낮았' if lower else '높았'}습니다. 12띠 가운데 **{direction} 쪽에서 {rank}번째**이며, "
    f"아무 해나 {len(sel_valid)}개를 골랐을 때 이만큼 치우칠 확률은 **{p_chance * 100:.1f}%** 정도입니다."
    + (" 우연으로 보기 어려운 수준입니다." if p_chance < 0.05 else " 우연으로도 충분히 나올 수 있는 차이입니다.")
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{zodiac}띠 해 (선택 기간)", f"{len(sel_valid)}개 해",
          ", ".join(str(y) for y in sel_valid["year"]), delta_color="off", delta_arrow="off", border=True)
c2.metric(f"평균 편차 · {metric}", f"{observed:+.1f} {dev_unit}",
          f"앞뒤 {k}년 평균 대비", delta_color="off", delta_arrow="off", border=True)
c3.metric("12띠 중 순위", f"{rank}위 / {len(by_zodiac)}", f"{direction} 쪽부터",
          delta_color="off", delta_arrow="off", border=True)
c4.metric("우연히 이만큼 치우칠 확률", f"{p_chance * 100:.1f}%", "무작위로 해 뽑기 20,000번",
          delta_color="off", delta_arrow="off", border=True)

if 2026 in zodiac_years(zodiac):
    st.caption("※ 2026년(병오)은 연간 확정치가 아직 없어 위 계산에서 빠집니다. ④ 탭에서 월별 잠정치로 확인하세요.")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 ① 띠 해에 정말 달랐나",
    "🐾 ② 12띠 비교",
    "👶 ③ 누가·어디서",
    "🐎 ④ 2026 병오년",
    "📋 ⑤ 데이터·검산·한계",
])

# ── ① 추이와 편차 ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{metric}: 연도별 추이와 {zodiac}띠 해")
    fig_line = go.Figure()
    if col == "srb":
        fig_line.add_hrect(y0=NORMAL_LO, y1=NORMAL_HI, fillcolor=BAND, opacity=1, line_width=0, layer="below")
    fig_line.add_trace(go.Scatter(
        x=view["year"], y=view[col], mode="lines", name=metric,
        line=dict(color=INK, width=2),
        hovertemplate="%{x}년<br>" + metric + ": %{y:" + fmt + "}<extra></extra>",
    ))
    fig_line.add_trace(go.Scatter(
        x=sel["year"], y=sel[col], mode="markers", name=f"{zodiac}띠 해",
        marker=dict(color=HIGHLIGHT, size=11, line=dict(color=BG, width=2)),
        hovertemplate="%{x}년 · " + zodiac + "띠<br>" + metric + ": %{y:" + fmt + "}<extra></extra>",
    ))
    for y_, v_, g_ in zip(sel["year"], sel[col], sel["ganji"]):   # 이름표는 선과 겹치지 않게 점 위/아래로 띄운다
        fig_line.add_annotation(x=y_, y=v_, text=f"{y_} {g_}", showarrow=False, font=dict(size=11, color=INK_STRONG),
                                yshift=-18 if lower else 18)
    fig_line.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10),
                           legend=dict(orientation="h", y=1.12, x=0))
    fig_line.update_yaxes(title=f"{metric} ({m['unit']})", tickformat=fmt)
    fig_line.update_xaxes(dtick=5)
    st.plotly_chart(fig_line, width="stretch")
    if col == "srb":
        st.caption("옅은 배경 띠 = 자연 상태의 출생성비 범위(103\\~107). 1970년대 값은 해마다 크게 오르내려 자료 특성도 함께 고려해야 합니다.")

    st.subheader(f"추세를 빼면? 앞뒤 {k}년 평균 대비 편차 ({dev_unit})")
    bar_colors = np.where(valid["zodiac"] == zodiac, HIGHLIGHT, MUTED)
    fig_dev = go.Figure(go.Bar(
        x=valid["year"], y=valid["dev"], marker_color=bar_colors, marker_line_width=0,
        customdata=np.stack([valid["ganji"], valid["zodiac"], valid[col].astype(float), valid["base"]], axis=1),
        hovertemplate="%{x}년 %{customdata[0]} (%{customdata[1]}띠)<br>값 %{customdata[2]:" + fmt + "}"
                      " · 앞뒤 평균 %{customdata[3]:" + fmt + "}<br>편차 %{y:+.1f}" + dev_unit + "<extra></extra>",
        name="편차",
    ))
    fig_dev.add_hline(y=0, line_color=INK, line_width=1)
    fig_dev.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), showlegend=False, bargap=0.25)
    fig_dev.update_yaxes(title=f"편차 ({dev_unit})", zeroline=False)
    fig_dev.update_xaxes(dtick=5)
    st.plotly_chart(fig_dev, width="stretch")
    st.caption(
        f"파란 막대 = {zodiac}띠 해, 회색 = 나머지 해. 0보다 아래면 앞뒤 해보다 낮았다(적었다)는 뜻입니다. "
        f"처음·마지막 {k}개 해는 비교할 앞뒤 해가 모자라 빠집니다."
    )
    # 모양 검사: 편차는 '앞뒤 평균'과의 차이라, 추세가 꺾이는 해에도 크게 나올 수 있다.
    # 바로 앞 해와 다음 해보다 모두 낮은(또는 높은) 해만 '움푹 팬(튀어나온)' 해로 센다.
    vs = sel_valid.dropna(subset=["prev1", "next1"])
    if lower:
        dip = vs[(vs[col] < vs["prev1"]) & (vs[col] < vs["next1"])]
    else:
        dip = vs[(vs[col] > vs["prev1"]) & (vs[col] > vs["next1"])]
    shape = "움푹 팬" if lower else "툭 튀어나온"
    st.markdown(
        f"**모양 검사:** {zodiac}띠 해 {len(vs)}개 중 바로 앞 해와 다음 해보다 모두 "
        f"{'낮은' if lower else '높은'} '{shape}' 해는 "
        f"**{', '.join(str(y) for y in dip['year']) or '없음'}** ({len(dip)}/{len(vs)})입니다. "
        "나머지는 추세가 꺾이는 길목일 수 있어 편차만으로 단정하기 어렵습니다."
    )

# ── ② 12띠 비교 ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader(f"12띠별 편차 — 점 하나가 한 해 ({metric}, 앞뒤 {k}년 기준)")
    left, right = st.columns([3, 1])
    with left:
        dot_colors = np.where(valid["zodiac"] == zodiac, HIGHLIGHT, MUTED)
        fig_strip = go.Figure()
        fig_strip.add_vline(x=0, line_color=INK, line_width=1)
        fig_strip.add_trace(go.Scatter(
            x=valid["dev"], y=valid["zodiac"], mode="markers", name="각 해",
            marker=dict(color=dot_colors, size=11, line=dict(color=BG, width=1.5)),
            customdata=np.stack([valid["year"], valid["ganji"]], axis=1),
            hovertemplate="%{customdata[0]}년 %{customdata[1]} (%{y}띠)<br>편차 %{x:+.1f}" + dev_unit + "<extra></extra>",
        ))
        means = by_zodiac.reindex(ZODIAC)
        fig_strip.add_trace(go.Scatter(
            x=means.values, y=means.index, mode="markers", name="띠별 평균",
            marker=dict(symbol="line-ns", size=22, line=dict(color=INK_STRONG, width=3)),
            hovertemplate="%{y}띠 평균 %{x:+.1f}" + dev_unit + "<extra></extra>",
        ))
        fig_strip.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(orientation="h", y=1.06, x=0))
        fig_strip.update_yaxes(categoryorder="array", categoryarray=ZODIAC[::-1], title=None)
        fig_strip.update_xaxes(title=f"앞뒤 {k}년 평균 대비 편차 ({dev_unit})")
        st.plotly_chart(fig_strip, width="stretch")
    with right:
        n_by = valid.groupby("zodiac")["dev"].agg(["count", "mean"]).reindex(ZODIAC)
        table = n_by.reset_index().rename(columns={"zodiac": "띠", "count": "해 수", "mean": f"평균 편차({dev_unit})"})
        table[f"평균 편차({dev_unit})"] = table[f"평균 편차({dev_unit})"].round(2)
        st.dataframe(table.sort_values(f"평균 편차({dev_unit})"), hide_index=True, width="stretch", height=460)
    st.caption(
        "흰 세로 막대 = 띠별 평균. 띠마다 해가 3\\~5개뿐이라 상자그림 대신 모든 해를 점으로 보여 줍니다. "
        f"'우연 확률' {p_chance * 100:.1f}%는 같은 기간의 모든 해 편차에서 {len(sel_valid)}개를 무작위로 2만 번 뽑아 "
        f"평균이 {zodiac}띠만큼 {direction} 경우의 비율입니다. 다만 12띠를 모두 본 뒤 하나를 고르면 우연히 걸릴 가능성도 "
        "12배 커진다는 점을 함께 고려해야 합니다."
    )

# ── ③ 누가(출산순위)·어디서(시도) ─────────────────────────────────────────────────
with tab3:
    st.subheader("누가? 출산순위별 출생성비 (여아 100명당 남아)")
    share = st.radio("세로축", ["순위마다 따로 (모양 비교)", "세 그림 모두 같게 (크기 비교)"], horizontal=True)
    o = order.loc[order["year"].between(years[0], years[1])]
    if o.empty:
        st.warning("출산순위별 자료는 1981년부터 있습니다. 연도 범위를 넓혀 주세요.")
    else:
        kfont = korean_font()
        en = {"첫째아": "1st child", "둘째아": "2nd child", "셋째아 이상": "3rd+ child"}
        dark_rc = {"font.family": kfont or "DejaVu Sans", "axes.unicode_minus": False,
                   "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                   "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK_STRONG,
                   "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED}
        with plt.rc_context(dark_rc):
            fig_o, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=share.startswith("세 그림"))
            for ax, name in zip(axes, ["첫째아", "둘째아", "셋째아 이상"]):
                s = o[o["order"] == name]
                h = s[s["zodiac"] == zodiac]
                ax.axhspan(NORMAL_LO, NORMAL_HI, color=BAND, zorder=0, lw=0)
                ax.plot(s["year"], s["srb"], color=INK, lw=1.8, zorder=2)
                ax.scatter(h["year"], h["srb"], s=55, color=HIGHLIGHT, edgecolor=BG, linewidth=1.5, zorder=3)
                for _, r in h.iterrows():
                    ax.annotate(f"{int(r['year'])}\n{r['srb']:.1f}", (r["year"], r["srb"]),
                                textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=INK_STRONG)
                ax.set_title(name if kfont else en[name], loc="left", fontsize=11)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color=GRID, lw=0.8)
                ax.margins(y=0.18)
            axes[0].set_ylabel("출생성비" if kfont else "Sex ratio at birth")
            fig_o.tight_layout()
        st.pyplot(fig_o, width="stretch")
        plt.close(fig_o)
        if not kfont:
            st.caption("※ 배포 서버에 한글 글꼴이 없어 위 Matplotlib 차트는 영어로 표시됩니다.")

        ob = order.copy()
        ob["base"] = ob.groupby("order")["srb"].transform(lambda s: neighbor_base(s, k))
        ob["dev"] = ob["srb"] - ob["base"]
        ob = ob[ob["year"].between(years[0], years[1]) & (ob["zodiac"] == zodiac)].dropna(subset=["dev"])
        if not ob.empty:
            summary = ob.groupby("order")["dev"].mean()
            st.markdown(
                f"**{zodiac}띠 해의 순위별 성비 편차 (앞뒤 {k}년 평균 대비):** "
                + " · ".join(f"{n} {summary.get(n, float('nan')):+.1f}pt" for n in ["첫째아", "둘째아", "셋째아 이상"])
            )
        st.caption("옅은 배경 띠 = 자연 상태 범위(103\\~107). 파란 점 = 선택한 띠 해. 출산순위 '미상'은 계산에서 뺐습니다.")

    st.divider()
    st.subheader("어디서? 시도별 총출생성비 (모든 시도를 같은 크기 칸으로)")
    yr_options = sorted(regional["year"].unique())
    map_year = st.select_slider("지도 연도", options=yr_options, value=1990)
    r = regional[regional["year"] == map_year].set_index("sido")
    nat = float(r.loc["전국", "srb"])
    rs = r.drop(index="전국")
    out_of_range = int((rs["srb"] > NORMAL_HI).sum())

    mcol, gcol = st.columns([1, 3])
    with mcol:
        short_of = {name: t[2] for name, t in TILES.items()}
        st.metric(f"{map_year}년 전국", f"{nat:.1f}", f"{ZODIAC[(map_year - 4) % 12]}띠 해",
                  delta_color="off", delta_arrow="off", border=True)
        st.metric("가장 높은 시도", f"{short_of[rs['srb'].idxmax()]} {rs['srb'].max():.1f}", border=True)
        st.metric("가장 낮은 시도", f"{short_of[rs['srb'].idxmin()]} {rs['srb'].min():.1f}", border=True)
        st.metric("107을 넘은 시도", f"{out_of_range}곳 / {len(rs)}곳", border=True)
    with gcol:
        z = np.full((6, 4), np.nan)
        hover = np.full((6, 4), "", dtype=object)
        lx, ly, lt = [], [], []
        missing = []
        for sido, (cx, cy, short) in TILES.items():
            if sido in rs.index:
                v = float(rs.loc[sido, "srb"])
                z[cy, cx] = v
                hover[cy, cx] = f"{sido} {map_year}년: {v:.1f}"
                lt.append(f"<b>{short}</b><br>{v:.1f}")
            else:
                missing.append((cx, cy))
                lt.append(f"{short}<br>자료 없음")
            lx.append(cx)
            ly.append(cy)
        fig_map = go.Figure(go.Heatmap(
            z=z, text=hover, hovertemplate="%{text}<extra></extra>", xgap=5, ygap=5,
            colorscale=[[0, HIGHLIGHT], [0.5, TILE_MID], [1, TILE_HI]],
            zmin=80, zmid=105, zmax=130,
            colorbar=dict(title="출생성비", tickvals=[80, 90, 105, 120, 130],
                          ticktext=["80", "90", "105(자연)", "120", "130+"]),
        ))
        for cx, cy in missing:
            fig_map.add_shape(type="rect", x0=cx - 0.45, x1=cx + 0.45, y0=cy - 0.45, y1=cy + 0.45,
                              line=dict(color=MUTED, dash="dot"), fillcolor="rgba(0,0,0,0)")
        fig_map.add_trace(go.Scatter(x=lx, y=ly, mode="text", text=lt, hoverinfo="skip",
                                     textfont=dict(color=INK_STRONG, size=13), showlegend=False))
        fig_map.update_xaxes(visible=False, range=[-0.6, 3.6])
        fig_map.update_yaxes(visible=False, autorange="reversed", scaleanchor="x")
        fig_map.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_map, width="stretch")
    st.caption(
        "색은 모든 연도에 같은 기준(가운데 105 = 자연 상태)으로 고정해 연도를 바꿔도 비교할 수 있습니다. "
        "울산은 1997년, 세종은 2012년부터 자료가 있습니다. 인구가 적은 시도는 해마다 값이 크게 흔들릴 수 있습니다."
    )

# ── ④ 월별 비교 (2026 병오년) ───────────────────────────────────────────────────
with tab4:
    st.subheader("같은 달끼리 비교: 선택한 해와 앞뒤 해의 월별 출생아 수")
    mon_years = sorted(monthly["year"].unique())
    choices = [y for y in mon_years if y > mon_years[0]]
    z_choices = [y for y in choices if (y - 4) % 12 == ZODIAC.index(zodiac)]
    default_year = z_choices[-1] if z_choices else choices[-1]
    cmp_year = st.selectbox(
        "비교할 해", choices, index=choices.index(default_year), key=f"cmp_{zodiac}",
        format_func=lambda y: f"{y}년 ({ZODIAC[(y - 4) % 12]}띠){' · 잠정' if y == mon_years[-1] else ''}",
    )
    mw = monthly.pivot(index="month", columns="year", values="births")
    months = mw[cmp_year].dropna().index.tolist()
    prev_y, next_y = cmp_year - 1, cmp_year + 1

    def growth(cur, base, ms):
        if not ms:
            return float("nan")
        b = mw.loc[ms, base].sum()
        return (mw.loc[ms, cur].sum() / b - 1) * 100

    early = [mm for mm in months if mm <= 2]
    late = [mm for mm in months if mm >= 3]
    g1, g2, g3 = st.columns(3)
    g1.metric(f"{cmp_year}년 1\\~{months[-1]}월 출생아", f"{mw.loc[months, cmp_year].sum():,.0f}명",
              f"{growth(cmp_year, prev_y, months):+.1f}% (전년 같은 달 대비)", delta_color="off", border=True)
    g2.metric("1\\~2월 (띠가 바뀌는 구간)", f"{growth(cmp_year, prev_y, early):+.1f}%", "전년 같은 달 대비",
              delta_color="off", delta_arrow="off", border=True)
    g3.metric(f"3\\~{months[-1]}월 (확실히 새 띠)", f"{growth(cmp_year, prev_y, late):+.1f}%", "전년 같은 달 대비",
              delta_color="off", delta_arrow="off", border=True)

    fig_m = go.Figure()
    fig_m.add_vrect(x0=0.5, x1=2.5, fillcolor=BAND, opacity=1, line_width=0,
                    annotation_text="설날·입춘 전후: 앞 띠와 섞임", annotation_position="top left", layer="below")
    series = [(prev_y, PREV, 2), (cmp_year, HIGHLIGHT, 3.5), (next_y, NEXT, 2)]
    for y, colr, w in series:
        if y not in mw.columns:
            continue
        s = mw[y].dropna()
        label = f"{y}년 ({ZODIAC[(y - 4) % 12]}띠){' 잠정' if y == mon_years[-1] else ''}"
        fig_m.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines+markers", name=label,
            line=dict(color=colr, width=w), marker=dict(size=7),
            hovertemplate=f"{y}년 " + "%{x}월: %{y:,.0f}명<extra></extra>",
        ))
        fig_m.add_annotation(x=s.index[-1], y=s.values[-1], text=f"{y}", showarrow=False,
                             xanchor="left", xshift=8, font=dict(color=INK_STRONG, size=12))
    fig_m.update_layout(height=400, margin=dict(l=10, r=40, t=40, b=10),
                        legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig_m.update_xaxes(tickvals=list(range(1, 13)), ticktext=[f"{i}월" for i in range(1, 13)], range=[0.5, 12.8])
    fig_m.update_yaxes(title="출생아 수 (명)", rangemode="tozero", tickformat=",")
    st.plotly_chart(fig_m, width="stretch")
    st.caption(
        "띠는 설날(또는 입춘)에 바뀌지만 통계는 양력 1월 1일 기준이라 1\\~2월생 일부는 앞 띠입니다. "
        "그래서 1\\~2월과 3월 이후를 나눠 봅니다. 2026년 값은 잠정치(p)이며 8월까지만 공표되었습니다. "
        "참고: 일본은 1966년 병오년에 출생아가 전년보다 약 25% 줄었습니다(nippon.com; Akabayashi, 2007)."
    )

# ── ⑤ 데이터·검산·한계 ──────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"{zodiac}띠 해 상세 ({metric}, 앞뒤 {k}년 기준)")
    detail = sel[["year", "ganji", "stem_color", col, "base", "dev"]].rename(columns={
        "year": "연도", "ganji": "간지", "stem_color": "천간 색", col: metric,
        "base": f"앞뒤 {k}년 평균", "dev": f"편차({dev_unit})",
    })
    # 표시 형식: 숫자는 그대로 두고(정렬 가능) 보이는 자릿수만 맞춘다. 빈칸은 Streamlit이 None으로 표시
    val_fmt = {"births": "{:,.0f}", "marriages": "{:,.0f}", "tfr": "{:.3f}", "srb": "{:.1f}"}[col]
    base_fmt = {"births": "{:,.0f}", "marriages": "{:,.0f}", "tfr": "{:.3f}", "srb": "{:.2f}"}[col]

    def styled(df: pd.DataFrame):
        return df.style.format(
            {metric: val_fmt, f"앞뒤 {k}년 평균": base_fmt, f"편차({dev_unit})": "{:+.2f}"}, na_rep="—")

    st.dataframe(styled(detail), hide_index=True, width="stretch")

    st.subheader("선택 기간 전체 표")
    full = view[["year", "ganji", "zodiac", col, "base", "dev"]].rename(columns={
        "year": "연도", "ganji": "간지", "zodiac": "띠", col: metric,
        "base": f"앞뒤 {k}년 평균", "dev": f"편차({dev_unit})",
    })
    st.dataframe(styled(full), hide_index=True, width="stretch", height=300)
    st.caption(f"None = 앞뒤 {k}년 자료가 모자라 기준값과 편차를 계산하지 않은 해입니다(기간의 처음·마지막 해).")
    st.download_button(
        "⬇️ 이 표를 CSV로 받기", full.round(3).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"birth_zodiac_{col}_{years[0]}_{years[1]}_k{k}.csv", mime="text/csv",
    )

    st.subheader("검산 (앱이 열릴 때마다 다시 계산)")
    full_years = monthly.groupby("year")["month"].count().pipe(lambda c: c[c == 12].index)
    msum = monthly[monthly["year"].isin(full_years)].groupby("year")["births"].sum()
    chk = msum.to_frame("월별 합").join(annual.set_index("year")["births"].rename("연간 표"))
    ok1 = bool((chk["월별 합"] == chk["연간 표"]).all())
    nat_reg = regional[regional["sido"] == "전국"].set_index("year")["srb"]
    ok2 = bool(((nat_reg - annual.set_index("year")["srb"].reindex(nat_reg.index)).abs() < 0.05).all())
    (st.success if ok1 else st.error)(
        f"월별 출생아 합계 = 연간 출생아 수: {len(chk)}개 연도 {'모두 일치' if ok1 else '불일치 있음'}")
    (st.success if ok2 else st.error)(
        f"시도표의 전국 출생성비 = 연간 표 출생성비: {len(nat_reg)}개 연도 {'모두 일치' if ok2 else '불일치 있음'}")

    st.subheader("출처와 집계 기준")
    st.markdown(
        """
- **INH_1B8000F_01** 출생아수, 합계출산율, 자연증가 등 (1970\\~2025, 자료갱신일 2026-09-22)
- **DT_1B8000G** 월·분기·연간 인구동향 — 전국 월별 출생아 수·혼인 건수 (1981.01\\~2026.08, 2026년 잠정, 자료갱신일 2026-10-28)
- **DT_1B80A01** 성/모의 연령/출산순위별 출생 — 남·여 출생아 수로 순위별 성비를 직접 계산 (1981\\~2025, 자료갱신일 2026-08-26)
- **DT_1B81A19** 시도/출산순위별 출생성비 — 시도별 총출생성비 (1990\\~2025, 자료갱신일 2026-08-26)
- 모두 국가데이터처 「인구동향조사」, KOSIS 국가통계포털에서 2026-11-24 열람
- 띠 = (양력 연도 − 4) mod 12. 출생성비 = 여아 100명당 남아 수. 연간 혼인 건수 = 월별 혼인 건수의 합
"""
    )
    st.subheader("해석의 한계")
    st.markdown(
        """
1. **상관이지 인과가 아닙니다.** 2007년 증가는 3차 베이비붐 등 인구학적 요인도 함께 작용했다고 통계청이 설명했습니다.
2. **띠마다 해가 3\\~5개뿐**이라 한두 해의 사건(경제위기 등)이 평균을 크게 흔듭니다.
3. **띠 경계 불일치:** 띠는 설날·입춘에 바뀌지만 통계는 양력 연도입니다.
4. **기준값의 영향:** 편차는 앞뒤 평균과의 차이라 추세가 꺾이는 해에도 커집니다. 예를 들어 1990년 출생아는 전년보다 1.6% 늘었는데, 다음 해가 9.2% 급증해 편차가 음수로 나옵니다. 2002년 감소 폭에도 2000년 용띠 정점이 기준값을 끌어올린 영향이 섞여 있습니다. 그래서 ① 탭의 '모양 검사'를 함께 봅니다.
5. **1970년대 출생성비는 해마다 크게 흔들립니다**(1973년 104.6 → 1975년 112.4 → 1977년 104.2). 출생신고 관행 등 자료 특성일 수 있어 이 시기 성비 해석은 조심해야 합니다.
6. **2026년은 잠정치**이며 8월까지만 있습니다. 2026년부터 행정구역 개편으로 시도 구분이 바뀔 수 있습니다.
7. 1980\\~90년대의 높은 성비는 **태아 성감별에 따른 선택이 있었던 시기**의 기록입니다. 이 앱은 그 사실을 통계로 보여 줄 뿐, 특정 선택을 평가하지 않습니다.
"""
    )

st.caption("※ 이 대시보드는 공개 통계로 띠 미신의 흔적을 살펴본 것이며, 특정 띠나 출생 연도에 대한 편견을 지지하지 않습니다.")
