"""Quant Lab：依照 Ernest Chan《Quantitative Trading》做研究的回測工具。

本機執行：  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt

import streamlit as st

from quantlab.data import load_prices
from quantlab.ui import backtest_page, chart_page, scanner_page, tester_page
from quantlab.ui import theme as T

st.set_page_config(page_title="Quant Lab", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(T.CSS, unsafe_allow_html=True)

PRESETS = {
    "黃金相關（書中範例）": ["GLD", "GDX", "GDXJ", "SLV"],
    "台股金控": ["2881", "2882", "2891", "2886", "2884", "2885"],
    "台股半導體": ["2330", "2303", "2454", "3711", "2379"],
    "美股能源": ["XOM", "CVX", "COP", "EOG"],
    "美股飲料（書中 Example 7.3）": ["KO", "PEP"],
    "外匯：商品貨幣": ["AUDUSD=X", "NZDUSD=X", "CAD=X"],
    "美股大盤 ETF": ["SPY", "QQQ", "IWM", "DIA"],
}
PAGES = ["策略測試器", "圖表", "書中範例", "配對掃描"]

if "watchlist" not in st.session_state:
    st.session_state["watchlist"] = PRESETS["黃金相關（書中範例）"]
if st.session_state.get("page") not in PAGES:
    st.session_state["page"] = "策略測試器"


def password_gate():
    """如果在 Streamlit 的 Secrets 設定了 APP_PASSWORD，就要求輸入密碼（程式碼會在伺服器上執行）。"""
    try:
        pw = st.secrets.get("APP_PASSWORD")
    except Exception:
        pw = None
    if not pw or st.session_state.get("authed"):
        return
    st.markdown('<div class="page-title">Quant Lab</div>', unsafe_allow_html=True)
    typed = st.text_input("密碼", type="password")
    if typed and typed == pw:
        st.session_state["authed"] = True
        st.rerun()
    elif typed:
        st.error("密碼錯誤")
    st.stop()


password_gate()


def load_preset():
    name = st.session_state.get("preset")
    if name in PRESETS:
        st.session_state["watchlist"] = list(PRESETS[name])


@st.cache_data(ttl=6 * 3600, show_spinner="下載價格資料中…")
def cached_prices(tickers: tuple[str, ...], start, end):
    return load_prices(list(tickers), start, end)


# ───────────────────────── 側欄 ─────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">QUANT<b>LAB</b></div>'
                '<div class="hint">Ernest Chan《Quantitative Trading》研究工具</div>',
                unsafe_allow_html=True)

    st.markdown("### 標的清單")
    known = sorted({t for v in PRESETS.values() for t in v} | set(st.session_state["watchlist"]))
    try:
        st.multiselect("標的", known, key="watchlist", accept_new_options=True,
                       label_visibility="collapsed", placeholder="輸入代號後按 Enter 新增",
                       help="美股如 KO；台股打數字如 2330；外匯如 EURUSD=X；期貨如 GC=F；指數如 ^TWII")
    except TypeError:   # 舊版 Streamlit 不支援自行輸入新選項
        raw = st.text_input("標的（逗號分隔）", ", ".join(st.session_state["watchlist"]))
        st.session_state["watchlist"] = [t.strip() for t in raw.split(",") if t.strip()]
    st.selectbox("快速載入清單", ["—"] + list(PRESETS), key="preset", on_change=load_preset)

    st.markdown("### 資料期間")
    periods = {"最近 3 年": 3, "最近 5 年": 5, "最近 10 年": 10,
               "書中期間（2006-05 ~ 2007-11）": None, "自訂": None}
    period = st.radio("期間", list(periods), index=1, label_visibility="collapsed")
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)   # yfinance 的 end 不含當天，所以加一天
    if periods[period]:
        start, end = today - dt.timedelta(days=round(365.25 * periods[period])), tomorrow
    elif period.startswith("書中"):
        start, end = dt.date(2006, 5, 23), dt.date(2007, 12, 1)
    else:
        start = st.date_input("開始", dt.date(2015, 1, 1), min_value=dt.date(1995, 1, 1), max_value=today)
        end = st.date_input("結束", today, min_value=dt.date(1995, 1, 1), max_value=today) + dt.timedelta(days=1)

    st.markdown("### 交易成本")
    cost_bps = st.slider("單邊交易成本（bp）", 0.0, 60.0, 5.0, 0.5, label_visibility="collapsed",
                         help="1 bp = 0.01%。美股大型股約 5 bp（書中 Example 3.7）；"
                              "台股約 25–30 bp（手續費 14.25 bp＋賣出證交稅 30 bp，平均到每邊）；"
                              "台股 ETF 約 15–20 bp；主要外匯約 0.5–2 bp。")
    st.markdown(f'<div class="hint">每邊 {cost_bps:g} bp，一趟來回 {2 * cost_bps:g} bp</div>',
                unsafe_allow_html=True)

# ───────────────────────── 頁首 + 頁面切換 ─────────────────────────
top_l, top_r = st.columns([3, 2])
with top_r:
    try:
        choice = st.segmented_control("頁面", PAGES, key="page", label_visibility="collapsed")
    except AttributeError:
        choice = st.radio("頁面", PAGES, key="page", horizontal=True, label_visibility="collapsed")
page = choice or "策略測試器"

watchlist = [t for t in st.session_state["watchlist"] if str(t).strip()]

if page == "圖表":
    chart_page.render(watchlist)
    st.stop()

if page == "策略測試器":
    if start >= end:
        st.error("開始日期必須早於結束日期。")
        st.stop()
    tester_page.render(watchlist, start, end, cost_bps)
    st.stop()

if not watchlist:
    st.info("左側標的清單是空的，請先加入標的。")
    st.stop()
if start >= end:
    st.error("開始日期必須早於結束日期。")
    st.stop()

try:
    all_prices, missing = cached_prices(tuple(watchlist), start, end)
except Exception as e:   # yfinance 偶爾會被限流
    st.error(f"資料下載失敗：{e}\n\n可能是 Yahoo 暫時限流，稍等一下再重新整理。")
    st.stop()
if missing:
    st.warning(f"這些代號抓不到資料，已略過：{', '.join(missing)}")
if all_prices.empty:
    st.stop()

with top_l:
    st.markdown(T.chips([f"Yahoo Finance · 已調整分割與配息",
                         f"{all_prices.index[0].date()} → {all_prices.index[-1].date()}"]),
                unsafe_allow_html=True)

if page == "配對掃描":
    scanner_page.render(all_prices, cost_bps)
else:
    backtest_page.render(all_prices, cost_bps)
