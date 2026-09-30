"""策略測試器：用 Python 寫策略 → 回測 → MT5 風格報告、最佳化、批次回測。"""
from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import engine, optimize, report
from ..data import INTERVAL_LIMIT_DAYS, load_ohlcv
from ..templates import TEMPLATES, load
from . import theme as T

INTERVALS = {"日線": "1d", "週線": "1wk", "1 小時": "1h", "30 分": "30m", "15 分": "15m", "5 分": "5m"}
MY = "我的策略（上傳）"

API_DOC = """
### 策略怎麼寫

```python
class MyStrategy(Strategy):
    params = {"n": 20, "sl_pct": 0.05}      # 可調參數（會自動出現在介面上，也能最佳化）

    def init(self):                          # 只跑一次：先算好指標
        self.ma = self.I(ta.sma(self.close, self.p.n), "MA")

    def next(self):                          # 每根 K 棒收盤後跑一次
        i = self.i
        if self.is_flat and self.close[i] > self.ma[i]:
            self.buy(sl_pct=self.p.sl_pct)   # 下一根開盤成交
        elif self.is_long and self.close[i] < self.ma[i]:
            self.close_position("跌破均線")
```

**資料**：`self.open` `self.high` `self.low` `self.close` `self.volume`（numpy 陣列）、
`self.i`（目前第幾根）、`self.time`（目前時間）、`self.data`（完整 DataFrame）、`self.p.參數名`

**下單**（全部在下一根開盤成交）
- `self.buy(size=1.0, sl=None, tp=None, sl_pct=None, tp_pct=None, tag="")`：做多，size 是淨值比例
- `self.sell(...)`：做空，參數同上；持有反向部位時會自動反手
- `self.close_position(tag="")`：平倉
- `self.set_sl(價格)`、`self.set_tp(價格)`：修改停損 / 停利（可做移動停損）

**部位狀態**：`self.is_flat` `self.is_long` `self.is_short` `self.position`（股數，空單為負）
`self.entry_price` `self.bars_in_trade` `self.open_pnl_pct` `self.equity`

**指標**（`ta.`）：`sma` `ema` `rsi` `macd`（回傳 3 條）`bollinger`（回傳 3 條）`atr` `highest` `lowest`
`stdev` `roc` `zscore` `shift`，交叉判斷用 `ta.crossover(a, b, i)` / `ta.crossunder(a, b, i)`。
`self.I(數列, "名稱", overlay=True)` 把指標畫到圖上（overlay=False 畫在副圖）。也可以用 `np`、`pd`。

**成交規則**：停損 / 停利用每根的最高、最低價判斷，跳空穿越時以開盤價成交；
同一根同時碰到停損和停利，保守假設先停損。手續費 = 成交金額 × 側欄的單邊成本。

**注意**：程式碼會在伺服器上執行，請把 app 設成私人（Streamlit 設定 → Sharing），只讓自己使用。
"""


# ───────────────────────── 工具 ─────────────────────────
def money(v):
    return "—" if v is None or pd.isna(v) else f"{v:,.0f}"


def pct(v, d=1):
    return "—" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def num(v, d=2):
    if v is None or pd.isna(v):
        return "—"
    return "∞" if np.isinf(v) else f"{v:.{d}f}"


def sign(v):
    return None if v is None or pd.isna(v) or v == 0 else ("up" if v > 0 else "down")


@st.cache_data(ttl=6 * 3600, show_spinner="下載 K 線資料中…", max_entries=64)
def cached_ohlcv(ticker, start, end, interval):
    return load_ohlcv(ticker, start, end, interval)


@st.cache_data(show_spinner=False, max_entries=32)
def cached_run(code: str, params_items: tuple, data: pd.DataFrame, cash: float, cost: float):
    cls = engine.load_strategy(code)
    res = engine.run(cls, data, dict(params_items), cash, cost)
    ok, bad = engine.lookahead_check(cls, data, dict(params_items), cash, cost)
    res["lookahead"] = (ok, bad)
    return res


def code_key(code: str) -> str:
    return hashlib.md5(code.encode()).hexdigest()[:8]


def code_editor(code: str, key: str) -> str:
    """有語法上色的編輯器（streamlit-ace）；套件不能用時退回純文字框。"""
    try:
        from streamlit_ace import st_ace
        out = st_ace(value=code, language="python", theme="tomorrow_night", key=key, height=430,
                     font_size=14, tab_size=4, show_gutter=True, wrap=False, auto_update=False,
                     placeholder="在這裡寫策略…")
        st.markdown('<div class="code-note">改完程式碼後按編輯器右下角的 APPLY（或 Ctrl/⌘ + Enter）才會套用。</div>',
                    unsafe_allow_html=True)
        return out if out else code
    except Exception:
        return st.text_area("策略程式碼", code, height=430, key=key + "_ta", label_visibility="collapsed")


def param_widgets(cls, tpl: str) -> dict:
    params = {}
    items = list(cls.params.items())
    if not items:
        st.caption("這個策略沒有可調參數（在 class 裡加上 `params = {...}` 就會出現在這裡）。")
        return params
    cols = st.columns(min(4, len(items)))
    for k, (name, default) in enumerate(items):
        c = cols[k % len(cols)]
        key = f"tp::{tpl}::{name}"
        # 已有值（例如最佳化「套用」設定過）就不要再給預設值，避免 Streamlit 警告
        dv = {} if key in st.session_state else {"value": default}
        if isinstance(default, bool):
            params[name] = c.toggle(name, key=key, **dv)
        elif isinstance(default, int):
            params[name] = int(c.number_input(name, step=1, key=key, **dv))
        elif isinstance(default, float):
            step = 0.01 if abs(default) < 1 else 0.1 if abs(default) < 10 else 1.0
            params[name] = float(c.number_input(name, step=step, format="%g", key=key,
                                                **({k: float(v) for k, v in dv.items()})))
        else:
            params[name] = c.text_input(name, key=key, **({k: str(v) for k, v in dv.items()}))
    return params


# ───────────────────────── 圖 ─────────────────────────
def price_chart(res: dict, title: str):
    data, trades, inds = res["data"], res["trades"], res["indicators"]
    sub = [d for d in inds if not d["overlay"]]
    rows = 2 if sub else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.72, 0.28] if sub else [1.0])
    x = data.index
    if len(data) <= 2500:
        fig.add_trace(go.Candlestick(x=x, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
                                     name="K 線", increasing=dict(line=dict(color=T.UP, width=1), fillcolor=T.UP),
                                     decreasing=dict(line=dict(color=T.DOWN, width=1), fillcolor=T.DOWN)), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=x, y=data["Close"], name="收盤價", line=dict(color=T.INK2, width=1.2)), row=1, col=1)
    for d, color in zip([d for d in inds if d["overlay"]], T.SERIES[2:] + T.SERIES[:2]):
        fig.add_trace(go.Scatter(x=x, y=d["values"], name=d["name"], line=dict(color=color, width=1.3)), row=1, col=1)
    for d, color in zip(sub, T.SERIES):
        fig.add_trace(go.Scatter(x=x, y=d["values"], name=d["name"], line=dict(color=color, width=1.3)), row=2, col=1)

    if len(trades):
        for won, color, name in [(True, T.UP, "獲利交易"), (False, T.DOWN, "虧損交易")]:
            t = trades[(trades["損益"] > 0) == won]
            xs, ys = [], []
            for _, r in t.iterrows():
                xs += [r["進場時間"], r["出場時間"], None]
                ys += [r["進場價"], r["出場價"], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=name, hoverinfo="skip",
                                     line=dict(color=color, width=1.5, dash="dot")), row=1, col=1)
        for side, symbol, color in [("多", "triangle-up", T.UP), ("空", "triangle-down", T.DOWN)]:
            t = trades[trades["方向"] == side]
            if len(t):
                fig.add_trace(go.Scatter(
                    x=t["進場時間"], y=t["進場價"], mode="markers", name=f"{side}單進場",
                    marker=dict(symbol=symbol, size=11, color=color, line=dict(color=T.SURFACE, width=1.5)),
                    customdata=np.stack([t["進場標籤"], t["報酬率"] * 100], axis=1),
                    hovertemplate=f"{side}單進場 %{{y:.2f}}<br>%{{customdata[0]}}<br>此筆報酬 %{{customdata[1]:.2f}}%<extra></extra>"),
                    row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades["出場時間"], y=trades["出場價"], mode="markers", name="出場",
            marker=dict(symbol="x-thin", size=10, color=T.INK, line=dict(color=T.INK, width=2)),
            customdata=np.stack([trades["出場原因"], trades["損益"]], axis=1),
            hovertemplate="出場 %{y:.2f}<br>%{customdata[0]}<br>損益 %{customdata[1]:,.0f}<extra></extra>"),
            row=1, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig = T.style(fig, 620 if sub else 520, title)
    return fig


def equity_chart(res: dict, split=None):
    eq, close, cash = res["equity"], res["data"]["Close"], res["cash"]
    bh = close / close.iloc[0] * cash
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bh.index, y=bh, name="買進持有", line=dict(color=T.MUTED, width=1.4, dash="dot")))
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="策略淨值", line=dict(color=T.BLUE, width=2)))
    if split is not None:
        fig.add_vline(x=split, line_dash="dash", line_color=T.AXIS)
    fig.update_yaxes(tickformat=",.0f")
    return T.style(fig, 360, "淨值曲線 vs 買進持有")


def drawdown_chart(eq):
    dd = eq / eq.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=T.DOWN, width=1.2),
                               fillcolor="rgba(230,103,103,0.18)", hovertemplate="%{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    return T.style(fig, 220, "回撤", legend=False)


def trade_bars(trades):
    colors = np.where(trades["損益"] > 0, T.UP, T.DOWN)
    fig = go.Figure(go.Bar(x=np.arange(1, len(trades) + 1), y=trades["損益"], marker_color=colors,
                           customdata=np.stack([trades["方向"], trades["出場原因"], trades["報酬率"] * 100], axis=1),
                           hovertemplate="第 %{x} 筆（%{customdata[0]}）<br>損益 %{y:,.0f}<br>%{customdata[2]:.2f}%"
                                         "<br>%{customdata[1]}<extra></extra>"))
    fig.update_xaxes(title_text="交易序號")
    fig = T.style(fig, 280, "每筆交易損益", legend=False)
    fig.update_layout(hovermode="closest", bargap=0.15)
    return fig


def monthly_heatmap(table: pd.DataFrame):
    z = table.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]
    lim = float(np.abs(finite).max()) if finite.size else 0.01
    text = np.where(np.isnan(z), "", np.char.mod("%.1f%%", np.nan_to_num(z) * 100))
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{m}月" for m in range(1, 13)] + ["全年"], y=[str(y) for y in table.index],
        colorscale=T.DIVERGING, zmid=0, zmin=-lim, zmax=lim, text=text, texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono, monospace", color=T.INK, size=11), xgap=2, ygap=2,
        hovertemplate="%{y} %{x}：%{z:.2%}<extra></extra>", showscale=False))
    fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
    fig.update_xaxes(showgrid=False, side="top")
    fig = T.style(fig, 90 + 34 * len(table), None, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


# ───────────────────────── 報告 ─────────────────────────
def overview(s: dict):
    bh = s.get("買進持有報酬")
    st.markdown(T.kpi_cards([
        dict(label="淨利", value=money(s["淨利"]), delta=f"總報酬 {pct(s['總報酬'])}", trend=sign(s["淨利"])),
        dict(label="年化報酬", value=pct(s["年化報酬"]),
             delta=f"買進持有 {pct(bh)}" if bh is not None else None,
             trend=sign(s["總報酬"] - bh) if bh is not None else None),
        dict(label="Sharpe", value=num(s["Sharpe"]), delta=f"Sortino {num(s['Sortino'])}"),
        dict(label="最大回撤", value=pct(s["最大回撤"]), delta=f"最長 {s['最長回撤天數']} 天"),
        dict(label="獲利因子", value=num(s.get("獲利因子"))),
        dict(label="勝率", value=pct(s.get("勝率"), 0), delta=f"{s['交易次數']} 筆交易"),
    ]), unsafe_allow_html=True)
    g = s.get
    st.markdown(T.report_table([
        ("收益", [
            ("起始資金", money(g("起始資金"))), ("最終淨值", money(g("最終淨值"))),
            ("淨利", money(g("淨利")), sign(g("淨利"))), ("毛利", money(g("毛利"))), ("毛損", money(g("毛損"))),
            ("總報酬", pct(g("總報酬")), sign(g("總報酬"))), ("年化報酬", pct(g("年化報酬"))),
            ("買進持有報酬", pct(g("買進持有報酬"))), ("總手續費", money(g("總手續費"))),
        ]),
        ("風險", [
            ("最大回撤", pct(g("最大回撤"))), ("最大回撤金額", money(g("最大回撤金額"))),
            ("最長回撤（日曆天）", f"{g('最長回撤天數', 0)}"), ("年化波動", pct(g("年化波動"))),
            ("Sharpe", num(g("Sharpe"))), ("Sortino", num(g("Sortino"))), ("Calmar", num(g("Calmar"))),
            ("回復因子", num(g("回復因子"))), ("持倉時間比例", pct(g("持倉時間比例"), 0)),
        ]),
        ("交易", [
            ("交易次數", f"{g('交易次數', 0)}"), ("勝率", pct(g("勝率"), 1)),
            ("多單（勝率）", f"{g('多單次數', 0)}（{pct(g('多單勝率'), 0)}）"),
            ("空單（勝率）", f"{g('空單次數', 0)}（{pct(g('空單勝率'), 0)}）"),
            ("獲利因子", num(g("獲利因子"))), ("期望收益（每筆）", money(g("期望收益")), sign(g("期望收益"))),
            ("平均獲利 / 平均虧損", f"{money(g('平均獲利'))} / {money(g('平均虧損'))}"),
            ("盈虧比", num(g("盈虧比"))), ("SQN", num(g("SQN"))),
        ]),
        ("極值與連續", [
            ("最大單筆獲利", money(g("最大單筆獲利")), sign(g("最大單筆獲利"))),
            ("最大單筆虧損", money(g("最大單筆虧損")), sign(g("最大單筆虧損"))),
            ("最大連續獲利", f"{g('最大連續獲利次數', 0)} 筆（{money(g('最大連續獲利金額'))}）"),
            ("最大連續虧損", f"{g('最大連續虧損次數', 0)} 筆（{money(g('最大連續虧損金額'))}）"),
            ("平均持有 K 棒", num(g("平均持有K棒"), 1)),
        ]),
    ]), unsafe_allow_html=True)


# ───────────────────────── 頁面 ─────────────────────────
def on_upload():
    up = st.session_state.get("tester_upload")
    if up is None:
        return
    st.session_state.setdefault("codes", {})[MY] = up.getvalue().decode("utf-8", errors="replace")
    vers = st.session_state.setdefault("code_ver", {})
    vers[MY] = vers.get(MY, 0) + 1
    st.session_state["tester_tpl"] = MY


def render(watchlist: list[str], start, end, cost_bps: float) -> None:
    st.markdown('<div class="page-title">策略測試器</div>'
                '<div class="page-ref">用 Python 寫策略 → 回測 → 報告、最佳化、批次測試（類似 MT5 Strategy Tester）</div>',
                unsafe_allow_html=True)

    # ── 設定列 ──
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        tpl_names = list(TEMPLATES) + ([MY] if MY in st.session_state.get("codes", {}) else [])
        tpl = c1.selectbox("策略", tpl_names, key="tester_tpl")
        sym_opts = list(dict.fromkeys((watchlist or []) + ["2330", "SPY"]))
        symbol = c2.selectbox("標的", sym_opts, key="tester_sym",
                              help="從左側標的清單選。要測其他代號，先加到清單裡。")
        iv_name = c3.selectbox("週期", list(INTERVALS), key="tester_iv")
        cash = c4.number_input("初始資金", min_value=1000.0, value=100_000.0, step=10_000.0, format="%.0f",
                               key="tester_cash")
    interval = INTERVALS[iv_name]
    if interval in INTERVAL_LIMIT_DAYS:
        st.caption(f"Yahoo 的{iv_name}最多只有約 {INTERVAL_LIMIT_DAYS[interval]} 天歷史，起始日會自動調整。")

    codes = st.session_state.setdefault("codes", {})
    if tpl not in codes:
        codes[tpl] = load(tpl)
    ver = st.session_state.setdefault("code_ver", {}).get(tpl, 0)

    with st.expander("策略程式碼", expanded=True):
        code = code_editor(codes[tpl], key=f"ace::{tpl}::{ver}")
        codes[tpl] = code
        b1, b2, b3 = st.columns([1, 1, 2])
        b1.download_button("下載 .py", code.encode("utf-8"), file_name=f"{tpl}.py", mime="text/x-python")
        if b2.button("還原範本", disabled=tpl == MY):
            codes[tpl] = load(tpl)
            st.session_state["code_ver"][tpl] = ver + 1
            st.rerun()
        b3.file_uploader("上傳 .py", type=["py", "txt"], label_visibility="collapsed", key="tester_upload",
                         on_change=on_upload)

    try:
        cls = engine.load_strategy(code)
    except engine.StrategyError as e:
        st.error(f"策略程式碼有錯誤：\n\n```\n{e}\n```")
        return

    with st.container(border=True):
        st.markdown(f'<div class="hint"><b>{T.esc(cls.__name__)}</b> — {T.esc((cls.__doc__ or "").strip())}</div>',
                    unsafe_allow_html=True)
        params = param_widgets(cls, tpl)

    # ── 資料 ──
    try:
        data, used = cached_ohlcv(symbol, start, end, interval)
    except Exception as e:
        st.error(f"資料下載失敗：{e}")
        return
    if data.empty:
        st.warning(f"抓不到 {symbol} 的{iv_name}資料。")
        return

    # ── 回測 ──
    try:
        res = cached_run(code, tuple(sorted(params.items())), data, float(cash), float(cost_bps))
    except engine.StrategyError as e:
        st.error(f"回測時發生錯誤：\n\n```\n{e}\n```")
        return
    s = report.compute(res["equity"], res["trades"], data["Close"], res["exposure"], float(cash))
    ok, bad = res["lookahead"]

    st.markdown(T.chips([used, iv_name, f"{data.index[0]:%Y-%m-%d} → {data.index[-1]:%Y-%m-%d}",
                         f"{len(data):,} 根 K 棒", f"成本 {cost_bps:g} bp / 邊"]), unsafe_allow_html=True)
    st.markdown(T.status(ok, "前視偏差檢查通過：截斷資料重跑，每一根下的單都和完整資料時相同" if ok else
                         f"前視偏差檢查失敗：有 {bad} 根 K 棒的下單在截斷未來資料後改變了，策略可能偷看了未來"),
                unsafe_allow_html=True)

    tabs = st.tabs(["總覽", "圖表", "交易明細", "月報酬", "最佳化", "批次回測", "寫法說明"])

    with tabs[0]:
        overview(s)
        if s["交易次數"] == 0:
            st.info("這段期間沒有任何交易。可以調整參數、拉長期間，或檢查進場條件。")

    with tabs[1]:
        st.plotly_chart(price_chart(res, f"{used} · 進出場位置"), theme=None)
        st.plotly_chart(equity_chart(res), theme=None)
        st.plotly_chart(drawdown_chart(res["equity"]), theme=None)

    with tabs[2]:
        t = res["trades"]
        if t.empty:
            st.info("沒有交易。")
        else:
            st.plotly_chart(trade_bars(t), theme=None)
            show = t.copy()
            show.insert(0, "#", range(1, len(show) + 1))
            for c in ["進場價", "出場價"]:
                show[c] = show[c].map(lambda v: f"{v:,.4g}" if abs(v) < 1 else f"{v:,.2f}")
            show["數量"] = show["數量"].map(lambda v: f"{v:,.2f}")
            show["損益"] = show["損益"].map(money)
            show["手續費"] = show["手續費"].map(money)
            for c in ["報酬率", "最大有利波動", "最大不利波動"]:
                show[c] = show[c].map(lambda v: pct(v, 2))
            st.dataframe(show, hide_index=True)
            st.download_button("下載交易明細（CSV）", t.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{used}_{cls.__name__}_trades.csv", mime="text/csv")

    with tabs[3]:
        mt = report.monthly_returns(res["equity"])
        st.plotly_chart(monthly_heatmap(mt), theme=None)
        st.markdown('<div class="hint">以月底淨值計算，藍色為正報酬、紅色為負報酬。</div>', unsafe_allow_html=True)

    with tabs[4]:
        optimization_tab(cls, tpl, data, params, float(cash), float(cost_bps))

    with tabs[5]:
        batch_tab(code, cls, params, watchlist, start, end, interval, float(cash), float(cost_bps))

    with tabs[6]:
        st.markdown(API_DOC)


# ───────────────────────── 最佳化 ─────────────────────────
def apply_params(tpl: str, values: dict):
    for k, v in values.items():
        st.session_state[f"tp::{tpl}::{k}"] = v


def optimization_tab(cls, tpl, data, params, cash, cost):
    numeric = {k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    bools = [k for k, v in params.items() if isinstance(v, bool)]
    if not numeric and not bools:
        st.info("這個策略沒有數值參數可以最佳化。")
        return
    st.markdown('<div class="hint">勾選要最佳化的參數並設定範圍。排序只看<b>樣本內</b>；'
                '「前推」會保留最後一段資料當作樣本外驗證（和 MT5 的 Forward 相同）。'
                '好的參數應該在樣本外也維持表現，而不是只有樣本內最高。</div>', unsafe_allow_html=True)
    rows = []
    defaults = cls.params    # 用策略的預設值建範圍表：套用最佳參數後表格才不會被重設
    for k in numeric:
        v = defaults.get(k, numeric[k])
        is_int = isinstance(v, int)
        if v:
            lo, hi = v * 0.5, v * 1.5
        else:                               # 預設 0（例如停損關閉）：小數試 0–10%，整數試 0–10
            lo, hi = 0, (10 if is_int else 0.1)
        step = max(1, round((hi - lo) / 5)) if is_int else round((hi - lo) / 5, 4) or 0.01
        rows.append(dict(參數=k, 最佳化=False, 起始=float(int(lo) if is_int else lo),
                         結束=float(int(hi) if is_int else hi), 間距=float(step)))
    for k in bools:
        rows.append(dict(參數=k, 最佳化=False, 起始=0.0, 結束=1.0, 間距=1.0))
    table = st.data_editor(pd.DataFrame(rows), hide_index=True, key=f"opt_tbl::{tpl}",
                           disabled=["參數"], column_config={"最佳化": st.column_config.CheckboxColumn()})
    c1, c2, c3 = st.columns(3)
    objective = c1.selectbox("最佳化目標", list(optimize.OBJECTIVES), key="opt_obj")
    fwd_name = c2.selectbox("前推（樣本外）", list(optimize.FORWARD), index=2, key="opt_fwd")
    ranges = {}
    for _, r in table[table["最佳化"]].iterrows():
        k = r["參數"]
        if k in bools:
            ranges[k] = [False, True]
        elif isinstance(numeric[k], int):
            ranges[k] = [int(v) for v in optimize.value_range(int(r["起始"]), int(r["結束"]), max(int(r["間距"]), 1))]
        else:
            ranges[k] = optimize.value_range(float(r["起始"]), float(r["結束"]), float(r["間距"]))
    n = int(np.prod([len(v) for v in ranges.values()])) if ranges else 0
    c3.metric("組合數", f"{n:,}")
    limit = 600
    run = st.button("開始最佳化", type="primary", disabled=n == 0 or n > limit, key="opt_run")
    if n > limit:
        st.caption(f"組合太多（上限 {limit}），請縮小範圍或加大間距。")

    key = (tpl, code_key(str(cls.params)), str(data.index[0]), str(data.index[-1]), len(data), cost, cash,
           objective, fwd_name, tuple((k, tuple(v)) for k, v in ranges.items()),
           tuple(sorted((k, v) for k, v in params.items() if k not in ranges)))
    if run:
        bar = st.progress(0.0, text="最佳化中…")
        df, split = optimize.optimize(cls, data, params, ranges, cash, cost, objective,
                                      optimize.FORWARD[fwd_name],
                                      progress=lambda f: bar.progress(f, text=f"最佳化中… {f:.0%}"))
        bar.empty()
        st.session_state["opt_result"] = dict(key=key, df=df, split=split, keys=list(ranges), objective=objective)

    saved = st.session_state.get("opt_result")
    if not saved:
        return
    if saved["key"] != key:
        st.info("設定已經改變，請重新執行最佳化。")
        return
    df, split, keys, obj_name = saved["df"], saved["split"], saved["keys"], saved["objective"]
    if split is not None:
        st.markdown(T.chips([f"樣本內：{data.index[0]:%Y-%m-%d} → {split:%Y-%m-%d}",
                             f"樣本外：{split:%Y-%m-%d} → {data.index[-1]:%Y-%m-%d}"]), unsafe_allow_html=True)
    show = df.head(50).copy()
    for c in show.columns:
        if c in keys:
            continue
        if any(x in c for x in ["報酬", "回撤", "勝率"]):
            show[c] = show[c].map(lambda v: pct(v))
        elif "淨利" in c:
            show[c] = show[c].map(money)
        elif "次數" in c:
            show[c] = show[c].map(lambda v: "—" if pd.isna(v) else f"{int(v)}")
        elif c != "錯誤":
            show[c] = show[c].map(num)
    show.insert(0, "名次", range(1, len(show) + 1))
    st.dataframe(show, hide_index=True)

    if "樣本外·Sharpe" in df:
        fig = go.Figure(go.Scatter(
            x=df["樣本內·Sharpe"], y=df["樣本外·Sharpe"], mode="markers",
            marker=dict(size=9, color=T.BLUE, line=dict(color=T.SURFACE, width=1.5)),
            text=[", ".join(f"{k}={r[k]}" for k in keys) for _, r in df.iterrows()],
            hovertemplate="%{text}<br>樣本內 %{x:.2f}<br>樣本外 %{y:.2f}<extra></extra>", name="參數組合"))
        v = pd.concat([df["樣本內·Sharpe"], df["樣本外·Sharpe"]]).replace([np.inf, -np.inf], np.nan).dropna()
        if not v.empty:
            fig.add_trace(go.Scatter(x=[v.min(), v.max()], y=[v.min(), v.max()], mode="lines", name="樣本內 = 樣本外",
                                     line=dict(color=T.AXIS, dash="dash", width=1), hoverinfo="skip"))
        fig.update_xaxes(title_text="樣本內 Sharpe")
        fig.update_yaxes(title_text="樣本外 Sharpe")
        fig = T.style(fig, 400, "樣本內表現 vs 樣本外表現（點越集中在虛線附近越穩健）")
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, theme=None)

    if len(keys) >= 2:
        c1, c2 = st.columns(2)
        xk = c1.selectbox("熱圖 X 軸", keys, index=0, key="opt_hx")
        yk = c2.selectbox("熱圖 Y 軸", keys, index=1, key="opt_hy")
        if xk != yk:
            obj_col = f"樣本內·{optimize.OBJECTIVES[obj_name]}"
            cols = st.columns(2 if "樣本外·Sharpe" in df else 1)
            for col, metric, title in [(cols[0], obj_col, f"樣本內 {obj_name}")] + \
                    ([(cols[1], "樣本外·Sharpe", "樣本外 Sharpe")] if "樣本外·Sharpe" in df else []):
                pv = df.pivot_table(index=yk, columns=xk, values=metric, aggfunc="max")
                z = pv.to_numpy(dtype=float)
                fin = z[np.isfinite(z)]
                lim = float(np.abs(fin).max()) if fin.size else 1.0
                fig = go.Figure(go.Heatmap(
                    z=z, x=[str(c) for c in pv.columns], y=[str(i) for i in pv.index], colorscale=T.DIVERGING,
                    zmid=0, zmin=-lim, zmax=lim, xgap=2, ygap=2,
                    text=np.where(np.isfinite(z), np.char.mod("%.2f", np.nan_to_num(z)), ""), texttemplate="%{text}",
                    textfont=dict(family="JetBrains Mono, monospace", color=T.INK, size=11),
                    hovertemplate=f"{yk}=%{{y}}<br>{xk}=%{{x}}<br>%{{z:.2f}}<extra></extra>", showscale=False))
                fig.update_xaxes(title_text=xk, type="category", showgrid=False)
                fig.update_yaxes(title_text=yk, type="category", showgrid=False)
                fig = T.style(fig, 380, title, legend=False)
                fig.update_layout(hovermode="closest")
                col.plotly_chart(fig, theme=None)
            st.caption("同一格有其他參數時取最大值。")

    ok_rows = df[df["錯誤"].isna()] if "錯誤" in df else df
    if len(ok_rows):
        c1, c2 = st.columns([3, 1])
        rank = c1.number_input("套用第幾名的參數", min_value=1, max_value=len(ok_rows), value=1, step=1, key="opt_rank")
        best = {k: ok_rows.iloc[int(rank) - 1][k] for k in keys}
        best = {k: (bool(v) if isinstance(params[k], bool) else int(v) if isinstance(params[k], int) else float(v))
                for k, v in best.items()}
        c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        c2.button("套用", on_click=apply_params, args=(tpl, best), type="primary", key="opt_apply")
        st.caption("套用：" + "、".join(f"{k} = {v}" for k, v in best.items()))


# ───────────────────────── 批次回測 ─────────────────────────
def batch_tab(code, cls, params, watchlist, start, end, interval, cash, cost):
    st.markdown('<div class="hint">用同一份策略和參數，對左側清單中的每個標的各跑一次，'
                '看看策略是普遍有效，還是只在某一檔剛好有效。</div>', unsafe_allow_html=True)
    if not watchlist:
        st.info("左側標的清單是空的。")
        return
    key = (code_key(code), tuple(sorted(params.items())), tuple(watchlist), str(start), str(end), interval, cash, cost)
    if st.button(f"對清單中的 {len(watchlist)} 個標的執行", type="primary", key="batch_run"):
        rows, bar = [], st.progress(0.0)
        for k, sym in enumerate(watchlist):
            bar.progress((k + 1) / len(watchlist), text=f"回測中：{sym}")
            try:
                d, used = cached_ohlcv(sym, start, end, interval)
                if d.empty:
                    rows.append(dict(標的=sym, 備註="沒有資料"))
                    continue
                r = engine.run(cls, d, params, cash, cost)
                s = report.compute(r["equity"], r["trades"], d["Close"], r["exposure"], cash)
                rows.append(dict(標的=used, 總報酬=s["總報酬"], 買進持有=s.get("買進持有報酬"),
                                 年化報酬=s["年化報酬"], Sharpe=s["Sharpe"], 最大回撤=s["最大回撤"],
                                 獲利因子=s.get("獲利因子"), 勝率=s.get("勝率"), 交易次數=s["交易次數"], 備註=""))
            except Exception as e:
                rows.append(dict(標的=sym, 備註=str(e).splitlines()[0][:80]))
        bar.empty()
        st.session_state["batch_result"] = (key, pd.DataFrame(rows))

    saved = st.session_state.get("batch_result")
    if not saved:
        return
    if saved[0] != key:
        st.info("策略、參數或清單已經改變，請重新執行。")
        return
    df = saved[1]
    good = df.dropna(subset=["Sharpe"]) if "Sharpe" in df else df.iloc[0:0]
    if len(good):
        beat = int((good["總報酬"] > good["買進持有"]).sum())
        st.markdown(T.kpi_cards([
            dict(label="平均 Sharpe", value=num(good["Sharpe"].mean())),
            dict(label="Sharpe > 0 的標的", value=f"{int((good['Sharpe'] > 0).sum())} / {len(good)}"),
            dict(label="打敗買進持有", value=f"{beat} / {len(good)}"),
            dict(label="平均最大回撤", value=pct(good["最大回撤"].mean())),
        ]), unsafe_allow_html=True)
        g2 = good.sort_values("Sharpe")
        fig = go.Figure(go.Bar(y=g2["標的"], x=g2["Sharpe"], orientation="h",
                               marker_color=np.where(g2["Sharpe"] > 0, T.BLUE, T.DOWN),
                               hovertemplate="%{y}：Sharpe %{x:.2f}<extra></extra>"))
        fig = T.style(fig, 120 + 28 * len(g2), "各標的 Sharpe", legend=False)
        fig.update_layout(hovermode="closest")
        st.plotly_chart(fig, theme=None)
    show = df.copy()
    for c in ["總報酬", "買進持有", "年化報酬", "最大回撤", "勝率"]:
        if c in show:
            show[c] = show[c].map(lambda v: pct(v))
    for c in ["Sharpe", "獲利因子"]:
        if c in show:
            show[c] = show[c].map(num)
    st.dataframe(show, hide_index=True)
