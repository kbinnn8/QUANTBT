"""策略測試器（類似 MT5 Strategy Tester）。

版面：上方「測試設定」＋「▶ 開始回測」按鈕；下方分頁切換
參數 / 程式碼 / 回測報告 / 圖表 / 交易分析 / 交易明細 / 日誌 / 最佳化 / 批次回測 / 說明。
結果只在按下「開始回測」時更新；設定改變後會提示重新回測。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import time

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
VIEWS = ["參數", "程式碼", "回測報告", "圖表", "交易分析", "交易明細", "日誌", "最佳化", "批次回測", "說明"]
RESULT_VIEWS = {"回測報告", "圖表", "交易分析", "交易明細", "日誌"}
MY = "我的策略（上傳）"
FORWARD = {"不做前推": 0.0, "後 1/2": 1 / 2, "後 1/3": 1 / 3, "後 1/4": 1 / 4}
WEEKDAYS = ["一", "二", "三", "四", "五", "六", "日"]

API_DOC = """
### 策略怎麼寫

```python
class MyStrategy(Strategy):
    params = {"n": 20, "sl_pct": 0.05}      # 可調參數（會出現在「參數」分頁，也能最佳化）

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
- `self.buy(size=None, sl=None, tp=None, sl_pct=None, tp_pct=None, tag="")`：做多。
  `size` 不填就用「測試設定」的部位大小；填數字代表淨值比例（1 = 100%）
- `self.sell(...)`：做空，參數同上；持有反向部位時會自動反手
- `self.close_position(tag="")`：平倉
- `self.set_sl(價格)`、`self.set_tp(價格)`：修改停損 / 停利（可做移動停損）
- `self.log("訊息", 數值)`：寫到「日誌」分頁（類似 MT5 的 Print）

**部位狀態**：`self.is_flat` `self.is_long` `self.is_short` `self.position`（數量，空單為負）
`self.entry_price` `self.bars_in_trade` `self.open_pnl_pct` `self.equity` `self.balance`

**指標**（`ta.`）：`sma` `ema` `rsi` `macd`（回傳 3 條）`bollinger`（回傳 3 條）`atr` `highest` `lowest`
`stdev` `roc` `zscore` `shift`；交叉判斷用 `ta.crossover(a, b, i)` / `ta.crossunder(a, b, i)`。
`self.I(數列, "名稱", overlay=True)` 把指標畫到圖上（overlay=False 畫在副圖）。也可以用 `np`、`pd`、`math`。

**成交規則**
- 市價單在下一根開盤成交；停損 / 停利用每根的最高、最低價判斷，跳空穿越時以開盤價成交；
  同一根同時碰到停損和停利，保守假設先停損
- 手續費 = 成交金額 × 手續費 bp；滑價 = 成交價往不利方向移動 滑價 bp
- 淨部位模式：同一時間只有一個方向的部位

**報告名詞**
- **獲利因子** = 毛利 ÷ |毛損|；**期望收益** = 平均每筆損益；**回復因子** = 淨利 ÷ |最大回撤金額|
- **餘額** 只算已平倉損益；**淨值** 含未平倉損益
- **Z 分數**：輸贏是否成串（|Z| > 2 代表有顯著的連續性）；**AHPR / GHPR**：每筆交易的算術 / 幾何平均持有期報酬
- **LR 相關係數**：淨值曲線和一條直線有多像（越接近 1 越平穩向上）

**注意**：程式碼會在伺服器上執行，請把 app 設成私人（Streamlit 設定 → Sharing），或在 Secrets 設定 `APP_PASSWORD`。
"""


# ───────────────────────── 格式 ─────────────────────────
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


def digest(*parts) -> str:
    return hashlib.md5(repr(parts).encode()).hexdigest()[:12]


# ───────────────────────── 快取 ─────────────────────────
@st.cache_data(ttl=6 * 3600, show_spinner="下載 K 線資料中…", max_entries=64)
def cached_ohlcv(ticker, start, end, interval):
    return load_ohlcv(ticker, start, end, interval)


# ───────────────────────── 程式碼編輯器與參數 ─────────────────────────
def code_editor(code: str, key: str) -> str:
    """有語法上色的編輯器（streamlit-ace，打字即時更新）；套件不能用時退回純文字框。"""
    try:
        from streamlit_ace import st_ace
        out = st_ace(value=code, language="python", theme="tomorrow_night", key=key, height=520,
                     font_size=14, tab_size=4, show_gutter=True, wrap=False, auto_update=True,
                     placeholder="在這裡寫策略…")
        return out if out else code
    except Exception:
        return st.text_area("策略程式碼", code, height=520, key=key + "_ta", label_visibility="collapsed")


def pvals(tpl: str) -> dict:
    """參數值存在這裡（不隨分頁切換消失）。"""
    return st.session_state.setdefault("pvals", {}).setdefault(tpl, {})


def current_params(cls, tpl: str) -> dict:
    """目前的參數值：只從 pvals 讀（widget 不在畫面上時，Streamlit 會把它的值重設成預設值，不能依賴）。"""
    store = pvals(tpl)
    out = {}
    for name, default in cls.params.items():
        v = store.get(name, default)
        if type(v) is not type(default) and not (isinstance(default, float) and isinstance(v, int)):
            v = default           # 程式碼裡改了參數型別：用新的預設值
        out[name] = v
    return out


def _save_param(tpl: str, name: str, key: str):
    """widget 一改動就存進 pvals（回呼在重新執行前觸發，所以按「開始回測」時一定拿得到新值）。"""
    pvals(tpl)[name] = st.session_state[key]


def param_widgets(cls, tpl: str) -> dict:
    params = current_params(cls, tpl)
    if not params:
        st.info("這個策略沒有可調參數（在 class 裡加上 `params = {...}` 就會出現在這裡）。")
        return params
    cols = st.columns(min(4, len(params)))
    for k, (name, value) in enumerate(params.items()):
        c = cols[k % len(cols)]
        key = f"w::{tpl}::{name}"
        st.session_state[key] = value          # 每次都用儲存的值初始化（使用者的修改已經由回呼存進去）
        default = cls.params[name]
        cb = dict(key=key, on_change=_save_param, args=(tpl, name, key))
        if isinstance(default, bool):
            c.toggle(name, **cb)
        elif isinstance(default, int):
            c.number_input(name, step=1, **cb)
        elif isinstance(default, float):
            step = 0.01 if abs(default) < 1 else 0.1 if abs(default) < 10 else 1.0
            c.number_input(name, step=step, format="%g", **cb)
        else:
            c.text_input(name, **cb)
    if st.button("全部還原為預設值", key=f"reset_params::{tpl}"):
        pvals(tpl).clear()
        st.rerun()
    return current_params(cls, tpl)


# ───────────────────────── 圖 ─────────────────────────
def price_chart(res: dict, title: str, split=None):
    data, trades, inds = res["data"], res["trades"], res["indicators"]
    sub = [d for d in inds if not d["overlay"]]
    fig = make_subplots(rows=2 if sub else 1, cols=1, shared_xaxes=True, vertical_spacing=0.04,
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
        span = float(data["High"].max() - data["Low"].min()) or 1.0
        for side, symbol, color in [("多", "triangle-up", T.UP), ("空", "triangle-down", T.DOWN)]:
            t = trades[trades["方向"] == side]
            if len(t):
                # 箭頭畫在 K 棒外側（多單在最低價下方、空單在最高價上方），才不會被 K 線蓋住
                lo = data["Low"].reindex(t["進場時間"]).to_numpy()
                hi = data["High"].reindex(t["進場時間"]).to_numpy()
                y = lo - span * 0.025 if side == "多" else hi + span * 0.025
                fig.add_trace(go.Scatter(
                    x=t["進場時間"], y=y, mode="markers", name=f"{side}單進場",
                    marker=dict(symbol=symbol, size=14, color=color, line=dict(color=T.INK, width=1)),
                    customdata=np.stack([t["進場標籤"], t["報酬率"] * 100, t["進場價"]], axis=1),
                    hovertemplate=f"{side}單進場 %{{customdata[2]:.2f}}<br>%{{customdata[0]}}"
                                  f"<br>此筆報酬 %{{customdata[1]:.2f}}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=trades["出場時間"], y=trades["出場價"], mode="markers", name="出場",
            marker=dict(symbol="x-thin", size=10, color=T.INK, line=dict(color=T.INK, width=2)),
            customdata=np.stack([trades["出場原因"], trades["損益"]], axis=1),
            hovertemplate="出場 %{y:.2f}<br>%{customdata[0]}<br>損益 %{customdata[1]:,.0f}<extra></extra>"),
            row=1, col=1)
    if split is not None:
        fig.add_vline(x=split, line_dash="dash", line_color=T.ORANGE)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig = T.style(fig, 680 if sub else 580, title)
    fig.update_layout(legend=dict(orientation="h", y=-0.06, x=0, xanchor="left", yanchor="top"), margin=dict(b=70))
    return fig


def balance_chart(res: dict, split=None, height=380):
    eq, bal, close, cash = res["equity"], res["balance"], res["data"]["Close"], res["cash"]
    bh = close / close.iloc[0] * cash
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bh.index, y=bh, name="買進持有", line=dict(color=T.MUTED, width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="淨值（含未平倉）", line=dict(color=T.SERIES[2], width=1.4)))
    fig.add_trace(go.Scatter(x=bal.index, y=bal, name="餘額（已平倉）", line=dict(color=T.BLUE, width=2.2, shape="hv")))
    if split is not None:
        fig.add_vline(x=split, line_dash="dash", line_color=T.ORANGE,
                      annotation_text="前推開始", annotation_font=dict(color=T.ORANGE, size=11))
    fig.update_yaxes(tickformat=",.0f")
    return T.style(fig, height, "餘額 / 淨值 vs 買進持有")


def drawdown_chart(eq):
    dd = eq / eq.cummax() - 1
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", line=dict(color=T.DOWN, width=1.2),
                               fillcolor="rgba(230,103,103,0.18)", hovertemplate="%{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%")
    return T.style(fig, 220, "淨值回撤", legend=False)


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


def scatter(x, y, color, text, title, xt, yt, xfmt=None, yfmt=".1%"):
    fig = go.Figure(go.Scatter(x=x, y=y, mode="markers", text=text,
                               marker=dict(size=9, color=color, line=dict(color=T.SURFACE, width=1.5)),
                               hovertemplate="%{text}<extra></extra>"))
    fig.add_hline(y=0, line_color=T.AXIS)
    fig.update_xaxes(title_text=xt, tickformat=xfmt)
    fig.update_yaxes(title_text=yt, tickformat=yfmt)
    fig = T.style(fig, 340, title, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


def grouped_bars(t: pd.DataFrame, key, labels, title):
    g = t.groupby(key)["損益"].agg(["sum", "count"]).reindex(range(len(labels))).fillna(0)
    fig = go.Figure(go.Bar(x=labels, y=g["sum"], marker_color=np.where(g["sum"] >= 0, T.BLUE, T.DOWN),
                           customdata=g["count"], hovertemplate="%{x}：損益 %{y:,.0f}（%{customdata:.0f} 筆）<extra></extra>"))
    fig = T.style(fig, 280, title, legend=False)
    fig.update_layout(hovermode="closest", bargap=0.25)
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
def report_view(R: dict):
    s, x, cfg = R["s"], R["x"], R["cfg"]
    g = lambda k, d=None: s.get(k, x.get(k, d))   # noqa: E731
    bh = s.get("買進持有報酬")
    st.markdown(T.kpi_cards([
        dict(label="淨利", value=money(s["淨利"]), delta=f"總報酬 {pct(s['總報酬'])}", trend=sign(s["淨利"])),
        dict(label="年化報酬", value=pct(s["年化報酬"]),
             delta=f"買進持有 {pct(bh)}" if bh is not None else None,
             trend=sign(s["總報酬"] - bh) if bh is not None else None),
        dict(label="獲利因子", value=num(g("獲利因子")), delta=f"期望收益 {money(g('期望收益'))}"),
        dict(label="Sharpe", value=num(s["Sharpe"]), delta=f"Sortino {num(s['Sortino'])}"),
        dict(label="最大回撤", value=pct(s["最大回撤"]), delta=f"{money(s['最大回撤金額'])}"),
        dict(label="交易次數", value=f"{s['交易次數']}", delta=f"勝率 {pct(g('勝率'), 1)}"),
    ]), unsafe_allow_html=True)

    st.markdown(T.report_table([
        ("測試設定", [
            ("策略", R["strategy"]), ("標的", R["used"]), ("週期", cfg["週期"]),
            ("期間", f"{x['開始']:%Y-%m-%d} → {x['結束']:%Y-%m-%d}"), ("K 棒數", f"{x['K棒數']:,}"),
            ("初始資金", money(cfg["初始資金"])), ("部位大小", cfg["部位大小"]),
            ("最小交易單位", cfg["最小交易單位"]), ("手續費 / 滑價", cfg["成本"]),
            ("參數", ", ".join(f"{k}={v}" for k, v in R["params"].items()) or "—"),
        ]),
        ("收益", [
            ("最終淨值", money(s["最終淨值"])), ("淨利", money(s["淨利"]), sign(s["淨利"])),
            ("毛利", money(g("毛利"))), ("毛損", money(g("毛損"))),
            ("總報酬", pct(s["總報酬"]), sign(s["總報酬"])), ("年化報酬", pct(s["年化報酬"])),
            ("買進持有報酬", pct(bh)), ("超額報酬（vs 買進持有）", pct(s["總報酬"] - bh) if bh is not None else "—",
                                   sign(s["總報酬"] - bh) if bh is not None else None),
            ("總手續費", money(g("總手續費"))), ("持倉時間比例", pct(s.get("持倉時間比例"), 0)),
        ]),
        ("回撤與風險", [
            ("餘額回撤絕對值", money(x.get("餘額回撤絕對值"))),
            ("餘額最大回撤", f"{money(x.get('餘額最大回撤金額'))}（{pct(x.get('餘額最大回撤'))}）"),
            ("淨值回撤絕對值", money(x.get("淨值回撤絕對值"))),
            ("淨值最大回撤", f"{money(s['最大回撤金額'])}（{pct(s['最大回撤'])}）"),
            ("最長回撤（日曆天）", f"{s['最長回撤天數']}"), ("年化波動", pct(s["年化波動"])),
            ("Sharpe", num(s["Sharpe"])), ("Sortino", num(s["Sortino"])), ("Calmar", num(s["Calmar"])),
            ("回復因子", num(s["回復因子"])),
        ]),
        ("交易統計", [
            ("交易次數", f"{s['交易次數']}"),
            ("獲利交易（占比）", f"{x.get('獲利交易數', 0)}（{pct(x.get('獲利交易比例'), 1)}）"),
            ("虧損交易（占比）", f"{x.get('虧損交易數', 0)}（{pct(x.get('虧損交易比例'), 1)}）"),
            ("多單（勝率）", f"{g('多單次數', 0)}（{pct(g('多單勝率'), 1)}）"),
            ("空單（勝率）", f"{g('空單次數', 0)}（{pct(g('空單勝率'), 1)}）"),
            ("獲利因子", num(g("獲利因子"))), ("期望收益（每筆）", money(g("期望收益")), sign(g("期望收益"))),
            ("平均每筆報酬", pct(g("平均每筆報酬"), 2)),
            ("平均獲利 / 平均虧損", f"{money(g('平均獲利'))} / {money(g('平均虧損'))}"),
            ("盈虧比", num(g("盈虧比"))),
        ]),
        ("極值與連續", [
            ("最大單筆獲利", money(g("最大單筆獲利")), sign(g("最大單筆獲利"))),
            ("最大單筆虧損", money(g("最大單筆虧損")), sign(g("最大單筆虧損"))),
            ("最多連續獲利（金額）", f"{g('最大連續獲利次數', 0)} 筆（{money(s.get('最大連續獲利金額'))}）"),
            ("最多連續虧損（金額）", f"{g('最大連續虧損次數', 0)} 筆（{money(s.get('最大連續虧損金額'))}）"),
            ("最大連續獲利金額（筆數）", f"{money(x.get('最大連續獲利金額'))}（{x.get('最大連續獲利金額次數', 0)} 筆）"),
            ("最大連續虧損金額（筆數）", f"{money(x.get('最大連續虧損金額'))}（{x.get('最大連續虧損金額次數', 0)} 筆）"),
            ("平均連續獲利 / 虧損", f"{num(x.get('平均連續獲利次數'), 1)} / {num(x.get('平均連續虧損次數'), 1)} 筆"),
        ]),
        ("統計與持有時間", [
            ("Z 分數（信賴度）", f"{num(x.get('Z 分數'))}（{pct(x.get('Z 信賴度'), 1)}）"),
            ("AHPR / GHPR", f"{num(x.get('AHPR'), 4)} / {num(x.get('GHPR'), 4)}"),
            ("LR 相關係數", num(x.get("LR 相關係數"))), ("LR 標準誤", money(x.get("LR 標準誤"))),
            ("SQN", num(g("SQN"))),
            ("平均持有", f"{report._fmt_td(x.get('平均持有時間'))}（{num(g('平均持有K棒'), 1)} 根）"),
            ("最長 / 最短持有", f"{report._fmt_td(x.get('最長持有時間'))} / {report._fmt_td(x.get('最短持有時間'))}"),
        ]),
    ]), unsafe_allow_html=True)

    if R.get("split") is not None:
        st.markdown("#### 回測期 vs 前推期")
        rows = {}
        for name, seg in [("回測期（樣本內）", R["seg_in"]), ("前推期（樣本外）", R["seg_out"])]:
            rows[name] = {"總報酬": pct(seg.get("總報酬")), "年化報酬": pct(seg.get("年化報酬")),
                          "Sharpe": num(seg.get("Sharpe")), "最大回撤": pct(seg.get("最大回撤")),
                          "獲利因子": num(seg.get("獲利因子")), "勝率": pct(seg.get("勝率")),
                          "交易次數": f"{seg.get('交易次數', 0)}", "買進持有": pct(seg.get("買進持有報酬"))}
        st.dataframe(pd.DataFrame(rows).T)
        st.markdown('<div class="hint">前推期是策略「沒看過」的資料。前推期的表現如果和回測期差很多，'
                    '代表參數可能過度貼合歷史（過擬合）。</div>', unsafe_allow_html=True)
    st.plotly_chart(balance_chart(R["res"], R.get("split"), 320), theme=None)


def analysis_view(R: dict):
    t = R["res"]["trades"]
    if t.empty:
        st.info("沒有交易可以分析。")
        return
    color = np.where(t["損益"] > 0, T.UP, T.DOWN)
    txt = [f"第 {k + 1} 筆 · {r['方向']}單 · 報酬 {r['報酬率']:.2%}" for k, r in t.reset_index(drop=True).iterrows()]
    c1, c2 = st.columns(2)
    c1.plotly_chart(scatter(t["最大有利波動"], t["報酬率"], color, txt, "MFE：持有期間最大浮盈 vs 最終報酬",
                            "最大有利波動", "最終報酬", ".0%"), theme=None)
    c2.plotly_chart(scatter(t["最大不利波動"], t["報酬率"], color, txt, "MAE：持有期間最大浮虧 vs 最終報酬",
                            "最大不利波動", "最終報酬", ".0%"), theme=None)
    st.markdown('<div class="hint">MFE 圖：很多虧損單曾經有大浮盈 → 可以考慮停利或移動停損。'
                'MAE 圖：獲利單的最大浮虧很少超過某個值 → 停損可以設在那附近。</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(scatter(t["持有K棒"], t["報酬率"], color, txt, "持有時間 vs 報酬", "持有 K 棒數", "報酬"),
                    theme=None)
    fig = go.Figure(go.Histogram(x=t["報酬率"] * 100, nbinsx=30, marker_color=T.BLUE,
                                 hovertemplate="報酬 %{x}%：%{y} 筆<extra></extra>"))
    fig.add_vline(x=0, line_color=T.AXIS)
    fig.update_xaxes(title_text="每筆報酬（%）")
    fig = T.style(fig, 340, "報酬分布", legend=False)
    fig.update_layout(hovermode="closest", bargap=0.05)
    c2.plotly_chart(fig, theme=None)

    et = pd.to_datetime(t["進場時間"])
    c1, c2 = st.columns(2)
    c1.plotly_chart(grouped_bars(t.assign(k=et.dt.weekday), "k", [f"週{d}" for d in WEEKDAYS],
                                 "依進場星期：損益合計"), theme=None)
    c2.plotly_chart(grouped_bars(t.assign(k=et.dt.month - 1), "k", [f"{m}月" for m in range(1, 13)],
                                 "依進場月份：損益合計"), theme=None)
    if R["cfg"]["interval"] not in ("1d", "1wk"):
        st.plotly_chart(grouped_bars(t.assign(k=et.dt.hour), "k", [f"{h}時" for h in range(24)],
                                     "依進場時段：損益合計"), theme=None)
    st.markdown("#### 月報酬")
    st.plotly_chart(monthly_heatmap(report.monthly_returns(R["res"]["equity"])), theme=None)
    reasons = t.groupby("出場原因")["損益"].agg(筆數="count", 損益合計="sum", 平均損益="mean")
    reasons["勝率"] = t.groupby("出場原因")["損益"].apply(lambda v: (v > 0).mean())
    st.markdown("#### 依出場原因")
    st.dataframe(pd.DataFrame({"筆數": reasons["筆數"], "損益合計": reasons["損益合計"].map(money),
                               "平均損益": reasons["平均損益"].map(money), "勝率": reasons["勝率"].map(pct)}))


def trades_view(R: dict):
    t = R["res"]["trades"]
    if t.empty:
        st.info("沒有交易。")
        return
    st.plotly_chart(trade_bars(t), theme=None)
    show = t.copy()
    show.insert(0, "#", range(1, len(show) + 1))
    for c in ["進場價", "出場價"]:
        show[c] = show[c].map(lambda v: f"{v:,.4g}" if abs(v) < 1 else f"{v:,.2f}")
    show["數量"] = show["數量"].map(lambda v: f"{v:,.2f}")
    for c in ["損益", "手續費", "餘額"]:
        show[c] = show[c].map(money)
    for c in ["報酬率", "最大有利波動", "最大不利波動"]:
        show[c] = show[c].map(lambda v: pct(v, 2))
    st.dataframe(show, hide_index=True)
    st.download_button("下載交易明細（CSV）", t.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{R['used']}_{R['strategy']}_trades.csv", mime="text/csv")


def journal_view(R: dict):
    j = R["res"]["journal"]
    if j.empty:
        st.info("日誌是空的。")
        return
    kinds = list(j["類型"].unique())
    pick = st.multiselect("顯示類型", kinds, default=kinds, key="journal_kinds")
    view = j[j["類型"].isin(pick)]
    st.caption(f"共 {len(view):,} 筆（最多記錄 20,000 筆）。在策略裡用 self.log(...) 可以寫入自己的訊息。")
    st.dataframe(view, hide_index=True)
    st.download_button("下載日誌（CSV）", j.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{R['used']}_{R['strategy']}_journal.csv", mime="text/csv")


# ───────────────────────── 回呼 ─────────────────────────
def on_upload():
    up = st.session_state.get("tester_upload")
    if up is None:
        return
    st.session_state.setdefault("codes", {})[MY] = up.getvalue().decode("utf-8", errors="replace")
    vers = st.session_state.setdefault("code_ver", {})
    vers[MY] = vers.get(MY, 0) + 1
    st.session_state["tester_tpl"] = MY


def on_start():
    st.session_state["tester_run"] = True
    if st.session_state.get("tester_view") not in RESULT_VIEWS:
        st.session_state["tester_view"] = "回測報告"


def apply_params(tpl: str, values: dict):
    pvals(tpl).update(values)
    st.session_state["tester_run"] = True
    st.session_state["tester_view"] = "回測報告"


# ───────────────────────── 頁面 ─────────────────────────
def render(watchlist: list[str], start, end, cost_bps: float) -> None:
    codes = st.session_state.setdefault("codes", {})
    st.session_state.setdefault("tester_view", "參數")

    st.markdown('<div class="page-title">策略測試器</div>'
                '<div class="page-ref">用 Python 寫策略 → 設定測試條件 → ▶ 開始回測 → 看報告、最佳化、批次測試</div>',
                unsafe_allow_html=True)

    # ── 測試設定（類似 MT5 的 Settings） ──
    with st.container(border=True):
        st.markdown("<div class='kpi-label'>測試設定</div>", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        tpl_names = list(TEMPLATES) + ([MY] if MY in codes else [])
        tpl = c1.selectbox("策略（EA）", tpl_names, key="tester_tpl")
        sym_opts = list(dict.fromkeys((watchlist or []) + ["2330", "SPY"]))
        symbol = c2.selectbox("標的", sym_opts, key="tester_sym", help="從左側標的清單選。要測其他代號，先加到清單裡。")
        iv_name = c3.selectbox("週期", list(INTERVALS), key="tester_iv")
        fwd_name = c4.selectbox("前推", list(FORWARD), key="tester_fwd",
                                help="保留最後一段資料當作「沒看過的資料」，報告會分開顯示（同 MT5 的 Forward）。")

        interval = INTERVALS[iv_name]
        today = dt.date.today()
        c1, c2, c3, c4 = st.columns(4)
        d_start = c1.date_input("開始日期", value=start if isinstance(start, dt.date) else today.replace(year=today.year - 5),
                                min_value=dt.date(1990, 1, 1), max_value=today, key="tester_start")
        d_end = c2.date_input("結束日期", value=min(end - dt.timedelta(days=1), today) if isinstance(end, dt.date) else today,
                              min_value=dt.date(1990, 1, 1), max_value=today, key="tester_end")
        cash = c3.number_input("初始資金", min_value=1000.0, value=100_000.0, step=10_000.0, format="%.0f", key="tester_cash")
        lot = c4.number_input("最小交易單位", min_value=0.0, value=0.0, step=1.0, format="%g", key="tester_lot",
                              help="0 = 可以買零碎數量；台股整張填 1000、零股填 1；美股填 1。")

        c1, c2, c3, c4 = st.columns(4)
        sizing_mode = c1.selectbox("部位大小", engine.SIZING_MODES, key="tester_sizing",
                                   help="策略的 buy()/sell() 沒指定 size 時使用。")
        default_val = {"淨值比例": 1.0, "固定金額": 100_000.0, "固定數量": 1000.0}[sizing_mode]
        sizing_value = c2.number_input({"淨值比例": "淨值比例（1 = 100%）", "固定金額": "每筆金額",
                                        "固定數量": "每筆數量"}[sizing_mode],
                                       min_value=0.0, value=default_val, step=0.1 if sizing_mode == "淨值比例" else 1000.0,
                                       format="%g", key=f"tester_sizeval::{sizing_mode}")
        commission = c3.number_input("手續費（bp / 邊）", min_value=0.0, value=float(cost_bps), step=0.5, format="%g",
                                     key="tester_comm", help="1 bp = 0.01%。預設值來自左側「交易成本」。")
        slippage = c4.number_input("滑價（bp / 邊）", min_value=0.0, value=0.0, step=0.5, format="%g", key="tester_slip",
                                   help="成交價往不利方向移動的幅度，用來模擬實際成交比預期差。")
        try:
            st.button("▶  開始回測", type="primary", width="stretch", on_click=on_start, key="tester_start_btn")
        except TypeError:   # 舊版 Streamlit
            st.button("▶  開始回測", type="primary", use_container_width=True, on_click=on_start,
                      key="tester_start_btn")

    if interval in INTERVAL_LIMIT_DAYS:
        st.caption(f"Yahoo 的{iv_name}最多只有約 {INTERVAL_LIMIT_DAYS[interval]} 天歷史，開始日期會自動調整。")

    if tpl not in codes:
        codes[tpl] = load(tpl)
    ver = st.session_state.setdefault("code_ver", {}).get(tpl, 0)

    # ── 分頁 ──
    try:
        view = st.segmented_control("檢視", VIEWS, key="tester_view", label_visibility="collapsed")
    except AttributeError:
        view = st.radio("檢視", VIEWS, key="tester_view", horizontal=True, label_visibility="collapsed")
    view = view or "參數"

    if view == "程式碼":
        codes[tpl] = code_editor(codes[tpl], key=f"ace::{tpl}::{ver}")
        b1, b2, b3 = st.columns([1, 1, 2])
        b1.download_button("下載 .py", codes[tpl].encode("utf-8"), file_name=f"{tpl}.py", mime="text/x-python")
        if b2.button("還原範本", disabled=tpl == MY):
            codes[tpl] = load(tpl)
            st.session_state["code_ver"][tpl] = ver + 1
            st.rerun()
        b3.file_uploader("上傳 .py", type=["py", "txt"], label_visibility="collapsed", key="tester_upload",
                         on_change=on_upload)
        st.markdown('<div class="code-note">程式碼會自動儲存在這次連線中；重新整理頁面前記得「下載 .py」。</div>',
                    unsafe_allow_html=True)

    code = codes[tpl]
    try:
        cls = engine.load_strategy(code)
        code_err = None
    except engine.StrategyError as e:
        cls, code_err = None, str(e)

    if code_err:
        st.error(f"策略程式碼有錯誤，修正後再按開始回測：\n\n```\n{code_err}\n```")
        if view != "程式碼":
            return
    if cls is None:
        return

    params = param_widgets(cls, tpl) if view == "參數" else current_params(cls, tpl)
    if view == "參數":
        st.markdown(f'<div class="hint"><b>{T.esc(cls.__name__)}</b> — {T.esc((cls.__doc__ or "").strip())}</div>',
                    unsafe_allow_html=True)

    cfg = {"週期": iv_name, "interval": interval, "初始資金": cash,
           "部位大小": f"{sizing_mode} {sizing_value:g}" + ("（淨值的 {:.0%}）".format(sizing_value) if sizing_mode == "淨值比例" else ""),
           "最小交易單位": f"{lot:g}" if lot else "不限（可零碎）",
           "成本": f"{commission:g} bp / {slippage:g} bp"}
    opts = dict(slippage_bps=slippage, sizing_mode=sizing_mode, sizing_value=sizing_value, lot_size=lot)
    snap = digest(code, sorted(params.items()), symbol, interval, str(d_start), str(d_end), cash, commission,
                  opts, fwd_name)

    # ── 執行回測（只有按下開始時） ──
    if st.session_state.pop("tester_run", False):
        run_backtest(cls, tpl, code, params, symbol, interval, d_start, d_end, cash, commission, opts, fwd_name, cfg, snap)

    R = st.session_state.get("tester_result")
    err = st.session_state.get("tester_error")
    if err:
        st.error(err)
    if R is None:
        if view in RESULT_VIEWS:
            st.info("還沒有回測結果。設定好之後按上方的「▶ 開始回測」。")
    elif R["snap"] != snap:
        st.warning("設定、參數或程式碼已經改變，下面是上一次的結果。按「▶ 開始回測」更新。")
    if R is not None and view in RESULT_VIEWS:
        ok, bad = R["lookahead"]
        st.markdown(T.chips([R["strategy"], R["used"], R["cfg"]["週期"],
                             f"{R['x']['開始']:%Y-%m-%d} → {R['x']['結束']:%Y-%m-%d}",
                             f"{R['x']['K棒數']:,} 根 K 棒", f"耗時 {R['elapsed']:.2f} 秒"]), unsafe_allow_html=True)
        st.markdown(T.status(ok, "前視偏差檢查通過：截斷資料重跑，每一根下的單都和完整資料時相同" if ok else
                             f"前視偏差檢查失敗：有 {bad} 根 K 棒的下單在截斷未來資料後改變了，策略可能偷看了未來"),
                    unsafe_allow_html=True)

    if view == "回測報告" and R:
        report_view(R)
        if R["s"]["交易次數"] == 0:
            st.info("這段期間沒有任何交易。可以調整參數、拉長期間，或到「日誌」看看訂單為什麼沒有成交。")
    elif view == "圖表" and R:
        st.plotly_chart(price_chart(R["res"], f"{R['used']} · 進出場位置", R.get("split")), theme=None)
        st.plotly_chart(balance_chart(R["res"], R.get("split")), theme=None)
        st.plotly_chart(drawdown_chart(R["res"]["equity"]), theme=None)
    elif view == "交易分析" and R:
        analysis_view(R)
    elif view == "交易明細" and R:
        trades_view(R)
    elif view == "日誌" and R:
        journal_view(R)
    elif view == "最佳化":
        data = load_data(symbol, d_start, d_end, interval)
        if data is not None:
            optimization_tab(cls, tpl, data, params, cash, commission, opts)
    elif view == "批次回測":
        batch_tab(code, cls, params, watchlist, d_start, d_end, interval, cash, commission, opts)
    elif view == "說明":
        st.markdown(API_DOC)


def load_data(symbol, d_start, d_end, interval):
    try:
        data, used = cached_ohlcv(symbol, d_start, d_end + dt.timedelta(days=1), interval)
    except Exception as e:
        st.error(f"資料下載失敗：{e}")
        return None
    if data.empty:
        st.warning(f"抓不到 {symbol} 的資料。")
        return None
    return data


def run_backtest(cls, tpl, code, params, symbol, interval, d_start, d_end, cash, commission, opts, fwd_name, cfg, snap):
    st.session_state.pop("tester_error", None)
    if d_start >= d_end:
        st.session_state["tester_error"] = "開始日期必須早於結束日期。"
        return
    try:
        data, used = cached_ohlcv(symbol, d_start, d_end + dt.timedelta(days=1), interval)
    except Exception as e:
        st.session_state["tester_error"] = f"資料下載失敗：{e}"
        return
    if data.empty:
        st.session_state["tester_error"] = f"抓不到 {symbol} 的資料（{cfg['週期']}）。"
        return
    t0 = time.time()
    try:
        with st.spinner("回測中…"):
            res = engine.run(cls, data, params, cash, commission, **opts)
            look = engine.lookahead_check(cls, data, params, cash, commission, **opts)
    except engine.StrategyError as e:
        st.session_state["tester_error"] = f"回測時發生錯誤：\n\n```\n{e}\n```"
        return
    s = report.compute(res["equity"], res["trades"], data["Close"], res["exposure"], cash)
    x = report.extended(res)
    R = dict(res=res, s=s, x=x, cfg=cfg, snap=snap, used=used, strategy=f"{cls.__name__}（{tpl}）",
             params=dict(params), lookahead=look, elapsed=time.time() - t0, split=None)
    frac = FORWARD[fwd_name]
    if frac:
        split = optimize.split_time(data, frac)
        R.update(split=split, seg_in=report.segment(res, end=split), seg_out=report.segment(res, start=split))
    st.session_state["tester_result"] = R


# ───────────────────────── 最佳化 ─────────────────────────
def optimization_tab(cls, tpl, data, params, cash, cost, opts):
    numeric = {k: v for k, v in params.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    bools = [k for k, v in params.items() if isinstance(v, bool)]
    if not numeric and not bools:
        st.info("這個策略沒有數值參數可以最佳化。")
        return
    st.markdown('<div class="hint">勾選要最佳化的參數並設定範圍（使用上方的測試設定：標的、期間、資金、成本）。'
                '排序只看<b>樣本內</b>；「前推」會保留最後一段資料當作樣本外驗證（同 MT5 的 Forward）。'
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
    if n == 0:
        st.caption("先在表格的「最佳化」欄勾選至少一個參數。")
    elif n > limit:
        st.caption(f"組合太多（上限 {limit}），請縮小範圍或加大間距。")

    key = digest(tpl, str(cls.params), str(data.index[0]), str(data.index[-1]), len(data), cost, cash, opts,
                 objective, fwd_name, ranges, sorted((k, v) for k, v in params.items() if k not in ranges))
    if run:
        bar = st.progress(0.0, text="最佳化中…")
        df, split = optimize.optimize(cls, data, params, ranges, cash, cost, objective, optimize.FORWARD[fwd_name],
                                      progress=lambda f: bar.progress(f, text=f"最佳化中… {f:.0%}"), **opts)
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
        if any(k in c for k in ["報酬", "回撤", "勝率"]):
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
        c2.button("套用並回測", on_click=apply_params, args=(tpl, best), type="primary", key="opt_apply")
        st.caption("套用：" + "、".join(f"{k} = {v}" for k, v in best.items()))


# ───────────────────────── 批次回測 ─────────────────────────
def batch_tab(code, cls, params, watchlist, d_start, d_end, interval, cash, cost, opts):
    st.markdown('<div class="hint">用同一份策略、參數和測試設定，對左側清單中的每個標的各跑一次，'
                '看看策略是普遍有效，還是只在某一檔剛好有效。</div>', unsafe_allow_html=True)
    if not watchlist:
        st.info("左側標的清單是空的。")
        return
    key = digest(code, sorted(params.items()), watchlist, str(d_start), str(d_end), interval, cash, cost, opts)
    if st.button(f"對清單中的 {len(watchlist)} 個標的執行", type="primary", key="batch_run"):
        rows, bar = [], st.progress(0.0)
        for k, sym in enumerate(watchlist):
            bar.progress((k + 1) / len(watchlist), text=f"回測中：{sym}")
            try:
                d, used = cached_ohlcv(sym, d_start, d_end + dt.timedelta(days=1), interval)
                if d.empty:
                    rows.append(dict(標的=sym, 備註="沒有資料"))
                    continue
                r = engine.run(cls, d, params, cash, cost, **opts)
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
        st.info("策略、參數、設定或清單已經改變，請重新執行。")
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
