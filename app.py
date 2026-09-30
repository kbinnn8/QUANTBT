"""Quant Lab：依照 Ernest Chan《Quantitative Trading》做研究的回測工具。

本機執行：  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import inspect

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quantlab import metrics
from quantlab.backtest import lookahead_check, run_backtest
from quantlab.data import load_prices
from quantlab.strategies import REGISTRY

# 顏色：驗證過的類別色（藍、橘）與藍↔紅發散色階，灰色當中點
BLUE, ORANGE, RED, GRAY = "#2a78d6", "#eb6834", "#d03b3b", "#898781"
DIVERGING = [[0.0, RED], [0.5, "#f0efec"], [1.0, BLUE]]

st.set_page_config(page_title="Quant Lab", page_icon="📈", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="下載價格資料中…")
def cached_prices(tickers: tuple[str, ...], start, end) -> pd.DataFrame:
    return load_prices(list(tickers), start, end)


def pct(v, digits=1):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{digits}f}%"


def num(v, digits=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


# ───────────────────────── 側欄：所有控制項 ─────────────────────────
with st.sidebar:
    st.header("📈 Quant Lab")
    strat_name = st.selectbox("策略", list(REGISTRY))
    strat = REGISTRY[strat_name]

    st.subheader("資料")
    cols = st.columns(strat.N_ASSETS)
    tickers = tuple(
        c.text_input(f"標的 {i + 1}", value=strat.DEFAULT_TICKERS[i], key=f"tk{i}",
                     help="Yahoo 代號。美股如 KO；台股直接打數字如 2330（自動判斷上市 / 上櫃）。").strip().upper()
        for i, c in enumerate(cols)
    )
    periods = {"最近 3 年": 3, "最近 5 年": 5, "最近 10 年": 10,
               "書中期間（2006-05 ~ 2007-11）": None, "自訂": None}
    period = st.radio("期間", list(periods), index=1,
                      help="資料來自 Yahoo，近期期間會抓到最新一個交易日。"
                           "選「書中期間」可以對照書上的結果。")
    today = dt.date.today()
    tomorrow = today + dt.timedelta(days=1)   # yfinance 的 end 不含當天，所以加一天
    if periods[period]:
        start, end = today - dt.timedelta(days=round(365.25 * periods[period])), tomorrow
    elif period.startswith("書中"):
        start, end = dt.date(2006, 5, 23), dt.date(2007, 12, 1)
    else:
        start = st.date_input("開始", dt.date(2015, 1, 1), min_value=dt.date(1995, 1, 1), max_value=today)
        end = st.date_input("結束", today, min_value=dt.date(1995, 1, 1), max_value=today) + dt.timedelta(days=1)

    st.subheader("策略參數")
    params = {}
    for p in strat.PARAMS:
        if p["kind"] == "choice":
            params[p["key"]] = st.selectbox(p["label"], p["options"],
                                            index=p["options"].index(p["default"]), help=p.get("help"))
        elif p["kind"] == "int":
            params[p["key"]] = st.slider(p["label"], int(p["min"]), int(p["max"]), int(p["default"]),
                                         step=int(p["step"]), help=p.get("help"))
        else:
            params[p["key"]] = st.slider(p["label"], float(p["min"]), float(p["max"]), float(p["default"]),
                                         step=float(p["step"]), help=p.get("help"))

    cost_bps = st.slider("單邊交易成本（bp）", 0.0, 60.0, 5.0, 0.5,
                         help="1 bp = 0.01%。書中 Example 3.7 用 5 bp（美股大型股）。"
                              "台股：手續費 14.25 bp（買賣各一次，券商常有折扣）＋ 賣出證交稅 30 bp"
                              "（ETF 10 bp），平均每邊約 25–30 bp；ETF 約 15–20 bp。")

# ───────────────────────── 資料 + 回測 ─────────────────────────
st.title(strat_name)
st.caption(f"📖 書中對應：{strat.BOOK_REF}")

if start >= end:
    st.error("開始日期必須早於結束日期。")
    st.stop()

try:
    prices = cached_prices(tickers, start, end)
except Exception as e:  # yfinance 偶爾會被限流，給使用者看得懂的訊息
    st.error(f"資料下載失敗：{e}\n\n可能是代號錯誤，或 Yahoo 暫時限流，稍等一下再重新整理。")
    st.stop()

if len(prices) <= params.get("train_days", 0) + 20:
    st.warning(f"這段期間只有 {len(prices)} 個交易日，扣掉訓練期後測試期太短。"
               "請拉長期間或縮短訓練期。")
    st.stop()

st.caption(f"📅 資料來源：Yahoo Finance（已調整分割與配息），"
           f"{prices.index[0].date()} ~ {prices.index[-1].date()}，共 {len(prices)} 個交易日")

out = strat.run(prices, params)
bt = run_backtest(prices, out["weights"], cost_bps)
is_train = out["is_train"]
ex = out["extras"]

# 書中計算訓練期 Sharpe 時跳過第一天（沒有前一天的價格）
train_net, test_net = bt.net[is_train].iloc[1:], bt.net[~is_train]
train_gross, test_gross = bt.gross[is_train].iloc[1:], bt.gross[~is_train]
s_tr, s_te = metrics.summary(train_net), metrics.summary(test_net)

# ───────────────────────── 頂部重點數字 ─────────────────────────
m = st.columns(4)
m[0].metric("測試期 Sharpe（扣成本）", num(s_te["Sharpe"]),
            delta=(f"{s_te['Sharpe'] - s_tr['Sharpe']:+.2f} vs 訓練期"
                   if not (np.isnan(s_te["Sharpe"]) or np.isnan(s_tr["Sharpe"])) else None))
m[1].metric("訓練期 Sharpe（扣成本）", num(s_tr["Sharpe"]),
            help="顯示「—」代表訓練期一次都沒進場（門檻太嚴），可以調低進場門檻。")
m[2].metric("測試期最大回撤", pct(s_te["最大回撤"]))
if "hedge_ratio" in ex:
    m[3].metric("避險比例", num(ex["hedge_ratio"], 3))
else:
    m[3].metric("測試期年化報酬", pct(s_te["年化報酬"]))

ok, bad_days = lookahead_check(strat.run, prices, params)
if ok:
    st.success("✅ 前視偏差檢查通過：砍掉最後 20 天資料重跑，重疊期間的部位完全一致。")
else:
    st.error(f"⚠️ 前視偏差檢查失敗：有 {bad_days} 天的部位在砍掉未來資料後改變了，策略可能偷看了未來。")

tab_perf, tab_sig, tab_sweep, tab_data, tab_help = st.tabs(
    ["績效", "訊號", "參數掃描", "資料", "策略說明"])

# ───────────────────────── 績效 ─────────────────────────
with tab_perf:
    rows = {
        "訓練期（未扣成本）": metrics.summary(train_gross),
        "訓練期（扣成本）": s_tr,
        "測試期（未扣成本）": metrics.summary(test_gross),
        "測試期（扣成本）": s_te,
    }
    table = pd.DataFrame(rows).T
    shown = pd.DataFrame({
        "Sharpe": table["Sharpe"].map(num),
        "年化報酬": table["年化報酬"].map(pct),
        "年化波動": table["年化波動"].map(pct),
        "最大回撤": table["最大回撤"].map(pct),
        "最長回撤天數": table["最長回撤天數"].astype(int),
        "總報酬": table["總報酬"].map(pct),
        "交易日數": table["天數"].astype(int),
    })
    st.dataframe(shown)

    split = prices.index[is_train][-1]
    eq_net = metrics.equity_curve(bt.net)
    eq_gross = metrics.equity_curve(bt.gross)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_gross.index, y=eq_gross, name="未扣成本",
                             line=dict(color=GRAY, width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=eq_net.index, y=eq_net, name="扣成本",
                             line=dict(color=BLUE, width=2)))
    fig.add_vrect(x0=prices.index[0], x1=split, fillcolor=GRAY, opacity=0.10, line_width=0,
                  annotation_text="訓練期", annotation_position="top left")
    fig.update_layout(title="淨值曲線（起始 = 1）", hovermode="x unified", height=380,
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    fig.update_yaxes(title_text="淨值")
    st.plotly_chart(fig)

    dd = metrics.drawdown(bt.net)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, name="回撤", fill="tozeroy",
                               line=dict(color=RED, width=1.5), hovertemplate="%{y:.1%}<extra></extra>"))
    fig.add_vline(x=split, line_dash="dash", line_color=GRAY)
    fig.update_layout(title="回撤（扣成本，距離歷史高點）", height=240, showlegend=False,
                      margin=dict(l=10, r=10, t=50, b=10))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)

    trades = int((bt.turnover > 0).sum())
    st.caption(f"調整部位的天數：{trades}；累計交易成本：{pct(bt.costs.sum(), 2)}（以資金百分比計）。"
               "虛線 / 灰底左側為訓練期，參數只用這段資料決定。")

# ───────────────────────── 訊號 ─────────────────────────
with tab_sig:
    if "zscore" in ex:
        z, pos = ex["zscore"], ex["spread_pos"]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                            vertical_spacing=0.06,
                            subplot_titles=("價差的 z-score 與進出場門檻", "價差部位（+1 多、−1 空）"))
        fig.add_trace(go.Scatter(x=z.index, y=z, name="z-score", line=dict(color=BLUE, width=1.5)),
                      row=1, col=1)
        for lvl, lab in [(ex["entry"], "進場"), (-ex["entry"], "進場"), (ex["exit"], "出場"), (-ex["exit"], "出場")]:
            fig.add_hline(y=lvl, line_dash="dash" if lab == "進場" else "dot",
                          line_color=ORANGE if lab == "進場" else GRAY, row=1, col=1)
        fig.add_trace(go.Scatter(x=pos.index, y=pos, name="部位", line=dict(color=ORANGE, width=2, shape="hv")),
                      row=2, col=1)
        fig.add_vline(x=split, line_dash="dash", line_color=GRAY)
        fig.update_layout(height=520, showlegend=False, hovermode="x unified",
                          margin=dict(l=10, r=10, t=50, b=10))
        fig.update_yaxes(tickvals=[-1, 0, 1], row=2, col=1)
        st.plotly_chart(fig)
        st.caption("橘色虛線 = 進場門檻，灰色點線 = 出場門檻。z-score 的平均和標準差只用訓練期計算。")

    norm = prices / prices.iloc[0]
    fig = go.Figure()
    for col, color in zip(norm.columns, [BLUE, ORANGE]):
        fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col, line=dict(color=color, width=2)))
    fig.update_layout(title="價格走勢（起點標準化為 1）", height=320, hovermode="x unified",
                      margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(fig)

# ───────────────────────── 參數掃描 ─────────────────────────
with tab_sweep:
    sweep = getattr(strat, "SWEEP", None)
    if not sweep:
        st.info("這個策略沒有設定參數掃描。")
    else:
        labels = {p["key"]: p["label"] for p in strat.PARAMS}
        st.markdown(
            "書中 Example 3.6 的重點：**參數只能在訓練期挑，再到測試期驗證。** "
            "如果某組參數只在訓練期特別好、測試期卻崩掉，就是過擬合。"
            "比較好的選擇是兩張圖都偏藍的一整片區域，而不是訓練期最亮的那一格。")
        sweep_key = (strat_name, tickers, str(start), str(end), cost_bps,
                     tuple(sorted((k, v) for k, v in params.items() if k not in (sweep["x"], sweep["y"]))))
        if st.button("執行參數掃描", type="primary"):
            xs, ys = sweep["x_values"], sweep["y_values"]
            grid_tr = np.full((len(ys), len(xs)), np.nan)
            grid_te = np.full_like(grid_tr, np.nan)
            prog = st.progress(0.0)
            for i, yv in enumerate(ys):
                for j, xv in enumerate(xs):
                    if sweep["x"] == "exit_z" and xv >= yv:   # 出場門檻必須小於進場門檻
                        continue
                    p2 = {**params, sweep["x"]: xv, sweep["y"]: yv}
                    o2 = strat.run(prices, p2)
                    b2 = run_backtest(prices, o2["weights"], cost_bps)
                    grid_tr[i, j] = metrics.sharpe_ratio(b2.net[o2["is_train"]].iloc[1:])
                    grid_te[i, j] = metrics.sharpe_ratio(b2.net[~o2["is_train"]])
                prog.progress((i + 1) / len(ys))
            prog.empty()
            st.session_state["sweep"] = (sweep_key, xs, ys, grid_tr, grid_te)

        saved = st.session_state.get("sweep")
        if saved and saved[0] != sweep_key:
            st.info("參數或資料已經改變，請重新執行掃描。")
        elif saved:
            _, xs, ys, grid_tr, grid_te = saved
            lim = float(np.nanmax(np.abs(np.concatenate([grid_tr.ravel(), grid_te.ravel()])))) or 1.0
            c1, c2 = st.columns(2)
            for col, grid, title in [(c1, grid_tr, "訓練期 Sharpe"), (c2, grid_te, "測試期 Sharpe")]:
                invalid = np.array([[sweep["x"] == "exit_z" and xv >= yv for xv in xs] for yv in ys])
                text = np.where(invalid, "", np.where(np.isnan(grid), "無交易", np.round(grid, 2).astype(str)))
                fig = go.Figure(go.Heatmap(
                    z=grid, x=[str(v) for v in xs], y=[str(v) for v in ys],
                    colorscale=DIVERGING, zmid=0, zmin=-lim, zmax=lim,
                    text=text, texttemplate="%{text}",
                    hovertemplate=f"{labels[sweep['y']]} %{{y}}<br>{labels[sweep['x']]} %{{x}}"
                                  "<br>Sharpe %{z:.2f}<extra></extra>",
                    colorbar=dict(title="Sharpe")))
                fig.update_layout(title=title, height=420, margin=dict(l=10, r=10, t=50, b=10))
                fig.update_xaxes(title_text=labels[sweep["x"]], type="category")
                fig.update_yaxes(title_text=labels[sweep["y"]], type="category")
                col.plotly_chart(fig)
            st.caption(f"目前的交易成本 {cost_bps} bp 與其他參數保持不變。空白格代表出場門檻 ≥ 進場門檻，不合理所以略過；「無交易」代表門檻太嚴，那段期間一次都沒進場。")

# ───────────────────────── 資料 ─────────────────────────
with tab_data:
    st.write(f"共 {len(prices)} 個交易日：{prices.index[0].date()} ~ {prices.index[-1].date()}"
             f"（訓練期 {int(is_train.sum())} 天，測試期 {int((~is_train).sum())} 天）")
    export = prices.copy()
    for c in bt.weights.columns:
        export[f"權重_{c}"] = bt.weights[c]
    if "zscore" in ex:
        export["zscore"] = ex["zscore"]
    export["每日報酬_未扣成本"] = bt.gross
    export["每日報酬_扣成本"] = bt.net
    export["訓練期"] = is_train
    st.dataframe(export.tail(300))
    st.download_button("下載完整結果（CSV）", export.to_csv().encode("utf-8-sig"),
                       file_name=f"{'_'.join(tickers)}_backtest.csv", mime="text/csv")

# ───────────────────────── 說明 ─────────────────────────
with tab_help:
    st.markdown(inspect.getdoc(strat) or "")
    st.markdown("""
**幾個要記得的陷阱（第 3 章）**
- **前視偏差**：用到當下還不知道的資料。上方的自動檢查會幫你抓。
- **資料窺探 / 過擬合**：參數在訓練期調到完美，測試期就失效。用「參數掃描」分頁檢查。
- **交易成本**：把側欄的交易成本調高，看策略還撐不撐得住。
- **倖存者偏差**：這裡用 ETF 影響不大；之後做個股策略時要特別注意。

**注意**：Yahoo 現在提供的調整後價格和書中 2007 年的資料不完全相同，所以 Sharpe 不會和書上一模一樣
（書中：進場 2、出場 1 時，訓練期約 2.1、測試期約 1.5；進場 1、出場 0.5 時約 2.9 與 3.0，皆未扣成本）。
""")
