"""回測頁：選策略、調參數、看績效與訊號、參數掃描。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from .. import metrics
from ..backtest import lookahead_check, run_backtest
from ..strategies import REGISTRY
from . import theme as T


def pct(v, d=1):
    return "—" if v is None or pd.isna(v) else f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "—" if v is None or pd.isna(v) else f"{v:.{d}f}"


def param_key(strat, key):
    return f"{strat.NAME}:{key}"


def render_params(strat, assets: list[str]) -> dict:
    """依策略的 PARAMS 規格產生控制元件（每列 3 個，平板也好按）。"""
    params: dict = {}
    cols = st.columns(3)
    i = 0
    for p in strat.PARAMS:
        cond = p.get("show_if")
        if cond and params.get(cond[0]) != cond[1]:
            params[p["key"]] = p.get("default")
            continue
        c = cols[i % 3]
        i += 1
        k = param_key(strat, p["key"])
        if p["kind"] == "asset":
            if k in st.session_state and st.session_state[k] not in assets:
                del st.session_state[k]
            if k in st.session_state:      # 可能是「配對掃描」頁幫你選好的
                params[p["key"]] = c.selectbox(p["label"], assets, key=k, help=p.get("help"))
            else:
                idx = min(p["default_index"], len(assets) - 1)
                params[p["key"]] = c.selectbox(p["label"], assets, index=idx, key=k, help=p.get("help"))
        elif p["kind"] == "choice":
            params[p["key"]] = c.selectbox(p["label"], p["options"], index=p["options"].index(p["default"]),
                                           key=k, help=p.get("help"))
        elif p["kind"] == "int":
            params[p["key"]] = c.slider(p["label"], int(p["min"]), int(p["max"]), int(p["default"]),
                                        step=int(p["step"]), key=k, help=p.get("help"))
        else:
            params[p["key"]] = c.slider(p["label"], float(p["min"]), float(p["max"]), float(p["default"]),
                                        step=float(p["step"]), key=k, help=p.get("help"))
    return params


def label_of(strat, key):
    return next((p["label"] for p in strat.PARAMS if p["key"] == key), key)


# ───────────────────────── 圖 ─────────────────────────
def equity_chart(bt, split):
    eq_n, eq_g = metrics.equity_curve(bt.net), metrics.equity_curve(bt.gross)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_g.index, y=eq_g, name="未扣成本", line=dict(color=T.MUTED, width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=eq_n.index, y=eq_n, name="扣成本", line=dict(color=T.BLUE, width=2)))
    fig.add_vrect(x0=eq_n.index[0], x1=split, fillcolor="#ffffff", opacity=0.04, line_width=0,
                  annotation_text="訓練期", annotation_position="top left",
                  annotation_font=dict(color=T.MUTED, size=11))
    fig.update_yaxes(title_text="淨值")
    return T.style(fig, 380, "淨值曲線（起始 = 1）")


def drawdown_chart(bt, split):
    dd = metrics.drawdown(bt.net)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, name="回撤", fill="tozeroy",
                               line=dict(color=T.DOWN, width=1.2), fillcolor="rgba(230,103,103,0.18)",
                               hovertemplate="%{y:.1%}<extra></extra>"))
    fig.add_vline(x=split, line_dash="dash", line_color=T.AXIS)
    fig.update_yaxes(tickformat=".0%")
    return T.style(fig, 220, "回撤（扣成本）", legend=False)


def spread_chart(ex, split):
    z, pos = ex["zscore"], ex["spread_pos"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=z.index, y=z, name="z-score", line=dict(color=T.BLUE, width=1.4)), row=1, col=1)
    for lvl, is_entry in [(ex["entry"], True), (-ex["entry"], True), (ex["exit"], False), (-ex["exit"], False)]:
        fig.add_hline(y=lvl, line_dash="dash" if is_entry else "dot",
                      line_color=T.ORANGE if is_entry else T.MUTED, row=1, col=1)
    fig.add_trace(go.Scatter(x=pos.index, y=pos, name="價差部位", line=dict(color=T.ORANGE, width=1.8, shape="hv")),
                  row=2, col=1)
    fig.add_vline(x=split, line_dash="dash", line_color=T.AXIS)
    fig.update_yaxes(tickvals=[-1, 0, 1], row=2, col=1)
    return T.style(fig, 500, "價差 z-score（橘虛線 = 進場，灰點線 = 出場）與部位", legend=False)


def asset_signal_chart(price, lines, sig, name):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=price.index, y=price, name=name, line=dict(color=T.INK2, width=1.4)), row=1, col=1)
    for (lab, s), color in zip(lines.items(), T.SERIES):
        dash = "dot" if "σ" in lab else None
        fig.add_trace(go.Scatter(x=s.index, y=s, name=lab, line=dict(color=color, width=1.4, dash=dash)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sig.index, y=sig, name="部位", line=dict(color=T.ORANGE, width=1.8, shape="hv"),
                             showlegend=False), row=2, col=1)
    fig.update_yaxes(tickvals=[-1, 0, 1], row=2, col=1)
    return T.style(fig, 500, f"{name}：價格、指標與部位")


def heatmap(grid, xs, ys, title, xlabel, ylabel, invalid):
    text = np.where(invalid, "", np.where(np.isnan(grid), "無交易", np.round(grid, 2).astype(str)))
    finite = grid[np.isfinite(grid)]
    lim = float(np.abs(finite).max()) if finite.size else 1.0
    fig = go.Figure(go.Heatmap(
        z=grid, x=[str(v) for v in xs], y=[str(v) for v in ys], colorscale=T.DIVERGING,
        zmid=0, zmin=-lim, zmax=lim, text=text, texttemplate="%{text}", xgap=2, ygap=2,
        textfont=dict(family="JetBrains Mono, monospace", color=T.INK),
        hovertemplate=f"{ylabel} %{{y}}<br>{xlabel} %{{x}}<br>Sharpe %{{z:.2f}}<extra></extra>",
        colorbar=dict(title="Sharpe", tickfont=dict(color=T.MUTED))))
    fig.update_xaxes(title_text=xlabel, type="category", showgrid=False)
    fig.update_yaxes(title_text=ylabel, type="category", showgrid=False)
    fig = T.style(fig, 400, title, legend=False)
    fig.update_layout(hovermode="closest")
    return fig


# ───────────────────────── 頁面 ─────────────────────────
def render(all_prices: pd.DataFrame, cost_bps: float) -> None:
    names = list(REGISTRY)
    if st.session_state.get("strategy") not in names:
        st.session_state["strategy"] = names[0]

    with st.container(border=True):
        strat_name = st.selectbox("策略", names, key="strategy")
        strat = REGISTRY[strat_name]
        assets = list(all_prices.columns)
        if len(assets) < strat.MIN_ASSETS:
            st.warning(f"「{strat_name}」至少需要 {strat.MIN_ASSETS} 檔標的，請在左側清單加入。")
            st.stop()
        params = render_params(strat, assets)

    st.markdown(f'<div class="page-title">{T.esc(strat_name)}</div>'
                f'<div class="page-ref">📖 {T.esc(strat.BOOK_REF)}</div>', unsafe_allow_html=True)

    if strat.MAX_ASSETS and len(assets) > strat.MAX_ASSETS:
        st.info(f"這個策略最多用 {strat.MAX_ASSETS} 檔，已取清單中的前 {strat.MAX_ASSETS} 檔。")
    if "asset_a" in params and params["asset_a"] == params["asset_b"]:
        st.warning("標的 A 和 B 不能相同。")
        st.stop()

    prices = strat.select(all_prices, params)
    if len(prices) <= params["train_days"] + 20:
        st.warning(f"這些標的共同有價格的交易日只有 {len(prices)} 天，扣掉訓練期後測試期太短。"
                   "請拉長期間、縮短訓練期，或移除上市較晚的標的。")
        st.stop()

    try:
        out = strat.run(prices, params)
    except Exception as e:
        st.error(f"策略計算失敗：{e}")
        st.stop()
    bt = run_backtest(prices, out["weights"], cost_bps)
    is_train, ex = out["is_train"], out["extras"]
    split = prices.index[is_train.to_numpy()][-1]
    tr_net, te_net = bt.net[is_train].iloc[1:], bt.net[~is_train]
    s_tr, s_te = metrics.summary(tr_net), metrics.summary(te_net)

    d = s_te["Sharpe"] - s_tr["Sharpe"]
    st.markdown(T.chips([f"{c}" for c in prices.columns] +
                        [f"{prices.index[0].date()} → {prices.index[-1].date()}",
                         f"訓練 {int(is_train.sum())} 天 / 測試 {int((~is_train).sum())} 天",
                         f"成本 {cost_bps:g} bp"]), unsafe_allow_html=True)
    st.markdown(T.kpi_cards([
        dict(label="測試期 Sharpe", value=num(s_te["Sharpe"]),
             delta=f"{d:+.2f} vs 訓練期" if np.isfinite(d) else None,
             trend=("up" if d >= 0 else "down") if np.isfinite(d) else None),
        dict(label="訓練期 Sharpe", value=num(s_tr["Sharpe"]),
             delta="訓練期沒有交易" if pd.isna(s_tr["Sharpe"]) else None),
        dict(label="測試期年化報酬", value=pct(s_te["年化報酬"])),
        dict(label="測試期最大回撤", value=pct(s_te["最大回撤"]),
             delta=f"最長 {s_te['最長回撤天數']} 天"),
        dict(label=ex["kpi"][0], value=ex["kpi"][1]),
    ]), unsafe_allow_html=True)

    ok, bad = lookahead_check(strat.run, prices, params)
    st.markdown(T.status(ok, "前視偏差檢查通過：砍掉最後 20 天資料重跑，重疊期間的部位完全一致" if ok
                         else f"前視偏差檢查失敗：有 {bad} 天的部位在砍掉未來資料後改變了"),
                unsafe_allow_html=True)

    t_perf, t_sig, t_sweep, t_data, t_help = st.tabs(["績效", "訊號", "參數掃描", "資料", "策略說明"])

    with t_perf:
        rows = {"訓練期 · 未扣成本": metrics.summary(bt.gross[is_train].iloc[1:]), "訓練期 · 扣成本": s_tr,
                "測試期 · 未扣成本": metrics.summary(bt.gross[~is_train]), "測試期 · 扣成本": s_te}
        tb = pd.DataFrame(rows).T
        st.dataframe(pd.DataFrame({
            "Sharpe": tb["Sharpe"].map(num), "年化報酬": tb["年化報酬"].map(pct),
            "年化波動": tb["年化波動"].map(pct), "最大回撤": tb["最大回撤"].map(pct),
            "最長回撤天數": tb["最長回撤天數"].astype(int), "總報酬": tb["總報酬"].map(pct),
            "交易日數": tb["天數"].astype(int)}))
        st.plotly_chart(equity_chart(bt, split), theme=None)
        st.plotly_chart(drawdown_chart(bt, split), theme=None)
        st.markdown(f'<div class="hint">調整部位的天數：{int((bt.turnover > 0).sum())}；'
                    f'累計交易成本：{pct(bt.costs.sum(), 2)}（以資金百分比計）。灰底 / 虛線左側為訓練期。</div>',
                    unsafe_allow_html=True)

    with t_sig:
        if ex["kind"] == "spread":
            hl = ex.get("half_life", np.nan)
            info = [f"訓練期半衰期：{num(hl, 1)} 天" if np.isfinite(hl) else "訓練期半衰期：無（價差沒有均值回歸）"]
            if "units" in ex:
                info.append("每單位價差 = " + "、".join(f"{u:+.3f} 股 {c}" for c, u in ex["units"].items()))
            st.markdown(T.chips(info), unsafe_allow_html=True)
            st.plotly_chart(spread_chart(ex, split), theme=None)
        else:
            per = []
            ret = prices.pct_change()
            for c in prices.columns:
                contrib = bt.weights[c].shift(1) * ret[c]
                per.append({"標的": c, "訓練期 Sharpe": metrics.sharpe_ratio(contrib[is_train].iloc[1:]),
                            "測試期 Sharpe": metrics.sharpe_ratio(contrib[~is_train]),
                            "持有天數比例": float((ex["signals"][c] != 0).mean())})
            per_df = pd.DataFrame(per)
            per_df["訓練期 Sharpe"] = per_df["訓練期 Sharpe"].map(num)
            per_df["測試期 Sharpe"] = per_df["測試期 Sharpe"].map(num)
            per_df["持有天數比例"] = per_df["持有天數比例"].map(lambda v: pct(v, 0))
            st.dataframe(per_df, hide_index=True)
            pick = st.selectbox("查看標的", list(prices.columns), key="sig_pick")
            st.plotly_chart(asset_signal_chart(prices[pick], ex["lines"][pick], ex["signals"][pick], pick),
                            theme=None)
        norm = prices / prices.iloc[0]
        fig = go.Figure()
        for (c, s), color in zip(norm.items(), T.SERIES):
            fig.add_trace(go.Scatter(x=s.index, y=s, name=c, line=dict(color=color, width=1.6)))
        if len(norm.columns) > len(T.SERIES):
            st.caption(f"只畫出前 {len(T.SERIES)} 檔的走勢。")
        st.plotly_chart(T.style(fig, 320, "價格走勢（起點 = 1）"), theme=None)

    with t_sweep:
        sweep = strat.sweep_for(params) if hasattr(strat, "sweep_for") else getattr(strat, "SWEEP", None)
        if not sweep:
            st.info("這個策略沒有設定參數掃描。")
        else:
            xl, yl = label_of(strat, sweep["x"]), label_of(strat, sweep["y"])
            st.markdown('<div class="hint">書中 Example 3.6 的重點：<b>參數只能在訓練期挑，再到測試期驗證。</b>'
                        '訓練期特別好、測試期卻崩掉，就是過擬合。好的選擇是兩張圖都偏藍的一整片區域，'
                        '而不是訓練期最亮的那一格。</div>', unsafe_allow_html=True)
            key = (strat_name, tuple(prices.columns), str(prices.index[0]), str(prices.index[-1]), cost_bps,
                   tuple(sorted((k, str(v)) for k, v in params.items() if k not in (sweep["x"], sweep["y"]))))
            if st.button("執行參數掃描", type="primary"):
                xs, ys = sweep["x_values"], sweep["y_values"]
                g_tr = np.full((len(ys), len(xs)), np.nan)
                g_te = g_tr.copy()
                bad_mask = np.zeros_like(g_tr, dtype=bool)
                prog = st.progress(0.0)
                for i, yv in enumerate(ys):
                    for j, xv in enumerate(xs):
                        p2 = {**params, sweep["x"]: xv, sweep["y"]: yv}
                        if hasattr(strat, "valid") and not strat.valid(p2):
                            bad_mask[i, j] = True
                            continue
                        pr2 = strat.select(all_prices, p2)
                        o2 = strat.run(pr2, p2)
                        b2 = run_backtest(pr2, o2["weights"], cost_bps)
                        g_tr[i, j] = metrics.sharpe_ratio(b2.net[o2["is_train"]].iloc[1:])
                        g_te[i, j] = metrics.sharpe_ratio(b2.net[~o2["is_train"]])
                    prog.progress((i + 1) / len(ys))
                prog.empty()
                st.session_state["sweep"] = (key, xs, ys, g_tr, g_te, bad_mask)
            saved = st.session_state.get("sweep")
            if saved and saved[0] != key:
                st.info("參數或資料已經改變，請重新執行掃描。")
            elif saved:
                _, xs, ys, g_tr, g_te, bad_mask = saved
                c1, c2 = st.columns(2)
                c1.plotly_chart(heatmap(g_tr, xs, ys, "訓練期 Sharpe", xl, yl, bad_mask), theme=None)
                c2.plotly_chart(heatmap(g_te, xs, ys, "測試期 Sharpe", xl, yl, bad_mask), theme=None)
                st.markdown('<div class="hint">空白格是不合理的參數組合（例如出場門檻 ≥ 進場門檻、短均線 ≥ 長均線）；'
                            '「無交易」代表那段期間一次都沒進場。</div>', unsafe_allow_html=True)

    with t_data:
        export = prices.copy()
        for c in bt.weights.columns:
            export[f"權重_{c}"] = bt.weights[c]
        if "zscore" in ex:
            export["zscore"] = ex["zscore"]
        export["報酬_未扣成本"], export["報酬_扣成本"], export["訓練期"] = bt.gross, bt.net, is_train
        st.dataframe(export.tail(300))
        st.download_button("下載完整結果（CSV）", export.to_csv().encode("utf-8-sig"),
                           file_name=f"{strat_name}_backtest.csv", mime="text/csv")

    with t_help:
        st.markdown(strat.__doc__ or "")
