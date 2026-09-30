"""配對掃描頁：清單中所有兩兩組合跑一次配對交易，依訓練期共整合程度排序。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..scanner import scan_pairs
from ..strategies import pair_trading
from . import theme as T
from .backtest_page import num, pct


def use_pair(a: str, b: str) -> None:
    """按鈕回呼：把選好的配對帶到回測頁。"""
    st.session_state["strategy"] = pair_trading.NAME
    st.session_state[f"{pair_trading.NAME}:asset_a"] = a
    st.session_state[f"{pair_trading.NAME}:asset_b"] = b
    st.session_state["page"] = "回測"


def render(all_prices: pd.DataFrame, cost_bps: float) -> None:
    st.markdown('<div class="page-title">配對掃描</div>'
                '<div class="page-ref">📖 第 7 章 Example 7.2（共整合檢定）、7.5（半衰期）；'
                '進出場規則同第 3 章 Example 3.6</div>', unsafe_allow_html=True)
    cols = list(all_prices.columns)
    n_pairs = len(cols) * (len(cols) - 1) // 2
    if len(cols) < 2:
        st.warning("至少需要 2 檔標的，請在左側清單加入。")
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        entry = c1.slider("進場門檻（標準差）", 0.5, 3.0, 2.0, 0.1, key="scan_entry")
        exit_ = c2.slider("出場門檻（標準差）", 0.0, 2.0, 1.0, 0.1, key="scan_exit")
        train_days = c3.slider("訓練期天數", 60, 1500, 252, 21, key="scan_train")
        st.markdown(f'<div class="hint">清單中有 {len(cols)} 檔，共 {n_pairs} 組配對。'
                    '排序只用<b>訓練期</b>的共整合 p 值；測試期 Sharpe 只用來驗證，'
                    '不要回頭用它挑配對，否則就是資料窺探。</div>', unsafe_allow_html=True)
        run = st.button("開始掃描", type="primary", disabled=n_pairs > 190)
        if n_pairs > 190:
            st.caption("配對太多（上限 190 組，約 20 檔），請縮減清單。")

    params = {"entry_z": entry, "exit_z": min(exit_, entry), "train_days": train_days,
              "weighting": "書中做法（兩邊等金額）"}
    key = (tuple(cols), str(all_prices.index[0]), str(all_prices.index[-1]), cost_bps,
           entry, exit_, train_days)
    if run:
        bar = st.progress(0.0)
        res = scan_pairs(all_prices, params, cost_bps,
                         progress=lambda f, txt: bar.progress(f, text=f"掃描中：{txt}"))
        bar.empty()
        st.session_state["scan"] = (key, res)

    saved = st.session_state.get("scan")
    if not saved:
        return
    if saved[0] != key:
        st.info("清單或參數已經改變，請重新掃描。")
        return
    res = saved[1]
    good = res[res["共整合 p 值"].notna()] if "共整合 p 值" in res else res.iloc[0:0]
    if good.empty:
        st.warning("沒有任何配對有足夠的資料。")
        st.dataframe(res, hide_index=True)
        return

    n_sig = int((good["共整合 p 值"] < 0.05).sum())
    best = good.iloc[0]
    st.markdown(T.kpi_cards([
        dict(label="掃描配對數", value=f"{len(res)}"),
        dict(label="共整合顯著（p < 0.05）", value=f"{n_sig}"),
        dict(label="訓練期最佳配對", value=f"{best['A']} / {best['B']}",
             delta=f"p = {best['共整合 p 值']:.3f}"),
        dict(label="最佳配對測試期 Sharpe", value=num(best["測試期 Sharpe"])),
    ]), unsafe_allow_html=True)

    show = res.copy()
    show.insert(0, "配對", show["A"] + " / " + show["B"])
    for c in ["共整合 p 值"]:
        show[c] = show[c].map(lambda v: num(v, 3))
    show["半衰期（天）"] = show["半衰期（天）"].map(lambda v: num(v, 1))
    show["報酬相關係數"] = show["報酬相關係數"].map(num)
    show["訓練期 Sharpe"] = show["訓練期 Sharpe"].map(num)
    show["測試期 Sharpe"] = show["測試期 Sharpe"].map(num)
    show["測試期最大回撤"] = show["測試期最大回撤"].map(pct)
    st.dataframe(show.drop(columns=["A", "B"]), hide_index=True)

    fig = go.Figure()
    sig = good["共整合 p 值"] < 0.05
    for mask, name, color in [(sig, "共整合顯著（p < 0.05）", T.BLUE), (~sig, "不顯著", T.MUTED)]:
        sub = good[mask]
        fig.add_trace(go.Scatter(
            x=sub["訓練期 Sharpe"], y=sub["測試期 Sharpe"], mode="markers", name=name,
            marker=dict(size=11, color=color, line=dict(color=T.SURFACE, width=2)),
            text=sub["A"] + " / " + sub["B"],
            hovertemplate="%{text}<br>訓練期 %{x:.2f}<br>測試期 %{y:.2f}<extra></extra>"))
    vals = pd.concat([good["訓練期 Sharpe"], good["測試期 Sharpe"]]).replace([np.inf, -np.inf], np.nan).dropna()
    if not vals.empty:
        lo, hi = float(vals.min()) - 0.3, float(vals.max()) + 0.3
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="訓練 = 測試",
                                 line=dict(color=T.AXIS, dash="dash", width=1), hoverinfo="skip"))
    fig.update_xaxes(title_text="訓練期 Sharpe")
    fig.update_yaxes(title_text="測試期 Sharpe")
    fig = T.style(fig, 420, "訓練期表現能預測測試期嗎？（越靠近虛線越穩定）")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, theme=None)

    options = [f"{a} / {b}" for a, b in zip(good["A"], good["B"])]
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("挑一組帶到回測頁細看", options, key="scan_pick")
    a, b = pick.split(" / ")
    c2.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c2.button("用這組做配對交易 →", on_click=use_pair, args=(a, b), type="primary")
