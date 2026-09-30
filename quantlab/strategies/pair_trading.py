"""配對交易（書中 Example 3.6：GLD vs GDX）。

步驟：
1. 在訓練期用線性迴歸（不含截距）求避險比例 hedge ratio：GLD ≈ h × GDX
2. 價差 spread = GLD − h × GDX
3. 用訓練期的平均和標準差把價差標準化成 z-score
4. z ≤ −進場門檻 → 買進價差（多 GLD、空 GDX）；z ≥ −出場門檻 → 平倉
   z ≥ +進場門檻 → 放空價差（空 GLD、多 GDX）；z ≤ +出場門檻 → 平倉
5. 所有參數只用訓練期決定，測試期完全沒碰過 —— 這是避免資料窺探的核心
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NAME = "配對交易（Pair Trading）"
BOOK_REF = "第 3 章 Example 3.6；共整合理論見第 7 章 Example 7.2、7.3"
DEFAULT_TICKERS = ["GLD", "GDX"]
N_ASSETS = 2

# 參數規格：app 會依這份清單自動產生控制元件
PARAMS = [
    dict(key="entry_z", label="進場門檻（標準差）", kind="float",
         min=0.5, max=3.0, step=0.1, default=2.0,
         help="價差偏離平均多少個標準差時進場。書中先用 2，再試 1。"),
    dict(key="exit_z", label="出場門檻（標準差）", kind="float",
         min=0.0, max=2.0, step=0.1, default=1.0,
         help="價差回到平均附近多少個標準差內就平倉。必須小於進場門檻。"),
    dict(key="train_days", label="訓練期天數", kind="int",
         min=60, max=1500, step=21, default=252,
         help="前 N 個交易日當作訓練期，用來估計避險比例和 z-score。書中用 252 天（約一年）。"),
    dict(key="weighting", label="部位權重", kind="choice",
         options=["書中做法（兩邊等金額）", "依避險比例"], default="書中做法（兩邊等金額）",
         help="書中程式碼兩邊各放 1 單位資金；「依避險比例」則是 1 股 GLD 對 h 股 GDX。"),
]

# 參數掃描用的兩個軸
SWEEP = dict(x="exit_z", y="entry_z",
             x_values=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
             y_values=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def _state_machine(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """進場訊號 → 1，出場訊號 → 0，其他時間延續前一天的部位。"""
    s = pd.Series(np.nan, index=enter.index)
    s[exit_] = 0.0
    s[enter] = 1.0
    return s.ffill().fillna(0.0)


def run(prices: pd.DataFrame, params: dict) -> dict:
    y_col, x_col = prices.columns[:2]
    y, x = prices[y_col], prices[x_col]
    n_train = int(min(params["train_days"], len(prices) - 1))
    train = prices.index[:n_train]

    # 1. 避險比例：不含截距的最小平方法，對應書中 ols(cl1, cl2)
    hedge = float((y[train] * x[train]).sum() / (x[train] ** 2).sum())

    # 2–3. 價差與 z-score（只用訓練期統計量）
    spread = y - hedge * x
    mu, sd = spread[train].mean(), spread[train].std(ddof=1)
    z = (spread - mu) / sd

    entry, exit_ = params["entry_z"], min(params["exit_z"], params["entry_z"])
    long_leg = _state_machine(z <= -entry, z >= -exit_)
    short_leg = _state_machine(z >= entry, z <= exit_)
    spread_pos = long_leg - short_leg          # +1 多價差、−1 空價差、0 空手

    # 4. 轉成權重，總曝險（多 + 空的絕對值）= 100% 資金
    if params["weighting"].startswith("依避險"):
        dollars = pd.DataFrame({y_col: y, x_col: -hedge * x})
    else:
        dollars = pd.DataFrame({y_col: 1.0, x_col: -1.0}, index=prices.index)
    unit = dollars.div(dollars.abs().sum(axis=1), axis=0)
    weights = unit.mul(spread_pos, axis=0)

    is_train = pd.Series(prices.index.isin(train), index=prices.index)
    return dict(
        weights=weights,
        is_train=is_train,
        extras=dict(hedge_ratio=hedge, spread=spread, zscore=z,
                    spread_pos=spread_pos, entry=entry, exit=exit_,
                    spread_mean=mu, spread_std=sd),
    )
