"""配對交易（書中 Example 3.6：GLD vs GDX）

**步驟**
1. 在訓練期用線性迴歸（不含截距）求避險比例 h：標的 A ≈ h × 標的 B
2. 價差 spread = A − h × B
3. 用訓練期的平均和標準差把價差標準化成 z-score
4. z ≤ −進場門檻 → 買進價差（多 A、空 B）；z 回到 −出場門檻以上 → 平倉
   z ≥ +進場門檻 → 放空價差（空 A、多 B）；z 回到 +出場門檻以下 → 平倉
5. 所有參數只用訓練期決定，測試期完全沒碰過，這是避免資料窺探的核心
"""
from __future__ import annotations

import pandas as pd

from .common import band_positions, half_life, spread_weights, train_mask

NAME = "配對交易"
BOOK_REF = "第 3 章 Example 3.6；共整合與半衰期見第 7 章 Example 7.2、7.3、7.5"
MIN_ASSETS, MAX_ASSETS = 2, None   # 從清單中挑兩檔

PARAMS = [
    dict(key="asset_a", label="標的 A", kind="asset", default_index=0,
         help="價差 = A − 避險比例 × B"),
    dict(key="asset_b", label="標的 B", kind="asset", default_index=1),
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
         help="書中程式碼兩邊各放 1 單位資金；「依避險比例」則是 1 股 A 對 h 股 B。"),
]

SWEEP = dict(x="exit_z", y="entry_z",
             x_values=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
             y_values=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def select(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    """這個策略實際用到的欄位（並對齊日期）。"""
    a, b = params["asset_a"], params["asset_b"]
    return prices[[a, b]].dropna()


def run(prices: pd.DataFrame, params: dict) -> dict:
    y_col, x_col = prices.columns[:2]
    y, x = prices[y_col], prices[x_col]
    is_train = train_mask(prices.index, params["train_days"])

    # 1. 避險比例：不含截距的最小平方法，對應書中 ols(cl1, cl2)
    hedge = float((y[is_train] * x[is_train]).sum() / (x[is_train] ** 2).sum())

    # 2–3. 價差與 z-score（只用訓練期統計量）
    spread = y - hedge * x
    mu, sd = spread[is_train].mean(), spread[is_train].std(ddof=1)
    z = (spread - mu) / sd

    entry, exit_ = params["entry_z"], min(params["exit_z"], params["entry_z"])
    spread_pos = band_positions(z, entry, exit_)

    if params["weighting"].startswith("依避險"):
        units = pd.Series({y_col: 1.0, x_col: -hedge})
        weights = spread_weights(prices, units, spread_pos)
    else:
        unit = pd.DataFrame({y_col: 0.5, x_col: -0.5}, index=prices.index)
        weights = unit.mul(spread_pos, axis=0)

    return dict(
        weights=weights, is_train=is_train,
        extras=dict(kind="spread", hedge_ratio=hedge, spread=spread, zscore=z,
                    spread_pos=spread_pos, entry=entry, exit=exit_,
                    half_life=half_life(spread[is_train]),
                    kpi=("避險比例", f"{hedge:.3f}")),
    )


def valid(params: dict) -> bool:
    return params["exit_z"] < params["entry_z"] and params["asset_a"] != params["asset_b"]
