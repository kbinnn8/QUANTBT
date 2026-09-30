"""通用的回測引擎。

約定：策略輸出的是「權重」DataFrame，欄位是各個標的，值是佔資金的比例
（正數做多、負數做空）。第 t 天收盤決定的權重，從第 t+1 天開始賺賠，
所以計算損益時一定要先 shift(1)，這是避免前視偏差（look-ahead bias）的關鍵。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    gross: pd.Series          # 未扣成本的每日報酬
    net: pd.Series            # 扣掉交易成本的每日報酬
    costs: pd.Series          # 每日交易成本
    weights: pd.DataFrame     # 每日持倉權重
    turnover: pd.Series       # 每日換手（權重變化的絕對值總和）
    extras: dict = field(default_factory=dict)  # 策略額外輸出（z-score、避險比例…）


def run_backtest(prices: pd.DataFrame, weights: pd.DataFrame,
                 cost_bps: float = 0.0) -> BacktestResult:
    """以權重計算每日報酬。

    cost_bps：單邊交易成本（基點，1 bp = 0.01%）。書中 Example 3.7 用 5 bp。
    成本 = 單邊成本 × 當天權重變化的絕對值總和（買或賣各算一次）。
    """
    weights = weights.reindex(prices.index).fillna(0.0).astype(float)
    asset_ret = prices.pct_change()
    held = weights.shift(1)                      # 昨天收盤的部位，賺今天的漲跌
    gross = (held * asset_ret).sum(axis=1, min_count=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    costs = turnover * cost_bps / 1e4
    net = gross - costs
    return BacktestResult(gross=gross, net=net, costs=costs,
                          weights=weights, turnover=turnover)


def lookahead_check(strategy_fn, prices: pd.DataFrame, params: dict,
                    cut: int = 20) -> tuple[bool, int]:
    """書中介紹的前視偏差檢查法：

    把最後 `cut` 天的資料砍掉再跑一次策略。如果策略沒有偷看未來，
    兩次在重疊期間的部位應該完全一樣。回傳（是否通過, 不一致的天數）。
    """
    full = strategy_fn(prices, params)["weights"]
    part = strategy_fn(prices.iloc[:-cut], params)["weights"]
    a = full.loc[part.index].fillna(0).to_numpy()
    b = part.fillna(0).to_numpy()
    mismatch = int((~np.isclose(a, b)).any(axis=1).sum())
    return mismatch == 0, mismatch
