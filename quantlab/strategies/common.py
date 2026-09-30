"""策略共用的工具：均值回歸的價差進出場邏輯。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def train_mask(index: pd.Index, train_days: int) -> pd.Series:
    """前 train_days 個交易日為訓練期（至少留 1 天給測試期）。"""
    n = int(min(max(train_days, 2), len(index) - 1))
    return pd.Series(np.arange(len(index)) < n, index=index)


def state_machine(enter: pd.Series, exit_: pd.Series) -> pd.Series:
    """進場訊號 → 1，出場訊號 → 0，其他時間延續前一天的部位。"""
    s = pd.Series(np.nan, index=enter.index)
    s[exit_.fillna(False).astype(bool)] = 0.0
    s[enter.fillna(False).astype(bool)] = 1.0
    return s.ffill().fillna(0.0)


def band_positions(z: pd.Series, entry: float, exit_: float) -> pd.Series:
    """z ≤ −entry 做多、z ≥ +entry 做空；回到 ±exit 以內平倉。回傳 +1 / 0 / −1。"""
    exit_ = min(exit_, entry)
    long_leg = state_machine(z <= -entry, z >= -exit_)
    short_leg = state_machine(z >= entry, z <= exit_)
    return long_leg - short_leg


def half_life(spread: pd.Series) -> float:
    """均值回歸的半衰期（書中 Example 7.5）：Δspread 對前一天 spread 迴歸，
    半衰期 = −ln(2) / 斜率。斜率 ≥ 0 代表沒有均值回歸，回傳 NaN。"""
    s = spread.dropna()
    lag = s.shift(1).iloc[1:]
    delta = s.diff().iloc[1:]
    x = lag - lag.mean()
    if len(x) < 10 or (x ** 2).sum() == 0:
        return float("nan")
    slope = float((x * (delta - delta.mean())).sum() / (x ** 2).sum())
    return float(-np.log(2) / slope) if slope < 0 else float("nan")


def spread_weights(prices: pd.DataFrame, units: pd.Series, spread_pos: pd.Series) -> pd.DataFrame:
    """把「每 1 單位價差 = units 股」轉成資金權重，總曝險（多空絕對值加總）= 100%。"""
    dollars = prices[units.index].mul(units, axis=1)
    unit = dollars.div(dollars.abs().sum(axis=1), axis=0)
    return unit.mul(spread_pos, axis=0)
