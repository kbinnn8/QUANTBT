"""績效指標：Sharpe ratio、最大回撤等（對應書中 Example 3.4、3.5）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sharpe_ratio(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化 Sharpe ratio。

    對多空中性（market-neutral）策略，書中說明不需要扣無風險利率，
    因為做空拿到的現金本身就有利息；所以這裡直接用 mean / std。
    """
    r = returns.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(np.sqrt(periods) * r.mean() / r.std(ddof=1))


def equity_curve(returns: pd.Series) -> pd.Series:
    """複利淨值曲線，從 1 開始。"""
    return (1 + returns.fillna(0)).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """每天距離歷史高點跌了多少（負數百分比）。"""
    eq = equity_curve(returns)
    return eq / eq.cummax() - 1


def max_drawdown(returns: pd.Series) -> tuple[float, int]:
    """回傳（最大回撤, 最長回撤天數）。

    最長回撤天數 = 淨值從創高到再次創高之間，最長經過幾個交易日。
    """
    if returns.dropna().empty:
        return float("nan"), 0
    dd = drawdown(returns)
    underwater = (dd < 0).astype(int).to_numpy()
    longest = cur = 0
    for u in underwater:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    return float(dd.min()), int(longest)


def summary(returns: pd.Series, periods: int = TRADING_DAYS) -> dict:
    """一組常用指標，給畫面上的表格用。"""
    r = returns.dropna()
    if r.empty:
        return {"年化報酬": np.nan, "年化波動": np.nan, "Sharpe": np.nan,
                "最大回撤": np.nan, "最長回撤天數": 0, "總報酬": np.nan, "天數": 0}
    eq = equity_curve(r)
    years = len(r) / periods
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    mdd, mdd_days = max_drawdown(r)
    return {
        "年化報酬": float(cagr),
        "年化波動": float(r.std(ddof=1) * np.sqrt(periods)),
        "Sharpe": sharpe_ratio(r, periods),
        "最大回撤": mdd,
        "最長回撤天數": mdd_days,
        "總報酬": float(eq.iloc[-1] - 1),
        "天數": int(len(r)),
    }
