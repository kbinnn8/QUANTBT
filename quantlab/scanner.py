"""配對掃描：把清單中所有兩兩組合都跑一次，找出最有潛力的配對。

排序只用「訓練期」的統計量（共整合 p 值），測試期 Sharpe 只拿來驗證，
避免「看了測試期結果才挑配對」的資料窺探。
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from . import metrics
from .backtest import run_backtest
from .strategies import pair_trading
from .strategies.common import half_life, train_mask


def coint_pvalue(y: pd.Series, x: pd.Series) -> float:
    """Engle-Granger 共整合檢定的 p 值（書中 Example 7.2 的 CADF 檢定）。"""
    from statsmodels.tsa.stattools import coint

    try:
        return float(coint(y.to_numpy(), x.to_numpy())[1])
    except Exception:
        return float("nan")


def scan_pairs(prices: pd.DataFrame, params: dict, cost_bps: float,
               min_days: int = 120, progress=None) -> pd.DataFrame:
    cols = list(prices.columns)
    pairs = list(combinations(cols, 2))
    rows = []
    for k, (a, b) in enumerate(pairs):
        if progress:
            progress((k + 1) / len(pairs), f"{a} / {b}")
        df = prices[[a, b]].dropna()
        is_train = train_mask(df.index, params["train_days"])
        if len(df) < params["train_days"] + min_days:
            rows.append(dict(A=a, B=b, 交易日數=len(df), 備註="資料太短"))
            continue
        tr = df[is_train]
        # 兩個方向都試：誰當 A 會影響避險比例，挑訓練期 p 值較低的方向
        p_ab, p_ba = coint_pvalue(tr[a], tr[b]), coint_pvalue(tr[b], tr[a])
        if not np.isnan(p_ba) and (np.isnan(p_ab) or p_ba < p_ab):
            a, b, p = b, a, p_ba
            df = df[[a, b]]
        else:
            p = p_ab
        p2 = {**params, "asset_a": a, "asset_b": b}
        out = pair_trading.run(df, p2)
        bt = run_backtest(df, out["weights"], cost_bps)
        m = out["is_train"]
        rows.append({
            "A": a, "B": b,
            "共整合 p 值": p,
            "半衰期（天）": out["extras"]["half_life"],
            "報酬相關係數": float(tr.pct_change().corr().iloc[0, 1]),
            "訓練期 Sharpe": metrics.sharpe_ratio(bt.net[m].iloc[1:]),
            "測試期 Sharpe": metrics.sharpe_ratio(bt.net[~m]),
            "測試期最大回撤": metrics.max_drawdown(bt.net[~m])[0],
            "交易日數": len(df),
            "備註": "",
        })
    res = pd.DataFrame(rows)
    if "共整合 p 值" in res:
        res = res.sort_values("共整合 p 值", na_position="last").reset_index(drop=True)
    return res
