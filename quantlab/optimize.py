"""參數最佳化（網格搜尋）與前推測試，類似 MT5 的 Optimization + Forward。

每組參數只在完整資料上跑一次，再把結果切成「樣本內」（最佳化用）與「樣本外」（前推驗證）。
策略本身不會偷看未來（有前視偏差檢查），所以這樣切和分開跑是等價的，但快很多。
排序只看樣本內；樣本外只用來驗證。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import engine, report

OBJECTIVES = {
    "Sharpe": "Sharpe", "淨利": "淨利", "獲利因子": "獲利因子", "回復因子": "回復因子",
    "年化報酬": "年化報酬", "勝率": "勝率", "Calmar": "Calmar",
}
FORWARD = {"不做前推": 0.0, "後 1/2": 1 / 2, "後 1/3": 1 / 3, "後 1/4": 1 / 4}


def value_range(start, stop, step) -> list:
    """包含終點的數列（整數或小數）。"""
    if step <= 0 or stop < start:
        return [start]
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    vals = [start + k * step for k in range(n)]
    if all(float(v).is_integer() for v in (start, stop, step)):
        return [int(round(v)) for v in vals]
    return [round(float(v), 10) for v in vals]


def grid(ranges: dict) -> list[dict]:
    keys = list(ranges)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(ranges[k] for k in keys))]


def split_time(data: pd.DataFrame, forward_frac: float):
    if not forward_frac:
        return None
    k = int(len(data) * (1 - forward_frac))
    return data.index[min(max(k, 1), len(data) - 1)]


def optimize(strategy_cls, data, base_params: dict, ranges: dict, cash: float, cost_bps: float,
             objective: str, forward_frac: float, progress=None) -> tuple[pd.DataFrame, object]:
    combos = grid(ranges)
    split = split_time(data, forward_frac)
    rows = []
    for k, combo in enumerate(combos):
        if progress:
            progress((k + 1) / len(combos))
        params = {**base_params, **combo}
        row = dict(combo)
        try:
            res = engine.run(strategy_cls, data, params, cash, cost_bps)
        except engine.StrategyError as e:
            row["錯誤"] = str(e).splitlines()[0]
            rows.append(row)
            continue
        ins = report.segment(res, end=split) if split is not None else report.compute(
            res["equity"], res["trades"], data["Close"], res["exposure"], cash)
        for key in ["Sharpe", "淨利", "總報酬", "最大回撤", "獲利因子", "回復因子", "勝率", "交易次數", "年化報酬", "Calmar"]:
            row[f"樣本內·{key}"] = ins.get(key, np.nan)
        if split is not None:
            oos = report.segment(res, start=split)
            for key in ["Sharpe", "總報酬", "最大回撤", "獲利因子", "交易次數"]:
                row[f"樣本外·{key}"] = oos.get(key, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    col = f"樣本內·{OBJECTIVES[objective]}"
    if col in df:
        df = df.sort_values(col, ascending=False, na_position="last").reset_index(drop=True)
    return df, split
