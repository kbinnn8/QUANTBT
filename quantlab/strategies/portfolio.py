"""單標的策略組合

每一檔標的各自套用同一種交易規則，再把所有標的等權重組成一個投資組合。
適合用來回答：「這個規則是只在某一檔有效，還是在一整類標的上都有效？」

**三種規則**
- **均線交叉**：短均線在長均線之上做多，之下做空（或空手）。屬於趨勢跟隨
- **動能**：過去 N 天漲就做多、跌就做空（或空手）。書中第 7 章提到動能與均值回歸是兩大類策略
- **布林通道（均值回歸）**：價格跌破 N 日均線以下 k 個標準差時做多，回到均線附近平倉；
  漲破上緣時做空。和配對交易同樣的想法，只是用單一標的自己的均線當作「平均」

**權重**：每檔分到 1/N 的資金，訊號為 +1（多）、−1（空）、0（空手）。
沒有訊號的標的，那部分資金就閒置，所以組合的總曝險會隨時間變動。

**注意**：這裡的參數是你手動挑的，沒有在訓練期估計任何東西；
訓練期 / 測試期的切分是讓你用「參數掃描」時，只在訓練期挑參數、再到測試期驗證。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import band_positions, train_mask

NAME = "單標的策略組合"
BOOK_REF = "第 7 章均值回歸 vs 動能策略；交易成本見第 3 章 Example 3.7"
MIN_ASSETS, MAX_ASSETS = 1, None

RULES = ["均線交叉", "動能", "布林通道（均值回歸）"]

PARAMS = [
    dict(key="rule", label="交易規則", kind="choice", options=RULES, default="均線交叉"),
    dict(key="fast", label="短均線（天）", kind="int", min=3, max=100, step=1, default=20,
         show_if=("rule", "均線交叉")),
    dict(key="slow", label="長均線（天）", kind="int", min=10, max=300, step=5, default=100,
         show_if=("rule", "均線交叉")),
    dict(key="lookback", label="回顧天數", kind="int", min=5, max=252, step=1, default=60,
         show_if=("rule", "動能"), help="過去幾天的報酬決定方向。常見：20、60、120、252。"),
    dict(key="bb_window", label="均線天數", kind="int", min=5, max=120, step=1, default=20,
         show_if=("rule", "布林通道（均值回歸）")),
    dict(key="entry_z", label="進場門檻（標準差）", kind="float", min=0.5, max=3.0, step=0.1,
         default=2.0, show_if=("rule", "布林通道（均值回歸）")),
    dict(key="exit_z", label="出場門檻（標準差）", kind="float", min=0.0, max=2.0, step=0.1,
         default=0.0, show_if=("rule", "布林通道（均值回歸）")),
    dict(key="direction", label="方向", kind="choice", options=["多空都做", "只做多"],
         default="多空都做", help="台股放空限制多，可以選「只做多」看看差別。"),
    dict(key="train_days", label="訓練期天數", kind="int", min=60, max=1500, step=21,
         default=252),
]


def sweep_for(params: dict):
    rule = params["rule"]
    if rule == "均線交叉":
        return dict(x="fast", y="slow", x_values=[5, 10, 20, 30, 50],
                    y_values=[50, 100, 150, 200, 250])
    if rule == "動能":
        return dict(x="lookback", y="train_days", x_values=[10, 20, 40, 60, 120, 180, 252],
                    y_values=[params["train_days"]])
    return dict(x="exit_z", y="entry_z", x_values=[0.0, 0.25, 0.5, 0.75, 1.0],
                y_values=[1.0, 1.5, 2.0, 2.5, 3.0])


def select(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    return prices.dropna()


def signals(price: pd.Series, params: dict) -> tuple[pd.Series, dict]:
    """單一標的的訊號（+1 / 0 / −1）與畫圖用的指標線。"""
    rule = params["rule"]
    if rule == "均線交叉":
        fast = price.rolling(params["fast"]).mean()
        slow = price.rolling(params["slow"]).mean()
        sig = pd.Series(np.sign(fast - slow), index=price.index).fillna(0.0)
        lines = {f"MA{params['fast']}": fast, f"MA{params['slow']}": slow}
    elif rule == "動能":
        ret = price.pct_change(params["lookback"])
        sig = pd.Series(np.sign(ret), index=price.index).fillna(0.0)
        lines = {}
    else:
        w = params["bb_window"]
        ma, sd = price.rolling(w).mean(), price.rolling(w).std(ddof=1)
        z = (price - ma) / sd
        sig = band_positions(z, params["entry_z"], params["exit_z"])
        sig[z.isna()] = 0.0
        k = params["entry_z"]
        lines = {f"MA{w}": ma, f"+{k}σ": ma + k * sd, f"−{k}σ": ma - k * sd}
    if params["direction"] == "只做多":
        sig = sig.clip(lower=0.0)
    return sig, lines


def run(prices: pd.DataFrame, params: dict) -> dict:
    is_train = train_mask(prices.index, params["train_days"])
    sigs, lines = {}, {}
    for col in prices.columns:
        sigs[col], lines[col] = signals(prices[col], params)
    sig_df = pd.DataFrame(sigs)
    weights = sig_df / len(prices.columns)
    return dict(
        weights=weights, is_train=is_train,
        extras=dict(kind="portfolio", signals=sig_df, lines=lines,
                    kpi=("平均曝險", f"{weights.abs().sum(axis=1).mean() * 100:.0f}%")),
    )


def valid(params: dict) -> bool:
    if params["rule"] == "均線交叉":
        return params["fast"] < params["slow"]
    if params["rule"].startswith("布林"):
        return params["exit_z"] < params["entry_z"]
    return True
