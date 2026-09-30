"""一籃子均值回歸（Johansen 共整合）

配對交易的多檔版本：不只兩檔，而是找出一組權重，讓 3 檔以上的組合價值會均值回歸。

**步驟**
1. 在訓練期對所有標的做 **Johansen 共整合檢定**，取第一個特徵向量當作每檔的股數比例
2. 組合價值 = Σ 股數ᵢ × 價格ᵢ，就是這一籃子的「價差」
3. 用訓練期的平均和標準差把組合價值標準化成 z-score
4. 進出場規則和配對交易一樣：偏離太多就反向進場，回到平均附近就平倉

**怎麼看檢定結果**：「共整合關係數」是 Johansen 在 95% 信心水準下找到的關係個數，
0 代表這組標的在訓練期沒有顯著的共整合，就算回測好看也要特別小心。
半衰期（書中 Example 7.5）代表價差偏離後大約幾天回到一半，太長的話資金會被卡很久。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import band_positions, half_life, spread_weights, train_mask

NAME = "一籃子均值回歸"
BOOK_REF = "第 7 章共整合與半衰期（Example 7.2、7.5）的多檔延伸"
MIN_ASSETS, MAX_ASSETS = 2, 12   # Johansen 檢定的臨界值表最多支援 12 檔

PARAMS = [
    dict(key="entry_z", label="進場門檻（標準差）", kind="float",
         min=0.5, max=3.0, step=0.1, default=1.5),
    dict(key="exit_z", label="出場門檻（標準差）", kind="float",
         min=0.0, max=2.0, step=0.1, default=0.5),
    dict(key="train_days", label="訓練期天數", kind="int",
         min=60, max=1500, step=21, default=252,
         help="Johansen 權重和 z-score 只用這段資料估計。"),
]

SWEEP = dict(x="exit_z", y="entry_z",
             x_values=[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
             y_values=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])


def select(prices: pd.DataFrame, params: dict) -> pd.DataFrame:
    return prices.iloc[:, :MAX_ASSETS].dropna()


def johansen(train_prices: pd.DataFrame) -> tuple[np.ndarray, int]:
    """回傳（第一個特徵向量, 95% 信心水準下的共整合關係數）。"""
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    res = coint_johansen(train_prices.to_numpy(), det_order=0, k_ar_diff=1)
    n_rel = int((res.lr1 > res.cvt[:, 1]).cumprod().sum())   # 依序比較 trace 統計量
    # numpy 的特徵值分解有時會回傳「虛部為 0 的複數」，這裡轉回實數
    return np.real(np.asarray(res.evec[:, 0])).astype(float), n_rel


def run(prices: pd.DataFrame, params: dict) -> dict:
    is_train = train_mask(prices.index, params["train_days"])
    vec, n_rel = johansen(prices[is_train])
    units = pd.Series(vec / np.abs(vec).max(), index=prices.columns)  # 最大的一檔 = ±1 股

    spread = prices.mul(units, axis=1).sum(axis=1)
    mu, sd = spread[is_train].mean(), spread[is_train].std(ddof=1)
    z = (spread - mu) / sd
    entry, exit_ = params["entry_z"], min(params["exit_z"], params["entry_z"])
    spread_pos = band_positions(z, entry, exit_)
    weights = spread_weights(prices, units, spread_pos)

    return dict(
        weights=weights, is_train=is_train,
        extras=dict(kind="spread", units=units, n_relations=n_rel, spread=spread, zscore=z,
                    spread_pos=spread_pos, entry=entry, exit=exit_,
                    half_life=half_life(spread[is_train]),
                    kpi=("共整合關係數", f"{n_rel} / {len(prices.columns)}")),
    )


def valid(params: dict) -> bool:
    return params["exit_z"] < params["entry_z"]
