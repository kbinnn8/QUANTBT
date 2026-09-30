"""技術指標（全部只用「當根與之前」的資料，不會偷看未來）。

在策略的 init() 裡使用，例如：
    self.ma = self.I(ta.sma(self.close, 20), "MA20")
輸入可以是 numpy 陣列或 pandas Series，回傳 numpy 陣列（長度和輸入相同，前面暖身期為 NaN）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _s(x) -> pd.Series:
    return x if isinstance(x, pd.Series) else pd.Series(np.asarray(x, dtype=float))


def _out(s: pd.Series) -> np.ndarray:
    return s.to_numpy(dtype=float)


def sma(x, n: int) -> np.ndarray:
    """簡單移動平均"""
    return _out(_s(x).rolling(int(n)).mean())


def ema(x, n: int) -> np.ndarray:
    """指數移動平均"""
    return _out(_s(x).ewm(span=int(n), adjust=False, min_periods=int(n)).mean())


def stdev(x, n: int) -> np.ndarray:
    """移動標準差"""
    return _out(_s(x).rolling(int(n)).std(ddof=1))


def rsi(x, n: int = 14) -> np.ndarray:
    """相對強弱指標（Wilder 平滑），0–100"""
    d = _s(x).diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out[(dn == 0) & up.notna()] = 100.0
    return _out(out)


def macd(x, fast: int = 12, slow: int = 26, signal: int = 9):
    """回傳（MACD 線, 訊號線, 柱狀體）"""
    m = _s(ema(x, fast)) - _s(ema(x, slow))
    sig = m.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return _out(m), _out(sig), _out(m - sig)


def bollinger(x, n: int = 20, k: float = 2.0):
    """回傳（中線, 上軌, 下軌）"""
    mid, sd = _s(sma(x, n)), _s(stdev(x, n))
    return _out(mid), _out(mid + k * sd), _out(mid - k * sd)


def atr(high, low, close, n: int = 14) -> np.ndarray:
    """平均真實區間（Wilder 平滑）"""
    h, l, c = _s(high), _s(low), _s(close)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return _out(tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean())


def highest(x, n: int) -> np.ndarray:
    """過去 n 根（含當根）的最高值"""
    return _out(_s(x).rolling(int(n)).max())


def lowest(x, n: int) -> np.ndarray:
    """過去 n 根（含當根）的最低值"""
    return _out(_s(x).rolling(int(n)).min())


def roc(x, n: int) -> np.ndarray:
    """n 根的變動率（報酬），例如 0.05 = 5%"""
    return _out(_s(x).pct_change(int(n)))


def zscore(x, n: int) -> np.ndarray:
    """價格相對 n 根均線的標準分數"""
    s = _s(x)
    return _out((s - s.rolling(n).mean()) / s.rolling(n).std(ddof=1))


def shift(x, n: int = 1) -> np.ndarray:
    """往後移 n 根（取前 n 根的值）"""
    return _out(_s(x).shift(int(n)))


def crossover(a, b, i: int) -> bool:
    """第 i 根時 a 由下往上穿越 b（b 可以是數字）"""
    if i < 1:
        return False
    a0, a1 = a[i - 1], a[i]
    b0, b1 = (b, b) if np.isscalar(b) else (b[i - 1], b[i])
    return bool(a0 <= b0 and a1 > b1)


def crossunder(a, b, i: int) -> bool:
    """第 i 根時 a 由上往下穿越 b（b 可以是數字）"""
    if i < 1:
        return False
    a0, a1 = a[i - 1], a[i]
    b0, b1 = (b, b) if np.isscalar(b) else (b[i - 1], b[i])
    return bool(a0 >= b0 and a1 < b1)
