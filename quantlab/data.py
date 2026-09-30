"""抓價格資料。

用 yfinance 的 auto_adjust=True 取得「已調整分割與配息」的收盤價，
這就是書中 Example 3.2 強調要用的價格；沒調整的話，配息日會被誤判成大跌。
"""
from __future__ import annotations

import pandas as pd


def load_prices(tickers: list[str], start, end) -> pd.DataFrame:
    import yfinance as yf

    tickers = [t.strip().upper() for t in tickers if t.strip()]
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=False)
    if raw is None or raw.empty:
        raise ValueError("抓不到資料，請確認代號和日期。")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # 單一標的時有些版本不回傳 MultiIndex
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.reindex(columns=tickers)
    missing = [t for t in tickers if close[t].isna().all()]
    if missing:
        raise ValueError(f"這些代號沒有資料：{', '.join(missing)}")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    # 只保留所有標的都有價格的交易日（對應書中的 intersect）
    return close.dropna(how="any")
