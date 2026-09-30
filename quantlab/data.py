"""抓價格資料。

用 yfinance 的 auto_adjust=True 取得「已調整分割與配息」的收盤價，
這就是書中 Example 3.2 強調要用的價格；沒調整的話，配息日會被誤判成大跌。

台股可以直接輸入數字代號（例如 2330），會自動先試上市（.TW），
找不到再試上櫃（.TWO）。
"""
from __future__ import annotations

import pandas as pd


def _download_close(tickers: list[str], start, end) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=tickers)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:  # 單一標的時有些版本不回傳 MultiIndex
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    return close.reindex(columns=tickers)


def _has_data(close: pd.DataFrame, t: str) -> bool:
    return t in close.columns and not close[t].isna().all()


def normalize_ticker(t: str) -> str:
    t = t.strip().upper()
    return f"{t}.TW" if t.isdigit() else t


def load_prices(tickers: list[str], start, end) -> tuple[pd.DataFrame, list[str]]:
    """回傳（收盤價 DataFrame, 抓不到資料的代號清單）。

    收盤價「沒有」對齊日期：每一檔保留自己的完整歷史，
    需要多檔同時有價格的地方再自行 dropna（對應書中的 intersect）。
    """
    raw_inputs = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    resolved = [normalize_ticker(t) for t in raw_inputs]
    close = _download_close(resolved, start, end) if resolved else pd.DataFrame()

    # 純數字代號在上市找不到時，改試上櫃
    for i, (orig, t) in enumerate(zip(raw_inputs, resolved)):
        if orig.isdigit() and not _has_data(close, t):
            alt = f"{orig}.TWO"
            alt_close = _download_close([alt], start, end)
            if _has_data(alt_close, alt):
                close = close.drop(columns=[t]).join(alt_close, how="outer")
                resolved[i] = alt

    missing = [t for t in resolved if not _has_data(close, t)]
    ok = [t for t in resolved if t not in missing]
    close = close[ok] if ok else pd.DataFrame()
    if not close.empty:
        close.index = pd.to_datetime(close.index).tz_localize(None)
        close = close.dropna(how="all").sort_index()
    return close, missing
