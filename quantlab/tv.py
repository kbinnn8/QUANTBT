"""把 Yahoo 代號轉成 TradingView 代號。"""
from __future__ import annotations

FUTURES = {
    "GC": "COMEX:GC1!", "SI": "COMEX:SI1!", "HG": "COMEX:HG1!",
    "CL": "NYMEX:CL1!", "NG": "NYMEX:NG1!",
    "ES": "CME_MINI:ES1!", "NQ": "CME_MINI:NQ1!", "YM": "CBOT_MINI:YM1!", "RTY": "CME_MINI:RTY1!",
    "ZB": "CBOT:ZB1!", "ZN": "CBOT:ZN1!", "ZC": "CBOT:ZC1!", "ZS": "CBOT:ZS1!", "ZW": "CBOT:ZW1!",
}
INDICES = {
    "^GSPC": "SP:SPX", "^IXIC": "NASDAQ:IXIC", "^DJI": "DJ:DJI", "^TWII": "TWSE:TAIEX",
    "^N225": "TVC:NI225", "^HSI": "TVC:HSI", "^VIX": "CBOE:VIX", "^SOX": "NASDAQ:SOX",
}


def to_tradingview(ticker: str) -> str:
    t = ticker.strip().upper()
    if ":" in t:                       # 已經是 TradingView 格式，例如 OANDA:XAUUSD
        return t
    if t.isdigit():
        return f"TWSE:{t}"
    if t.endswith(".TWO"):
        return f"TPEX:{t[:-4]}"
    if t.endswith(".TW"):
        return f"TWSE:{t[:-3]}"
    if t.endswith("=X"):               # 外匯：EURUSD=X；JPY=X 代表 USD/JPY
        pair = t[:-2]
        return f"FX_IDC:{pair if len(pair) == 6 else 'USD' + pair}"
    if t.endswith("=F"):
        root = t[:-2]
        return FUTURES.get(root, f"{root}1!")
    if t.startswith("^"):
        return INDICES.get(t, t[1:])
    if t.endswith("-USD"):             # 加密貨幣：BTC-USD
        return f"COINBASE:{t.replace('-', '')}"
    return t
