"""策略清單。

新增策略的方法：
1. 在這個資料夾新增一個 .py 檔，照 pair_trading.py 的格式寫：
   NAME、BOOK_REF、DEFAULT_TICKERS、N_ASSETS、PARAMS、SWEEP（可省略）、run(prices, params)
2. run 要回傳 dict(weights=..., is_train=..., extras=...)
3. 在下面的 REGISTRY 加一行
"""
from . import pair_trading

REGISTRY = {
    pair_trading.NAME: pair_trading,
}
