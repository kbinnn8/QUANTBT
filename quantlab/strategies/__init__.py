"""策略清單。

新增策略的方法：
1. 在這個資料夾新增一個 .py 檔，照現有策略的格式寫：
   NAME、BOOK_REF、MIN_ASSETS、MAX_ASSETS、PARAMS、select(prices, params)、run(prices, params)
   選用：SWEEP 或 sweep_for(params)、valid(params)
2. run 要回傳 dict(weights=..., is_train=..., extras=...)
3. 在下面的 REGISTRY 加一行

PARAMS 支援的 kind：float、int、choice、asset（從標的清單中挑一檔）；
加上 show_if=(參數, 值) 可以只在特定條件下顯示。
"""
from . import basket_mr, pair_trading, portfolio

REGISTRY = {
    pair_trading.NAME: pair_trading,
    portfolio.NAME: portfolio,
    basket_mr.NAME: basket_mr,
}
