# Quant Lab

依照 Ernest Chan《Quantitative Trading》（第二版）做研究的回測工具。
研究邏輯用 Python 寫，介面用 Streamlit，部署在 Streamlit Community Cloud，平板或手機都能隨時打開。

## 三個頁面
- **圖表**：嵌入 TradingView 進階圖表，可以畫趨勢線、加指標（均線、布林、RSI、MACD…）。
  也能直接輸入 TradingView 代號看 CFD，例如 `OANDA:XAUUSD`
- **回測**：選策略、調參數，看訓練期 / 測試期績效（扣成本前後）、淨值、回撤、訊號、參數掃描熱圖，
  並自動做前視偏差檢查
- **配對掃描**：清單中所有兩兩組合跑一次配對交易，依訓練期共整合 p 值排序，
  再用測試期 Sharpe 驗證；可以一鍵把配對帶到回測頁

## 策略
| 策略 | 標的數 | 書中對應 |
|---|---|---|
| 配對交易 | 從清單挑 2 檔 | 第 3 章 Example 3.6 |
| 單標的策略組合（均線交叉 / 動能 / 布林通道） | 1 檔以上，等權重 | 第 7 章均值回歸 vs 動能 |
| 一籃子均值回歸（Johansen） | 2–12 檔 | 第 7 章共整合的多檔延伸 |

## 標的代號
- 美股：`KO`、`SPY`；台股：直接打數字 `2330`（自動判斷上市 .TW / 上櫃 .TWO）
- 外匯：`EURUSD=X`；期貨：`GC=F`；指數：`^TWII`；加密貨幣：`BTC-USD`

## 檔案結構
```
app.py                         進入點：側欄、頁面切換
quantlab/data.py               抓價格（Yahoo，已調整分割與配息）
quantlab/metrics.py            Sharpe、回撤等指標
quantlab/backtest.py           通用回測引擎 + 前視偏差檢查
quantlab/scanner.py            配對掃描
quantlab/tv.py                 Yahoo → TradingView 代號轉換
quantlab/strategies/           每個策略一個檔案
quantlab/ui/                   介面：主題、三個頁面
```

## 新增策略
在 `quantlab/strategies/` 新增一個檔案（參考 `portfolio.py`），然後在 `__init__.py` 的 `REGISTRY` 註冊。
介面會依照 `PARAMS` 自動產生控制元件。

## 本機執行
```bash
pip install -r requirements.txt
streamlit run app.py
```
