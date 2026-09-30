# Quant Lab

依照 Ernest Chan《Quantitative Trading》（第二版）做研究的回測工具。
研究邏輯用 Python 寫，介面用 Streamlit，部署在 Streamlit Community Cloud，平板或手機都能隨時打開。

## 頁面
- **策略測試器**（主要功能，類似 MT5 Strategy Tester）：在 app 裡用 Python 寫策略（`init()` 算指標、`next()` 每根 K 棒下單），
  支援停損 / 停利 / 移動停損、日線到 5 分 K；報告包含淨利、獲利因子、期望收益、回復因子、Sharpe / Sortino / Calmar、
  最大回撤、勝率（多空分開）、連續盈虧、交易明細、K 線標註進出場、月報酬熱圖；
  另有參數最佳化（含前推測試）與整份清單的批次回測。內建 5 個範本
- **圖表**：嵌入 TradingView 進階圖表，可以畫線、加指標，也能輸入 CFD 代號（例如 `OANDA:XAUUSD`）
- **書中範例**：Chan 書中的配對交易、單標的策略組合、一籃子均值回歸
- **配對掃描**：清單中所有兩兩組合，依訓練期共整合 p 值排序，用測試期驗證

## 策略寫法
```python
class MyStrategy(Strategy):
    params = {"n": 20, "sl_pct": 0.05}          # 可調參數，會自動出現在介面上

    def init(self):                              # 只跑一次：先算指標
        self.ma = self.I(ta.sma(self.close, self.p.n), "MA")

    def next(self):                              # 每根 K 棒收盤後跑一次，下的單在下一根開盤成交
        i = self.i
        if self.is_flat and self.close[i] > self.ma[i]:
            self.buy(sl_pct=self.p.sl_pct)
        elif self.is_long and self.close[i] < self.ma[i]:
            self.close_position("跌破均線")
```
完整 API 在 app 的「寫法說明」分頁；範本在 `quantlab/templates/`。

**安全**：策略程式碼會在伺服器上執行。請把 app 設成私人，或在 Streamlit 的 Secrets 設定 `APP_PASSWORD = "你的密碼"` 啟用密碼。

## 書中範例策略
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
quantlab/engine.py             事件驅動回測引擎（策略測試器）
quantlab/report.py             MT5 風格績效統計
quantlab/optimize.py           網格最佳化 + 前推測試
quantlab/ta.py                 技術指標
quantlab/templates/            策略範本
quantlab/data.py               抓價格（Yahoo，已調整分割與配息；OHLCV 多週期）
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
