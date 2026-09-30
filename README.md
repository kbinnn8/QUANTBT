# Quant Lab

依照 Ernest Chan《Quantitative Trading》（第二版）做研究的回測工具。
研究邏輯用 Python 寫，介面用 Streamlit，部署到 Streamlit Community Cloud 後，平板或手機都能隨時打開調整。

## 目前的功能
- **配對交易（書中 Example 3.6，GLD / GDX）**
  - 訓練期 / 測試期分開：參數只用訓練期決定
  - 可調整進出場門檻、訓練期長度、部位權重、交易成本
  - 績效：Sharpe、年化報酬、最大回撤、最長回撤天數（未扣成本 / 扣成本）
  - 自動前視偏差檢查（書中介紹的「砍掉未來資料重跑」法）
  - 參數掃描熱圖：並排比較訓練期和測試期，用來檢查過擬合
  - 結果可下載 CSV

## 檔案結構
```
app.py                        Streamlit 介面
quantlab/data.py              抓價格（已調整分割與配息）
quantlab/metrics.py           Sharpe、回撤等指標
quantlab/backtest.py          通用回測引擎 + 前視偏差檢查
quantlab/strategies/          每個策略一個檔案
```

## 新增策略
1. 在 `quantlab/strategies/` 複製 `pair_trading.py`，改成你的策略
2. 保留 `NAME`、`BOOK_REF`、`DEFAULT_TICKERS`、`N_ASSETS`、`PARAMS`、`run(prices, params)`
3. 在 `quantlab/strategies/__init__.py` 的 `REGISTRY` 加一行

介面會依照 `PARAMS` 自動產生滑桿和選單。

## 部署到 Streamlit Community Cloud
1. 到 https://share.streamlit.io 用 GitHub 帳號登入
2. 點 **Create app**，選擇從 GitHub 部署（repository 設成 private 也可以）
3. Repository 選這個專案，Branch 選 `main`，Main file path 填 `app.py`
4. 按 **Deploy**，幾分鐘後會得到網址，在平板上加入書籤即可

## 本機執行
```bash
pip install -r requirements.txt
streamlit run app.py
```
