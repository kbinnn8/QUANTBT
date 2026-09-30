"""圖表頁：嵌入 TradingView 進階圖表（可畫線、加指標）。"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from ..tv import to_tradingview

INTERVALS = {"1 分": "1", "5 分": "5", "15 分": "15", "1 小時": "60", "4 小時": "240",
             "日": "D", "週": "W", "月": "M"}
STUDIES = {
    "均線（SMA）": "STD;SMA", "指數均線（EMA）": "STD;EMA", "布林通道": "STD;Bollinger_Bands",
    "RSI": "STD;RSI", "MACD": "STD;MACD", "KD（隨機指標）": "STD;Stochastic",
    "成交量加權均價（VWAP）": "STD;VWAP", "ATR": "STD;Average_True_Range",
}


def widget_html(symbol: str, interval: str, studies: list[str], height: int) -> str:
    config = {
        "autosize": True, "symbol": symbol, "interval": interval,
        "timezone": "Asia/Taipei", "theme": "dark", "style": "1", "locale": "zh_TW",
        "backgroundColor": "#161615", "gridColor": "rgba(44,44,42,0.6)",
        "allow_symbol_change": True, "hide_side_toolbar": False, "withdateranges": True,
        "save_image": True, "details": False, "calendar": False,
        "studies": studies, "support_host": "https://www.tradingview.com",
    }
    # 注意：TradingView 的程式會把容器高度改成 100%，所以 html / body 也必須是 100%，
    # 否則圖表會縮成預設的 150px。
    return f"""
<style>html,body{{margin:0;padding:0;height:100%;background:#161615;overflow:hidden;}}
.tradingview-widget-container,.tradingview-widget-container__widget{{height:100%!important;width:100%!important;}}</style>
<div class="tradingview-widget-container" style="height:100%;width:100%">
  <div class="tradingview-widget-container__widget" style="height:100%;width:100%"></div>
  <script type="text/javascript"
    src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {json.dumps(config)}
  </script>
</div>
"""


def render(watchlist: list[str]) -> None:
    st.markdown('<div class="page-title">圖表</div>'
                '<div class="page-ref">TradingView 進階圖表：左側工具列畫線，上方「指標」按鈕加指標</div>',
                unsafe_allow_html=True)

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        options = watchlist or ["2330"]
        pick = c1.selectbox("標的清單", options, key="chart_pick",
                            format_func=lambda t: f"{t}  →  {to_tradingview(t)}")
        custom = c2.text_input("或直接輸入 TradingView 代號", key="chart_custom",
                               placeholder="例如 OANDA:XAUUSD、TWSE:2330、FX:EURUSD",
                               help="CFD 可以用券商代號，例如 OANDA:XAUUSD（黃金）、OANDA:SPX500USD（標普 500）。")
        interval = c3.selectbox("週期", list(INTERVALS), index=5, key="chart_interval")
        c4, c5 = st.columns([4, 1])
        studies = c4.multiselect("預設指標", list(STUDIES), default=["均線（SMA）", "RSI"],
                                 key="chart_studies")
        height = c5.select_slider("圖表高度", [500, 650, 800, 950, 1100], value=800, key="chart_height")

    symbol = custom.strip().upper() or to_tradingview(pick)
    components.html(widget_html(symbol, INTERVALS[interval], [STUDIES[s] for s in studies], height),
                    height=height)
    st.link_button("在 TradingView 開啟全螢幕 ↗", f"https://www.tradingview.com/chart/?symbol={symbol}")
    st.markdown(
        '<div class="hint">資料來自 TradingView，和回測用的 Yahoo 資料是分開的，兩邊的價格可能有些微差異。'
        '免費嵌入版的限制：重新整理頁面後，畫的線與手動加的指標不會保留。'
        '想保留畫線，可以按上方按鈕到 TradingView 網站開啟（登入後會自動儲存）。</div>',
        unsafe_allow_html=True)
