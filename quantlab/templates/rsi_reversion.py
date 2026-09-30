class RSIReversion(Strategy):
    """RSI 均值回歸：RSI 跌破超賣線時做多，回到中線以上平倉；
    可選擇只在長期均線之上做多（順大勢、逆小勢）。"""

    params = {
        "rsi_len": 14,
        "oversold": 30,        # 低於這個值視為超賣 → 做多
        "overbought": 70,      # 高於這個值視為超買 → 做空（若允許）
        "exit_level": 50,      # RSI 回到這附近就出場
        "trend_filter": 200,   # 只在價格高於此均線時做多（0 = 不過濾）
        "allow_short": False,
        "max_bars": 20,        # 最多持有幾根，時間到就出場（0 = 不限）
    }

    def init(self):
        self.rsi = self.I(ta.rsi(self.close, self.p.rsi_len), "RSI", overlay=False)
        n = self.p.trend_filter
        self.trend = self.I(ta.sma(self.close, n), f"MA{n}") if n else None

    def next(self):
        i, r = self.i, self.rsi[self.i]
        if np.isnan(r):
            return
        uptrend = self.trend is None or self.close[i] > self.trend[i]
        downtrend = self.trend is None or self.close[i] < self.trend[i]

        if self.is_flat:
            if r < self.p.oversold and uptrend:
                self.buy(tag="超賣")
            elif self.p.allow_short and r > self.p.overbought and downtrend:
                self.sell(tag="超買")
            return

        timeout = self.p.max_bars and self.bars_in_trade >= self.p.max_bars
        if self.is_long and (r > self.p.exit_level or timeout):
            self.close_position("時間到" if timeout else "RSI 回升")
        elif self.is_short and (r < self.p.exit_level or timeout):
            self.close_position("時間到" if timeout else "RSI 回落")
