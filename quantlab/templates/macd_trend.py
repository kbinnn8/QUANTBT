class MACDTrend(Strategy):
    """MACD 交叉 + 長期均線趨勢濾網：
    只在價格高於長均線時做 MACD 黃金交叉的多單，並用移動停損保護獲利。"""

    params = {
        "fast": 12, "slow": 26, "signal": 9,
        "trend_len": 200,      # 趨勢濾網
        "trail_pct": 0.08,     # 移動停損：從持倉期間最高價回落 8% 出場
    }

    def init(self):
        m, s, h = ta.macd(self.close, self.p.fast, self.p.slow, self.p.signal)
        self.macd = self.I(m, "MACD", overlay=False)
        self.sig = self.I(s, "訊號線", overlay=False)
        self.trend = self.I(ta.sma(self.close, self.p.trend_len), f"MA{self.p.trend_len}")
        self.peak = np.nan

    def next(self):
        i, c = self.i, self.close[self.i]
        if self.is_flat:
            if c > self.trend[i] and ta.crossover(self.macd, self.sig, i):
                self.buy(tag="MACD 黃金交叉")
                self.peak = c
            return
        # 持有多單：更新最高價與移動停損
        self.peak = max(self.peak, self.high[i])
        self.set_sl(self.peak * (1 - self.p.trail_pct))
        if ta.crossunder(self.macd, self.sig, i) and c < self.trend[i]:
            self.close_position("MACD 死亡交叉且跌破趨勢線")
