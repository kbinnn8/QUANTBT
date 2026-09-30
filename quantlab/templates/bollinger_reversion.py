class BollingerReversion(Strategy):
    """布林通道均值回歸：收盤跌破下軌做多、突破上軌做空，回到中線平倉。
    和書中配對交易的 z-score 進出場是同一個想法，只是用單一商品自己的均線。"""

    params = {
        "window": 20,     # 均線天數
        "k": 2.0,         # 通道寬度（幾個標準差）
        "allow_short": True,
        "sl_pct": 0.05,   # 停損 5%，避免遇到趨勢時一路虧
    }

    def init(self):
        mid, up, lo = ta.bollinger(self.close, self.p.window, self.p.k)
        self.mid = self.I(mid, "中線")
        self.up = self.I(up, "上軌")
        self.lo = self.I(lo, "下軌")

    def next(self):
        i, c = self.i, self.close[self.i]
        sl = self.p.sl_pct or None
        if self.is_flat:
            if c < self.lo[i]:
                self.buy(sl_pct=sl, tag="跌破下軌")
            elif self.p.allow_short and c > self.up[i]:
                self.sell(sl_pct=sl, tag="突破上軌")
        elif self.is_long and c >= self.mid[i]:
            self.close_position("回到中線")
        elif self.is_short and c <= self.mid[i]:
            self.close_position("回到中線")
