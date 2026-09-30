class DonchianBreakout(Strategy):
    """唐奇安通道突破（海龜交易法的簡化版）：
    收盤創 N 日新高做多、創 N 日新低做空；用 ATR 設定初始停損，並用較短通道做移動出場。"""

    params = {
        "entry_len": 55,   # 進場通道
        "exit_len": 20,    # 出場通道（較短）
        "atr_len": 20,
        "atr_stop": 2.0,   # 停損 = 進場價 ± 2 倍 ATR
        "allow_short": True,
    }

    def init(self):
        # 用「前一根為止」的高低點，避免拿當根自己跟自己比
        self.hi = self.I(ta.shift(ta.highest(self.high, self.p.entry_len)), "進場上緣")
        self.lo = self.I(ta.shift(ta.lowest(self.low, self.p.entry_len)), "進場下緣")
        self.exit_hi = ta.shift(ta.highest(self.high, self.p.exit_len))
        self.exit_lo = ta.shift(ta.lowest(self.low, self.p.exit_len))
        self.atr = ta.atr(self.high, self.low, self.close, self.p.atr_len)

    def next(self):
        i, c, a = self.i, self.close[self.i], self.atr[self.i]
        if np.isnan(a):
            return
        stop = self.p.atr_stop * a
        if self.is_flat:
            if c > self.hi[i]:
                self.buy(sl=c - stop, tag="突破新高")
            elif self.p.allow_short and c < self.lo[i]:
                self.sell(sl=c + stop, tag="跌破新低")
        elif self.is_long and c < self.exit_lo[i]:
            self.close_position("跌破出場通道")
        elif self.is_short and c > self.exit_hi[i]:
            self.close_position("突破出場通道")
