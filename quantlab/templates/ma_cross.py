class MACross(Strategy):
    """均線交叉：短均線上穿長均線做多，下穿做空（或平倉）。趨勢跟隨的經典範例。"""

    # 可調參數：會自動出現在介面上，也可以拿去做最佳化
    params = {
        "fast": 20,          # 短均線天數
        "slow": 60,          # 長均線天數
        "allow_short": True, # 是否做空
        "sl_pct": 0.0,       # 停損（0.05 = 5%，0 = 不設）
        "tp_pct": 0.0,       # 停利（0 = 不設）
    }

    def init(self):
        # init() 只跑一次：先把指標算好。self.I() 會把指標畫到圖上
        self.fast = self.I(ta.sma(self.close, self.p.fast), f"MA{self.p.fast}")
        self.slow = self.I(ta.sma(self.close, self.p.slow), f"MA{self.p.slow}")

    def next(self):
        # next() 每根 K 棒收盤後跑一次；self.i 是目前這根的位置
        i = self.i
        sl = self.p.sl_pct or None
        tp = self.p.tp_pct or None
        if ta.crossover(self.fast, self.slow, i):
            self.buy(sl_pct=sl, tp_pct=tp, tag="黃金交叉")       # 持有空單時會自動反手
        elif ta.crossunder(self.fast, self.slow, i):
            if self.p.allow_short:
                self.sell(sl_pct=sl, tp_pct=tp, tag="死亡交叉")
            elif self.is_long:
                self.close_position("死亡交叉")
