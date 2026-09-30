"""事件驅動回測引擎（類似 MT5 策略測試器的「每根 K 棒」模式）。

時間順序（每根 K 棒 i）：
1. 以開盤價執行上一根 next() 下的單（市價單在「下一根開盤」成交，不會用到當根收盤價 → 避免前視偏差）
2. 用當根的最高 / 最低價檢查停損、停利（同一根兩者都碰到時，保守假設先碰停損）
3. 以收盤價計算淨值
4. 呼叫策略的 next()，決定下一步
帳戶是「淨部位」模式：同時只有一個方向的部位；持有空單時下買單會先平空再做多（反手）。
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import ta

STRATEGY_FILENAME = "strategy.py"


class StrategyError(Exception):
    """使用者策略程式碼出錯（訊息已整理成只顯示策略檔的行號）。"""


# ───────────────────────── 策略基底類別 ─────────────────────────
class Strategy:
    """所有策略都繼承這個類別，覆寫 init() 和 next()。"""

    params: dict = {}

    def __init__(self, data: pd.DataFrame, params: dict, broker: "Broker"):
        self.data = data
        self.open = data["Open"].to_numpy(dtype=float)
        self.high = data["High"].to_numpy(dtype=float)
        self.low = data["Low"].to_numpy(dtype=float)
        self.close = data["Close"].to_numpy(dtype=float)
        self.volume = data["Volume"].to_numpy(dtype=float) if "Volume" in data else np.zeros(len(data))
        self.p = SimpleNamespace(**{**type(self).params, **(params or {})})
        self.i = 0
        self._broker = broker
        self._indicators: list[dict] = []

    # 使用者覆寫
    def init(self):
        pass

    def next(self):
        pass

    # 指標註冊（會畫在圖上）
    def I(self, values, name: str | None = None, overlay: bool = True):
        """註冊指標：overlay=True 畫在 K 線上（均線、通道），False 畫在下方副圖（RSI、MACD）。"""
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != len(self.data):
            raise ValueError(f"指標 {name} 的長度 {arr.shape[0]} 和資料長度 {len(self.data)} 不同")
        self._indicators.append(dict(name=name or f"指標{len(self._indicators) + 1}", values=arr,
                                     overlay=overlay))
        return arr

    # 狀態
    @property
    def time(self):
        return self.data.index[self.i]

    @property
    def position(self) -> float:
        """目前部位的股數 / 單位：正數 = 多單，負數 = 空單，0 = 空手"""
        return self._broker.position

    @property
    def is_long(self) -> bool:
        return self._broker.position > 0

    @property
    def is_short(self) -> bool:
        return self._broker.position < 0

    @property
    def is_flat(self) -> bool:
        return self._broker.position == 0

    @property
    def equity(self) -> float:
        return self._broker.cash + self._broker.position * self.close[self.i]

    @property
    def entry_price(self) -> float:
        return self._broker.entry_price

    @property
    def bars_in_trade(self) -> int:
        b = self._broker
        return 0 if b.entry_i is None else self.i - b.entry_i

    @property
    def open_pnl_pct(self) -> float:
        """目前部位的未實現報酬（不含手續費），空手時為 0"""
        b = self._broker
        if b.position == 0:
            return 0.0
        r = self.close[self.i] / b.entry_price - 1
        return r if b.position > 0 else -r

    # 下單（下一根開盤成交）
    def buy(self, size: float = 1.0, sl=None, tp=None, sl_pct=None, tp_pct=None, tag: str = ""):
        """做多。size = 使用淨值的比例（1 = 100%）。
        sl / tp 為價格；sl_pct / tp_pct 為相對成交價的比例（例如 0.05 = 5%）。"""
        self._broker.submit(self.i, "buy", size, sl, tp, sl_pct, tp_pct, tag)

    def sell(self, size: float = 1.0, sl=None, tp=None, sl_pct=None, tp_pct=None, tag: str = ""):
        """做空。參數同 buy()。"""
        self._broker.submit(self.i, "sell", size, sl, tp, sl_pct, tp_pct, tag)

    def close_position(self, tag: str = ""):
        """平倉（下一根開盤成交）"""
        self._broker.submit(self.i, "close", 0, None, None, None, None, tag)

    def set_sl(self, price):
        """修改目前部位的停損價（下一根開始生效），可用來做移動停損"""
        self._broker.sl = None if price is None or np.isnan(price) else float(price)

    def set_tp(self, price):
        """修改目前部位的停利價"""
        self._broker.tp = None if price is None or np.isnan(price) else float(price)


# ───────────────────────── 經紀商（成交與帳務） ─────────────────────────
@dataclass
class Order:
    i: int
    kind: str
    size: float
    sl: float | None
    tp: float | None
    sl_pct: float | None
    tp_pct: float | None
    tag: str


@dataclass
class Broker:
    data: pd.DataFrame
    cash: float
    cost_bps: float
    position: float = 0.0
    entry_price: float = float("nan")
    entry_i: int | None = None
    entry_comm: float = 0.0
    entry_tag: str = ""
    sl: float | None = None
    tp: float | None = None
    pending: list = field(default_factory=list)
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    def __post_init__(self):
        d = self.data
        self.o, self.h = d["Open"].to_numpy(float), d["High"].to_numpy(float)
        self.l, self.c = d["Low"].to_numpy(float), d["Close"].to_numpy(float)
        self.idx = d.index
        self.rate = self.cost_bps / 1e4

    def submit(self, i, kind, size, sl, tp, sl_pct, tp_pct, tag):
        if kind in ("buy", "sell") and not (size and size > 0):
            raise ValueError("size 必須大於 0（1 = 使用 100% 淨值）")
        o = Order(i, kind, float(size or 0), sl, tp, sl_pct, tp_pct, tag)
        self.pending.append(o)
        self.orders.append((i, kind, round(float(size or 0), 10)))

    # 成交
    def _open(self, i, direction: int, o: Order):
        price = self.o[i]
        if not np.isfinite(price) or price <= 0:
            return
        equity = self.cash + self.position * price
        units = o.size * equity / price
        if units <= 0:
            return
        comm = units * price * self.rate
        self.cash -= direction * units * price + comm
        self.position = direction * units
        self.entry_price, self.entry_i, self.entry_comm, self.entry_tag = price, i, comm, o.tag
        sl, tp = o.sl, o.tp
        if o.sl_pct:
            sl = price * (1 - direction * o.sl_pct)
        if o.tp_pct:
            tp = price * (1 + direction * o.tp_pct)
        self.sl = float(sl) if sl is not None and np.isfinite(sl) else None
        self.tp = float(tp) if tp is not None and np.isfinite(tp) else None

    def close_at(self, i, price, reason):
        if self.position == 0:
            return
        units = self.position
        comm = abs(units) * price * self.rate
        self.cash += units * price - comm
        direction = 1 if units > 0 else -1
        gross = units * (price - self.entry_price)
        pnl = gross - self.entry_comm - comm
        seg_h, seg_l = self.h[self.entry_i:i + 1], self.l[self.entry_i:i + 1]
        if direction > 0:
            mfe, mae = np.nanmax(seg_h) / self.entry_price - 1, np.nanmin(seg_l) / self.entry_price - 1
        else:
            mfe, mae = 1 - np.nanmin(seg_l) / self.entry_price, 1 - np.nanmax(seg_h) / self.entry_price
        self.trades.append(dict(
            方向="多" if direction > 0 else "空", 數量=abs(units),
            進場時間=self.idx[self.entry_i], 進場價=self.entry_price,
            出場時間=self.idx[i], 出場價=price,
            損益=pnl, 報酬率=pnl / (abs(units) * self.entry_price),
            手續費=self.entry_comm + comm, 持有K棒=i - self.entry_i,
            最大有利波動=mfe, 最大不利波動=mae,
            進場標籤=self.entry_tag, 出場原因=reason,
        ))
        self.position, self.entry_price, self.entry_i = 0.0, float("nan"), None
        self.entry_comm, self.sl, self.tp = 0.0, None, None

    def execute_pending(self, i):
        for o in self.pending:
            if o.kind == "close":
                self.close_at(i, self.o[i], o.tag or "訊號平倉")
            elif o.kind == "buy":
                if self.position < 0:
                    self.close_at(i, self.o[i], o.tag or "反手")
                if self.position == 0:
                    self._open(i, +1, o)
            elif o.kind == "sell":
                if self.position > 0:
                    self.close_at(i, self.o[i], o.tag or "反手")
                if self.position == 0:
                    self._open(i, -1, o)
        self.pending = []

    def check_stops(self, i):
        if self.position == 0:
            return
        o, h, l = self.o[i], self.h[i], self.l[i]
        if self.position > 0:
            if self.sl is not None and l <= self.sl:
                return self.close_at(i, min(o, self.sl), "停損")
            if self.tp is not None and h >= self.tp:
                return self.close_at(i, max(o, self.tp), "停利")
        else:
            if self.sl is not None and h >= self.sl:
                return self.close_at(i, max(o, self.sl), "停損")
            if self.tp is not None and l <= self.tp:
                return self.close_at(i, min(o, self.tp), "停利")


# ───────────────────────── 載入使用者程式碼 ─────────────────────────
def load_strategy(code: str) -> type:
    """執行策略程式碼，回傳其中定義的 Strategy 子類別。"""
    import linecache
    # 讓錯誤訊息能顯示出錯的那一行程式碼
    linecache.cache[STRATEGY_FILENAME] = (len(code), None, code.splitlines(True), STRATEGY_FILENAME)
    ns = {"Strategy": Strategy, "ta": ta, "np": np, "pd": pd, "__name__": "user_strategy"}
    try:
        exec(compile(code, STRATEGY_FILENAME, "exec"), ns)
    except SyntaxError as e:
        raise StrategyError(f"語法錯誤（第 {e.lineno} 行）：{e.msg}\n{(e.text or '').rstrip()}") from None
    except Exception:
        raise StrategyError(format_user_error()) from None
    classes = [v for v in ns.values() if isinstance(v, type) and issubclass(v, Strategy) and v is not Strategy]
    if not classes:
        raise StrategyError("找不到策略類別：請定義一個繼承 Strategy 的 class，例如 `class MyStrategy(Strategy):`")
    return classes[-1]


def format_user_error() -> str:
    """只保留策略檔裡的錯誤行，讓訊息好讀。"""
    tb = traceback.extract_tb(__import__("sys").exc_info()[2])
    exc = __import__("sys").exc_info()[1]
    lines = [f"第 {fr.lineno} 行（{fr.name}）：{fr.line}" for fr in tb if fr.filename == STRATEGY_FILENAME]
    where = "\n".join(lines[-3:]) if lines else "（錯誤發生在策略程式碼之外）"
    return f"{type(exc).__name__}: {exc}\n{where}"


# ───────────────────────── 執行回測 ─────────────────────────
def run(strategy_cls: type, data: pd.DataFrame, params: dict | None = None,
        cash: float = 100_000.0, cost_bps: float = 5.0) -> dict:
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    if len(data) < 2:
        raise StrategyError("資料不足（少於 2 根 K 棒）")
    broker = Broker(data=data, cash=float(cash), cost_bps=float(cost_bps))
    strat = strategy_cls(data, params or {}, broker)
    n = len(data)
    equity = np.empty(n)
    exposure = np.zeros(n, dtype=bool)
    try:
        strat.init()
        for i in range(n):
            if broker.pending:
                broker.execute_pending(i)
            broker.check_stops(i)
            equity[i] = broker.cash + broker.position * broker.c[i]
            exposure[i] = broker.position != 0
            strat.i = i
            strat.next()
    except StrategyError:
        raise
    except Exception:
        raise StrategyError(format_user_error()) from None
    if broker.position != 0:
        broker.close_at(n - 1, broker.c[n - 1], "回測結束")
        equity[n - 1] = broker.cash
    trades = pd.DataFrame(broker.trades)
    return dict(
        data=data,
        equity=pd.Series(equity, index=data.index, name="淨值"),
        exposure=pd.Series(exposure, index=data.index),
        trades=trades,
        indicators=strat._indicators,
        orders=broker.orders,
        params=vars(strat.p).copy(),
        cash=float(cash),
    )


def lookahead_check(strategy_cls, data, params, cash, cost_bps, cuts=(20, 37, 61, 97)) -> tuple[bool, int]:
    """書中的前視偏差檢查法：砍掉最後幾根重跑。

    如果策略沒偷看未來，截斷後資料的每一根（包含最後一根）下的單都應該和完整資料時相同。
    用幾個不同的截斷點各檢查一次，偷看的策略很難每次都剛好躲過。回傳（是否通過, 不一致的 K 棒數）。
    """
    full = run(strategy_cls, data, params, cash, cost_bps)["orders"]
    bad_bars: set = set()
    for cut in cuts:
        if len(data) <= cut + 30:
            continue
        part_data = data.iloc[:-cut]
        part = run(strategy_cls, part_data, params, cash, cost_bps)["orders"]
        last = len(part_data) - 1
        a = [o for o in full if o[0] <= last]
        b = [o for o in part if o[0] <= last]
        if a != b:
            bad_bars |= {o[0] for o in set(a) ^ set(b)}
    return not bad_bars, len(bad_bars)
