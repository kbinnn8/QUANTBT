"""MT5 風格的回測報告：從淨值曲線與交易紀錄計算各種統計。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(equity: pd.Series) -> pd.Series:
    """不論 K 棒週期，都先轉成日淨值再算報酬，讓 Sharpe 可以互相比較。"""
    eq = equity.dropna()
    if len(eq) < 2:
        return pd.Series(dtype=float)
    d = eq.groupby(eq.index.normalize()).last()
    return d.pct_change().dropna()


def _streaks(wins: np.ndarray, pnl: np.ndarray, want: bool) -> tuple[int, float]:
    best_n, best_amt, n, amt = 0, 0.0, 0, 0.0
    for w, p in zip(wins, pnl):
        if w == want:
            n, amt = n + 1, amt + p
            if n > best_n:
                best_n, best_amt = n, amt
        else:
            n, amt = 0, 0.0
    return best_n, best_amt


def max_drawdown(equity: pd.Series) -> dict:
    eq = equity.dropna()
    peak = eq.cummax()
    dd_amt = eq - peak
    dd_pct = eq / peak - 1
    # 最長回撤期間（日曆天）：從創高到下次創高
    longest, start = pd.Timedelta(0), None
    for t, underwater in (dd_pct < 0).items():
        if underwater and start is None:
            start = t
        elif not underwater and start is not None:
            longest, start = max(longest, t - start), None
    if start is not None:
        longest = max(longest, eq.index[-1] - start)
    return dict(pct=float(dd_pct.min()) if len(eq) else np.nan,
                amount=float(dd_amt.min()) if len(eq) else np.nan,
                longest_days=int(longest.days))


def compute(equity: pd.Series, trades: pd.DataFrame, bench_close: pd.Series | None,
            exposure: pd.Series | None = None, base: float | None = None) -> dict:
    """回傳一個 dict，key 為中文統計名稱。base = 起始資金（預設用淨值第一個值）。"""
    eq = equity.dropna()
    if eq.empty:
        return {}
    base = float(base if base is not None else eq.iloc[0])
    final = float(eq.iloc[-1])
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = days / 365.25
    total_ret = final / base - 1
    cagr = (final / base) ** (1 / years) - 1 if final > 0 and years > 0 else np.nan
    dr = daily_returns(eq)
    vol = dr.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(dr) > 1 else np.nan
    sharpe = dr.mean() / dr.std(ddof=1) * np.sqrt(TRADING_DAYS) if len(dr) > 1 and dr.std(ddof=1) > 0 else np.nan
    downside = dr[dr < 0]
    dstd = np.sqrt((downside ** 2).sum() / len(dr)) if len(dr) > 1 else np.nan
    sortino = dr.mean() / dstd * np.sqrt(TRADING_DAYS) if dstd and dstd > 0 else np.nan
    dd = max_drawdown(eq)
    calmar = cagr / abs(dd["pct"]) if dd["pct"] and dd["pct"] < 0 and np.isfinite(cagr) else np.nan
    net = final - base

    s = {
        "起始資金": base, "最終淨值": final, "淨利": net, "總報酬": total_ret, "年化報酬": cagr,
        "年化波動": vol, "Sharpe": sharpe, "Sortino": sortino, "Calmar": calmar,
        "最大回撤": dd["pct"], "最大回撤金額": dd["amount"], "最長回撤天數": dd["longest_days"],
        "回復因子": net / abs(dd["amount"]) if dd["amount"] and dd["amount"] < 0 else np.nan,
    }
    if bench_close is not None and len(bench_close.dropna()) > 1:
        b = bench_close.dropna()
        s["買進持有報酬"] = float(b.iloc[-1] / b.iloc[0] - 1)
    if exposure is not None and len(exposure):
        s["持倉時間比例"] = float(exposure.mean())

    t = trades if trades is not None else pd.DataFrame()
    n = len(t)
    s["交易次數"] = n
    if n:
        pnl = t["損益"].to_numpy(float)
        wins = pnl > 0
        gp, gl = pnl[wins].sum(), pnl[~wins].sum()
        longs, shorts = t[t["方向"] == "多"], t[t["方向"] == "空"]
        avg_w = pnl[wins].mean() if wins.any() else np.nan
        avg_l = pnl[~wins].mean() if (~wins).any() else np.nan
        cw_n, cw_amt = _streaks(wins, pnl, True)
        cl_n, cl_amt = _streaks(wins, pnl, False)
        r = t["報酬率"].to_numpy(float)
        s.update({
            "毛利": gp, "毛損": gl,
            "獲利因子": gp / abs(gl) if gl < 0 else np.inf if gp > 0 else np.nan,
            "期望收益": pnl.mean(), "平均每筆報酬": r.mean(),
            "勝率": wins.mean(),
            "多單次數": len(longs), "多單勝率": (longs["損益"] > 0).mean() if len(longs) else np.nan,
            "空單次數": len(shorts), "空單勝率": (shorts["損益"] > 0).mean() if len(shorts) else np.nan,
            "平均獲利": avg_w, "平均虧損": avg_l,
            "盈虧比": avg_w / abs(avg_l) if np.isfinite(avg_w) and np.isfinite(avg_l) and avg_l < 0 else np.nan,
            "最大單筆獲利": pnl.max(), "最大單筆虧損": pnl.min(),
            "最大連續獲利次數": cw_n, "最大連續獲利金額": cw_amt,
            "最大連續虧損次數": cl_n, "最大連續虧損金額": cl_amt,
            "平均持有K棒": float(t["持有K棒"].mean()),
            "總手續費": float(t["手續費"].sum()),
            "SQN": np.sqrt(n) * r.mean() / r.std(ddof=1) if n > 1 and r.std(ddof=1) > 0 else np.nan,
        })
    return s


def segment(result: dict, start=None, end=None) -> dict:
    """計算某一段期間（例如最佳化的樣本內 / 樣本外）的統計。
    淨值以該段第一根為基準；交易以「進場時間」歸屬。"""
    eq, data = result["equity"], result["data"]
    mask = pd.Series(True, index=eq.index)
    if start is not None:
        mask &= eq.index >= start
    if end is not None:
        mask &= eq.index < end
    t = result["trades"]
    if len(t):
        tm = pd.Series(True, index=t.index)
        if start is not None:
            tm &= t["進場時間"] >= start
        if end is not None:
            tm &= t["進場時間"] < end
        t = t[tm]
    return compute(eq[mask], t, data["Close"][mask], result["exposure"][mask])


def monthly_returns(equity: pd.Series) -> pd.DataFrame:
    """年 × 月 的報酬表，最後一欄為全年。"""
    eq = equity.dropna()
    m = eq.groupby([eq.index.year, eq.index.month]).last()
    first = eq.iloc[0]
    prev = m.shift(1)
    prev.iloc[0] = first
    ret = (m / prev - 1).rename("r").reset_index()
    ret.columns = ["年", "月", "r"]
    table = ret.pivot(index="年", columns="月", values="r").reindex(columns=range(1, 13))
    y = eq.groupby(eq.index.year).last()
    py = y.shift(1)
    py.iloc[0] = first
    table["全年"] = (y / py - 1).to_numpy()
    return table
