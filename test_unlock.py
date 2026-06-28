"""
test_unlock.py — 解禁超跌反弹的【策略级终审】

策略：某股【大解禁】(占流通市值≥thr) → 解禁后第 ENTRY_LAG 日买入 → 持有 HOLD 日 → 卖出。
(对应事件研究里显著的 [+5,+25] 反弹窗：跳过紧邻抛售、等出清再进。)
长仓、T+1、含成本、时间止损出场(无趋势信号)。

终审口径：α/β vs 沪深300全收益、PSR、walk-forward 逐窗 OOS(purge+embargo)。
对比不同 thr(含 thr=0 全解禁基线)看"解禁量越大越强"是否传导到策略级。
另出逐笔收益左尾，间接感受幸存者偏差(解禁后崩盘退市样本缺失)。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import backtest as BT
import benchmark as BM
import optimize as OPT
import test_hma as TH

ENTRY_LAG, HOLD = 5, 20
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlock_calendar.csv")


def build_signals(df_trim: pd.DataFrame, dates) -> pd.DataFrame:
    idx = df_trim.index; n = len(idx)
    entry = np.zeros(n, dtype=bool)
    for d in dates:
        pos = int(idx.searchsorted(d, side="left")) + ENTRY_LAG
        if 0 <= pos < n:
            entry[pos] = True
    sig = pd.DataFrame(index=idx)
    sig["entry"] = entry
    sig["exit"] = False                      # 只靠时间止损出场
    return sig


def run(stocks: dict, cal: pd.DataFrame, thr: float) -> BT.BacktestResult:
    risk = C.RiskParams(use_hard_stop=False, use_atr_trailing=False,
                        use_time_stop=True, max_holding_days=HOLD)
    start, end = DL.backtest_window()
    sleeve = C.INIT_CAPITAL / len(stocks)
    per = {}
    for c, df in stocks.items():
        dft = DL.trim_to_window(df)
        ev = cal[(cal["code"] == c) & (cal["pct_float"] >= thr)]
        if start is not None:
            ev = ev[(ev["date"] >= start) & (ev["date"] <= (end or ev["date"].max()))]
        per[c] = BT.backtest_stock(c, dft, build_signals(dft, ev["date"].tolist()), sleeve, risk=risk)
    cal_idx = None
    for r in per.values():
        cal_idx = r.equity.index if cal_idx is None else cal_idx.union(r.equity.index)
    parts = [r.equity.reindex(cal_idx).ffill().fillna(sleeve) for r in per.values()]
    pe = pd.concat(parts, axis=1).sum(axis=1); pe.name = "portfolio"
    return BT.BacktestResult(per_stock=per, portfolio_equity=pe, calendar=cal_idx,
                             init_capital=C.INIT_CAPITAL)


def main():
    cal = pd.read_csv(CACHE, dtype={"code": str}, parse_dates=["date"])
    stocks = DL.load_pool(DL.get_universe(verbose=False), verbose=False)
    cal0 = None
    for c in stocks:
        t = DL.trim_to_window(stocks[c])
        cal0 = t.index if cal0 is None else cal0.union(t.index)
    split = OPT.holdout_split_date(cal0)
    bench = BM.benchmark_equity(cal0, C.BENCHMARK_CODE)

    variants = [("thr=0(全解禁)", 0.0), ("thr≥5%(大)", 0.05), ("thr≥10%(很大)", 0.10)]
    print("=" * 96)
    print(f"解禁超跌反弹 策略级终审 | 120只 | 入场=解禁后{ENTRY_LAG}日 持{HOLD}日 | 切分 {split.date()} | 基准=沪深300全收益")
    print("=" * 96)
    print(f"{'变体':<16}{'段':<6}{'年化':>8}{'最大回撤':>9}{'Sharpe':>7}{'Calmar':>7}"
          f"{'β':>6}{'α(年化)':>9}{'PSR':>6}{'交易':>7}")
    for name, thr in variants:
        res = run(stocks, cal, thr)
        seg = TH.evaluate(res, bench, split)
        trades = BT.trades_to_frame(res)
        for sname in ("full", "OOS"):
            m = seg[sname]
            nt = f"{seg['n_trades']}" if sname == "full" else ""
            print(f"{name if sname=='full' else '':<16}{sname:<6}"
                  f"{m['ann_return']:>8.2%}{m['max_dd']:>9.2%}{m['sharpe']:>7.2f}{m['calmar']:>7.2f}"
                  f"{m.get('beta',np.nan):>6.2f}{m.get('alpha_ann',np.nan):>9.2%}{m['psr0']:>6.0%}{nt:>7}")
        wf = seg["wf"]
        rt = trades["ret"] if len(trades) else pd.Series([np.nan])
        print(f"{'':<16}{'WF':<6}  OOS α中位 {wf['alpha_med']:>7.2%} | α>0窗 {wf['alpha_pos']:>4.0%} | "
              f"逐笔: 均{rt.mean():>6.2%} 中{rt.median():>6.2%} P5{rt.quantile(.05):>7.2%} 胜{(rt>0).mean():>3.0%}")
        print("-" * 96)
    print("判定：大解禁版 WF OOS α 为正且高于全解禁版 / PSR 高 → 解禁反转过组合级考卷。")
    print("  逐笔 P5(左尾)若很深，说明幸存者偏差掩盖的崩盘样本之外、真实左尾更糟。")


if __name__ == "__main__":
    main()
