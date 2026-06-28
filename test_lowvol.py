"""
test_lowvol.py — “低量 flip_up”反转信号的策略级终审

事件级已发现：flip_up 在【低量】(成交量<自身10日均量) 入场，未来超额显著更高(t=2.22)。
本测试把它做成策略，过组合级考卷：WF 逐窗 OOS α(purge+embargo) + PSR + α/β vs 沪深300全收益。

受控对比（唯一变量=入场过滤）：
  ① 裸 flip_up（基线，WF OOS α 已知 +2.74%）
  B  低量 flip_up
  C  安静抄底 flip_up（低量 且 比前60日高点低≥10%）
  D  低量 flip_up + 硬止损8%（用户方法论）
出场均为 flip_down。参数固定(n=10/m=3)，不调参。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import backtest as BT
import benchmark as BM
import optimize as OPT
import test_hma as TH

N_ST, M_ST = 10, 3.0
RISK_OFF = C.RiskParams(use_hard_stop=False, use_atr_trailing=False)
RISK_HARD = C.RiskParams(use_hard_stop=True, hard_stop_pct=0.08, use_atr_trailing=False)


def build_panel(stocks: dict) -> dict:
    params = C.StrategyParams(n=N_ST, m=M_ST, w=10, k=1.5)
    panel = {}
    for c, df in stocks.items():
        fdf = F.compute_factors(df, params).copy()
        v = fdf["volume"]
        fdf["lowvol"] = (v < v.rolling(10).mean())
        prior_h60 = fdf["high"].shift(1).rolling(60).max()
        fdf["pullback"] = (fdf["close"] < 0.90 * prior_h60)
        panel[c] = DL.trim_to_window(fdf)
    return panel


def signals(fdf: pd.DataFrame, kind: str) -> pd.DataFrame:
    fu = fdf["flip_up"].to_numpy(bool)
    if kind == "raw":
        entry = fu
    elif kind == "lowvol":
        entry = fu & fdf["lowvol"].to_numpy(bool)
    elif kind == "quiet_dip":
        entry = fu & fdf["lowvol"].to_numpy(bool) & fdf["pullback"].to_numpy(bool)
    else:
        raise ValueError(kind)
    sig = pd.DataFrame(index=fdf.index)
    sig["entry"] = entry
    sig["exit"] = fdf["flip_down"].to_numpy(bool)
    return sig


def run_variant(panel: dict, kind: str, risk) -> BT.BacktestResult:
    sleeve = C.INIT_CAPITAL / len(panel)
    per = {c: BT.backtest_stock(c, panel[c], signals(panel[c], kind), sleeve, risk=risk) for c in panel}
    cal = None
    for r in per.values():
        cal = r.equity.index if cal is None else cal.union(r.equity.index)
    parts = [r.equity.reindex(cal).ffill().fillna(sleeve) for r in per.values()]
    pe = pd.concat(parts, axis=1).sum(axis=1); pe.name = "portfolio"
    return BT.BacktestResult(per_stock=per, portfolio_equity=pe, calendar=cal, init_capital=C.INIT_CAPITAL)


def main():
    stocks = DL.load_pool(DL.get_universe(verbose=False) if C.USE_UNIVERSE else C.POOL, verbose=False)
    panel = build_panel(stocks)
    cal = None
    for fdf in panel.values():
        cal = fdf.index if cal is None else cal.union(fdf.index)
    split = OPT.holdout_split_date(cal)
    bench = BM.benchmark_equity(cal, C.BENCHMARK_CODE)

    variants = [
        ("① 裸flip_up", "raw", RISK_OFF),
        ("B 低量flip_up", "lowvol", RISK_OFF),
        ("C 安静抄底flip_up", "quiet_dip", RISK_OFF),
        ("D 低量+硬止损8%", "lowvol", RISK_HARD),
    ]
    results = {name: TH.evaluate(run_variant(panel, k, r), bench, split) for name, k, r in variants}

    print("=" * 100)
    print(f"低量 flip_up 策略级终审 | 120只 | {cal.min().date()}~{cal.max().date()} | 切分 {split.date()} | "
          f"基准=沪深300全收益")
    print("=" * 100)
    print(f"{'变体':<18}{'段':<6}{'年化':>8}{'最大回撤':>9}{'Sharpe':>7}{'Calmar':>7}"
          f"{'β':>6}{'α(年化)':>9}{'PSR':>6}{'交易':>7}")
    for name, seg in results.items():
        for sname in ("full", "OOS"):
            m = seg[sname]
            nt = f"{seg['n_trades']}" if sname == "full" else ""
            print(f"{name if sname=='full' else '':<18}{sname:<6}"
                  f"{m['ann_return']:>8.2%}{m['max_dd']:>9.2%}{m['sharpe']:>7.2f}{m['calmar']:>7.2f}"
                  f"{m.get('beta',np.nan):>6.2f}{m.get('alpha_ann',np.nan):>9.2%}{m['psr0']:>6.0%}{nt:>7}")
        wf = seg["wf"]
        print(f"{'':<18}{'WF':<6}  OOS α中位 {wf['alpha_med']:>7.2%} | α>0窗口 {wf['alpha_pos']:>4.0%} | "
              f"OOS年化中位 {wf['ann_med']:>6.2%}（{wf['n']}窗）")
        print("-" * 100)
    print("判定：低量版 WF OOS α 明显高于裸版、且为正 / PSR 高 → 信号过组合级考卷。")
    print("  注意低量交易稀疏(~8%)，WF 逐窗样本少、α 噪声大，需结合点估计与方向一致性看。")


if __name__ == "__main__":
    main()
