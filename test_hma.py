"""
test_hma.py — SuperTrend + HMA200 策略的严谨受控测试

用户策略：入场 = SuperTrend 翻多 且 收盘价 > HMA(200)；出场 = SuperTrend 翻空（仅此，故止损关闭）。
参数：SuperTrend 默认 n=10/m=3.0；HMA=200。

受控对比（同股池/窗口/评估框架，唯一变量=入场过滤器）：
  ① 裸 SuperTrend（entry=flip_up）
  ② SuperTrend + 量能突破（原版 entry=flip_up & vol_break）
  ③ SuperTrend + HMA200（本次 entry=flip_up & close>HMA200）
三者出场均为 flip_down、止损均关闭，再与沪深300全收益基准比 α。

严谨口径：前复权、无前视/T+1/涨跌停、闲置现金计息、基准全收益、holdout IS/OOS、
walk-forward 逐窗 OOS（purge+embargo）、PSR（夏普显著性）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import backtest as BT
import report as R
import benchmark as BM
import optimize as OPT

N, M, HMA_P = 10, 3.0, 200
RISK_OFF = C.RiskParams(use_hard_stop=False, use_atr_trailing=False)   # 出场只看 SuperTrend


def build_factor_panel(stocks: dict) -> dict:
    params = C.StrategyParams(n=N, m=M, w=10, k=1.5)
    panel = {}
    for c, df in stocks.items():
        fdf = DL.trim_to_window(F.compute_factors(df, params)).copy()
        fdf["hma"] = F.hma(fdf["close"].to_numpy(float), HMA_P)
        panel[c] = fdf
    return panel


def make_signals(fdf: pd.DataFrame, kind: str) -> pd.DataFrame:
    flip_up = fdf["flip_up"].to_numpy(bool)
    flip_dn = fdf["flip_down"].to_numpy(bool)
    if kind == "raw":
        entry = flip_up
    elif kind == "vol":
        entry = flip_up & fdf["vol_break"].to_numpy(bool)
    elif kind == "hma":
        entry = flip_up & (fdf["close"].to_numpy(float) > fdf["hma"].to_numpy(float))
    else:
        raise ValueError(kind)
    sig = pd.DataFrame(index=fdf.index)
    sig["entry"] = entry
    sig["exit"] = flip_dn
    return sig


def run_strategy(panel: dict, kind: str) -> BT.BacktestResult:
    sleeve = C.INIT_CAPITAL / len(panel)
    per = {c: BT.backtest_stock(c, panel[c], make_signals(panel[c], kind), sleeve, risk=RISK_OFF)
           for c in panel}
    cal = None
    for r in per.values():
        cal = r.equity.index if cal is None else cal.union(r.equity.index)
    parts = [r.equity.reindex(cal).ffill().fillna(sleeve) for r in per.values()]
    pe = pd.concat(parts, axis=1).sum(axis=1); pe.name = "portfolio"
    return BT.BacktestResult(per_stock=per, portfolio_equity=pe, calendar=cal,
                             init_capital=C.INIT_CAPITAL)


def _psr0(equity: pd.Series) -> float:
    m = R.metrics_from_equity(equity)
    d = R.deflated_sharpe_ratio(equity.pct_change(), [m["sharpe"]], n_trials=1)
    return d.get("psr0", float("nan"))


def evaluate(res: BT.BacktestResult, bench: pd.Series, split: pd.Timestamp) -> dict:
    pe = res.portfolio_equity
    trades = BT.trades_to_frame(res)
    segs = {"full": (None, None), "IS": (None, split), "OOS": (split, None)}
    seg = {}
    for name, (s, e) in segs.items():
        eq = R.slice_equity(pe, s, e)
        td = R.slice_trades(trades, s, e)
        bq = R.slice_equity(bench, s, e)
        m = R.compute_metrics(eq, td, benchmark=bq)
        m["psr0"] = _psr0(eq)
        seg[name] = m
    # walk-forward 逐窗 OOS（固定策略，不再寻优；purge+embargo）
    windows = OPT.walkforward_windows(res.calendar)
    run = OPT.ComboRun(params=C.StrategyParams(n=N, m=M), per_stock_equity={}, per_stock_trades={},
                       portfolio_equity=pe, trades_all=trades)
    wf_alpha, wf_ann = [], []
    for (tr_s, tr_e, te_s, te_e) in windows:
        ptf = OPT._purged_test_metrics(run, te_s, te_e, bench)
        wf_alpha.append(ptf.get("alpha_ann", np.nan))
        wf_ann.append(ptf.get("ann_return", np.nan))
    wf_alpha = pd.Series(wf_alpha).dropna(); wf_ann = pd.Series(wf_ann).dropna()
    seg["wf"] = dict(alpha_med=float(wf_alpha.median()) if len(wf_alpha) else np.nan,
                     ann_med=float(wf_ann.median()) if len(wf_ann) else np.nan,
                     alpha_pos=float((wf_alpha > 0).mean()) if len(wf_alpha) else np.nan,
                     n=len(wf_alpha))
    seg["n_trades"] = len(trades)
    return seg


def main():
    pool = DL.get_universe(verbose=False) if C.USE_UNIVERSE else C.POOL
    stocks = DL.load_pool(pool, verbose=False)
    panel = build_factor_panel(stocks)
    cal = None
    for fdf in panel.values():
        cal = fdf.index if cal is None else cal.union(fdf.index)
    split = OPT.holdout_split_date(cal)
    bench = BM.benchmark_equity(cal, C.BENCHMARK_CODE)
    bench_m = R.metrics_from_equity(bench)

    strats = {"① 裸SuperTrend": "raw", "② ST+量能(原版)": "vol", "③ ST+HMA200(本次)": "hma"}
    results = {name: evaluate(run_strategy(panel, kind), bench, split) for name, kind in strats.items()}

    print("=" * 104)
    print(f"SuperTrend+HMA200 严谨受控测试 | 120只 | 窗口 {cal.min().date()}~{cal.max().date()} | "
          f"切分 {split.date()} | 止损关 | 基准=沪深300全收益")
    print("=" * 104)
    print(f"{'策略':<20}{'段':<6}{'年化':>9}{'最大回撤':>10}{'Sharpe':>8}{'Calmar':>8}"
          f"{'β':>6}{'α(年化)':>9}{'超额':>9}{'PSR':>7}{'交易':>7}")
    for name, seg in results.items():
        for sname in ("full", "IS", "OOS"):
            m = seg[sname]
            nt = f"{seg['n_trades']}" if sname == "full" else ""
            print(f"{name if sname=='full' else '':<20}{sname:<6}"
                  f"{m['ann_return']:>9.2%}{m['max_dd']:>10.2%}{m['sharpe']:>8.2f}{m['calmar']:>8.2f}"
                  f"{m.get('beta',np.nan):>6.2f}{m.get('alpha_ann',np.nan):>9.2%}"
                  f"{m.get('excess_ann',np.nan):>9.2%}{m['psr0']:>7.0%}{nt:>7}")
        wf = seg["wf"]
        print(f"{'':<20}{'WF':<6}  OOS α中位数 {wf['alpha_med']:>7.2%} | OOS年化中位数 {wf['ann_med']:>7.2%} | "
              f"α>0窗口占比 {wf['alpha_pos']:>5.0%}（{wf['n']}窗）")
        print("-" * 104)
    print(f"{'沪深300(全收益)':<20}{'full':<6}{bench_m['ann_return']:>9.2%}{bench_m['max_dd']:>10.2%}"
          f"{bench_m['sharpe']:>8.2f}{bench_m['calmar']:>8.2f}")
    print("=" * 104)
    print("读法：α=剔沪深300后的真本事；PSR=夏普显著>0的概率(>95%才显著)；WF=跨13窗的OOS稳健性。")
    print("  对比②③即'量能 vs HMA 过滤'的纯效果（唯一变量）。")


if __name__ == "__main__":
    main()
