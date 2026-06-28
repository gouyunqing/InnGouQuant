"""
optimize_hma.py — SuperTrend + HMA200 策略的【因子式】网格寻优 + 严谨评估

策略：入场 = SuperTrend 翻多 且 close > HMA(period)；出场 = SuperTrend 翻空（止损关）。
网格：n∈{7,10,14} × m∈{2,2.5,3,3.5} × HMA∈{50,100,150,200,250} = 60 组。

【关键】因子口径：全部 120 只股票【共用同一组参数】（不是一只一套）。寻优选择分 =
跨股 Calmar 中位数（抗单股暴富），选出唯一一组全局参数 —— 这与现有主网格的做法一致。

复用现有评估机器：score_run / walk-forward(purge+embargo) / _wf_summary / DSR / PBO / α-β。
"""
from __future__ import annotations

import os
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import backtest as BT
import report as R
import benchmark as BM
import optimize as OPT

RISK_OFF = C.RiskParams(use_hard_stop=False, use_atr_trailing=False)   # 出场只看 SuperTrend

HMA_GRID = dict(n=[7, 10, 14], m=[2.0, 2.5, 3.0, 3.5], hma=[50, 100, 150, 200, 250])


@dataclass
class HMAParams:
    n: int
    m: float
    hma: int

    def key(self) -> str:
        return f"n{self.n}_m{self.m}_hma{self.hma}"


def _factor_df(cache: OPT.FactorCache, hma_cache: dict, code: str, p: HMAParams) -> pd.DataFrame:
    df = cache.stocks[code]
    st = cache.supertrend(code, p.n, p.m)                 # 缓存：只依赖 (n,m)
    hkey = (code, p.hma)
    if hkey not in hma_cache:
        hma_cache[hkey] = F.hma(df["close"].to_numpy(float), p.hma)   # 缓存：只依赖 period
    fdf = pd.DataFrame(index=df.index)
    for col in ["open", "high", "low", "close", "one_word_up", "one_word_down"]:
        fdf[col] = df[col]
    fdf["flip_up"] = st["flip_up"]
    fdf["flip_down"] = st["flip_down"]
    fdf["hma"] = hma_cache[hkey]
    return DL.trim_to_window(fdf)


def _signals(fdf: pd.DataFrame) -> pd.DataFrame:
    entry = fdf["flip_up"].to_numpy(bool) & (fdf["close"].to_numpy(float) > fdf["hma"].to_numpy(float))
    sig = pd.DataFrame(index=fdf.index)
    sig["entry"] = entry
    sig["exit"] = fdf["flip_down"].to_numpy(bool)
    return sig


def run_combo(cache: OPT.FactorCache, hma_cache: dict, p: HMAParams) -> OPT.ComboRun:
    codes = list(cache.stocks.keys())
    sleeve = C.INIT_CAPITAL / len(codes)
    per = {}
    for c in codes:
        fdf = _factor_df(cache, hma_cache, c, p)
        per[c] = BT.backtest_stock(c, fdf, _signals(fdf), sleeve, risk=RISK_OFF)
    cal = None
    for r in per.values():
        cal = r.equity.index if cal is None else cal.union(r.equity.index)
    parts = [r.equity.reindex(cal).ffill().fillna(sleeve) for r in per.values()]
    pe = pd.concat(parts, axis=1).sum(axis=1); pe.name = "portfolio"
    res = BT.BacktestResult(per_stock=per, portfolio_equity=pe, calendar=cal, init_capital=C.INIT_CAPITAL)
    trades_all = BT.trades_to_frame(res)
    return OPT.ComboRun(
        params=p,
        per_stock_equity={c: per[c].equity for c in codes},
        per_stock_trades={c: trades_all[trades_all["code"] == c] for c in codes},
        portfolio_equity=pe, trades_all=trades_all)


def _seg_metrics(pe: pd.Series, trades: pd.DataFrame, bench: pd.Series, s, e) -> dict:
    eq = R.slice_equity(pe, s, e)
    m = R.compute_metrics(eq, R.slice_trades(trades, s, e), benchmark=R.slice_equity(bench, s, e))
    m["psr0"] = R.deflated_sharpe_ratio(eq.pct_change(), [m["sharpe"]], n_trials=1).get("psr0", np.nan)
    return m


def main():
    pool = DL.get_universe(verbose=False) if C.USE_UNIVERSE else C.POOL
    stocks = DL.load_pool(pool, verbose=False)
    cache = OPT.FactorCache(stocks)
    hma_cache: dict = {}
    combos = [HMAParams(n, m, h) for n, m, h in
              itertools.product(HMA_GRID["n"], HMA_GRID["m"], HMA_GRID["hma"])]
    print(f"预回测 {len(combos)} 组 × {len(stocks)} 只（全股共用一组参数，止损关）...")
    runs = []
    for i, p in enumerate(combos, 1):
        runs.append(run_combo(cache, hma_cache, p))
        if i % 12 == 0 or i == len(combos):
            print(f"  ...{i}/{len(combos)}")

    cal = OPT._calendar(stocks)
    bench = BM.benchmark_equity(cal, C.BENCHMARK_CODE)
    split = OPT.holdout_split_date(cal)

    # ---- holdout 选参（跨股 Calmar 中位数）----
    best = None
    rows = []
    for run in runs:
        res = OPT.score_run(run, (None, split), (split, None), benchmark=bench)
        rows.append({"params": run.params.key(), "score": res["score"],
                     "is_calmar": res["train_portfolio"].get("calmar", np.nan),
                     "oos_ann": res["test_portfolio"].get("ann_return", np.nan),
                     "oos_alpha": res["test_portfolio"].get("alpha_ann", np.nan)})
        if best is None or res["score"] > best["score"]:
            best = res
    bp = best["params"]

    # ---- walk-forward（逐窗重选 + purged OOS）----
    win_rows = []
    for (tr_s, tr_e, te_s, te_e) in OPT.walkforward_windows(cal):
        b = None
        for run in runs:
            res = OPT.score_run(run, (tr_s, tr_e), (te_s, te_e), benchmark=bench)
            if b is None or res["score"] > b["score"]:
                b = res
        brun = next(r for r in runs if r.params.key() == b["params"].key())
        ptf = OPT._purged_test_metrics(brun, te_s, te_e, bench)
        win_rows.append({"best": b["params"].key(),
                         "oos_ann": ptf.get("ann_return", np.nan), "oos_sharpe": ptf.get("sharpe", np.nan),
                         "oos_calmar": ptf.get("calmar", np.nan), "oos_alpha": ptf.get("alpha_ann", np.nan),
                         "oos_beta": ptf.get("beta", np.nan), "oos_excess": ptf.get("excess_ann", np.nan)})
    wf = OPT._wf_summary(pd.DataFrame(win_rows))

    # ---- DSR + PBO ----
    trial_sharpes = [R.metrics_from_equity(r.portfolio_equity).get("sharpe", np.nan) for r in runs]
    brun = next(r for r in runs if r.params.key() == bp.key())
    dsr = R.deflated_sharpe_ratio(brun.portfolio_equity.pct_change(), trial_sharpes)
    ret_mat = pd.concat({r.params.key(): r.portfolio_equity.pct_change() for r in runs}, axis=1)
    pbo = R.probability_of_backtest_overfitting(ret_mat, n_blocks=C.PBO_BLOCKS)

    # ---- best 的 IS/OOS/full α-β ----
    seg = {nm: _seg_metrics(brun.portfolio_equity, brun.trades_all, bench, s, e)
           for nm, (s, e) in (("full", (None, None)), ("IS", (None, split)), ("OOS", (split, None)))}

    print("\n" + "=" * 92)
    print(f"SuperTrend+HMA 因子式网格寻优（{len(combos)}组，全股共用一组参数，止损关）")
    print(f"窗口 {cal.min().date()}~{cal.max().date()} | 切分 {split.date()} | 基准=沪深300全收益")
    print("=" * 92)
    print(f"holdout 最优参数：{bp.key()}（n={bp.n}, m={bp.m}, HMA={bp.hma}）")
    print(f"{'段':<6}{'年化':>9}{'最大回撤':>10}{'Sharpe':>8}{'Calmar':>8}{'β':>6}{'α(年化)':>9}{'超额':>9}{'PSR':>7}")
    for nm in ("IS", "OOS", "full"):
        m = seg[nm]
        print(f"{nm:<6}{m['ann_return']:>9.2%}{m['max_dd']:>10.2%}{m['sharpe']:>8.2f}{m['calmar']:>8.2f}"
              f"{m.get('beta',np.nan):>6.2f}{m.get('alpha_ann',np.nan):>9.2%}{m.get('excess_ann',np.nan):>9.2%}"
              f"{m['psr0']:>7.0%}")
    print("-" * 92)
    print("Walk-forward（主判据）：")
    print(f"  OOS α中位数 {wf['oos_alpha_median']:.2%} | OOS年化中位数 {wf['oos_ann_median']:.2%} | "
          f"α>0窗口 {wf['oos_alpha_positive_rate']:.0%} | {wf['n_windows']}窗")
    print(f"  参数稳定性：选出 {wf['param_unique']} 种不同最优（众数 {wf['modal_param']} 占 {wf['param_stability']:.0%}）")
    print(f"  判定：{wf['verdict']}")
    print("-" * 92)
    print(f"抗过拟合：DSR={dsr['dsr']:.0%}（观测夏普{dsr['sr_obs_ann']:.2f} vs 运气门槛{dsr['sr_star_ann']:.2f}，≥95%才显著） | "
          f"PBO={pbo['pbo']:.0%}（>50%=高过拟合）")
    print("=" * 92)

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(C.OUTPUT_DIR, "grid_hma.csv"), index=False, encoding="utf-8-sig")
    print(f"网格明细已存：{os.path.join(C.OUTPUT_DIR, 'grid_hma.csv')}")


if __name__ == "__main__":
    main()
