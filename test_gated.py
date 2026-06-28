"""
test_gated.py — Regime 门控对 SuperTrend+HMA 的受控测试

核心假设（用户）：SuperTrend 在【震荡 regime】被反复打脸，只在【趋势 regime】有效。
门控：ER/ADX/R² 三专家 2/3 投票为趋势市，才放行买入；不通过则永不买入（出场不受门控影响）。

受控对比（唯一变量逐个加）：
  ③ HMA（无门控/无止损）          —— 基线
  ④ HMA + 门控（无止损）          —— 隔离“门控”纯效果（验证震荡市失效假设）
  ⑤ HMA + 门控 + 硬止损8%         —— 用户完整方法论（谨慎买入+狠截亏损+让盈利奔跑+果断离场）

参数全部固定（n=10/m=3/HMA=200；门控用文献标准阈值 ER>0.3/ADX>25/R²>0.5），不调参=不引入过拟合。
评估复用 test_hma：α/β vs 沪深300全收益、PSR、walk-forward 逐窗 OOS（purge+embargo）。
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

N, M, HMA_P = 10, 3.0, 200
RISK_OFF = C.RiskParams(use_hard_stop=False, use_atr_trailing=False)
RISK_HARD = C.RiskParams(use_hard_stop=True, hard_stop_pct=0.08, use_atr_trailing=False)  # 狠截亏损+让盈利奔跑


def build_panel(stocks: dict) -> dict:
    params = C.StrategyParams(n=N, m=M, w=10, k=1.5)
    panel = {}
    for c, df in stocks.items():
        fdf = F.compute_factors(df, params).copy()
        fdf["hma"] = F.hma(fdf["close"].to_numpy(float), HMA_P)
        gate, _, _, _, _ = F.regime_gate(fdf)
        fdf["gate"] = gate
        panel[c] = DL.trim_to_window(fdf)
    return panel


def signals(fdf: pd.DataFrame, use_gate: bool, use_hma: bool = True) -> pd.DataFrame:
    entry = fdf["flip_up"].to_numpy(bool)
    if use_hma:
        entry = entry & (fdf["close"].to_numpy(float) > fdf["hma"].to_numpy(float))
    if use_gate:
        entry = entry & fdf["gate"].to_numpy(bool)
    sig = pd.DataFrame(index=fdf.index)
    sig["entry"] = entry
    sig["exit"] = fdf["flip_down"].to_numpy(bool)
    return sig


def run_variant(panel: dict, use_gate: bool, risk, use_hma: bool = True) -> BT.BacktestResult:
    sleeve = C.INIT_CAPITAL / len(panel)
    per = {c: BT.backtest_stock(c, panel[c], signals(panel[c], use_gate, use_hma), sleeve, risk=risk)
           for c in panel}
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
    grate = float(np.mean([fdf["gate"].mean() for fdf in panel.values()]))

    # SuperTrend + HMA + 门控（去掉量价因子），并与裸ST/裸ST+门控对照，看 HMA 在门控框架里的作用
    variants = [
        ("① 裸ST(无门控)", dict(use_gate=False, use_hma=False, risk=RISK_OFF)),
        ("⑥ 裸ST+门控", dict(use_gate=True, use_hma=False, risk=RISK_OFF)),
        ("④ ST+HMA+门控", dict(use_gate=True, use_hma=True, risk=RISK_OFF)),
        ("⑤ ST+HMA+门控+硬止损8%", dict(use_gate=True, use_hma=True, risk=RISK_HARD)),
    ]
    results = {name: TH.evaluate(run_variant(panel, kw["use_gate"], kw["risk"], kw["use_hma"]), bench, split)
               for name, kw in variants}

    print("=" * 100)
    print(f"Regime门控受控测试 | 120只 | 窗口 {cal.min().date()}~{cal.max().date()} | 切分 {split.date()} | "
          f"基准=沪深300全收益")
    print(f"门控放行比例（趋势bar占比，全池均值）：{grate:.0%}")
    print("=" * 100)
    print(f"{'变体':<22}{'段':<6}{'年化':>8}{'最大回撤':>9}{'Sharpe':>7}{'Calmar':>7}"
          f"{'β':>6}{'α(年化)':>9}{'PSR':>6}{'交易':>7}")
    for name, seg in results.items():
        for sname in ("full", "OOS"):
            m = seg[sname]
            nt = f"{seg['n_trades']}" if sname == "full" else ""
            print(f"{name if sname=='full' else '':<22}{sname:<6}"
                  f"{m['ann_return']:>8.2%}{m['max_dd']:>9.2%}{m['sharpe']:>7.2f}{m['calmar']:>7.2f}"
                  f"{m.get('beta',np.nan):>6.2f}{m.get('alpha_ann',np.nan):>9.2%}{m['psr0']:>6.0%}{nt:>7}")
        wf = seg["wf"]
        print(f"{'':<22}{'WF':<6}  OOS α中位 {wf['alpha_med']:>7.2%} | α>0窗口 {wf['alpha_pos']:>4.0%} | "
              f"OOS年化中位 {wf['ann_med']:>6.2%}（{wf['n']}窗）")
        print("-" * 100)
    print("读法：对比 ③→④ 即'加门控'的纯效果——若 WF OOS α 由负转正/明显改善 → 震荡市失效假设成立。")
    print("  ⑤ 再叠加硬止损（你的完整方法论）。PSR>95%/WF α>0 才算真有 edge。")


if __name__ == "__main__":
    main()
