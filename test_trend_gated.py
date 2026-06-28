"""
test_trend_gated.py — 趋势因子 × 指数级 regime 闸（条件 alpha 能否被收获）

v1 证实：横截面趋势因子无条件 alpha≈0，但 WF 显示强 regime 依赖(趋势市Q5−Q1 +85% vs 震荡市 −43%)。
本测试把"趋势工具只在趋势 regime 用"这条假设摆到【正确的单位=指数】上：
  · regime 闸 = 沪深300 close > 其 200 日均线(Faber GTAA 同款,parameter-light,不调参)。在信号日判定,无前视。
  · 闸开(指数上升趋势)→ 持 Top 档趋势股；闸关(指数跌破200日线)→ 空仓(或退守HS300)。
  · 对比 4 条净值：未闸 Top档 / 闸控Top(关→现金) / 闸控Top(关→HS300) / HS300，外加 Q5−Q1 价差闸前后。
  · 先验证条件分裂是否真实：spread|闸开 vs spread|闸关 的年化差。再看闸控能否把WF稳住。
判据：闸控Top 的 Sharpe/Calmar 显著高于未闸 Top 且回撤大降、WF更稳 → 条件alpha可收获，进v2 sizing。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import benchmark as BM
import optimize as OPT
import test_trend_xs as XS

SMA_WIN = 200


def maxdd(r):
    eq = (1 + r.dropna()).cumprod()
    return (eq / eq.cummax() - 1).min() if len(eq) else np.nan


def stats(r, label, bench=None):
    r = r.dropna()
    ann, shp = XS.ann_from_monthly(r)
    dd = maxdd(r)
    cal = ann / abs(dd) if dd and dd < 0 else np.nan
    line = f"  {label:<26} 年化 {ann:>+7.2%}  Sharpe {shp:>+5.2f}  回撤 {dd:>+7.2%}  Calmar {cal:>+5.2f}"
    print(line)
    return ann, shp, dd


def main():
    start, end = DL.backtest_window()
    end = end or pd.Timestamp("2026-12-31")
    print("[1] 复用 v1 面板 + 综合趋势因子")
    codes = DL.get_universe(verbose=False)
    panel = XS.build_panel(codes)
    master = panel.index
    bench_close = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"]
    bench_full = bench_close.reindex(master).ffill()

    sig_days = pd.DatetimeIndex(sorted(
        g.index[-1] for _, g in panel.groupby([panel.index.year, panel.index.month])))
    bench_m = pd.Series(bench_full.reindex(sig_days).pct_change().shift(-1).values, index=sig_days)

    prox = XS.sig_prox_high(panel).rank(axis=1, pct=True)
    mom = XS.sig_momentum(panel).rank(axis=1, pct=True)
    resid = XS.sig_resid_mom(panel, bench_full).rank(axis=1, pct=True)
    composite = (prox + mom + resid) / 3.0

    B, ic = XS.monthly_eval(composite, panel, sig_days, master, start, end, "综合")
    top = B[f"Q{XS.N_BUCKETS}"]
    spread = B[f"Q{XS.N_BUCKETS}"] - B["Q1"]

    # —— 指数 regime 闸：close > SMA200，信号日判定（防前视）——
    sma = bench_full.rolling(SMA_WIN, min_periods=SMA_WIN // 2).mean()
    on_daily = (bench_full > sma)
    on = on_daily.reindex(B.index).fillna(False)        # 各信号日闸状态
    benchm = bench_m.reindex(B.index)

    print(f"\n[2] 闸状态：{on.mean():.0%} 月份闸开(指数在200日线上方)，{1-on.mean():.0%} 闸关")

    # —— 条件分裂校验 ——
    sp_on = spread[on].mean() * 12
    sp_off = spread[~on].mean() * 12
    top_on = top[on].mean() * 12
    top_off = top[~on].mean() * 12
    print(f"[3] 条件分裂校验（年化）:")
    print(f"    Q5−Q1价差 | 闸开 {sp_on:+.2%}   闸关 {sp_off:+.2%}   差 {sp_on - sp_off:+.2%}")
    print(f"    Top档收益 | 闸开 {top_on:+.2%}   闸关 {top_off:+.2%}   差 {top_on - top_off:+.2%}")

    print("\n[4] 净值对比（月频，2010+，gross）")
    print("=" * 92)
    stats(top, "未闸 Top档", )
    stats(top.where(on, 0.0), "闸控Top(关→现金)")
    stats(top.where(on, benchm), "闸控Top(关→HS300)")
    stats(spread, "未闸 Q5−Q1价差")
    stats(spread.where(on, 0.0), "闸控 Q5−Q1(关→0)")
    stats(benchm, "HS300基准")
    print("=" * 92)

    # —— WF：闸控 Top 逐 OOS 窗 ——
    print("\n[5] Walk-forward OOS（闸控Top 关→现金 vs 未闸Top）")
    wins = OPT.walkforward_windows(master[master >= start])
    print(f"    {'OOS窗':<24}{'未闸年化':>10}{'闸控年化':>10}{'未闸回撤':>10}{'闸控回撤':>10}")
    rg, ru = [], []
    for (_, _, os_, oe) in wins:
        seg = (B.index >= os_) & (B.index <= oe)
        if seg.sum() < 3:
            continue
        tu, tg = top[seg], top[seg].where(on[seg], 0.0)
        au, _ = XS.ann_from_monthly(tu)
        ag, _ = XS.ann_from_monthly(tg)
        print(f"    {str(os_.date())+'~'+str(oe.date()):<24}{au:>+10.1%}{ag:>+10.1%}{maxdd(tu):>+10.1%}{maxdd(tg):>+10.1%}")
        ru.append(au); rg.append(ag)
    print(f"    {'中位':<24}{np.median(ru):>+10.1%}{np.median(rg):>+10.1%}")

    print("\n判据：闸控Top 比未闸 Top 的 Calmar↑、回撤↓、WF更稳 → 指数regime闸把条件alpha收住了，进v2(vol-scaling+sizing)。")


if __name__ == "__main__":
    main()
