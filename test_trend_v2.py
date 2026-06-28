"""
test_trend_v2.py — 方案1：β中性 + regime闸 + 波动率缩放（收获已证实的条件趋势α）

v1+gated 已证实：横截面趋势α是【多空性质、regime切换】的(Q5−Q1 闸开+17%/闸关−15%)，long-only收不干净
(闸关仍赚β)。本测试按业界正道把它做成可收获形态——日频回测、加上我们一直没用的两台引擎：
  · 中性化(消β/size)：① Q5−Q1分位多空(dollar-neutral纯因子)  ② 多Top−空HS300(IF期货代理,可交易)
  · 个股层 inverse-vol 加权(风险平价) + 组合层 vol-target 10%(trailing实现波动定杠杆,封顶2x)=TSMOM引擎
  · regime闸(沪深300<200日线→空仓,再平衡日定)
  · 扣成本(单边15bp×换手)，gross/net都报；验证策略对HS300的β≈0(真中性)
变体对比：A未闸LS / B闸控LS / C闸控+voltgt LS(net) / D闸控+voltgt IF对冲(net) / HS300，外加WF。
判据：净值Sharpe明显高于v1的0.48、β≈0、净扣成本仍正、WF多数OOS窗为正 → 条件α被收住，可落地市场中性。
局限：①gross→net已扣线性成本但未含冲击/做空成本/期货基差;②幸存者偏差仍在;③vol-target用事后实现波动近似。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import benchmark as BM
import optimize as OPT
import test_trend_xs as XS

TARGET_VOL = 0.10
VOL_LB = 60         # 组合实现波动回看(交易日)
STK_VOL_LB = 40     # 个股inverse-vol回看
BETA_LB = 120       # 对冲β回看
LEV_CAP = 2.0
COST_PER_SIDE = 0.0015
SMA_WIN = 200
TD = C.TRADING_DAYS_PER_YEAR


def inv_vol_w(r, codes, asof, lb=STK_VOL_LB):
    sub = r.iloc[max(0, asof - lb):asof][codes]
    vol = sub.std()
    w = (1.0 / vol.replace(0, np.nan)).fillna(0.0)
    s = w.sum()
    return (w / s) if s > 0 else w


def build_daily(panel, composite, sig_days, master):
    """构建未闸未杠杆的日频:Q5−Q1多空、Top多腿、换手。返回Series对齐master。"""
    r = panel.pct_change()
    pos = {d: i for i, d in enumerate(master)}
    nC = len(master)
    longr = np.zeros(nC); shortr = np.zeros(nC); turn = np.zeros(nC)
    prevL = None
    for k in range(len(sig_days) - 1):
        sd, nd = sig_days[k], sig_days[k + 1]
        sp, ep = pos[sd], pos[nd]
        sc = composite.loc[sd].dropna()
        if len(sc) < XS.N_BUCKETS * 3 or sp + 1 >= nC:
            continue
        q = pd.qcut(sc.rank(method="first"), XS.N_BUCKETS, labels=False)
        B5 = sc.index[q == XS.N_BUCKETS - 1].tolist()
        B1 = sc.index[q == 0].tolist()
        wL = inv_vol_w(r, B5, sp + 1); wS = inv_vol_w(r, B1, sp + 1)
        rr = r.iloc[sp + 1:min(ep + 1, nC)]
        lv = rr[B5].mul(wL, axis=1).sum(axis=1).values
        sv = rr[B1].mul(wS, axis=1).sum(axis=1).values
        idx = np.arange(sp + 1, sp + 1 + len(lv))
        longr[idx] = lv; shortr[idx] = sv
        if prevL is not None:
            turn[sp + 1] = len(set(B5) ^ set(prevL)) / (2 * len(B5))
        else:
            turn[sp + 1] = 1.0
        prevL = B5
    longs = pd.Series(longr, index=master)
    ls = pd.Series(longr - shortr, index=master)
    return ls, longs, pd.Series(turn, index=master)


def regime_lever(strat_daily, sig_days, master, bench_full):
    """每再平衡日(用≤sd信息)定:gate(指数>200日线) × lever(voltgt)。返回日频step序列。"""
    sma = bench_full.rolling(SMA_WIN, min_periods=SMA_WIN // 2).mean()
    on_sd = (bench_full > sma).reindex(sig_days).fillna(False)
    realvol = strat_daily.rolling(VOL_LB, min_periods=VOL_LB // 2).std() * np.sqrt(TD)
    rv_sd = realvol.reindex(sig_days).ffill()
    lev_sd = (TARGET_VOL / rv_sd.replace(0, np.nan)).clip(0, LEV_CAP).fillna(0.0)
    gate_lev_sd = lev_sd.where(on_sd, 0.0)
    # 广播到日频:每个交易日取≤它的最近再平衡日的(gate×lever)
    s = pd.Series(gate_lev_sd.values, index=sig_days).reindex(
        master.union(sig_days)).ffill().reindex(master).fillna(0.0)
    return s, on_sd


def apply_strategy(strat_daily, gate_lev, turn, master, lev_for_cost):
    """net = gate_lev*strat − 成本(再平衡日 换手×2腿×单边成本×杠杆)。"""
    gross = gate_lev * strat_daily
    cost = turn * 2 * COST_PER_SIDE * lev_for_cost   # 多空两腿
    return gross - cost


def metrics(daily, bench_r=None, label=""):
    d = daily.dropna()
    ann = d.mean() * TD
    vol = d.std() * np.sqrt(TD)
    shp = ann / vol if vol > 0 else np.nan
    eq = (1 + d).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cal = ann / abs(dd) if dd < 0 else np.nan
    beta = np.nan
    if bench_r is not None:
        b = bench_r.reindex(d.index)
        v = b.var()
        beta = d.cov(b) / v if v > 0 else np.nan
    print(f"  {label:<30} 年化{ann:>+7.2%} Sharpe{shp:>+5.2f} 回撤{dd:>+7.2%} Calmar{cal:>+5.2f} β{beta:>+5.2f}")
    return ann, shp, dd, beta


def main():
    start, end = DL.backtest_window()
    print("[1] 面板 + 因子（复用 v1）")
    codes = DL.get_universe(verbose=False)
    panel = XS.build_panel(codes)
    master = panel.index
    bench_full = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"].reindex(master).ffill()
    bench_r = bench_full.pct_change()

    sig_days = pd.DatetimeIndex(sorted(
        g.index[-1] for _, g in panel.groupby([panel.index.year, panel.index.month])))

    prox = XS.sig_prox_high(panel).rank(axis=1, pct=True)
    mom = XS.sig_momentum(panel).rank(axis=1, pct=True)
    resid = XS.sig_resid_mom(panel, bench_full).rank(axis=1, pct=True)
    composite = (prox + mom + resid) / 3.0

    print("[2] 构建日频多空 / 多腿 / 换手")
    ls, longs, turn = build_daily(panel, composite, sig_days, master)
    # IF对冲多腿:多Top − β·HS300(β滚动估)
    beta_roll = longs.rolling(BETA_LB, min_periods=BETA_LB // 2).cov(bench_r) / \
        bench_r.rolling(BETA_LB, min_periods=BETA_LB // 2).var()
    hl = longs - beta_roll.shift(1).fillna(1.0) * bench_r

    # 限定窗口
    win = (master >= start) & (master <= (end or master.max()))

    print("[3] regime闸 × vol-target")
    gl_ls, on_sd = regime_lever(ls, sig_days, master, bench_full)
    gl_hl, _ = regime_lever(hl, sig_days, master, bench_full)
    lev_ls = (gl_ls / 1.0)  # gate_lev already = gate×lever；成本用其量级近似杠杆
    print(f"    闸开月份占比 {on_sd.mean():.0%}")

    print("\n[4] 变体净值对比（日频，2010+）")
    print("=" * 96)
    metrics(ls[win], bench_r[win], "A 未闸 Q5−Q1多空(gross)")
    metrics((1.0 * (gl_ls > 0) * ls)[win], bench_r[win], "B 闸控 Q5−Q1多空(gross,未杠杆)")
    c_gross = (gl_ls * ls)
    metrics(c_gross[win], bench_r[win], "C 闸控+voltgt LS(gross)")
    c_net = apply_strategy(ls, gl_ls, turn, master, gl_ls.clip(0, LEV_CAP))
    metrics(c_net[win], bench_r[win], "C 闸控+voltgt LS(net扣15bp)")
    d_net = apply_strategy(hl, gl_hl, turn, master, gl_hl.clip(0, LEV_CAP))
    metrics(d_net[win], bench_r[win], "D 闸控+voltgt IF对冲(net)")
    metrics(bench_r[win], bench_r[win], "HS300基准")
    print("=" * 96)

    # WF
    print("\n[5] Walk-forward OOS（C 闸控+voltgt LS net）")
    wins = OPT.walkforward_windows(master[win])
    print(f"    {'OOS窗':<24}{'年化':>9}{'Sharpe':>8}{'回撤':>9}")
    anns, shps = [], []
    for (_, _, os_, oe) in wins:
        seg = (master >= os_) & (master <= oe)
        a, s, dd, _ = (lambda x: x)(metrics_silent(c_net[seg]))
        print(f"    {str(os_.date())+'~'+str(oe.date()):<24}{a:>+9.1%}{s:>+8.2f}{dd:>+9.1%}")
        anns.append(a); shps.append(s)
    print(f"    {'中位':<24}{np.median(anns):>+9.1%}{np.median(shps):>+8.2f}"
          f"   正窗 {np.mean([a>0 for a in anns]):.0%}")
    print("\n判据：C/D 的Sharpe>>0.48、β≈0、net仍正、WF多数窗正 → 条件趋势α被市场中性化收住，方案1成立。")


def metrics_silent(daily):
    d = daily.dropna()
    if len(d) < 5:
        return np.nan, np.nan, np.nan, np.nan
    ann = d.mean() * TD; vol = d.std() * np.sqrt(TD)
    shp = ann / vol if vol > 0 else np.nan
    eq = (1 + d).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return ann, shp, dd, np.nan


if __name__ == "__main__":
    main()
