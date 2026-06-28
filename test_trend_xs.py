"""
test_trend_xs.py — 横截面相对强度趋势因子 v1（纯选股层验证）

业界调研落地：A股单名价格趋势≈噪声/反转，趋势真正活在【相对强度】口径，且大盘股里
52周高点动量是唯一稳的那一格——正中我们的120大盘池。本测试只做"选什么"，不做
timing/sizing/止损(留v2)，用最干净的对照确认 alpha 在不在：
  · 月频，120大盘池，所有股票共用一套参数 = 纯因子。
  · 综合趋势分 = 截面rank等权平均(52周高点贴近 + 中期动量skip月集成{63,126,252} + 残差动量126d)。
  · 评估：5档forward收益是否【单调】；Q5−Q1多空价差(size中性,消小盘溢价);Top档long-only vs HS300;
        Rank-IC(均值/IR/t);Walk-forward逐OOS窗的IC与价差中位。
  · 消融：每个子信号单独跑，看谁在扛(prox52w / mom / resid)。
局限：①幸存者偏差(仍在,标注);②gross未扣成本/冲击;③暂无行业动量(缺板块数据,留v2)。
防前视：信号用 close[t]，【次日】进场，持有到下个月再平衡的次日(执行滞后1日)。
"""
from __future__ import annotations

import os
import pickle
import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import benchmark as BM
import optimize as OPT

HOLD_LABEL = "21d≈1月"
PANEL_PKL = os.path.join(C.CACHE_DIR, "trend_xs_panel.pkl")   # 价格面板缓存(依赖本地行情,换机重建)
N_BUCKETS = 5


# --------------------------------------------------------------------------- #
# 价格面板（前复权 close，全历史，列=股票）
# --------------------------------------------------------------------------- #
def build_panel(codes):
    if os.path.exists(PANEL_PKL):
        panel = pickle.load(open(PANEL_PKL, "rb"))
        print(f"  载入价格面板缓存：{panel.shape[1]} 只 × {panel.shape[0]} 日")
        return panel
    print(f"  构建价格面板（加载 {len(codes)} 只，首次慢）...")
    series = {}
    for i, c in enumerate(codes, 1):
        try:
            df = DL.build_stock(c, verbose=False)
        except Exception:
            df = None
        if df is not None and len(df) > 300:
            series[c] = df["close"].astype(float)
        if i % 40 == 0:
            print(f"    {i}/{len(codes)} 成功 {len(series)}")
    panel = pd.DataFrame(series).sort_index()
    pickle.dump(panel, open(PANEL_PKL, "wb"))
    print(f"  已缓存 {panel.shape[1]} 只 × {panel.shape[0]} 日")
    return panel


# --------------------------------------------------------------------------- #
# 三个子信号（全历史因果计算，列=股票；越大=趋势越强）
# --------------------------------------------------------------------------- #
def sig_prox_high(close, ns=(120, 250)):
    """52周(及半年)高点贴近度 = close / rolling_max。集成多窗取均值。"""
    parts = [close / close.rolling(n, min_periods=n // 2).max() for n in ns]
    return sum(parts) / len(parts)


def sig_momentum(close, lbs=(63, 126, 252), skip=21):
    """中期动量，剔除最近 skip 日(躲短期反转毒区)。集成多 lookback。"""
    parts = [close.shift(skip) / close.shift(lb) - 1.0 for lb in lbs]
    return sum(parts) / len(parts)


def sig_resid_mom(close, bench_close, win=126, beta_win=252):
    """残差动量：对沪深300滚动回归取 β，剔除市场后特质收益累计。"""
    r = close.pct_change()
    rm = bench_close.reindex(close.index).ffill().pct_change()
    var_m = rm.rolling(beta_win, min_periods=beta_win // 2).var()
    out = {}
    for col in close.columns:
        cov = r[col].rolling(beta_win, min_periods=beta_win // 2).cov(rm)
        beta = cov / var_m
        resid = r[col] - beta * rm
        out[col] = resid.rolling(win, min_periods=win // 2).sum()
    return pd.DataFrame(out, index=close.index)


# --------------------------------------------------------------------------- #
# 截面 rank 工具
# --------------------------------------------------------------------------- #
def xs_rank(row):
    """单日截面分位 rank ∈ [0,1]（NaN 保持 NaN）。"""
    return row.rank(pct=True)


def spearman(a, b):
    """rank-IC：先排名再 Pearson（不依赖 scipy）。"""
    m = (~a.isna()) & (~b.isna())
    if m.sum() < 8:
        return np.nan
    ar, br = a[m].rank(), b[m].rank()
    return ar.corr(br)


# --------------------------------------------------------------------------- #
# 月频再平衡 → 各档 forward 收益 / IC
# --------------------------------------------------------------------------- #
def monthly_eval(score, panel, sig_days, master, start, end, label):
    """score: 截面分(DataFrame, index=日, 列=股票)。返回各档月收益矩阵、月度IC、价差。"""
    pos = {d: i for i, d in enumerate(master)}
    bucket_rows, ic_rows, periods = [], [], []
    for k in range(len(sig_days) - 1):
        sd, nd = sig_days[k], sig_days[k + 1]
        if sd < start or sd > end:
            continue
        ip, xp = pos[sd] + 1, pos[nd] + 1            # 次日进场，下月次日出场（防前视）
        if xp >= len(master):
            continue
        sc = score.loc[sd]
        fwd = panel.iloc[xp] / panel.iloc[ip] - 1.0  # 各股本期forward收益
        valid = (~sc.isna()) & (~fwd.isna())
        sc, fwd = sc[valid], fwd[valid]
        if len(sc) < N_BUCKETS * 3:
            continue
        q = pd.qcut(sc.rank(method="first"), N_BUCKETS, labels=False)
        brow = [fwd[q == b].mean() for b in range(N_BUCKETS)]
        bucket_rows.append(brow)
        ic_rows.append(spearman(sc, fwd))
        periods.append(sd)
    B = pd.DataFrame(bucket_rows, index=pd.DatetimeIndex(periods),
                     columns=[f"Q{b+1}" for b in range(N_BUCKETS)])
    ic = pd.Series(ic_rows, index=pd.DatetimeIndex(periods)).dropna()
    return B, ic


def ann_from_monthly(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan, np.nan
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    shp = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    return ann, shp


def report(B, ic, label, bench_m=None):
    spread = B[f"Q{N_BUCKETS}"] - B["Q1"]
    mono = "↑单调" if list(B.mean().values) == sorted(B.mean().values) else "非单调"
    print(f"\n【{label}】 月数 {len(B)}  Rank-IC 均值 {ic.mean():+.4f}  IC-IR {ic.mean()/ic.std():+.2f}  "
          f"IC-t {ic.mean()/ic.std()*np.sqrt(len(ic)):+.2f}  IC>0占比 {(ic>0).mean():.0%}")
    avg = B.mean() * 12
    print("  各档年化(等权,forward):  " + "  ".join(f"{c} {avg[c]:+.1%}" for c in B.columns) + f"   [{mono}]")
    sp_ann, sp_shp = ann_from_monthly(spread)
    tq_ann, tq_shp = ann_from_monthly(B[f"Q{N_BUCKETS}"])
    line = (f"  Q{N_BUCKETS}−Q1价差(size中性): 年化 {sp_ann:+.2%}  Sharpe {sp_shp:+.2f}   |   "
            f"Top档long-only: 年化 {tq_ann:+.2%}  Sharpe {tq_shp:+.2f}")
    if bench_m is not None:
        bret = bench_m.reindex(B.index).dropna()
        b_ann, _ = ann_from_monthly(bret)
        line += f"   |   HS300同期 年化 {b_ann:+.2%}"
    print(line)
    return spread


def walkforward(score, panel, sig_days, master, label):
    """逐 OOS 窗(walkforward_windows)报 IC 与 Q5−Q1 价差年化。"""
    wins = OPT.walkforward_windows(master[(master >= DL.backtest_window()[0])])
    rows = []
    for (_, _, os_, oe) in wins:
        B, ic = monthly_eval(score, panel, sig_days, master, os_, oe, label)
        if len(B) < 3:
            continue
        sp = (B[f"Q{N_BUCKETS}"] - B["Q1"])
        sp_ann, _ = ann_from_monthly(sp)
        rows.append((os_.date(), oe.date(), len(B), ic.mean(), sp_ann))
    print(f"\n  Walk-forward OOS（{label}）:")
    print(f"    {'OOS窗':<26}{'月':>4}{'IC均值':>10}{'Q5-Q1年化':>12}")
    ics, sps = [], []
    for (a, b, n, icm, spa) in rows:
        print(f"    {str(a)+'~'+str(b):<26}{n:>4}{icm:>+10.4f}{spa:>+12.2%}")
        ics.append(icm); sps.append(spa)
    if ics:
        print(f"    {'中位/占正':<26}{'':>4}{np.median(ics):>+10.4f}{np.median(sps):>+12.2%}"
              f"   IC>0窗 {np.mean([i>0 for i in ics]):.0%} | 价差>0窗 {np.mean([s>0 for s in sps]):.0%}")


# --------------------------------------------------------------------------- #
def main():
    start, end = DL.backtest_window()
    end = end or pd.Timestamp("2026-12-31")
    print("[1] 股池 + 价格面板")
    codes = DL.get_universe(verbose=False)
    panel = build_panel(codes)
    master = panel.index
    bench = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"]
    bench_full = bench.reindex(master).ffill()

    # 月末信号日（全历史，供 lookback；评估只取窗口内）
    sig_days = [g.index[-1] for _, g in panel.groupby([panel.index.year, panel.index.month])]
    sig_days = pd.DatetimeIndex(sorted(sig_days))
    # 月度基准收益（用于 long-only 对照）
    bser = bench_full.reindex(sig_days).pct_change().shift(-1)  # sd_m→sd_{m+1} 近似
    bench_m = pd.Series(bser.values, index=sig_days)

    print(f"    股票 {panel.shape[1]} 只，交易日 {len(master)}，月末信号日 {len(sig_days)}")
    print(f"    回测窗口 {start.date()} ~ {end.date()}，持有 {HOLD_LABEL}")

    print("\n[2] 计算三个子信号（全历史因果，越大趋势越强）")
    prox = sig_prox_high(panel)
    mom = sig_momentum(panel)
    resid = sig_resid_mom(panel, bench_full)

    # 截面 rank（逐日）
    print("[3] 截面 rank + 综合分（等权平均三子信号的截面分位）")
    R_prox = prox.rank(axis=1, pct=True)
    R_mom = mom.rank(axis=1, pct=True)
    R_resid = resid.rank(axis=1, pct=True)
    composite = (R_prox + R_mom + R_resid) / 3.0

    print("\n" + "=" * 100)
    print("综合趋势因子（52周高点贴近 + 中期动量skip月集成 + 残差动量，截面rank等权）")
    print("=" * 100)
    B, ic = monthly_eval(composite, panel, sig_days, master, start, end, "综合")
    report(B, ic, "综合趋势因子", bench_m)
    walkforward(composite, panel, sig_days, master, "综合")

    print("\n" + "=" * 100)
    print("消融：各子信号单独（看谁在扛）")
    print("=" * 100)
    for nm, sc in [("52周高点贴近 prox", R_prox), ("中期动量 mom(skip月,集成)", R_mom),
                   ("残差动量 resid(126d)", R_resid)]:
        Bs, ics = monthly_eval(sc, panel, sig_days, master, start, end, nm)
        report(Bs, ics, nm, bench_m)

    print("\n" + "=" * 100)
    print("判定：综合因子 IC-t>2 且各档单调↑、Q5−Q1价差年化显著正、WF多数OOS窗为正 → 相对强度趋势alpha成立，进v2(加timing+vol-scaling)。")
    print("      若 Q5−Q1≈0 或 IC≈0 → 即便相对强度口径，这个大盘池里趋势也washout，需另寻单位(行业/残差加权)。")


if __name__ == "__main__":
    main()
