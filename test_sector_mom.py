"""
test_sector_mom.py — 行业/板块动量（A股调研里最强的趋势效应，v1 未测的那条腿）

调研：A股"行业动量在日、月频非常明显""行业内效应是大多数异象的驱动力"。单股价格趋势washout，
但趋势可能活在【板块】这个单位。本测试把趋势的单位从单股换成行业：
  · 拉 120 池每只股票的所属行业(akshare,缓存)。按行业建等权板块指数。
  · 板块动量 = 板块EW指数 trailing 收益(skip月,集成{63,126,252})。个股信号 = 其所属板块的动量。
  · 用与 v1 同一台 monthly_eval：5档forward收益单调性 + Rank-IC + Q5−Q1价差 + Top档long-only。
  · 再叠指数regime闸看条件版。
判据：板块动量 IC-t>2 且 Q5−Q1价差显著正、单调 → 趋势活在行业单位，落地为行业轮动；否则washout。
局限：板块仅由我们120大盘池成员构成(非全行业)，幸存者偏差仍在。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import benchmark as BM
import test_trend_xs as XS

SECTOR_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sector_map.csv")


FULL_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sector_map_full.csv")


def fetch_sector_map(codes):
    """读全市场 申万一级 映射缓存(fetch_sectors.py 产)，取本股池子集。"""
    m = pd.read_csv(FULL_CSV, dtype={"code": str})
    full = {str(c).zfill(6): s for c, s in zip(m["code"], m["sector"])}
    hit = sum(str(c).zfill(6) in full for c in codes)
    print(f"  全市场映射 {len(full)} 只 / {m['sector'].nunique()} 行业；本池命中 {hit}/{len(codes)}")
    return {c: full.get(str(c).zfill(6), "未知") for c in codes}


def build_sector_signal(panel, sector_map, lbs=(63, 126, 252), skip=21):
    """每个板块建等权指数→trailing动量(集成)→广播回成员股票。返回 score(日×股票)。"""
    r = panel.pct_change()
    sectors = {}
    for c in panel.columns:
        sectors.setdefault(sector_map.get(c, "未知"), []).append(c)
    sec_mom = {}
    for sec, members in sectors.items():
        sec_ret = r[members].mean(axis=1)                 # 等权板块日收益
        sec_idx = (1 + sec_ret).cumprod()
        parts = [sec_idx.shift(skip) / sec_idx.shift(lb) - 1.0 for lb in lbs]
        sec_mom[sec] = sum(parts) / len(parts)
    # 广播：每只股票取其板块动量
    score = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
    for c in panel.columns:
        score[c] = sec_mom[sector_map.get(c, "未知")]
    return score, {s: len(m) for s, m in sectors.items()}


def main():
    start, end = DL.backtest_window()
    end = end or pd.Timestamp("2026-12-31")
    print("[1] 股池 + 面板 + 行业映射")
    codes = DL.get_universe(verbose=False)
    panel = XS.build_panel(codes)
    master = panel.index
    bench_full = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"].reindex(master).ffill()
    sector_map = fetch_sector_map(list(panel.columns))
    nsec = len(set(sector_map.get(c, "未知") for c in panel.columns))
    print(f"    {panel.shape[1]} 只覆盖 {nsec} 个行业")

    sig_days = pd.DatetimeIndex(sorted(
        g.index[-1] for _, g in panel.groupby([panel.index.year, panel.index.month])))
    bench_m = pd.Series(bench_full.reindex(sig_days).pct_change().shift(-1).values, index=sig_days)

    print("\n[2] 板块动量信号（等权板块指数 trailing 动量，集成多窗，广播回个股）")
    score, sizes = build_sector_signal(panel, sector_map)
    big = sorted(sizes.items(), key=lambda kv: -kv[1])[:8]
    print("    行业成员数(前8): " + "  ".join(f"{s}{n}" for s, n in big))
    score_rank = score.rank(axis=1, pct=True)

    print("\n" + "=" * 100)
    print("行业/板块动量因子（个股=所属板块动量，截面rank分5档）")
    print("=" * 100)
    B, ic = XS.monthly_eval(score_rank, panel, sig_days, master, start, end, "行业动量")
    XS.report(B, ic, "行业动量", bench_m)
    XS.walkforward(score_rank, panel, sig_days, master, "行业动量")

    # 叠指数 regime 闸
    print("\n[3] 叠加指数regime闸（沪深300>200日线）")
    sma = bench_full.rolling(200, min_periods=100).mean()
    on = (bench_full > sma).reindex(B.index).fillna(False)
    top = B[f"Q{XS.N_BUCKETS}"]; spread = B[f"Q{XS.N_BUCKETS}"] - B["Q1"]
    a1, s1 = XS.ann_from_monthly(top)
    a2, s2 = XS.ann_from_monthly(top.where(on, 0.0))
    sp_on = spread[on].mean() * 12; sp_off = spread[~on].mean() * 12
    print(f"    Top档: 未闸 年化{a1:+.2%}/Sharpe{s1:+.2f}   闸控(关→现金) 年化{a2:+.2%}/Sharpe{s2:+.2f}")
    print(f"    Q5−Q1价差: 闸开{sp_on:+.2%}  闸关{sp_off:+.2%}  差{sp_on-sp_off:+.2%}")

    print("\n判据：板块动量 IC-t>2、Q5−Q1显著正且单调、且闸开闸关分裂明显 → 趋势活在行业单位，可落地行业轮动。")


if __name__ == "__main__":
    main()
