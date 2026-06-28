"""
test_unlock_ls.py — 解禁反转的【日历时间多空对冲】隔离测试

全市场组合那 +7% α 几乎肯定被 size因子 + 幸存者 吹高。本测试用多空对冲把它们消掉：
  · 多腿 = 解禁后[+5,+25]反弹窗的股票(等权)
  · 空腿 = 同一批 survivor 全市场的等权平均(=等权全市场基准)
  · 价差 = 多腿 − 等权全市场 → 消掉:市场β、小盘size因子、两腿共有的幸存者成分。
  · 按解禁量分档(全/≥5%/≥10%/≥20%)看价差是否随解禁量【单调上升】(=事件级剂量效应在中性化后回归)。

判定：价差显著为正(t>2)且随解禁量单调 → 解禁反转是真 edge；价差≈0 → 那7%全是size+幸存者。
残留偏差：解禁后退市的输家仍只缺在多腿(unlock-specific幸存者)，会让价差仍偏乐观——不可消，标注。
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

ENTRY_LAG, HOLD = 5, 20
FULLCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlock_calendar_full.csv")
PKL = os.path.join(C.CACHE_DIR, "unlock_ret_cache.pkl")   # 收益面板缓存(依赖本地行情,换机重建)


def main():
    cal = pd.read_csv(FULLCACHE, dtype={"code": str}, parse_dates=["date"])
    start, end = DL.backtest_window()
    end = end or cal["date"].max()
    cal = cal[(cal["date"] >= start) & (cal["date"] <= end)]
    bdf = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)
    common = bdf.index[(bdf.index >= start) & (bdf.index <= end)]
    nC = len(common)

    codes = sorted(cal["code"].unique())
    if os.path.exists(PKL):
        ret = pickle.load(open(PKL, "rb"))
        print(f"  载入收益面板缓存：{len(ret)} 只")
    else:
        print(f"  构建收益面板（加载 {len(codes)} 只，首次慢）...")
        ret = {}
        for i, c in enumerate(codes, 1):
            try:
                df = DL.build_stock(c, verbose=False)
            except Exception:
                df = None
            if df is not None and len(df) > 50:
                ret[c] = df["close"].reindex(common).ffill().pct_change().to_numpy()
            if i % 400 == 0:
                print(f"    {i}/{len(codes)} 成功 {len(ret)}")
        pickle.dump(ret, open(PKL, "wb"))
        print(f"  已缓存 {len(ret)} 只")

    mat = np.vstack([ret[c] for c in ret])
    uni_ret = np.nanmean(mat, axis=0)                 # 等权全市场日收益(=空腿/基准)
    uni_ann = np.nanmean(uni_ret) * C.TRADING_DAYS_PER_YEAR
    bench_ret = bdf["close"].reindex(common).ffill().pct_change().to_numpy()
    print(f"\n等权全市场年化 {uni_ann:.2%}  vs  沪深300价格年化 "
          f"{np.nanmean(bench_ret)*C.TRADING_DAYS_PER_YEAR:.2%}  → 二者之差≈小盘/size溢价(就是被误记成α的东西)")

    def long_basket(sub):
        sumr = np.zeros(nC); cnt = np.zeros(nC)
        for _, ev in sub.iterrows():
            c = ev["code"]
            if c not in ret:
                continue
            pos = int(common.searchsorted(ev["date"], side="left")) + ENTRY_LAG
            e, x = pos + 1, min(pos + HOLD, nC - 1)
            if e >= nC or e > x:
                continue
            seg = ret[c][e:x + 1]; v = ~np.isnan(seg)
            sumr[e:x + 1] += np.where(v, seg, 0.0); cnt[e:x + 1] += v
        lr = np.where(cnt > 0, sumr / np.where(cnt > 0, cnt, 1), np.nan)
        return lr, cnt

    def spread_stats(lr, cnt):
        active = cnt > 0
        sp = lr - uni_ret                              # 多空价差(中性化后)
        spa = sp[active & ~np.isnan(sp)]
        sd = spa.std(ddof=1)
        ann = spa.mean() * C.TRADING_DAYS_PER_YEAR
        t = spa.mean() / sd * np.sqrt(len(spa)) if sd > 0 else np.nan
        shp = spa.mean() / sd * np.sqrt(C.TRADING_DAYS_PER_YEAR) if sd > 0 else np.nan
        spser = pd.Series(np.where(active & ~np.isnan(sp), sp, 0.0), index=common)
        wf = [spser[(spser.index >= o) & (spser.index <= e)].mean() * C.TRADING_DAYS_PER_YEAR
              for (_, _, o, e) in OPT.walkforward_windows(common)]
        wf = pd.Series(wf).dropna()
        return ann, t, shp, wf.median(), (wf > 0).mean(), int(cnt[active].mean()) if active.any() else 0

    print("\n" + "=" * 96)
    print("解禁反转 多空对冲隔离（多腿=解禁后[+5,+25]股票，空腿=等权全市场）")
    print("=" * 96)
    print(f"{'解禁量档':<14}{'事件':>7}{'日均持':>7}{'价差年化':>10}{'t值':>8}{'Sharpe':>8}{'WF价差中位':>11}{'WF>0窗':>8}")
    for label, thr in [("全解禁", 0.0), ("≥5%", 0.05), ("≥10%", 0.10), ("≥20%", 0.20)]:
        sub = cal[cal["pct_float"] >= thr]
        lr, cnt = long_basket(sub)
        ann, t, shp, wfm, wfp, brd = spread_stats(lr, cnt)
        print(f"{label:<14}{len(sub):>7}{brd:>7}{ann:>10.2%}{t:>8.2f}{shp:>8.2f}{wfm:>11.2%}{wfp:>8.0%}")
    print("=" * 96)
    print("判定：价差年化显著正(t>2)且随解禁量单调↑ → 解禁反转是size/幸存者之上的真α；价差≈0 → 7%全是混杂。")
    print("（残留：解禁后退市输家仍只缺在多腿，价差仍略偏乐观，不可消。）")


if __name__ == "__main__":
    main()
