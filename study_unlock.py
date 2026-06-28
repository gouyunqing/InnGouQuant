"""
study_unlock.py — 解禁(限售解禁)事件研究：解禁后是否超跌反弹？

机制假设：限售股解禁 = 被锁股东(IPO前/定增方)到期被迫抛售 → 下行压力/超跌 → 之后反弹。
对手方=不得不卖的解禁股东(纯结构性)。契合本项目认知：A股奖励"被迫抛售后抄底"。

做法（复用低量信号那套事件研究范式）：
  · 抓全窗口解禁日历(akshare 东财)，落到我们120股池，缓存。
  · 用本地前复权价 + 沪深300，算【市场调整后】事件窗收益：
      pre[-20,0]   = 解禁前20日(看抛压下行)
      post[0,+20]  / post[0,+40]  = 解禁后(看反弹)
      post[+5,+25] = 跳过紧邻解禁日的持续抛售，抓延迟反弹
  · 按【解禁占流通市值比例】分档看单调、按【年】看衰减、对0做 t 检验。
判定：post 显著为正(t>2)、且随解禁量单调/未衰减 → 超跌反弹是真结构 edge，值得做策略级终审。

局限：①幸存者偏差(股池只含在市股)；②gross；③事件级≠策略级(过了才谈终审)。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import benchmark as BM

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlock_calendar.csv")


def fetch_calendar(force: bool = False) -> pd.DataFrame:
    if os.path.exists(CACHE) and not force:
        cal = pd.read_csv(CACHE, dtype={"code": str}, parse_dates=["date"])
        print(f"  载入解禁缓存 {CACHE}：{len(cal)} 起")
        return cal
    import akshare as ak
    uni = set(DL.get_universe(verbose=False))
    parts = []
    for y in range(2010, 2027):
        try:
            df = ak.stock_restricted_release_detail_em(start_date=f"{y}0101", end_date=f"{y}1231")
        except Exception as ex:
            print(f"  {y} 抓取失败: {repr(ex)[:80]}"); continue
        df = df[df["股票代码"].astype(str).isin(uni)]
        if len(df):
            parts.append(df[["股票代码", "解禁时间", "占解禁前流通市值比例", "限售股类型"]])
        print(f"  {y}: 落池 {len(df)} 起")
    cal = pd.concat(parts, ignore_index=True)
    cal.columns = ["code", "date", "pct_float", "type"]
    cal["code"] = cal["code"].astype(str)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.dropna(subset=["date"]).sort_values("date")
    cal.to_csv(CACHE, index=False, encoding="utf-8-sig")
    print(f"  已缓存 {CACHE}：{len(cal)} 起")
    return cal


def event_returns(cal: pd.DataFrame, stocks: dict, bench_close: pd.Series) -> pd.DataFrame:
    start, end = DL.backtest_window()
    rows = []
    for code, grp in cal.groupby("code"):
        if code not in stocks:
            continue
        df = stocks[code]; idx = df.index
        cl = df["close"].to_numpy(float)
        bb = bench_close.reindex(idx).ffill().to_numpy(float)
        n = len(cl)

        def madj(a, b):
            if a < 0 or b >= n or bb[a] <= 0 or bb[b] <= 0 or cl[a] <= 0:
                return np.nan
            return (cl[b] / cl[a] - 1) - (bb[b] / bb[a] - 1)

        for _, ev in grp.iterrows():
            d = ev["date"]
            if (start is not None and d < start) or (end is not None and d > end):
                continue
            i = int(idx.searchsorted(d, side="left"))
            if i >= n:
                continue
            rows.append({
                "code": code, "year": idx[i].year, "pct_float": ev["pct_float"],
                "pre_-20_0": madj(i - 20, i),
                "post_0_20": madj(i, i + 20),
                "post_0_40": madj(i, i + 40),
                "post_5_25": madj(i + 5, i + 25),
            })
    return pd.DataFrame(rows)


def _t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else np.nan


def main():
    print("[1] 解禁日历 ...")
    cal = fetch_calendar()
    print("[2] 加载股池价格 + 基准 ...")
    stocks = DL.load_pool(DL.get_universe(verbose=False), verbose=False)
    bench = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"]
    ev = event_returns(cal, stocks, bench)
    ev = ev.dropna(subset=["post_0_20"])

    print("\n" + "=" * 92)
    print(f"解禁事件研究（市场调整后超额）| 事件数 {len(ev)} | 窗口 2010-2026 | 120只")
    print("=" * 92)
    wins = [("pre_-20_0", "解禁前[-20,0]"), ("post_0_20", "解禁后[0,+20]"),
            ("post_0_40", "解禁后[0,+40]"), ("post_5_25", "解禁后[+5,+25]")]
    print(f"{'事件窗':<16}{'均值':>9}{'中位':>9}{'t(vs0)':>9}{'胜率':>8}{'n':>7}")
    for c, nm in wins:
        s = ev[c].dropna()
        print(f"{nm:<16}{s.mean():>9.2%}{s.median():>9.2%}{_t(s):>9.2f}{(s>0).mean():>8.0%}{len(s):>7}")
    print("-" * 92)
    print("按【解禁占流通市值比例】分档（post[0,+20] 与 post[+5,+25]）——量越大反弹越强？")
    ev = ev.copy()
    ev["q"] = pd.qcut(ev["pct_float"].rank(method="first"), 4, labels=["Q1小", "Q2", "Q3", "Q4大"])
    for q in ["Q1小", "Q2", "Q3", "Q4大"]:
        sub = ev[ev["q"] == q]
        print(f"  {q}: 解禁量中位 {sub['pct_float'].median():.1%} | post[0,20] {sub['post_0_20'].mean():>7.2%}(t{_t(sub['post_0_20']):>5.2f}) "
              f"| post[5,25] {sub['post_5_25'].mean():>7.2%}(t{_t(sub['post_5_25']):>5.2f}) | n={len(sub)}")
    print("-" * 92)
    print("分年（post[0,+20] 均值，看是否衰减）：")
    yr = ev.groupby("year")["post_0_20"].agg(["mean", "count"])
    line = "  " + " ".join(f"{int(y)}:{r['mean']:+.1%}" for y, r in yr.iterrows() if r["count"] >= 5)
    print(line)
    print("=" * 92)
    print("判定：post 显著正(t>2)、随解禁量单调、未衰减 → 解禁超跌反弹是真结构edge，进策略级终审。")


if __name__ == "__main__":
    main()
