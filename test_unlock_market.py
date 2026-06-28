"""
test_unlock_market.py — 解禁超跌反弹【全市场宽股池】终审（日历时间组合）

120只大盘上解禁反转事件级真、组合级washout(太稀疏/肥尾)。本测试搬到全市场看广度能否救活：
  · 抓全市场解禁日历(主板+创业板，剔688/北交所)，筛大解禁(占流通≥THR)。
  · 加载所有涉及股票的前复权收盘。
  · 【日历时间组合】：每天等权持有所有处于解禁后[+ENTRY_LAG, +ENTRY_LAG+HOLD]反弹窗的股票，
    事件密集时同时持几十上百只 → 肥尾被分散。组合日收益=当日在持股票日收益等权均值，空仓日=0。
  · 评估：α/β vs 沪深300全收益、Sharpe、最大回撤、PSR、walk-forward 逐窗 OOS α、日均在持只数(广度)。

对比 120只稀疏版(WF α≈0)：广度上来后 WF OOS α 是否转正/显著。
局限：①幸存者偏差(解禁后退市崩盘样本缺，偏乐观)；②gross(未扣成本)。
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import benchmark as BM
import report as R
import optimize as OPT

THR = 0.05
ENTRY_LAG, HOLD = 5, 20
FULLCACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlock_calendar_full.csv")


def _valid(code: str) -> bool:
    if len(code) != 6 or not code.isdigit() or code.startswith("688"):
        return False
    return code[:3] in C.VALID_SH_PREFIX or code[:3] in C.VALID_SZ_PREFIX


def fetch_full(force: bool = False) -> pd.DataFrame:
    if os.path.exists(FULLCACHE) and not force:
        cal = pd.read_csv(FULLCACHE, dtype={"code": str}, parse_dates=["date"])
        print(f"  载入全市场解禁缓存：{len(cal)} 起")
        return cal
    import akshare as ak
    parts = []
    for y in range(2010, 2027):
        try:
            df = ak.stock_restricted_release_detail_em(start_date=f"{y}0101", end_date=f"{y}1231")
        except Exception as ex:
            print(f"  {y} 失败 {repr(ex)[:60]}"); continue
        df = df[["股票代码", "解禁时间", "占解禁前流通市值比例"]].copy()
        df.columns = ["code", "date", "pct_float"]
        df["code"] = df["code"].astype(str)
        df = df[df["code"].map(_valid)]
        parts.append(df)
        print(f"  {y}: {len(df)} 起(主板+创业板)")
    cal = pd.concat(parts, ignore_index=True)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.dropna(subset=["date", "pct_float"]).sort_values("date")
    cal.to_csv(FULLCACHE, index=False, encoding="utf-8-sig")
    print(f"  已缓存 {FULLCACHE}：{len(cal)} 起")
    return cal


def load_closes(codes: list, verbose: bool = True) -> dict:
    out = {}
    for i, c in enumerate(codes, 1):
        try:
            df = DL.build_stock(c, verbose=False)
        except Exception:
            df = None
        if df is not None and len(df) > 50:
            out[c] = df["close"].astype(float)
        if verbose and i % 300 == 0:
            print(f"    加载 {i}/{len(codes)}，成功 {len(out)}")
    return out


def build_portfolio(cal: pd.DataFrame, closes: dict, common: pd.DatetimeIndex):
    nC = len(common)
    sumr = np.zeros(nC); cnt = np.zeros(nC)
    ret_cache = {}
    used = 0
    for _, ev in cal.iterrows():
        code = ev["code"]
        if code not in closes:
            continue
        if code not in ret_cache:
            ret_cache[code] = closes[code].reindex(common).ffill().pct_change().to_numpy()
        r = ret_cache[code]
        pos = int(common.searchsorted(ev["date"], side="left")) + ENTRY_LAG
        e, x = pos + 1, min(pos + HOLD, nC - 1)        # 持有期收益 [entry+1, exit]
        if e >= nC or e > x:
            continue
        seg = r[e:x + 1]
        valid = ~np.isnan(seg)
        sumr[e:x + 1] += np.where(valid, seg, 0.0)
        cnt[e:x + 1] += valid
        used += 1
    port_ret = np.where(cnt > 0, sumr / np.where(cnt > 0, cnt, 1), 0.0)
    equity = pd.Series((1.0 + port_ret).cumprod(), index=common, name="unlock_mkt")
    return equity, cnt, used


def evaluate(equity, bench_eq, label):
    m = R.metrics_from_equity(equity)
    mb = R.metrics_vs_benchmark(equity, bench_eq)
    psr = R.deflated_sharpe_ratio(equity.pct_change(), [m["sharpe"]], n_trials=1).get("psr0", np.nan)
    # walk-forward OOS α
    wf = []
    for (ts, te, os_, oe) in OPT.walkforward_windows(equity.index):
        a = R.metrics_vs_benchmark(R.slice_equity(equity, os_, oe), R.slice_equity(bench_eq, os_, oe))
        wf.append(a.get("alpha_ann", np.nan))
    wf = pd.Series(wf).dropna()
    print(f"[{label}] 年化{m['ann_return']:.2%} 回撤{m['max_dd']:.2%} Sharpe{m['sharpe']:.2f} "
          f"Calmar{m['calmar']:.2f} β{mb['beta']:.2f} α{mb['alpha_ann']:.2%} 超额{mb['excess_ann']:.2%} PSR{psr:.0%}")
    print(f"        WF OOS α中位 {wf.median():.2%} | α>0窗 {(wf>0).mean():.0%}（{len(wf)}窗）")


def main():
    print("[1] 全市场解禁日历 ...")
    cal = fetch_full()
    start, end = DL.backtest_window()
    cal = cal[(cal["date"] >= start) & (cal["date"] <= (end or cal["date"].max()))]
    big = cal[cal["pct_float"] >= THR]
    print(f"    窗口内: 全解禁 {len(cal)} 起 / 大解禁(≥{THR:.0%}) {len(big)} 起 | 大解禁涉及股票 {big['code'].nunique()} 只")

    print("[2] 加载相关股票收盘（前复权）...")
    codes = sorted(big["code"].unique())
    closes = load_closes(codes)
    print(f"    成功加载 {len(closes)}/{len(codes)} 只")

    bench_eq = BM.benchmark_equity(pd.DatetimeIndex(sorted(set().union(*[c.index for c in list(closes.values())[:50]]))), C.BENCHMARK_CODE) if False else None
    # 公共交易日历 = 沪深300 窗口内日期
    bdf = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)
    common = bdf.index[(bdf.index >= start) & (bdf.index <= (end or bdf.index.max()))]
    bench_eq = BM.benchmark_equity(common, C.BENCHMARK_CODE)

    print("[3] 日历时间组合 + 评估 ...")
    print("=" * 96)
    eqB, cntB, usedB = build_portfolio(big, closes, common)
    print(f"日均在持只数(大解禁): {cntB[cntB>0].mean():.0f}（最多 {int(cntB.max())}），用到事件 {usedB}")
    evaluate(eqB, bench_eq, f"大解禁≥{THR:.0%}")
    print("-" * 96)
    eqA, cntA, usedA = build_portfolio(cal, closes, common)  # 全解禁(同一批closes,小解禁的票可能没load)
    print(f"日均在持只数(全解禁，仅已加载票): {cntA[cntA>0].mean():.0f}")
    evaluate(eqA, bench_eq, "全解禁(参考)")
    print("=" * 96)
    print("判定：大解禁版 WF OOS α 转正/显著、且 PSR 高 → 广度救活了解禁反转，可落地为真策略。")


if __name__ == "__main__":
    main()
