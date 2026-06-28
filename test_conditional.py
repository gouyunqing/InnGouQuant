"""
test_conditional.py — 锐利条件检验（事件研究）：regime 能否预测 SuperTrend 入场质量？

绕开组合回测（去掉现金拖累/仓位/出场/成本等污染），只问一件最纯粹的事：
  在每一次 SuperTrend 买入信号(flip_up)的那一刻，"当时是不是趋势 regime"，
  能不能预测这笔买入【之后】涨得好不好？

做法：
  · 扫 120 只全历史所有 flip_up 事件，记录入场时三专家票数(0~3) 与 trend(票≥2)。
  · 未来收益用【市场调整】口径：减去同期沪深300涨幅 —— 否则"趋势regime多在牛市"会把结果带偏。
  · 多个口径：5/10/20/40 交易日前向收益，以及"持有到下一次 flip_down"的真实交易收益。
  · 对比 趋势组 vs 震荡组 的均值(Welch t 检验)与胜率；再看是否随票数单调。

判定：趋势组未来收益显著更高 → 假设成立(regime 用法待改)；不高/更低 → 假设证伪。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import benchmark as BM

N_ST, M_ST = 10, 3.0
HORIZONS = [5, 10, 20, 40]


def collect_events(stocks: dict, bench_close: pd.Series) -> pd.DataFrame:
    start, end = DL.backtest_window()
    rows = []
    for code, df in stocks.items():
        st = F.supertrend(df, N_ST, M_ST)
        flip_up = st["flip_up"].to_numpy(bool)
        flip_dn = st["flip_down"].to_numpy(bool)
        _, votes, er, ad, r2 = F.regime_gate(df)
        close = df["close"].to_numpy(float)
        idx = df.index
        b = bench_close.reindex(idx).ffill().to_numpy(float)
        n = len(close)
        dn_pos = np.where(flip_dn)[0]
        for i in np.where(flip_up)[0]:
            if (start is not None and idx[i] < start) or (end is not None and idx[i] > end):
                continue
            if not (np.isfinite(er[i]) and np.isfinite(ad[i]) and np.isfinite(r2[i])):
                continue                                  # 跳过 regime warmup 不全的事件
            rec = {"votes": int(votes[i]), "trend": bool(votes[i] >= 2)}
            for N in HORIZONS:
                if i + N < n:
                    fwd = close[i + N] / close[i] - 1.0
                    bf = (b[i + N] / b[i] - 1.0) if (b[i] > 0 and np.isfinite(b[i + N])) else np.nan
                    rec[f"exc{N}"] = fwd - bf
                    rec[f"fwd{N}"] = fwd
            nx = dn_pos[dn_pos > i]
            if len(nx):
                j = nx[0]
                tr = close[j] / close[i] - 1.0
                bf = (b[j] / b[i] - 1.0) if b[i] > 0 else np.nan
                rec["trade_exc"] = tr - bf
                rec["trade"] = tr
            rows.append(rec)
    return pd.DataFrame(rows)


def welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else np.nan


def main():
    stocks = DL.load_pool(DL.get_universe(verbose=False) if C.USE_UNIVERSE else C.POOL, verbose=False)
    bench_close = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"]
    ev = collect_events(stocks, bench_close)
    tr = ev[ev["trend"]]
    rg = ev[~ev["trend"]]

    print("=" * 96)
    print(f"锐利条件检验：regime 能否预测 SuperTrend 入场质量（事件研究，市场调整后）")
    print(f"flip_up 事件总数 {len(ev)} | 趋势组 {len(tr)}（{len(tr)/len(ev):.0%}）| 震荡组 {len(rg)}")
    print("=" * 96)
    cols = [("exc5", "5日"), ("exc10", "10日"), ("exc20", "20日"), ("exc40", "40日"), ("trade_exc", "到flip_down")]
    print(f"{'未来超额(市场调整)':<14}{'趋势组均值':>11}{'震荡组均值':>11}{'差(趋势-震荡)':>13}{'t值':>8}"
          f"{'趋势胜率':>9}{'震荡胜率':>9}")
    for c, nm in cols:
        a, b = tr[c], rg[c]
        diff = a.mean() - b.mean()
        t = welch_t(a, b)
        wr_a = (a.dropna() > 0).mean(); wr_b = (b.dropna() > 0).mean()
        print(f"{nm:<14}{a.mean():>11.2%}{b.mean():>11.2%}{diff:>13.2%}{t:>8.2f}{wr_a:>9.0%}{wr_b:>9.0%}")
    print("-" * 96)
    print("按票数分层（20日市场调整超额，看是否随 regime 强度单调）：")
    for v in (0, 1, 2, 3):
        sub = ev[ev["votes"] == v]["exc20"].dropna()
        if len(sub):
            print(f"  票数={v}（{'趋势' if v>=2 else '震荡'}）：均值 {sub.mean():>7.2%} | 胜率 {(sub>0).mean():>4.0%} | n={len(sub)}")
    print("=" * 96)
    print("判定：趋势组超额显著更高(t>2) → 假设成立(regime 用法待改)；不高/更低/t≤0 → 假设证伪。")


if __name__ == "__main__":
    main()
