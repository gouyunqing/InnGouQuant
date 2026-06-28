"""
test_contrarian.py — 反转线索条件检验：flip_up 在"安静/便宜/回调后"入场是否更好？

动机：探测器筛选发现"确认型"信号(放量/ATR跳升/破箱体)全部让 flip_up 入场变差，而"未命中"
(低量等)组系统性更好。这暗示 A 股大盘惩罚追高、奖励安静=反转型。本测试把那些条件【反过来】：
  低量 / 波动收缩 / 远离箱顶(未破位) / 低于VWAP / 深跌后(回调) —— 看 flip_up 命中这些时未来超额是否更高。

口径同前：市场调整后未来超额(10/20/40日 + 到flip_down)，命中 vs 未命中 做 Welch t。
判定：命中组超额显著更高(t>2) → 反转/安静入场是可放大的方向。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import benchmark as BM

N_ST, M_ST = 10, 3.0
HORIZONS = [10, 20, 40]


def detectors(df: pd.DataFrame) -> dict:
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    c = pd.Series(close); h = pd.Series(high); v = pd.Series(vol)
    out = {}
    out["低量"] = (v < v.rolling(10).mean()).to_numpy()
    atr = pd.Series(F.wilder_atr(high, low, close, 14))
    out["波动收缩"] = ((atr / atr.rolling(20).mean()) < 0.9).to_numpy()
    prior_h20 = h.shift(1).rolling(20).max()
    out["远离箱顶"] = (c < 0.95 * prior_h20).to_numpy()        # 收盘比前20日高点低≥5%（未破位）
    out["低于VWAP"] = (c < F.rolling_vwap(close, vol, 20)).to_numpy()
    prior_h60 = h.shift(1).rolling(60).max()
    out["深跌后回调"] = (c < 0.90 * prior_h60).to_numpy()       # 比前60日高点低≥10%（明显回调）
    out["最安静(低量&波动收缩)"] = out["低量"] & out["波动收缩"]
    out["安静抄底(低量&深跌后)"] = out["低量"] & out["深跌后回调"]
    return {k: np.nan_to_num(v_).astype(bool) for k, v_ in out.items()}


def welch_t(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else np.nan


def main():
    stocks = DL.load_pool(DL.get_universe(verbose=False) if C.USE_UNIVERSE else C.POOL, verbose=False)
    bench_close = BM.load_benchmark(C.BENCHMARK_CODE, verbose=False)["close"]
    start, end = DL.backtest_window()
    names = ["低量", "波动收缩", "远离箱顶", "低于VWAP", "深跌后回调",
             "最安静(低量&波动收缩)", "安静抄底(低量&深跌后)"]

    rows = []
    for code, df in stocks.items():
        st = F.supertrend(df, N_ST, M_ST)
        flip_up = st["flip_up"].to_numpy(bool)
        flip_dn = st["flip_down"].to_numpy(bool)
        dets = detectors(df)
        _, votes, er, ad, r2 = F.regime_gate(df)
        close = df["close"].to_numpy(float)
        idx = df.index
        b = bench_close.reindex(idx).ffill().to_numpy(float)
        n = len(close); dn_pos = np.where(flip_dn)[0]
        for i in np.where(flip_up)[0]:
            if (start is not None and idx[i] < start) or (end is not None and idx[i] > end):
                continue
            if not (np.isfinite(er[i]) and np.isfinite(ad[i]) and np.isfinite(r2[i])):
                continue
            rec = {d: bool(dets[d][i]) for d in names}
            for N in HORIZONS:
                if i + N < n and b[i] > 0 and np.isfinite(b[i + N]):
                    rec[f"exc{N}"] = (close[i + N] / close[i] - 1) - (b[i + N] / b[i] - 1)
            nx = dn_pos[dn_pos > i]
            if len(nx) and b[i] > 0:
                j = nx[0]
                rec["trade_exc"] = (close[j] / close[i] - 1) - (b[j] / b[i] - 1)
            rows.append(rec)
    ev = pd.DataFrame(rows)
    base20 = ev["exc20"].mean()

    print("=" * 100)
    print(f"反转线索条件检验 | flip_up 事件 {len(ev)} | 基线 exc20={base20:.2%}")
    print("=" * 100)
    print(f"{'反转条件':<22}{'命中数':>7}{'命中率':>7}{'exc20命中':>10}{'exc20未中':>10}{'提升':>9}{'t值':>7}"
          f"{'到flip命中':>11}{'命中胜率20':>10}")
    res = []
    for d in names:
        hit = ev[ev[d]]; miss = ev[~ev[d]]
        if len(hit) < 30:
            continue
        a, bb = hit["exc20"], miss["exc20"]
        res.append((d, len(hit), len(hit) / len(ev), a.mean(), bb.mean(), a.mean() - bb.mean(),
                    welch_t(a, bb), hit["trade_exc"].mean(), (a.dropna() > 0).mean()))
    for d, nh, hr, eh, em, lf, t, th, wr in sorted(res, key=lambda x: -x[6]):
        print(f"{d:<22}{nh:>7}{hr:>7.0%}{eh:>10.2%}{em:>10.2%}{lf:>9.2%}{t:>7.2f}{th:>11.2%}{wr:>10.0%}")
    print("-" * 100)
    print("判定：提升>0 且 t>2 = 安静/反转入场显著更好 → 可放大的方向。否则反转线索也不成立。")


if __name__ == "__main__":
    main()
