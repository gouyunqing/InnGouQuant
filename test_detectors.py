"""
test_detectors.py — 领先“变化点”探测器筛选（条件检验）

目标：SuperTrend 的 flip_up 有一丝真 α，但滞后门控会追高。改用【领先探测器】在趋势诞生处入场。
方法：在每次 flip_up，检查各探测器是否“近期(≤3根)触发”，比较【命中 vs 未命中】的【市场调整后未来超额】。
     命中组超额显著更高(t>2) = 该探测器能把 flip_up 的早期 α 抬起来，值得进策略。

探测器（Tier A 领先 + 用户新增）：
  量能扩张 / VWAP重夺 / ATR跳升(用户) / 箱体突破Donchian(用户) / ADX上穿20 / ER转升 / 门控刚开(转折点)
另测【用户组合】：flip_up 且 量能扩张 且 ATR跳升 且 箱体突破。
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


def _recent(b: np.ndarray, k: int = 3) -> np.ndarray:
    """近 k 根内是否触发过（含当根）。"""
    return (pd.Series(np.asarray(b, dtype=float)).rolling(k, min_periods=1).max() > 0).to_numpy()


def detectors(df: pd.DataFrame) -> dict:
    close = df["close"].to_numpy(float); high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    vol = df["volume"].to_numpy(float)
    out = {}
    out["量能扩张"] = F.volume_break(df["volume"], 10, 1.5).to_numpy()
    out["VWAP重夺"] = close > F.rolling_vwap(close, vol, 20)
    out["ATR跳升"] = F.atr_spike(high, low, close, 14, 20, 1.2)
    out["箱体突破D20"] = F.donchian_upper_break(high, close, 20)
    _, votes, er, ad, r2 = F.regime_gate(df)
    adx_s = pd.Series(ad)
    out["ADX上穿20"] = ((adx_s > 20) & (adx_s.shift(1) <= 20)).to_numpy()
    er_s = pd.Series(er)
    out["ER转升"] = (er_s > er_s.shift(5)).to_numpy()
    gate_s = pd.Series((votes >= 2).astype(float))
    out["门控刚开转折"] = ((gate_s > 0) & (gate_s.shift(1) == 0)).to_numpy()
    return {k: _recent(np.nan_to_num(v.astype(float)).astype(bool)) for k, v in out.items()}


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
    det_names = ["量能扩张", "VWAP重夺", "ATR跳升", "箱体突破D20", "ADX上穿20", "ER转升", "门控刚开转折"]

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
            rec = {d: bool(dets[d][i]) for d in det_names}
            rec["组合_量+ATR+箱"] = bool(dets["量能扩张"][i] and dets["ATR跳升"][i] and dets["箱体突破D20"][i])
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
    base_tr = ev["trade_exc"].mean()
    print("=" * 100)
    print(f"领先探测器筛选 | flip_up 事件 {len(ev)} | 基线 exc20={base20:.2%} 到flip_down={base_tr:.2%}")
    print("=" * 100)
    print(f"{'探测器':<16}{'命中数':>7}{'命中率':>7}{'exc20命中':>10}{'exc20未中':>10}{'提升':>9}{'t值':>7}"
          f"{'到flip命中':>11}{'命中胜率20':>10}")
    cand = det_names + ["组合_量+ATR+箱"]
    res = []
    for d in cand:
        hit = ev[ev[d]]; miss = ev[~ev[d]]
        if len(hit) < 30:
            continue
        a, bb = hit["exc20"], miss["exc20"]
        lift = a.mean() - bb.mean()
        t = welch_t(a, bb)
        res.append((d, len(hit), len(hit) / len(ev), a.mean(), bb.mean(), lift, t,
                    hit["trade_exc"].mean(), (a.dropna() > 0).mean()))
    for d, nh, hr, eh, em, lf, t, th, wr in sorted(res, key=lambda x: -x[6]):
        print(f"{d:<16}{nh:>7}{hr:>7.0%}{eh:>10.2%}{em:>10.2%}{lf:>9.2%}{t:>7.2f}{th:>11.2%}{wr:>10.0%}")
    print("-" * 100)
    print("判定：提升>0 且 t>2 = 该探测器显著把 flip_up 早期超额抬高 → 值得进策略。")
    print("  对比基线 exc20={:.2%}：命中列高于它=有增益。".format(base20))


if __name__ == "__main__":
    main()
