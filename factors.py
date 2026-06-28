"""
factors.py — M2 因子层

在【前复权日线】上实现：
  · SuperTrend（ATR 用 Wilder 平滑；参数 n、m）—— 完全按文末公式自实现，不依赖第三方默认。
  · 量能突破：vol > k × SMA(vol, w)

输出干净的【因子矩阵】DataFrame（策略层与未来 ML 都吃这张表）：
  原始/前复权 OHLCV + dir / supertrend / flip_up / flip_down / vol_break + 涨跌停标记透传。

SuperTrend 公式（与需求文末一致）：
  TR_t      = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
  ATR_t     = (ATR_{t-1}*(n-1) + TR_t)/n        # 种子 = 前 n 根 TR 的均值
  HL2_t     = (H+L)/2
  UB_t      = HL2_t + m*ATR_t ;  LB_t = HL2_t - m*ATR_t
  FinalUB_t = (UB_t<FinalUB_{t-1}) or (C_{t-1}>FinalUB_{t-1}) ? UB_t : FinalUB_{t-1}
  FinalLB_t = (LB_t>FinalLB_{t-1}) or (C_{t-1}<FinalLB_{t-1}) ? LB_t : FinalLB_{t-1}
  dir_t     = (C_t>FinalUB_{t-1})? +1 : (C_t<FinalLB_{t-1})? -1 : dir_{t-1}
  ST_t      = (dir_t==+1)? FinalLB_t : FinalUB_t
  flip_up   = (dir_{t-1}==-1 and dir_t==+1) ;  flip_down = (dir_{t-1}==+1 and dir_t==-1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Wilder 平滑 ATR。种子 = 前 n 根 TR 的均值（TR[0] 因无前收为 NaN，从 TR[1] 起算）。

    返回与输入等长的 ATR 数组，前 n 个为 NaN（warmup）。
    """
    size = len(close)
    tr = np.full(size, np.nan)
    for t in range(1, size):
        hl = high[t] - low[t]
        hc = abs(high[t] - close[t - 1])
        lc = abs(low[t] - close[t - 1])
        tr[t] = max(hl, hc, lc)

    atr = np.full(size, np.nan)
    # 种子放在 index=n：用 TR[1..n]（共 n 根）的均值。若该段全 NaN（早年坏数据），种子保持 NaN。
    if size > n:
        seed_seg = tr[1:n + 1]
        if np.any(~np.isnan(seed_seg)):
            atr[n] = np.nanmean(seed_seg)
            for t in range(n + 1, size):
                atr[t] = (atr[t - 1] * (n - 1) + tr[t]) / n
    return atr


def supertrend(df: pd.DataFrame, n: int, m: float) -> pd.DataFrame:
    """计算 SuperTrend。输入需含前复权 high/low/close。返回 dir/supertrend/flip_up/flip_down。"""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    size = len(close)

    atr = wilder_atr(high, low, close, n)
    hl2 = (high + low) / 2.0
    ub = hl2 + m * atr
    lb = hl2 - m * atr

    final_ub = np.full(size, np.nan)
    final_lb = np.full(size, np.nan)
    direction = np.zeros(size, dtype=int)
    st = np.full(size, np.nan)

    # 起点：第一个 ATR 有效的位置
    valid = np.where(~np.isnan(atr))[0]
    if len(valid) == 0:
        out = pd.DataFrame(index=df.index)
        out["dir"] = 0
        out["supertrend"] = np.nan
        out["atr"] = np.nan
        out["flip_up"] = False
        out["flip_down"] = False
        return out

    start = int(valid[0])
    final_ub[start] = ub[start]
    final_lb[start] = lb[start]
    direction[start] = 1            # 初始方向约定为 +1（多）
    st[start] = final_lb[start]

    for t in range(start + 1, size):
        # FinalUB
        if (ub[t] < final_ub[t - 1]) or (close[t - 1] > final_ub[t - 1]):
            final_ub[t] = ub[t]
        else:
            final_ub[t] = final_ub[t - 1]
        # FinalLB
        if (lb[t] > final_lb[t - 1]) or (close[t - 1] < final_lb[t - 1]):
            final_lb[t] = lb[t]
        else:
            final_lb[t] = final_lb[t - 1]
        # 方向
        if close[t] > final_ub[t - 1]:
            direction[t] = 1
        elif close[t] < final_lb[t - 1]:
            direction[t] = -1
        else:
            direction[t] = direction[t - 1]
        # 轨价
        st[t] = final_lb[t] if direction[t] == 1 else final_ub[t]

    dir_series = pd.Series(direction, index=df.index)
    prev_dir = dir_series.shift(1)
    flip_up = (prev_dir == -1) & (dir_series == 1)
    flip_down = (prev_dir == 1) & (dir_series == -1)

    # warmup 段（ATR 无效）方向标 0，flip 置 False
    warm = np.isnan(atr)
    dir_series[warm] = 0
    flip_up[warm] = False
    flip_down[warm] = False

    out = pd.DataFrame(index=df.index)
    out["dir"] = dir_series.astype(int)
    out["supertrend"] = st
    out["atr"] = atr                       # 供回测引擎做吊灯(ATR)止损
    out["flip_up"] = flip_up.fillna(False)
    out["flip_down"] = flip_down.fillna(False)
    return out


def wma(values: np.ndarray, period: int) -> np.ndarray:
    """线性加权移动平均（权重 1..period，最新值权重最大）。前 period-1 个为 NaN。"""
    n = len(values)
    out = np.full(n, np.nan)
    if period <= 0 or n < period:
        return out
    # np.convolve 会翻转核：用 [period..1] 使 convolve 后 a[t-period+1..t] 的权重恰为 [1..period]
    kernel = np.arange(period, 0, -1, dtype=float)
    denom = period * (period + 1) / 2.0
    out[period - 1:] = np.convolve(values.astype(float), kernel, mode="valid") / denom
    return out


def hma(values: np.ndarray, period: int) -> np.ndarray:
    """Hull 移动平均：HMA = WMA( 2*WMA(n/2) − WMA(n), √n )。前 ~period 个为 NaN（warmup）。"""
    n = len(values)
    out = np.full(n, np.nan)
    if period < 2 or n < period:
        return out
    half = max(1, period // 2)
    sq = max(1, int(np.sqrt(period)))
    w_half = wma(values, half)
    w_full = wma(values, period)
    raw = 2.0 * w_half - w_full          # 前 period-1 个为 NaN（w_full 决定）
    start = period - 1
    if n - start >= sq:
        out[start:] = wma(raw[start:], sq)   # raw[start:] 无 NaN，可直接卷积
    return out


# --------------------------------------------------------------------------- #
# Regime 门控三专家：ER / ADX / R²（判断趋势市 vs 震荡市）
# --------------------------------------------------------------------------- #
def efficiency_ratio(close: np.ndarray, n: int = 10) -> np.ndarray:
    """Kaufman 效率比 ER = |净变化| / 路径总长度 ∈[0,1]。≈1=强趋势(直来直去)，≈0=震荡(来回磨)。"""
    c = pd.Series(close, dtype=float)
    change = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n).sum()
    return (change / path.replace(0, np.nan)).to_numpy()


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ADX（用 ewm(alpha=1/n) 近似 Wilder 平滑）。ADX>25 常作趋势阈值。"""
    h, l, c = pd.Series(high, dtype=float), pd.Series(low, dtype=float), pd.Series(close, dtype=float)
    up, down = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / n, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def r_squared(close: np.ndarray, n: int = 20) -> np.ndarray:
    """收盘价对时间做线性回归的 R²（=与时间的相关系数平方）。高=直线趋势，低=杂乱无趋势。"""
    c = pd.Series(close, dtype=float)
    t = pd.Series(np.arange(len(c), dtype=float), index=c.index)
    corr = c.rolling(n).corr(t)
    return (corr ** 2).to_numpy()


def regime_gate(df: pd.DataFrame, er_n: int = 10, adx_n: int = 14, r2_n: int = 20,
                er_thr: float = 0.3, adx_thr: float = 25.0, r2_thr: float = 0.5):
    """三专家 2/3 投票的趋势 regime 门控。返回 (gate_bool, votes, er, adx, r2)。
    warmup 期(NaN) 比较为 False → 票=0 → 门关(保守，不买)。"""
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    er = efficiency_ratio(close, er_n)
    ad = adx(high, low, close, adx_n)
    r2 = r_squared(close, r2_n)
    votes = ((er > er_thr).astype(int) + (ad > adx_thr).astype(int) + (r2 > r2_thr).astype(int))
    gate = votes >= 2
    return gate, votes, er, ad, r2


# --------------------------------------------------------------------------- #
# 领先型“变化点”探测器（用于在趋势诞生处入场，而非确认后追高）
# --------------------------------------------------------------------------- #
def atr_spike(high: np.ndarray, low: np.ndarray, close: np.ndarray,
              n_atr: int = 14, n_avg: int = 20, thr: float = 1.2) -> np.ndarray:
    """波动率突然放大：ATR 相对其均值跳升（ATR/SMA(ATR) > thr）。常伴随盘整→突破的趋势诞生。"""
    atr = pd.Series(wilder_atr(high, low, close, n_atr))
    ratio = atr / atr.rolling(n_avg).mean()
    return (ratio > thr).to_numpy()


def donchian_upper_break(high: np.ndarray, close: np.ndarray, n: int = 20) -> np.ndarray:
    """突破前期区间：收盘价上破【前 n 根】最高价（Donchian/箱体上轨）。"""
    prior_high = pd.Series(high).shift(1).rolling(n).max()
    return (pd.Series(close) > prior_high).to_numpy()


def rolling_vwap(close: np.ndarray, volume: np.ndarray, n: int = 20) -> np.ndarray:
    """滚动 n 日成交量加权均价（VWAP 近似，前复权价×量）。close 上穿它=重夺价值区。"""
    c, v = pd.Series(close, dtype=float), pd.Series(volume, dtype=float)
    return ((c * v).rolling(n).sum() / v.rolling(n).sum().replace(0, np.nan)).to_numpy()


def volume_break(volume: pd.Series, w: int, k: float) -> pd.Series:
    """量能突破：vol_t > k * SMA(vol, w)_t。warmup（不足 w 根）为 False。"""
    sma = volume.rolling(window=w, min_periods=w).mean()
    vb = volume > (k * sma)
    return vb.fillna(False)


def compute_factors(df: pd.DataFrame, params: C.StrategyParams) -> pd.DataFrame:
    """组装单只股票的因子矩阵：前复权 OHLCV + SuperTrend + vol_break + 涨跌停标记透传。"""
    st = supertrend(df, n=params.n, m=params.m)
    vb = volume_break(df["volume"], w=params.w, k=params.k)

    fdf = pd.DataFrame(index=df.index)
    # 价/量（前复权 + 原始量额）
    for col in ["open", "high", "low", "close", "volume", "amount", "prev_close"]:
        fdf[col] = df[col]
    # SuperTrend
    fdf["dir"] = st["dir"]
    fdf["supertrend"] = st["supertrend"]
    fdf["atr"] = st["atr"]
    fdf["flip_up"] = st["flip_up"]
    fdf["flip_down"] = st["flip_down"]
    # 量能
    fdf["vol_break"] = vb
    # 涨跌停 / 一字板透传（回测引擎用）
    for col in ["limit_up", "limit_down", "one_word_up", "one_word_down"]:
        fdf[col] = df[col]
    return fdf


def compute_factor_panel(stocks: dict, params: C.StrategyParams) -> dict:
    """对整池逐股计算因子矩阵，返回 {code: factor_df}。"""
    return {code: compute_factors(df, params) for code, df in stocks.items()}
