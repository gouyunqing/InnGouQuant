"""
factor_test.py — 纯动量/反转 因子检验（消融式：剥离实现摩擦，只测信号预测力）

目的：判断 (A)edge真实、只是SuperTrend实现糙  还是  (B)动量信号在A股本就无预测力。
做法：同一120只股池，月频、横截面排序，剥掉 SuperTrend/止损/单股择时/只做多 等一切实现层杂质，
     只保留“过去N月涨幅 → 未来1月收益”这一个信号，看它裸的预测力。

三个业界标准标尺（Fama-French / Grinold-Kahn / Alphalens 范式）：
  · IC（信息系数）：每月 signal 与 下月收益 的横截面 Spearman 相关，再按月平均。IC≠0 且稳定=有预测力。
    IC_IR = mean(IC)/std(IC)；t = IC_IR×√月数（|t|>2 ≈ 5%显著）。
  · 多空价差：最强20%(Q5) − 最弱20%(Q1) 的收益（A股不能裸卖空，故这是“信号是否带信息”的纸面检验）。
  · 分层单调性：Q1..Q5 收益是否单调（稳健性）。
方向：同时测【1月=反转探针】——若短周期 IC 显著为负，则A股是反转市而非动量市。

局限：①幸存者偏差（股池只含当前在市股，偏乐观）；②未做市值/行业中性化（Level-1，非Level-2纯因子）；
     ③IC为gross口径（月频换手高，扣成本后多空价差会下降，但IC衡量的是“有无信息”本身）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import data_loader as DL


def build_close_panel(stocks: dict) -> pd.DataFrame:
    """前复权收盘价面板：index=日期，columns=股票代码，并裁剪到回测窗口。"""
    panel = pd.DataFrame({c: df["close"] for c, df in stocks.items()}).sort_index()
    start, end = DL.backtest_window()
    if start is not None:
        panel = panel[panel.index >= start]
    if end is not None:
        panel = panel[panel.index <= end]
    return panel


def monthly_panel(close_panel: pd.DataFrame) -> pd.DataFrame:
    """月末前复权价（取每月最后一个有效价）。"""
    return close_panel.resample("ME").last()


def momentum_signal(mp: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """过去 lookback 个月、跳过最近 skip 个月的累计收益。skip=1 即经典 12-1 动量的跳月。"""
    return mp.shift(skip) / mp.shift(lookback + skip) - 1.0


def ic_series(signal: pd.DataFrame, fwd_ret: pd.DataFrame, min_names: int = 25) -> pd.Series:
    """逐月横截面 Spearman 秩相关（rank IC）。"""
    out = {}
    for dt in signal.index:
        df = pd.concat([signal.loc[dt], fwd_ret.loc[dt]], axis=1).dropna()
        if len(df) >= min_names:
            # Spearman = 对两列各自取秩后求 Pearson（免 scipy 依赖）
            out[dt] = df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())
    return pd.Series(out).dropna()


def quintile_forward_returns(signal: pd.DataFrame, fwd_ret: pd.DataFrame,
                             q: int = 5, min_names: int = 25) -> pd.DataFrame:
    """每月按 signal 分 q 档，算各档【下月】等权收益。返回 index=月, cols=0..q-1（0=最弱）。"""
    rows = {}
    for dt in signal.index:
        s = signal.loc[dt].dropna()
        r = fwd_ret.loc[dt]
        common = s.index.intersection(r.dropna().index)
        if len(common) < min_names:
            continue
        s, r = s.loc[common], r.loc[common]
        labels = pd.qcut(s.rank(method="first"), q, labels=False)
        rows[dt] = r.groupby(labels).mean()
    return pd.DataFrame(rows).T.sort_index()


def _ann(mean_monthly: float) -> float:
    return (1.0 + mean_monthly) ** 12 - 1.0


def _t_stat(x: pd.Series) -> float:
    x = x.dropna()
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(len(x))) if sd > 1e-12 and len(x) > 2 else 0.0


def turnover(signal: pd.DataFrame, fwd_ret: pd.DataFrame, q: int = 5, min_names: int = 25) -> float:
    """Q5（最强档）成分的月均换手（衡量交易成本压力）。"""
    prev = None
    tos = []
    for dt in signal.index:
        s = signal.loc[dt].dropna()
        r = fwd_ret.loc[dt]
        common = s.index.intersection(r.dropna().index)
        if len(common) < min_names:
            continue
        s = s.loc[common]
        labels = pd.qcut(s.rank(method="first"), q, labels=False)
        top = set(s.index[labels == q - 1])
        if prev is not None and top:
            tos.append(len(top - prev) / len(top))
        prev = top
    return float(np.mean(tos)) if tos else float("nan")


def evaluate_factor(name: str, signal: pd.DataFrame, fwd_ret: pd.DataFrame,
                    q: int = 5) -> dict:
    ics = ic_series(signal, fwd_ret)
    qf = quintile_forward_returns(signal, fwd_ret, q=q)
    ls = qf[q - 1] - qf[0]                       # Q5 - Q1 多空价差（月）
    q_ann = [_ann(qf[i].mean()) for i in range(q)]
    monotonic = all(q_ann[i] <= q_ann[i + 1] for i in range(q - 1)) or \
                all(q_ann[i] >= q_ann[i + 1] for i in range(q - 1))
    ic_ir = float(ics.mean() / ics.std(ddof=1)) if ics.std(ddof=1) > 1e-12 else 0.0
    # 长多超额：Q5 − 全池等权（A股不能卖空时的可交易代理）
    uni_mean = fwd_ret.reindex(qf.index).mean(axis=1)
    long_excess = qf[q - 1] - uni_mean
    return dict(
        name=name, n_months=len(ics),
        mean_ic=float(ics.mean()), ic_ir=ic_ir, ic_t=_t_stat(ics),
        ls_ann=_ann(ls.mean()), ls_t=_t_stat(ls),
        ls_sharpe=float(ls.mean() / ls.std(ddof=1) * np.sqrt(12)) if ls.std(ddof=1) > 1e-12 else 0.0,
        q_ann=q_ann, monotonic=monotonic,
        long_excess_ann=_ann(long_excess.mean()), long_excess_t=_t_stat(long_excess),
        turnover=turnover(signal, fwd_ret, q=q),
    )


def run(verbose: bool = True) -> pd.DataFrame:
    pool = DL.get_universe(verbose=False) if C.USE_UNIVERSE else C.POOL
    stocks = DL.load_pool(pool, verbose=False)
    close = build_close_panel(stocks)
    mp = monthly_panel(close)
    mret = mp.pct_change(fill_method=None)
    fwd = mret.shift(-1)                          # 下月收益（无前视：signal用到t，收益用t→t+1）

    # 待测信号：1月(反转探针)/3月/6月/12-1月动量
    specs = [
        ("1月(反转探针)", momentum_signal(mp, 1)),
        ("3月动量", momentum_signal(mp, 3)),
        ("6月动量", momentum_signal(mp, 6)),
        ("12-1月动量", momentum_signal(mp, 12, skip=1)),
    ]
    rows = [evaluate_factor(nm, sig, fwd) for nm, sig in specs]
    df = pd.DataFrame(rows)

    if verbose:
        print("=" * 96)
        print(f"纯因子检验（120只股池 / 月频 / 窗口 {close.index.min().date()}~{close.index.max().date()} / {len(mp)}个月）")
        print("=" * 96)
        print(f"{'信号':<14}{'月数':>5}{'IC均值':>9}{'IC_IR':>8}{'IC_t':>7}"
              f"{'多空年化':>10}{'多空t':>7}{'多空SR':>8}{'单调':>6}{'长多超额':>10}{'超额t':>7}{'换手':>7}")
        for r in rows:
            print(f"{r['name']:<14}{r['n_months']:>5}{r['mean_ic']:>9.3f}{r['ic_ir']:>8.2f}{r['ic_t']:>7.2f}"
                  f"{r['ls_ann']:>10.2%}{r['ls_t']:>7.2f}{r['ls_sharpe']:>8.2f}"
                  f"{'是' if r['monotonic'] else '否':>6}{r['long_excess_ann']:>10.2%}{r['long_excess_t']:>7.2f}"
                  f"{r['turnover']:>7.0%}")
        print("-" * 96)
        print("分层年化收益（Q1最弱 → Q5最强）：")
        for r in rows:
            print(f"  {r['name']:<14}" + "  ".join(f"Q{i+1}:{v:>7.2%}" for i, v in enumerate(r['q_ann'])))
        print("=" * 96)
        print("读法：|IC_t|>2 且 |多空t|>2 = 信号显著有预测力。")
        print("  · 1月若 IC 显著为负 → A股短期是【反转】市。")
        print("  · 动量周期若 IC 显著为正且分层单调 → 动量edge真实(=A，去优化做法)。")
        print("  · 若全部不显著 → 信号本身无预测力(=B，该换edge)。")
    return df


if __name__ == "__main__":
    import os
    out = run(verbose=True)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(C.OUTPUT_DIR, "factor_test.csv")
    out.drop(columns=["q_ann"]).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{path}")
