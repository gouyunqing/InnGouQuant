"""
report.py — M6 报告（指标 / 图 / markdown）

指标：年化收益、最大回撤、Sharpe、Calmar、胜率、盈亏比、交易次数、平均持仓天数（分 in-sample / out-of-sample）。
图：每股(价格+买卖点) + 每股/组合净值曲线；参数寻优热力图。
输出：outputs/ 下 csv 指标表 + png 图 + 一份 markdown 汇总。

指标函数同时供 optimize.py 复用（目标函数 = Calmar/Sharpe）。
"""
from __future__ import annotations

import os
from math import exp
from statistics import NormalDist
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_NORM = NormalDist()   # 标准正态（stdlib，免 scipy）：DSR/PSR 用其 cdf/inv_cdf

import matplotlib
matplotlib.use("Agg")          # 无界面后端
import matplotlib.pyplot as plt
from matplotlib import font_manager

import config as C
from backtest import BacktestResult, StockResult, trades_to_frame

# 中文字体：注册一个系统里真实存在的 CJK 字体文件（macOS）。找不到则退英文（不致命，仅图中文变方块）。
_CJK_FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]
for _p in _CJK_FONT_CANDIDATES:
    if os.path.exists(_p):
        try:
            font_manager.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_p).get_name()
            break
        except Exception:
            continue
matplotlib.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def _daily_rf(periods_per_year: int) -> float:
    """年化无风险利率 → 每期（日）无风险利率。"""
    return (1.0 + C.RISK_FREE_ANNUAL) ** (1.0 / periods_per_year) - 1.0


def _empty_equity_metrics(n: int) -> dict:
    return dict(total_return=0.0, ann_return=0.0, ann_vol=0.0, max_dd=0.0,
                sharpe=0.0, sortino=0.0, calmar=0.0,
                skew=0.0, kurtosis=0.0, tail_ratio=0.0, var95=0.0, cvar95=0.0, n_days=n)


def metrics_from_equity(equity: pd.Series, periods_per_year: int = C.TRADING_DAYS_PER_YEAR) -> dict:
    """从净值曲线算机构级绝对指标：
    总收益、年化(CAGR)、年化波动率、最大回撤、Sharpe(扣无风险)、Sortino(只罚下行)、Calmar、
    偏度、峰度(超额)、尾比、VaR95、CVaR95(日度历史法)。"""
    equity = equity.dropna()
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return _empty_equity_metrics(len(equity))

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    years = len(equity) / periods_per_year
    ann_return = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0

    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    max_dd = float(dd.min())   # 负数

    rets = equity.pct_change().dropna()
    rf_d = _daily_rf(periods_per_year)
    sd = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0
    ann_vol = sd * np.sqrt(periods_per_year)
    excess = rets - rf_d
    sharpe = float(excess.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0

    # Sortino：只罚低于无风险的下行波动
    downside = np.minimum(rets - rf_d, 0.0)
    dd_dev = float(np.sqrt((downside ** 2).mean())) if len(rets) else 0.0
    sortino = float(excess.mean() / dd_dev * np.sqrt(periods_per_year)) if dd_dev > 1e-12 else 0.0

    calmar = ann_return / abs(max_dd) if max_dd < -1e-12 else 0.0

    skew = float(rets.skew()) if len(rets) > 2 else 0.0
    kurt = float(rets.kurtosis()) if len(rets) > 3 else 0.0   # pandas=超额峰度(正态=0)
    if len(rets) >= 20:
        p95 = np.percentile(rets, 95); p5 = np.percentile(rets, 5)
        tail_ratio = float(abs(p95) / abs(p5)) if abs(p5) > 1e-12 else 0.0
        var95 = float(p5)                                   # 日度 5% 分位（负数=亏损）
        cvar95 = float(rets[rets <= p5].mean()) if (rets <= p5).any() else var95
    else:
        tail_ratio = var95 = cvar95 = 0.0

    return dict(total_return=float(total_return), ann_return=float(ann_return),
                ann_vol=float(ann_vol), max_dd=max_dd, sharpe=sharpe, sortino=sortino,
                calmar=float(calmar), skew=skew, kurtosis=kurt, tail_ratio=tail_ratio,
                var95=var95, cvar95=cvar95, n_days=len(equity))


def metrics_from_trades(trades_df: pd.DataFrame) -> dict:
    """从逐笔交易算：交易次数、胜率、盈亏比、平均持仓天数、平均单笔收益。"""
    n = len(trades_df)
    if n == 0:
        return dict(n_trades=0, win_rate=0.0, profit_factor=0.0, avg_holding=0.0, avg_ret=0.0)
    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] < 0]
    gross_win = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_win > 0 else 0.0)
    return dict(
        n_trades=int(n),
        win_rate=float(len(wins) / n),
        profit_factor=float(profit_factor),
        avg_holding=float(trades_df["holding_days"].mean()),
        avg_ret=float(trades_df["ret"].mean()),
    )


def metrics_vs_benchmark(equity: pd.Series, bench_equity: pd.Series,
                         periods_per_year: int = C.TRADING_DAYS_PER_YEAR) -> dict:
    """基准相对指标（vs 沪深300）：β、Jensen's α(年化)、超额收益(年化)、跟踪误差、
    信息比率 IR、上行/下行捕获、与基准相关性。把'跟大盘搭便车(β)'和'真本事(α)'分开。"""
    empty = dict(beta=np.nan, alpha_ann=np.nan, excess_ann=np.nan, tracking_error=np.nan,
                 info_ratio=np.nan, up_capture=np.nan, down_capture=np.nan,
                 bench_corr=np.nan, bench_ann=np.nan)
    if equity is None or bench_equity is None:
        return empty
    p = equity.dropna().pct_change()
    b = bench_equity.dropna().pct_change()
    df = pd.concat([p, b], axis=1, join="inner").dropna()
    df.columns = ["p", "b"]
    if len(df) < 20:
        return empty
    p = df["p"].to_numpy(); b = df["b"].to_numpy()
    n = len(p)
    rf_d = _daily_rf(periods_per_year)

    var_b = float(np.var(b, ddof=1))
    beta = float(np.cov(p, b, ddof=1)[0, 1] / var_b) if var_b > 1e-18 else np.nan
    # Jensen's α（CAPM）：α_d = E[p] - (rf + β(E[b]-rf))，再几何年化
    alpha_d = float(np.mean(p) - (rf_d + beta * (np.mean(b) - rf_d))) if np.isfinite(beta) else np.nan
    alpha_ann = float((1.0 + alpha_d) ** periods_per_year - 1.0) if np.isfinite(alpha_d) else np.nan

    ann_p = float(np.prod(1.0 + p) ** (periods_per_year / n) - 1.0)
    ann_b = float(np.prod(1.0 + b) ** (periods_per_year / n) - 1.0)
    excess_ann = ann_p - ann_b

    d = p - b
    te = float(np.std(d, ddof=1) * np.sqrt(periods_per_year))
    info_ratio = float(np.mean(d) / np.std(d, ddof=1) * np.sqrt(periods_per_year)) if np.std(d, ddof=1) > 1e-18 else np.nan

    up = b > 0; dn = b < 0
    def _cap(mask):
        if mask.sum() < 3:
            return np.nan
        cb = float(np.prod(1.0 + b[mask]) - 1.0)
        cp = float(np.prod(1.0 + p[mask]) - 1.0)
        return float(cp / cb) if abs(cb) > 1e-12 else np.nan
    up_capture, down_capture = _cap(up), _cap(dn)
    bench_corr = float(np.corrcoef(p, b)[0, 1]) if n > 2 else np.nan

    return dict(beta=beta, alpha_ann=alpha_ann, excess_ann=excess_ann, tracking_error=te,
                info_ratio=info_ratio, up_capture=up_capture, down_capture=down_capture,
                bench_corr=bench_corr, bench_ann=ann_b)


def deflated_sharpe_ratio(returns: pd.Series, trial_sharpes_ann: List[float],
                          periods_per_year: int = C.TRADING_DAYS_PER_YEAR,
                          n_trials: Optional[int] = None) -> dict:
    """去偏夏普比率 DSR（Bailey & López de Prado 2014）：在‘网格里挑了 N 组、报最好那组’的
    选择偏差下，观测到的夏普有多大概率是【真本事】而非【N 次试验里蒙到的运气】。

    步骤：① 由 N 组试验的夏普方差，算‘纯运气下期望的最大夏普’SR*（去偏门槛）；
         ② PSR(SR*) = 观测夏普显著高于 SR* 的概率，并对偏度/峰度（非正态）做修正。
    DSR≥0.95 才算在 5% 显著性下‘真有 edge’。"""
    out = dict(dsr=np.nan, psr0=np.nan, sr_obs_ann=np.nan, sr_star_ann=np.nan,
               n_trials=0, n_obs=0)
    r = returns.dropna()
    n = len(r)
    if n < 30 or not trial_sharpes_ann:
        return out
    rf_d = _daily_rf(periods_per_year)
    sd = float(r.std(ddof=1))
    if sd <= 1e-18:
        return out
    sr_obs = float((r.mean() - rf_d) / sd)            # 每期（非年化）夏普
    g3 = float(r.skew())
    g4 = float(r.kurtosis()) + 3.0                    # 转为完整峰度（正态=3）

    N = int(n_trials or len(trial_sharpes_ann))
    sr_trials_p = np.array(trial_sharpes_ann, dtype=float) / np.sqrt(periods_per_year)
    var_tr = float(np.var(sr_trials_p[np.isfinite(sr_trials_p)], ddof=1)) if N > 1 else 0.0
    gamma = 0.5772156649015329
    if N >= 2 and var_tr > 0:
        z1 = _NORM.inv_cdf(1.0 - 1.0 / N)
        z2 = _NORM.inv_cdf(1.0 - 1.0 / (N * exp(1)))
        sr_star = float(np.sqrt(var_tr) * ((1.0 - gamma) * z1 + gamma * z2))
    else:
        sr_star = 0.0

    def _psr(sr_ref: float) -> float:
        denom = 1.0 - g3 * sr_obs + (g4 - 1.0) / 4.0 * sr_obs ** 2
        if denom <= 1e-12:
            return float("nan")
        z = (sr_obs - sr_ref) * np.sqrt(n - 1) / np.sqrt(denom)
        return float(_NORM.cdf(z))

    out.update(dsr=_psr(sr_star), psr0=_psr(0.0),
               sr_obs_ann=sr_obs * np.sqrt(periods_per_year),
               sr_star_ann=sr_star * np.sqrt(periods_per_year),
               n_trials=N, n_obs=n)
    return out


def probability_of_backtest_overfitting(returns_df: pd.DataFrame,
                                        n_blocks: int = 16) -> dict:
    """回测过拟合概率 PBO（CSCV 法，Bailey/Borwein/López de Prado/Zhu 2014）。

    输入：T×N 的【每个参数组合的日收益】矩阵（列=72组参数，行=交易日）。
    做法：把时间切成 S 块，枚举所有 C(S,S/2) 种‘一半当 IS、一半当 OOS’的组合；每种里
         用 IS 选出夏普最高的参数，再看它在 OOS 的相对排名。
    PBO = 「IS 最优参数在 OOS 排到中位数以下」的组合占比 —— 越高越说明‘最优’是过拟合。
    """
    out = dict(pbo=float("nan"), n_combos=0, n_blocks=0, n_configs=0, med_logit=float("nan"))
    M = returns_df.dropna(how="any").to_numpy(dtype=float)
    T, N = M.shape
    S = n_blocks - (n_blocks % 2)                 # 取偶数
    if N < 3 or S < 4 or T < S * 4:
        return out
    blocks = np.array_split(np.arange(T), S)
    bsum = np.vstack([M[b].sum(axis=0) for b in blocks])          # S×N
    bsq = np.vstack([(M[b] ** 2).sum(axis=0) for b in blocks])    # S×N
    bn = np.array([len(b) for b in blocks], dtype=float)          # S

    from itertools import combinations
    logits = []
    n_below = 0
    for is_b in combinations(range(S), S // 2):
        is_b = list(is_b)
        oos_b = [j for j in range(S) if j not in is_b]
        n_is = bn[is_b].sum(); n_oos = bn[oos_b].sum()
        m_is = bsum[is_b].sum(0) / n_is
        v_is = bsq[is_b].sum(0) / n_is - m_is ** 2
        m_oos = bsum[oos_b].sum(0) / n_oos
        v_oos = bsq[oos_b].sum(0) / n_oos - m_oos ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            sr_is = np.where(v_is > 1e-18, m_is / np.sqrt(v_is), -np.inf)
            sr_oos = np.where(v_oos > 1e-18, m_oos / np.sqrt(v_oos), np.nan)
        nstar = int(np.argmax(sr_is))                # IS 最优
        srn = sr_oos[nstar]
        if not np.isfinite(srn):
            continue
        valid = np.isfinite(sr_oos)
        rank = int((sr_oos[valid] < srn).sum())      # OOS 里有多少组比它差
        omega = (rank + 1) / (int(valid.sum()) + 1)  # 相对排名 ∈(0,1)
        omega = min(max(omega, 1e-6), 1 - 1e-6)
        lam = float(np.log(omega / (1 - omega)))     # logit
        logits.append(lam)
        if lam <= 0.0:                               # OOS 排名 ≤ 中位
            n_below += 1
    if not logits:
        return out
    out.update(pbo=n_below / len(logits), n_combos=len(logits),
               n_blocks=S, n_configs=N, med_logit=float(np.median(logits)))
    return out


def compute_metrics(equity: pd.Series, trades_df: pd.DataFrame,
                    periods_per_year: int = C.TRADING_DAYS_PER_YEAR,
                    benchmark: Optional[pd.Series] = None) -> dict:
    m = metrics_from_equity(equity, periods_per_year)
    m.update(metrics_from_trades(trades_df))
    if benchmark is not None:
        m.update(metrics_vs_benchmark(equity, benchmark, periods_per_year))
    return m


def slice_equity(equity: pd.Series, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.Series:
    s = equity
    if start is not None:
        s = s[s.index >= start]
    if end is not None:
        s = s[s.index <= end]
    return s


def slice_trades(trades_df: pd.DataFrame, start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> pd.DataFrame:
    if trades_df.empty:
        return trades_df
    ed = pd.to_datetime(trades_df["exit_date"])
    mask = pd.Series(True, index=trades_df.index)
    if start is not None:
        mask &= ed >= start
    if end is not None:
        mask &= ed <= end
    return trades_df[mask]


# --------------------------------------------------------------------------- #
# 指标表（每股 + 组合，分 IS/OOS）
# --------------------------------------------------------------------------- #
def build_metrics_table(result: BacktestResult,
                        split_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """每股 + 组合的 in-sample / out-of-sample 指标表。split_date=None 则只出 full。"""
    trades_all = trades_to_frame(result)
    rows = []

    def add(name: str, equity: pd.Series, tdf: pd.DataFrame):
        for seg, (s, e) in segments.items():
            eq = slice_equity(equity, s, e)
            td = slice_trades(tdf, s, e)
            m = compute_metrics(eq, td)
            rows.append({"name": name, "segment": seg, **m})

    if split_date is None:
        segments = {"full": (None, None)}
    else:
        segments = {"in_sample": (None, split_date), "out_sample": (split_date, None)}

    for code, r in result.per_stock.items():
        add(code, r.equity, trades_all[trades_all["code"] == code])
    add("PORTFOLIO", result.portfolio_equity, trades_all)

    df = pd.DataFrame(rows)
    cols = ["name", "segment", "total_return", "ann_return", "ann_vol", "max_dd",
            "sharpe", "sortino", "calmar",
            "n_trades", "win_rate", "profit_factor", "avg_holding", "avg_ret", "n_days"]
    return df[cols]


# --------------------------------------------------------------------------- #
# 画图
# --------------------------------------------------------------------------- #
def plot_stock(code: str, factor_df: pd.DataFrame, sr: StockResult, outdir: str,
               split_date: Optional[pd.Timestamp] = None) -> str:
    """单只股票：上图=前复权收盘+SuperTrend+买卖点；下图=该股净值。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    px = factor_df["close"]
    ax1.plot(px.index, px.values, color="#333", lw=0.9, label="前复权收盘")
    if "supertrend" in factor_df:
        ax1.plot(factor_df.index, factor_df["supertrend"].values, color="#1f77b4",
                 lw=0.8, alpha=0.7, label="SuperTrend")
    # 买卖点
    for t in sr.trades:
        ax1.scatter(t.entry_date, factor_df["close"].get(t.entry_date, np.nan),
                    marker="^", color="red", s=55, zorder=5)
        ax1.scatter(t.exit_date, factor_df["close"].get(t.exit_date, np.nan),
                    marker="v", color="green", s=55, zorder=5)
    if split_date is not None:
        ax1.axvline(split_date, color="orange", ls="--", lw=1, alpha=0.8, label="训练/测试切分")
    ax1.set_title(f"{code}  价格与买卖点（▲买 ▼卖）")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(alpha=0.25)

    ax2.plot(sr.equity.index, sr.equity.values, color="#d62728", lw=1.0)
    if split_date is not None:
        ax2.axvline(split_date, color="orange", ls="--", lw=1, alpha=0.8)
    ax2.axhline(sr.sleeve, color="#999", ls=":", lw=0.8)
    ax2.set_title(f"{code}  净值（起始 {sr.sleeve:,.0f}）")
    ax2.grid(alpha=0.25)

    fig.tight_layout()
    path = os.path.join(outdir, f"stock_{code}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_portfolio(result: BacktestResult, outdir: str,
                   split_date: Optional[pd.Timestamp] = None,
                   benchmark: Optional[pd.Series] = None,
                   benchmark_name: str = "基准") -> str:
    fig, ax = plt.subplots(figsize=(13, 6))
    eq = result.portfolio_equity / result.init_capital
    # 叠加各股归一净值（浅色、不入图例——股池可能上百只）
    for code, r in result.per_stock.items():
        e = (r.equity / r.sleeve).reindex(result.calendar).ffill()
        ax.plot(e.index, e.values, lw=0.5, alpha=0.18, color="#888")
    ax.plot(eq.index, eq.values, color="#111", lw=1.6, label="组合等权净值", zorder=5)
    if benchmark is not None:
        be = (benchmark / benchmark.dropna().iloc[0]).reindex(result.calendar).ffill()
        ax.plot(be.index, be.values, color="#1f77b4", lw=1.4, ls="-",
                label=benchmark_name, zorder=4)
    if split_date is not None:
        ax.axvline(split_date, color="orange", ls="--", lw=1.2, label="训练/测试切分")
    ax.axhline(1.0, color="#999", ls=":", lw=0.8)
    ax.set_title(f"股池等权组合净值（归一化，{len(result.per_stock)} 只）")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = os.path.join(outdir, "portfolio_equity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_heatmap(grid_df: pd.DataFrame, outdir: str, metric: str = "objective",
                 row: str = "n", col: str = "m") -> Optional[str]:
    """参数寻优热力图：对 (row,col) 两参数取其余参数的最优值后画热力图。"""
    if grid_df.empty or metric not in grid_df:
        return None
    piv = grid_df.pivot_table(index=row, columns=col, values=metric, aggfunc="max")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(piv.values, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    ax.set_title(f"参数寻优热力图（{metric}，其余参数取最优）")
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = os.path.join(outdir, "optimize_heatmap.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Markdown 汇总
# --------------------------------------------------------------------------- #
def _fmt_pct(x: float) -> str:
    return f"{x*100:.2f}%" if np.isfinite(x) else "—"


def _metrics_md_table(df: pd.DataFrame) -> str:
    head = ("| 标的 | 段 | 总收益 | 年化 | 年化波动 | 最大回撤 | Sharpe | Sortino | Calmar | 交易数 | 胜率 | 盈亏比 | 均持仓 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for _, r in df.iterrows():
        pf = r["profit_factor"]
        pf_s = "∞" if not np.isfinite(pf) else f"{pf:.2f}"
        lines.append(
            f"| {r['name']} | {r['segment']} | {_fmt_pct(r['total_return'])} | {_fmt_pct(r['ann_return'])} "
            f"| {_fmt_pct(r['ann_vol'])} | {_fmt_pct(r['max_dd'])} | {r['sharpe']:.2f} | {r['sortino']:.2f} "
            f"| {r['calmar']:.2f} | {int(r['n_trades'])} | {_fmt_pct(r['win_rate'])} | {pf_s} | {r['avg_holding']:.1f} |"
        )
    return head + "\n".join(lines)


def _benchmark_md(bench_summary: dict) -> str:
    """渲染‘组合 vs 基准’的 α/β/信息比率表 + 抗过拟合去偏夏普 DSR。"""
    name = bench_summary.get("benchmark_name", "基准")
    segs = bench_summary.get("segments", {})
    md = [f"## 组合 vs {name}（α/β/信息比率）\n",
          f"> 本策略只做多、长期持有，天生有 β。**α 才是剔掉‘跟{name}搭便车’之后的真本事**；"
          "β 越低=越像现金/越不随大盘；信息比率=单位主动风险的超额。\n",
          (f"| 段 | 策略年化 | {name}年化 | 超额(年化) | β | α(年化) | 信息比率 | 跟踪误差 | 上行捕获 | 下行捕获 | 相关性 |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|\n")]
    order = [("in_sample", "训练IS"), ("out_sample", "测试OOS"), ("full", "全样本")]
    for key, label in order:
        s = segs.get(key)
        if not s:
            continue
        md.append(
            f"| {label} | {_fmt_pct(s.get('ann_return', np.nan))} | {_fmt_pct(s.get('bench_ann', np.nan))} "
            f"| {_fmt_pct(s.get('excess_ann', np.nan))} | {s.get('beta', np.nan):.2f} "
            f"| {_fmt_pct(s.get('alpha_ann', np.nan))} | {s.get('info_ratio', np.nan):.2f} "
            f"| {_fmt_pct(s.get('tracking_error', np.nan))} | {s.get('up_capture', np.nan):.2f} "
            f"| {s.get('down_capture', np.nan):.2f} | {s.get('bench_corr', np.nan):.2f} |"
        )
    d = bench_summary.get("dsr")
    if d and np.isfinite(d.get("dsr", np.nan)):
        md.append(f"\n### 抗过拟合：去偏夏普比率 DSR\n")
        md.append(f"- 网格试验次数 N={d['n_trials']}，样本 {d['n_obs']} 日")
        md.append(f"- 观测年化夏普 {d['sr_obs_ann']:.2f}，纯运气下期望的‘最大夏普’门槛 SR\\* {d['sr_star_ann']:.2f}（年化）")
        md.append(f"- **DSR = {d['dsr']:.1%}**：在 {d['n_trials']} 次网格挑最优的选择偏差下，"
                  f"该夏普为【真本事】的概率。**DSR≥95% 才算显著**。")
        verdict = ("✅ 通过：edge 大概率真实" if d["dsr"] >= 0.95
                   else f"❌ 未通过：{d['dsr']:.0%} 远低于 95%，**最优夏普几乎可判定为 {d['n_trials']} 次试验里蒙到的运气**")
        md.append(f"- 判定：{verdict}")

    p = bench_summary.get("pbo")
    if p and np.isfinite(p.get("pbo", np.nan)):
        md.append(f"\n### 抗过拟合：回测过拟合概率 PBO（CSCV）\n")
        md.append(f"- 把全样本切 {p['n_blocks']} 块、组合式分 IS/OOS（共 {p['n_combos']} 种切法），"
                  f"在 {p['n_configs']} 组参数上做")
        md.append(f"- **PBO = {p['pbo']:.0%}**：样本内最优参数，在样本外掉到中位数【以下】的概率。"
                  f"**PBO 越高=过拟合越重；>50% 基本等于没有可持续的‘最优’**。")
        verd = ("✅ 低过拟合" if p["pbo"] < 0.3
                else ("⚠️ 中等过拟合" if p["pbo"] < 0.5
                      else f"❌ 高过拟合：{p['pbo']:.0%} 的切法里，IS 最优在 OOS 都跑输一半同行"))
        md.append(f"- 判定：{verd}")
        md.append(f"- 与 DSR 互证：DSR 从‘选择偏差’角度、PBO 从‘IS最优在OOS的排名分布’角度，"
                  f"指向同一结论。")
    md.append("")
    return "\n".join(md)


def write_markdown_report(outdir: str, *, params, metrics_table: pd.DataFrame,
                          result: BacktestResult, split_date: Optional[pd.Timestamp],
                          opt_summary: Optional[dict] = None,
                          bench_summary: Optional[dict] = None,
                          stock_pngs: Optional[Dict[str, str]] = None,
                          portfolio_png: Optional[str] = None,
                          heatmap_png: Optional[str] = None,
                          extra_notes: str = "") -> str:
    """生成 markdown 汇总，引用已保存的 png/csv。"""
    md = []
    md.append("# InnGouQuant · A股日线策略回测报告（V1）\n")
    md.append(f"- 股池：{', '.join(result.per_stock.keys())}")
    md.append(f"- 交易日历：{result.calendar.min().date()} ~ {result.calendar.max().date()}（{len(result.calendar)} 交易日）")
    md.append(f"- 策略参数：n={params.n}, m={params.m}, w={params.w}, k={params.k}")
    md.append(f"- 初始资金：{result.init_capital:,.0f}（等权分配 {len(result.per_stock)} 只）")
    if split_date is not None:
        md.append(f"- 训练/测试切分日：**{split_date.date()}**（左 in-sample / 右 out-of-sample）")
    md.append("")

    if opt_summary:
        md.append("## 参数寻优结果\n")
        md.append(f"- 目标函数：**{opt_summary.get('objective')}**，跨股聚合：{opt_summary.get('robust_agg')}")

        # —— Walk-forward 为主判据（跨 regime 真 OOS + 参数稳定性）——
        wf = opt_summary.get("wf_summary")
        if wf and wf.get("n_windows", 0) > 0:
            md.append("\n### ① Walk-forward（主判据：滚动窗口真 OOS，已 purge 跨界交易+embargo）\n")
            md.append(f"- 窗口数：**{wf['n_windows']}**（每窗各自训练→紧邻 OOS 检验）")
            md.append(f"- OOS 年化中位数 {_fmt_pct(wf['oos_ann_median'])}，"
                      f"OOS Sharpe 中位数 {wf['oos_sharpe_median']:.2f}，"
                      f"OOS Calmar 中位数 {wf['oos_calmar_median']:.2f}")
            md.append(f"- **OOS α中位数 {_fmt_pct(wf.get('oos_alpha_median', float('nan')))}**，"
                      f"OOS β中位数 {wf.get('oos_beta_median', float('nan')):.2f}，"
                      f"OOS 超额中位数 {_fmt_pct(wf.get('oos_excess_median', float('nan')))}"
                      f"（**α=剔除沪深300后的真本事**）")
            md.append(f"- OOS 年化为正窗口占比 {_fmt_pct(wf['oos_positive_rate'])}，"
                      f"**OOS α为正窗口占比 {_fmt_pct(wf.get('oos_alpha_positive_rate', float('nan')))}**")
            md.append(f"- 参数稳定性：{wf['n_windows']} 窗选出 **{wf['param_unique']}** 套不同最优参数"
                      f"（众数 `{wf['modal_param']}` 仅占 {_fmt_pct(wf['param_stability'])}）")
            md.append(f"- **判定：{wf['verdict']}**")
            md.append("\n> 逐窗明细见 `walkforward.csv`。参数每窗都换 = 寻优在拟合噪声；"
                      "OOS 中位数才是策略真实期望。")

        md.append("\n### ② Holdout 70/30（参考主线，用于单一净值图）\n")
        md.append(f"- 切分方式：{opt_summary.get('split_mode')}")
        md.append(f"- 最优参数：**{opt_summary.get('best_params')}**")
        bt = opt_summary.get("best_train", {})
        bo = opt_summary.get("best_test", {})
        md.append(f"- 训练集(IS)：年化 {_fmt_pct(bt.get('ann_return',0))}，最大回撤 {_fmt_pct(bt.get('max_dd',0))}，"
                  f"Calmar {bt.get('calmar',0):.2f}，Sharpe {bt.get('sharpe',0):.2f}，交易 {bt.get('n_trades',0)}")
        md.append(f"- 测试集(OOS)：年化 {_fmt_pct(bo.get('ann_return',0))}，最大回撤 {_fmt_pct(bo.get('max_dd',0))}，"
                  f"Calmar {bo.get('calmar',0):.2f}，Sharpe {bo.get('sharpe',0):.2f}，交易 {bo.get('n_trades',0)}")
        verdict = opt_summary.get("overfit_verdict")
        if verdict:
            md.append(f"- 过拟合判定（holdout）：{verdict}")
        if heatmap_png:
            md.append(f"\n![heatmap]({os.path.basename(heatmap_png)})")
        md.append("")

    if bench_summary:
        md.append(_benchmark_md(bench_summary))

    md.append("## 指标明细（每股 + 组合）\n")
    md.append(_metrics_md_table(metrics_table))
    md.append("")

    if portfolio_png:
        md.append("## 组合净值\n")
        md.append(f"![portfolio]({os.path.basename(portfolio_png)})\n")

    if stock_pngs:
        md.append("## 各股价格与买卖点 / 净值\n")
        for code, p in stock_pngs.items():
            md.append(f"### {code}\n\n![{code}]({os.path.basename(p)})\n")

    # 风控（Tier 1 止损）
    md.append("## 风控（Tier 1 止损）\n")
    tot_stop = sum(getattr(r, "stop_exits", 0) for r in result.per_stock.values())
    trades_all = trades_to_frame(result)
    n_all = len(trades_all)
    md.append(f"- 止损配置：硬止损 {C.RISK.hard_stop_pct:.0%}（启用={C.RISK.use_hard_stop}）；"
              f"吊灯 {C.RISK.atr_trail_mult}×ATR（启用={C.RISK.use_atr_trailing}）；"
              f"收盘触发→次日开盘成交")
    if n_all:
        by_reason = trades_all["exit_reason"].value_counts().to_dict()
        mix = "，".join(f"{k}={v}" for k, v in by_reason.items())
        md.append(f"- 离场原因分布（共 {n_all} 笔）：{mix}")
        md.append(f"- **因止损离场：{tot_stop} 笔（占 {tot_stop/n_all:.0%}）**")
    md.append("")

    # A股约束自检摘要
    md.append("## A股约束自检\n")
    tot_skip = sum(r.skipped_buys for r in result.per_stock.values())
    tot_defer = sum(r.deferred_sells for r in result.per_stock.values())
    md.append(f"- 因一字涨停【跳过买入】次数合计：{tot_skip}")
    md.append(f"- 因一字跌停【顺延卖出】天数合计：{tot_defer}")
    md.append("- 无前视：t 日收盘信号 → t+1 日开盘成交（引擎显式挂单到下一根 bar）。")
    md.append("- T+1：买入次日才可卖（卖出仅在持仓日严格晚于买入日时触发）。")
    md.append("- 停牌：无 bar 视为停牌，持仓穿越。")
    md.append("")

    if extra_notes:
        md.append("## 备注\n")
        md.append(extra_notes)

    text = "\n".join(md)
    path = os.path.join(outdir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
