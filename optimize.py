"""
optimize.py — M5 参数寻优（训练）

待优化参数（可配，见 config.PARAM_GRID）：
    n ∈ {7,10,14}, m ∈ {2,2.5,3,3.5}, w ∈ {5,10}, k ∈ {1.3,1.5,2.0}

时间序列切分（严禁随机打乱）：
    (a) holdout：顺序切 train/test（默认 70/30）
    (b) walkforward：滚动窗口

防过拟合硬约束：
    ① 最少交易次数门槛（训练集低于则该参数组判无效）。
    ② 跨股稳健性：用【每股目标值的中位数】做选择分（抗“只在单只爆赚”），并要求多数股票有效。
    ③ 必报 OOS：用训练集选出的最优参数，在测试集复核，报告 IS vs OOS。

效率要点：
    · 每个参数组合的整池回测【只跑一次】，缓存其每股净值/交易与组合净值；
      不同切分(window)只是对缓存做【日期切片】算指标 —— 故 walk-forward 也很快。
    · 因子缓存：SuperTrend 只依赖 (n,m)，量能突破只依赖 (w,k)，分别缓存。
    · 因子在【全序列】上因果计算（指标只用 t 及之前），再按日期切 train/test —— 无前视、无 warmup 缺口。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import strategy as S
import backtest as BT
import report as R
import benchmark as BM

Seg = Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]


# --------------------------------------------------------------------------- #
# 因子缓存
# --------------------------------------------------------------------------- #
class FactorCache:
    def __init__(self, stocks: Dict[str, pd.DataFrame]):
        self.stocks = stocks
        self._st: Dict[Tuple[str, int, float], pd.DataFrame] = {}
        self._vb: Dict[Tuple[str, int, float], pd.Series] = {}

    def supertrend(self, code: str, n: int, m: float) -> pd.DataFrame:
        key = (code, n, m)
        if key not in self._st:
            self._st[key] = F.supertrend(self.stocks[code], n=n, m=m)
        return self._st[key]

    def vol_break(self, code: str, w: int, k: float) -> pd.Series:
        key = (code, w, k)
        if key not in self._vb:
            self._vb[key] = F.volume_break(self.stocks[code]["volume"], w=w, k=k)
        return self._vb[key]

    def factor_df(self, code: str, p: C.StrategyParams) -> pd.DataFrame:
        df = self.stocks[code]
        st = self.supertrend(code, p.n, p.m)
        vb = self.vol_break(code, p.w, p.k)
        fdf = pd.DataFrame(index=df.index)
        for col in ["open", "high", "low", "close", "volume", "amount", "prev_close",
                    "limit_up", "limit_down", "one_word_up", "one_word_down"]:
            fdf[col] = df[col]
        fdf["dir"] = st["dir"]
        fdf["supertrend"] = st["supertrend"]
        fdf["atr"] = st["atr"]
        fdf["flip_up"] = st["flip_up"]
        fdf["flip_down"] = st["flip_down"]
        fdf["vol_break"] = vb
        return fdf


# --------------------------------------------------------------------------- #
# 一个参数组合的整池回测结果（只跑一次，重复切片）
# --------------------------------------------------------------------------- #
@dataclass
class ComboRun:
    params: C.StrategyParams
    per_stock_equity: Dict[str, pd.Series]
    per_stock_trades: Dict[str, pd.DataFrame]
    portfolio_equity: pd.Series
    trades_all: pd.DataFrame


def run_combo(cache: FactorCache, p: C.StrategyParams) -> ComboRun:
    codes = list(cache.stocks.keys())
    # 因子在全历史因果计算 → 再裁剪到回测窗口（无 warmup 断层、无前视，只是窗口外不交易）。
    factor_panel = {c: DL.trim_to_window(cache.factor_df(c, p)) for c in codes}
    signals_panel = {c: S.generate_signals(factor_panel[c], p) for c in codes}
    result = BT.run_backtest(factor_panel, signals_panel)
    trades_all = BT.trades_to_frame(result)
    return ComboRun(
        params=p,
        per_stock_equity={c: result.per_stock[c].equity for c in codes},
        per_stock_trades={c: trades_all[trades_all["code"] == c] for c in codes},
        portfolio_equity=result.portfolio_equity,
        trades_all=trades_all,
    )


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #
def _calendar(stocks: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    cal = None
    for df in stocks.values():
        cal = df.index if cal is None else cal.union(df.index)
    # 切分与滚动窗口都只在【回测损益窗口】内进行（与 run_combo 的裁剪一致）。
    start, end = DL.backtest_window()
    if start is not None:
        cal = cal[cal >= start]
    if end is not None:
        cal = cal[cal <= end]
    return cal


def holdout_split_date(calendar: pd.DatetimeIndex, train_ratio: float = C.TRAIN_RATIO) -> pd.Timestamp:
    cal = calendar.sort_values()
    i = int(len(cal) * train_ratio)
    i = min(max(i, 1), len(cal) - 1)
    return cal[i - 1]


def walkforward_windows(calendar: pd.DatetimeIndex) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    cal = calendar.sort_values()
    wins = []
    start = 0
    while start + C.WF_TRAIN_DAYS + 1 < len(cal):
        tr_s = start
        tr_e = start + C.WF_TRAIN_DAYS - 1
        te_s = tr_e + 1
        te_e = min(te_s + C.WF_TEST_DAYS - 1, len(cal) - 1)
        if te_s > len(cal) - 1:
            break
        wins.append((cal[tr_s], cal[tr_e], cal[te_s], cal[te_e]))
        start += C.WF_STEP_DAYS
    return wins


# --------------------------------------------------------------------------- #
# 评分（纯切片，不再回测）
# --------------------------------------------------------------------------- #
def _objective_value(metrics: dict, objective: str) -> float:
    v = metrics.get(objective, 0.0)
    return v if np.isfinite(v) else 0.0


def score_run(run: ComboRun, train_seg: Seg, test_seg: Seg,
              objective: str = C.OBJECTIVE, robust_agg: str = C.ROBUST_AGG,
              benchmark: Optional[pd.Series] = None) -> dict:
    """对已缓存的整池回测结果，在给定 train/test 切分上算选择分与组合指标。
    传入 benchmark（基准净值）时，组合指标会附带 β/α/信息比率等基准相对项。"""
    codes = list(run.per_stock_equity.keys())
    tr_s, tr_e = train_seg
    te_s, te_e = test_seg

    per_stock_obj = []
    valid_stocks = 0
    total_train_trades = 0
    for c in codes:
        eq = R.slice_equity(run.per_stock_equity[c], tr_s, tr_e)
        td = R.slice_trades(run.per_stock_trades[c], tr_s, tr_e)
        total_train_trades += len(td)
        if len(td) >= C.MIN_TRADES_PER_STOCK:
            valid_stocks += 1
            m = R.compute_metrics(eq, td)
            per_stock_obj.append(_objective_value(m, objective))

    tr_bench = R.slice_equity(benchmark, tr_s, tr_e) if benchmark is not None else None
    te_bench = R.slice_equity(benchmark, te_s, te_e) if benchmark is not None else None
    train_pf = R.compute_metrics(R.slice_equity(run.portfolio_equity, tr_s, tr_e),
                                 R.slice_trades(run.trades_all, tr_s, tr_e), benchmark=tr_bench)
    test_pf = R.compute_metrics(R.slice_equity(run.portfolio_equity, te_s, te_e),
                                R.slice_trades(run.trades_all, te_s, te_e),
                                benchmark=te_bench) if te_s is not None else {}

    majority = (valid_stocks >= int(np.ceil(len(codes) / 2)))
    enough_total = total_train_trades >= C.MIN_TRADES_TOTAL
    if per_stock_obj and majority and enough_total:
        agg = np.median if robust_agg == "median" else np.mean
        score = float(agg(per_stock_obj))
    else:
        score = float("-inf")

    return dict(params=run.params, score=score, valid_stocks=valid_stocks,
                total_train_trades=total_train_trades,
                train_portfolio=train_pf, test_portfolio=test_pf,
                train_objective=_objective_value(train_pf, objective))


def _purged_test_metrics(run: ComboRun, te_s: pd.Timestamp, te_e: pd.Timestamp,
                         benchmark: Optional[pd.Series], embargo: int = None) -> dict:
    """OOS 指标做 purge + embargo：只统计【完全落在测试窗内】的交易（剔除跨界：训练段买入、
    测试段才卖出的单），净值切片再跳过测试窗开头 embargo 个交易日（去掉跨界持仓的净值尾巴/自相关）。"""
    embargo = C.EMBARGO_DAYS if embargo is None else embargo
    eq = R.slice_equity(run.portfolio_equity, te_s, te_e)
    if embargo > 0 and len(eq) > embargo + 2:
        eq = eq.iloc[embargo:]
    td = run.trades_all
    if len(td):
        en = pd.to_datetime(td["entry_date"]); ex = pd.to_datetime(td["exit_date"])
        td = td[(en >= te_s) & (ex <= te_e)]            # 完全包含才算 OOS 交易
    be = (R.slice_equity(benchmark, eq.index.min(), te_e)
          if (benchmark is not None and len(eq)) else None)
    return R.compute_metrics(eq, td, benchmark=be)


def grid_combos(grid: Dict[str, list] = None) -> List[C.StrategyParams]:
    grid = grid or C.PARAM_GRID
    out = []
    for n, m, w, k in itertools.product(grid["n"], grid["m"], grid["w"], grid["k"]):
        out.append(C.StrategyParams(n=n, m=m, w=w, k=k))
    return out


def _precompute_runs(stocks: Dict[str, pd.DataFrame], combos: List[C.StrategyParams],
                     verbose: bool = True) -> List[ComboRun]:
    """对每个参数组合做一次整池回测并缓存（核心提速点）。"""
    cache = FactorCache(stocks)
    runs = []
    for i, p in enumerate(combos, 1):
        runs.append(run_combo(cache, p))
        if verbose and (i % 12 == 0 or i == len(combos)):
            print(f"  ...{i}/{len(combos)} 组合回测完毕")
    return runs


def precompute_all_runs(stocks: Dict[str, pd.DataFrame], verbose: bool = True) -> List[ComboRun]:
    """整批参数组合只回测一次，holdout 与 walk-forward 共用（避免重复跑全池回测）。"""
    combos = grid_combos()
    if verbose:
        print(f"  预回测 {len(combos)} 个参数组合 × {len(stocks)} 只股票（holdout/WF 共用）...")
    return _precompute_runs(stocks, combos, verbose)


# --------------------------------------------------------------------------- #
# Holdout 寻优
# --------------------------------------------------------------------------- #
def optimize_holdout(stocks: Dict[str, pd.DataFrame],
                     objective: str = C.OBJECTIVE, verbose: bool = True,
                     runs: Optional[List[ComboRun]] = None) -> dict:
    calendar = _calendar(stocks)
    split = holdout_split_date(calendar)
    train_seg, test_seg = (None, split), (split, None)

    combos = grid_combos()
    if verbose:
        print(f"\n========== 参数寻优（holdout 70/30，切分日 {split.date()}）==========")
        print(f"网格组合数：{len(combos)}，目标：{objective}，跨股聚合：{C.ROBUST_AGG}")
    if runs is None:
        runs = _precompute_runs(stocks, combos, verbose)

    rows, best = [], None
    for run in runs:
        res = score_run(run, train_seg, test_seg, objective)
        p = run.params
        rows.append({
            "n": p.n, "m": p.m, "w": p.w, "k": p.k,
            "objective": res["score"] if np.isfinite(res["score"]) else np.nan,
            "train_obj": res["train_objective"], "valid_stocks": res["valid_stocks"],
            "train_trades": res["total_train_trades"],
            "test_calmar": res["test_portfolio"].get("calmar", np.nan),
        })
        if best is None or res["score"] > best["score"]:
            best = res
    grid_df = pd.DataFrame(rows)

    if best is None or not np.isfinite(best["score"]):
        if verbose:
            print("  [告警] 无参数组合通过最少交易/稳健性门槛，回退默认参数。")
        default_run = run_combo(FactorCache(stocks), C.DEFAULT_PARAMS)
        best = score_run(default_run, train_seg, test_seg, objective)

    bp, bt, bo = best["params"], best["train_portfolio"], best["test_portfolio"]
    verdict = _overfit_verdict(bt, bo, objective)
    if verbose:
        print(f"\n最优参数：n={bp.n}, m={bp.m}, w={bp.w}, k={bp.k}")
        print(f"  IS  : 年化 {bt.get('ann_return',0):.2%}  回撤 {bt.get('max_dd',0):.2%}  "
              f"Calmar {bt.get('calmar',0):.2f}  交易 {bt.get('n_trades',0)}")
        print(f"  OOS : 年化 {bo.get('ann_return',0):.2%}  回撤 {bo.get('max_dd',0):.2%}  "
              f"Calmar {bo.get('calmar',0):.2f}  交易 {bo.get('n_trades',0)}")
        print(f"  过拟合判定：{verdict}")
        print("=================================================================\n")

    return dict(best_params=bp, split_date=split, grid_df=grid_df,
                best_train=bt, best_test=bo, objective=objective,
                robust_agg=C.ROBUST_AGG, split_mode="holdout 70/30",
                overfit_verdict=verdict,
                best_params_str=f"n={bp.n}, m={bp.m}, w={bp.w}, k={bp.k}")


def _overfit_verdict(train_pf: dict, test_pf: dict, objective: str) -> str:
    tis = _objective_value(train_pf, objective)
    toos = _objective_value(test_pf, objective)
    oos_ret = test_pf.get("ann_return", 0.0)
    if test_pf.get("n_trades", 0) == 0:
        return "测试集无交易，无法评估（样本不足）"
    if oos_ret <= 0:
        return f"疑似过拟合：OOS 年化为负({oos_ret:.1%})"
    if tis > 0 and toos < 0.5 * tis:
        return f"疑似过拟合：OOS {objective}({toos:.2f}) 不足 IS({tis:.2f}) 的一半"
    return f"OOS 表现可接受（IS {objective} {tis:.2f} → OOS {toos:.2f}）"


# --------------------------------------------------------------------------- #
# Walk-forward 寻优（复用同一批缓存回测）
# --------------------------------------------------------------------------- #
def optimize_walkforward(stocks: Dict[str, pd.DataFrame],
                         objective: str = C.OBJECTIVE, verbose: bool = True,
                         runs: Optional[List[ComboRun]] = None) -> dict:
    calendar = _calendar(stocks)
    bench = BM.benchmark_equity(calendar)        # 沪深300，逐窗算 OOS α/β
    windows = walkforward_windows(calendar)
    combos = grid_combos()
    if verbose:
        print(f"\n========== 参数寻优（walk-forward，{len(windows)} 窗 × {len(combos)} 组合）==========")
    if runs is None:
        runs = _precompute_runs(stocks, combos, verbose)

    win_rows = []
    for wi, (tr_s, tr_e, te_s, te_e) in enumerate(windows, 1):
        best = None
        for run in runs:
            res = score_run(run, (tr_s, tr_e), (te_s, te_e), objective, benchmark=bench)
            if best is None or res["score"] > best["score"]:
                best = res
        bp = best["params"]
        # 选参用未 purge 的训练分（不动）；但报告的 OOS 用 purged+embargo 的干净版本。
        best_run = next((r for r in runs if r.params.key() == bp.key()), None)
        ptf = (_purged_test_metrics(best_run, te_s, te_e, bench)
               if best_run is not None else best["test_portfolio"])
        win_rows.append({
            "window": wi, "train": f"{tr_s.date()}~{tr_e.date()}",
            "test": f"{te_s.date()}~{te_e.date()}", "best": f"n{bp.n}_m{bp.m}_w{bp.w}_k{bp.k}",
            "is_calmar": best["train_portfolio"].get("calmar", np.nan),
            "oos_ann": ptf.get("ann_return", np.nan),
            "oos_ann_raw": best["test_portfolio"].get("ann_return", np.nan),  # 未purge，看污染量
            "oos_sharpe": ptf.get("sharpe", np.nan),
            "oos_calmar": ptf.get("calmar", np.nan),
            "oos_alpha": ptf.get("alpha_ann", np.nan),
            "oos_beta": ptf.get("beta", np.nan),
            "oos_excess": ptf.get("excess_ann", np.nan),
            "oos_trades": ptf.get("n_trades", 0),
        })
        if verbose:
            r = win_rows[-1]
            ic = r["is_calmar"]; oa = r["oos_ann"]
            print(f"  窗{wi} [{r['test']}] best={r['best']}  "
                  f"IS_Calmar={ic:.2f}  OOS_年化={oa:.2%}" if np.isfinite(ic) and np.isfinite(oa)
                  else f"  窗{wi} [{r['test']}] best={r['best']}")

    wf_df = pd.DataFrame(win_rows)
    summary = _wf_summary(wf_df)
    if verbose and len(wf_df):
        print("\n  —— Walk-forward 汇总（OOS 才是真考卷）——")
        print(f"  窗口数：{summary['n_windows']}")
        print(f"  OOS 年化中位数：{summary['oos_ann_median']:.2%}    "
              f"OOS Sharpe 中位数：{summary['oos_sharpe_median']:.2f}    "
              f"OOS Calmar 中位数：{summary['oos_calmar_median']:.2f}")
        print(f"  OOS α中位数：{summary['oos_alpha_median']:.2%}    "
              f"OOS β中位数：{summary['oos_beta_median']:.2f}    "
              f"OOS 超额中位数：{summary['oos_excess_median']:.2%}（剔大盘后的真本事）")
        print(f"  OOS 年化为正的窗口占比：{summary['oos_positive_rate']:.0%}    "
              f"OOS α为正的窗口占比：{summary['oos_alpha_positive_rate']:.0%}")
        print(f"  参数稳定性：{summary['n_windows']} 窗里选出 {summary['param_unique']} 种不同最优参数"
              f"（众数 {summary['modal_param']} 仅占 {summary['param_stability']:.0%}）")
        print(f"  判定：{summary['verdict']}")
        print("====================================================================\n")
    return dict(split_mode="walkforward", wf_table=wf_df, objective=objective,
                wf_summary=summary)


def _wf_summary(wf_df: pd.DataFrame) -> dict:
    """从逐窗 OOS 结果汇总：OOS 中位数、为正占比、参数稳定性，并给出过拟合判定。

    过拟合的两个铁证：① OOS 中位数差/为负；② 每个窗口选出的最优参数都在变（说明
    寻优在拟合每段噪声，没有可持续的 edge）。这里把两者量化成可读的判定。
    """
    nan = float("nan")
    if wf_df is None or wf_df.empty:
        return dict(n_windows=0, oos_ann_median=nan, oos_sharpe_median=nan,
                    oos_calmar_median=nan, oos_alpha_median=nan, oos_beta_median=nan,
                    oos_excess_median=nan, oos_positive_rate=nan, oos_alpha_positive_rate=nan,
                    param_unique=0, modal_param="—", param_stability=nan,
                    verdict="无窗口，样本不足")
    n = len(wf_df)
    oos_ann = wf_df["oos_ann"].dropna()
    oos_alpha = wf_df["oos_alpha"].dropna()
    counts = wf_df["best"].value_counts()
    modal_param = str(counts.index[0]) if len(counts) else "—"
    param_unique = int(wf_df["best"].nunique())
    param_stability = float(counts.iloc[0] / n) if len(counts) else nan
    oos_ann_median = float(oos_ann.median()) if len(oos_ann) else nan
    oos_positive_rate = float((oos_ann > 0).mean()) if len(oos_ann) else nan
    oos_alpha_median = float(oos_alpha.median()) if len(oos_alpha) else nan
    oos_alpha_positive_rate = float((oos_alpha > 0).mean()) if len(oos_alpha) else nan

    # 判定逻辑：α 才是真本事——优先看剔除大盘后的超额是否稳定为正
    if np.isfinite(oos_alpha_median) and oos_alpha_median <= 0:
        verdict = (f"无 α/无 edge：剔除大盘后 OOS α 中位数为负({oos_alpha_median:.1%})——"
                   f"正收益主要来自 β（搭大盘便车），策略本身不创造超额")
    elif np.isfinite(param_stability) and param_stability < 0.4 and param_unique >= max(3, n // 2):
        verdict = (f"参数严重不稳定：{n} 窗选出 {param_unique} 套不同最优参数"
                   f"（众数仅占 {param_stability:.0%}）——寻优在拟合每段噪声，OOS 不可信")
    elif np.isfinite(oos_alpha_positive_rate) and oos_alpha_positive_rate < 0.5:
        verdict = (f"边际：α 中位数为正但仅 {oos_alpha_positive_rate:.0%} 的窗口有正 α，"
                   f"超额不稳健")
    else:
        verdict = (f"相对稳健：OOS α 中位数 {oos_alpha_median:.1%}，"
                   f"{oos_alpha_positive_rate:.0%} 的窗口有正 α，参数众数占 {param_stability:.0%}")
    return dict(n_windows=n, oos_ann_median=oos_ann_median,
                oos_sharpe_median=float(wf_df["oos_sharpe"].median()),
                oos_calmar_median=float(wf_df["oos_calmar"].median()),
                oos_alpha_median=oos_alpha_median,
                oos_beta_median=float(wf_df["oos_beta"].median()),
                oos_excess_median=float(wf_df["oos_excess"].median()),
                oos_positive_rate=oos_positive_rate,
                oos_alpha_positive_rate=oos_alpha_positive_rate, param_unique=param_unique,
                modal_param=modal_param, param_stability=param_stability,
                verdict=verdict)
