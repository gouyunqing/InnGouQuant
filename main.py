"""
main.py — 端到端流水线：体检 → 复权 → 因子 → 回测 → 寻优 → 报告

用法：
    python main.py                # 基线 + holdout 寻优 + 全套报告
    python main.py --no-optimize  # 只跑基线回测与报告（最快验证全链路）
    python main.py --walkforward  # 用 walk-forward 寻优
    python main.py --objective sharpe

产物在 outputs/：metrics_*.csv、trades.csv、grid_search.csv、各 png、report.md
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import pandas as pd

import config as C
import data_loader as DL
import factors as F
import strategy as S
import backtest as BT
import report as R
import optimize as OPT
import benchmark as BM


def set_seed(seed: int = C.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _build_bench_summary(best_result, bench_eq, split_date, best_params, runs):
    """组装‘组合 vs 沪深300’的分段(IS/OOS/全样本) α/β/IR + 抗过拟合 DSR。"""
    pe = best_result.portfolio_equity
    if split_date is not None:
        segs = {"in_sample": (None, split_date), "out_sample": (split_date, None),
                "full": (None, None)}
    else:
        segs = {"full": (None, None)}

    seg_metrics = {}
    for key, (s, e) in segs.items():
        eq = R.slice_equity(pe, s, e)
        bq = R.slice_equity(bench_eq, s, e)
        m = R.metrics_from_equity(eq)                    # 取策略年化
        m.update(R.metrics_vs_benchmark(eq, bq))         # β/α/超额/IR/捕获
        seg_metrics[key] = m

    # 去偏夏普 DSR：用 72 组试验的夏普分布，对‘挑最优’的选择偏差做修正
    dsr = None
    pbo = None
    if runs:
        trial_sharpes = [R.metrics_from_equity(r.portfolio_equity).get("sharpe", float("nan"))
                         for r in runs]
        best_run = next((r for r in runs if r.params.key() == best_params.key()), None)
        if best_run is not None:
            dsr = R.deflated_sharpe_ratio(best_run.portfolio_equity.pct_change(), trial_sharpes)
        # PBO（CSCV）：72 组参数的日收益矩阵 → 回测过拟合概率
        ret_mat = pd.concat({r.params.key(): r.portfolio_equity.pct_change() for r in runs},
                            axis=1)
        pbo = R.probability_of_backtest_overfitting(ret_mat, n_blocks=C.PBO_BLOCKS)

    return dict(benchmark_name=C.BENCHMARK_NAME, segments=seg_metrics, dsr=dsr, pbo=pbo)


def run(args) -> None:
    set_seed()
    outdir = C.ensure_output_dir()
    print("=" * 70)
    print("InnGouQuant · A股日线策略研究系统 V1")
    print("=" * 70)

    # ---------- M1 加载 + 复权 + 体检 ----------
    print("\n[1/6] 加载数据（前复权）...")
    pool = DL.get_universe(verbose=True) if C.USE_UNIVERSE else C.POOL
    print(f"    股池：{'程序化全市场 ' if C.USE_UNIVERSE else '手挑 '}{len(pool)} 只"
          f"；回测窗口：{C.BACKTEST_START or '数据起'} ~ {C.BACKTEST_END or '数据末'}")
    stocks = DL.load_pool(pool, verbose=False)
    if not stocks:
        print("没有可用股票，终止。")
        return
    health = DL.health_check(stocks)
    health.to_csv(os.path.join(outdir, "health_check.csv"), encoding="utf-8-sig")

    # ---------- M2 因子（基线参数） ----------
    params = C.DEFAULT_PARAMS
    print(f"[2/6] 计算因子（基线 n={params.n}, m={params.m}, w={params.w}, k={params.k}）...")
    # 因子在全历史因果计算，再裁剪到回测窗口（无 warmup 断层、无前视，窗口外不交易）。
    factor_panel = {c: DL.trim_to_window(fdf)
                    for c, fdf in F.compute_factor_panel(stocks, params).items()}

    # ---------- M3 信号 ----------
    print("[3/6] 生成信号（SuperTrend 翻多 + 量能突破）...")
    signals_panel = {c: S.generate_signals(factor_panel[c], params) for c in stocks}

    # ---------- M4 回测（基线） ----------
    print("[4/6] 回测（A股约束：无前视 / T+1 / 涨跌停 / 停牌 / 成本）...")
    result = BT.run_backtest(factor_panel, signals_panel)
    trades_df = BT.trades_to_frame(result)
    trades_df.to_csv(os.path.join(outdir, "trades.csv"), index=False, encoding="utf-8-sig")
    print(f"    基线总交易笔数：{len(trades_df)}；组合期末净值："
          f"{result.portfolio_equity.iloc[-1]:,.0f}（初始 {result.init_capital:,.0f}）")

    # ---------- M5 寻优 ----------
    opt_summary = None
    heatmap_png = None
    split_date = None
    best_result = result
    best_params = params

    wf_summary = None
    runs = None
    if not args.no_optimize:
        print("\n[5/6] 参数寻优 ...")
        # 72 组 × 全池回测只跑一次，holdout 与 walk-forward 共用，避免重复回测。
        runs = OPT.precompute_all_runs(stocks)
        # walk-forward 现为主判据（跨 regime 的真 OOS + 参数稳定性）。
        wf = OPT.optimize_walkforward(stocks, objective=args.objective, runs=runs)
        wf["wf_table"].to_csv(os.path.join(outdir, "walkforward.csv"),
                              index=False, encoding="utf-8-sig")
        wf_summary = wf.get("wf_summary")
        # holdout 70/30 作为参考主线（出单一 best 净值图与逐股报告）。
        opt = OPT.optimize_holdout(stocks, objective=args.objective, runs=runs)

        opt["grid_df"].to_csv(os.path.join(outdir, "grid_search.csv"),
                              index=False, encoding="utf-8-sig")
        best_params = opt["best_params"]
        split_date = opt["split_date"]
        heatmap_png = R.plot_heatmap(opt["grid_df"], outdir, metric="objective", row="n", col="m")

        # 用最优参数重算一遍（同样裁剪到窗口），作为报告主结果
        factor_panel = {c: DL.trim_to_window(fdf)
                        for c, fdf in F.compute_factor_panel(stocks, best_params).items()}
        signals_panel = {c: S.generate_signals(factor_panel[c], best_params) for c in stocks}
        best_result = BT.run_backtest(factor_panel, signals_panel)
        trades_df = BT.trades_to_frame(best_result)
        trades_df.to_csv(os.path.join(outdir, "trades.csv"), index=False, encoding="utf-8-sig")

        opt_summary = dict(
            objective=opt["objective"], robust_agg=opt["robust_agg"],
            split_mode=opt["split_mode"], best_params=opt["best_params_str"],
            best_train=opt["best_train"], best_test=opt["best_test"],
            overfit_verdict=opt["overfit_verdict"], wf_summary=wf_summary,
        )
    else:
        print("\n[5/6] 跳过寻优（--no-optimize）。")

    # ---------- M6 报告 ----------
    print("[6/6] 生成报告与图 ...")
    metrics_table = R.build_metrics_table(best_result, split_date=split_date)
    metrics_table.to_csv(os.path.join(outdir, "metrics.csv"), index=False, encoding="utf-8-sig")

    # —— 基准相对指标（组合 vs 沪深300）+ 抗过拟合去偏夏普 DSR ——
    bench_eq = BM.benchmark_equity(best_result.calendar, C.BENCHMARK_CODE)
    bench_summary = _build_bench_summary(best_result, bench_eq, split_date, best_params, runs)

    # 大股池下逐股出图无意义且慢，只画前 MAX_STOCK_PLOTS 只作抽样展示。
    stock_pngs = {}
    plot_codes = list(best_result.per_stock)[:C.MAX_STOCK_PLOTS]
    for code in plot_codes:
        stock_pngs[code] = R.plot_stock(code, factor_panel[code], best_result.per_stock[code],
                                        outdir, split_date=split_date)
    portfolio_png = R.plot_portfolio(best_result, outdir, split_date=split_date,
                                     benchmark=bench_eq, benchmark_name=C.BENCHMARK_NAME)

    notes = (
        "- 数据源：日线=`a股分钟线/日k/全部日k.zip`（按股票CSV，全历史至2026-04-30，直接读zip成员不解压）。\n"
        "- 前复权锚定到【各股窗口末日】后复权因子（防前视，末日前复权价=原始价）。\n"
        "- **【已修·复盘①】闲置现金按无风险利率 2%/年 计息**（空仓期不再 0 收益，修正 α 被低估）。\n"
        "- **【已修·复盘②】基准用沪深300【全收益】口径**（价格指数日收益加回 2%/年股息），避免高估超额/α。\n"
        "- **【数据硬限制·复盘③】幸存者偏差无法修复**：本数据集只含【当前仍上市】的股票，"
        "退市/爆雷股完全缺失（抽样600只末日全部≥2024）。故所有回测结果对真实历史【系统性偏乐观】，"
        "需要含退市股的数据源才能修正。\n"
        "- **【未修·复盘】股池按全窗口成交额选取，非 point-in-time**（轻微前视；因幸存者偏差更主导，暂缓）。\n"
        "- ST/涨跌停按板块 ±10%/±20%；量/额未复权（量能突破为相对自身均量比值，可接受）。"
    )
    report_path = R.write_markdown_report(
        outdir, params=best_params, metrics_table=metrics_table, result=best_result,
        split_date=split_date, opt_summary=opt_summary, bench_summary=bench_summary,
        stock_pngs=stock_pngs, portfolio_png=portfolio_png, heatmap_png=heatmap_png,
        extra_notes=notes,
    )

    print("\n完成 ✅  产物目录：", outdir)
    print("  - report.md（汇总）")
    print("  - metrics.csv / trades.csv / health_check.csv" +
          ("" if args.no_optimize else " / grid_search.csv"))
    print("  - portfolio_equity.png / stock_*.png" +
          ("" if heatmap_png is None else " / optimize_heatmap.png"))
    print("  报告：", report_path)


def parse_args():
    ap = argparse.ArgumentParser(description="InnGouQuant A股日线策略研究系统 V1")
    ap.add_argument("--no-optimize", action="store_true", help="只跑基线回测与报告")
    ap.add_argument("--walkforward", action="store_true", help="用 walk-forward 寻优（额外输出）")
    ap.add_argument("--objective", default=C.OBJECTIVE, choices=["calmar", "sharpe"],
                    help="寻优目标函数")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
