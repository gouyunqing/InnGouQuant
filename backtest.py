"""
backtest.py — M4 回测引擎（A股约束齐全）

核心约束（逐条对应需求）：
  · 无前视：t 日收盘信号 → t+1 日开盘成交（显式“挂单到下一根 bar 的 open”执行，等价 shift）。
  · T+1：当日买入次日才能卖（卖出仅在持仓日严格晚于买入日时发生）。
  · 涨跌停：主板 ±10% / 创业(300)·科创(688) ±20% / ST ±5%（由 data_loader 标好停板价与一字板）。
        一字涨停（开盘≥涨停价）→ 买不进 → 跳过该笔买入；
        一字跌停（开盘≤跌停价）→ 卖不出 → 顺延到下一可成交 bar。
  · 停牌：无 bar 的交易日视为停牌，持仓穿越（按股自身 bar 迭代，挂单自然顺延到下一根真实 bar）。
  · 成本：佣金双边 0.025%(最低5元) + 印花税卖出 0.05% + 过户费双边 0.001% + 滑点 0.1%~0.2%。
  · 长仓 only；每股同一时间至多 1 个仓位。

输出：逐笔交易日志、每股净值曲线、持仓序列、股池等权组合净值。

口径说明：每只股分配等额“资金小账户(sleeve)”独立交易（账户内复利），
组合净值 = 各股净值在统一交易日历上对齐(ffill) 后求和。简单、无跨股现金耦合。
量/额未做复权调整（V1 已知简化），量能突破是相对自身均量的比值，可接受。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C


@dataclass
class Trade:
    code: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float        # 前复权成交价（含滑点）
    exit_price: float
    shares: int
    gross_pnl: float          # 不含费的价差盈亏
    fees: float               # 买+卖总费用
    net_pnl: float            # 含费净盈亏
    holding_days: int         # 持仓自然交易日数（bar 数）
    ret: float                # 该笔净收益率（相对买入成本）
    exit_reason: str          # 'signal' / 'eod'(数据末尾强平)


@dataclass
class StockResult:
    code: str
    equity: pd.Series                 # 每股净值（绝对金额，索引=该股 bar 日期）
    position: pd.Series               # 持仓股数（0 或 lot 的整数倍）
    trades: List[Trade] = field(default_factory=list)
    sleeve: float = 0.0               # 该股初始小账户资金
    skipped_buys: int = 0             # 因一字涨停被跳过的买入次数
    deferred_sells: int = 0           # 因一字跌停被顺延的卖出次数（累计天数）
    stop_exits: int = 0               # 因止损（硬/吊灯）离场的笔数
    open_position: bool = False       # 回测末尾是否仍持仓


@dataclass
class BacktestResult:
    per_stock: Dict[str, StockResult]
    portfolio_equity: pd.Series       # 组合净值（绝对金额）
    calendar: pd.DatetimeIndex
    init_capital: float


# --------------------------------------------------------------------------- #
# 费用
# --------------------------------------------------------------------------- #
def _buy_fees(notional: float, cost: C.CostModel) -> float:
    return max(cost.commission_rate * notional, cost.commission_min) + cost.transfer_rate * notional


def _sell_fees(notional: float, cost: C.CostModel) -> float:
    return (max(cost.commission_rate * notional, cost.commission_min)
            + cost.stamp_rate * notional
            + cost.transfer_rate * notional)


# --------------------------------------------------------------------------- #
# 单只股票回测
# --------------------------------------------------------------------------- #
def backtest_stock(code: str, factor_df: pd.DataFrame, signals: pd.DataFrame,
                   sleeve: float, cost: C.CostModel = C.COST,
                   lot: int = C.LOT_SIZE, risk: C.RiskParams = C.RISK,
                   credit_cash: bool = None) -> StockResult:
    """对单只股票做事件驱动回测（信号 t 收盘 → t+1 开盘成交）。

    退出有三类（收盘触发、次日开盘成交，互相取先到者）：
      · 'signal' = SuperTrend 翻空 (flip_down)
      · 'stop'   = 硬止损(-hard_stop_pct) 或 吊灯止损(峰值回落 atr_trail_mult×ATR)
      · 'eod'    = 数据末尾强平
    """
    df = factor_df
    idx = df.index
    n = len(idx)

    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    atr_arr = df["atr"].to_numpy(dtype=float) if "atr" in df else np.full(n, np.nan)
    ow_up = df["one_word_up"].to_numpy(dtype=bool)      # 一字涨停（买不进）
    ow_down = df["one_word_down"].to_numpy(dtype=bool)  # 一字跌停（卖不出）
    entry_sig = signals["entry"].to_numpy(dtype=bool)
    exit_sig = signals["exit"].to_numpy(dtype=bool)

    credit_cash = C.CREDIT_CASH_INTEREST if credit_cash is None else credit_cash
    rf_d = ((1.0 + C.RISK_FREE_ANNUAL) ** (1.0 / C.TRADING_DAYS_PER_YEAR) - 1.0) if credit_cash else 0.0

    cash = float(sleeve)
    shares = 0
    entry_price = 0.0
    entry_i = -1
    entry_fees = 0.0
    hwm_high = -np.inf           # 持仓期最高价（吊灯止损用）

    pending_buy = False
    pending_sell = False
    pending_reason = ""          # 待成交卖单的离场原因（'signal'/'stop'）

    equity = np.empty(n, dtype=float)
    pos_arr = np.zeros(n, dtype=float)
    trades: List[Trade] = []
    skipped_buys = 0
    deferred_sells = 0
    stop_exits = 0

    for i in range(n):
        # ---- 0) 在手现金按无风险利率计息（空仓期不再 0 收益）----
        if i > 0 and rf_d > 0.0 and cash > 0.0:
            cash *= (1.0 + rf_d)

        # ---- 1) 在 i 日开盘，处理上一根 bar 挂出的待成交单 ----
        if pending_buy and shares == 0:
            if ow_up[i]:
                skipped_buys += 1          # 一字涨停，买不进，撤单
                pending_buy = False
            else:
                fill = op[i] * (1 + cost.slippage)
                # 预留费用后按手数取整
                max_lots = int(cash / (fill * lot * (1 + cost.commission_rate + cost.transfer_rate)))
                if max_lots >= 1:
                    qty = max_lots * lot
                    notional = qty * fill
                    fee = _buy_fees(notional, cost)
                    cash -= (notional + fee)
                    shares = qty
                    entry_price = fill
                    entry_i = i
                    entry_fees = fee
                    hwm_high = hi[i]       # 重置持仓期最高价
                pending_buy = False

        if pending_sell and shares > 0 and entry_i < i:  # T+1：严格晚于买入日才可卖
            if ow_down[i]:
                deferred_sells += 1        # 一字跌停，卖不出，顺延（保留 pending_sell）
            else:
                fill = op[i] * (1 - cost.slippage)
                notional = shares * fill
                fee = _sell_fees(notional, cost)
                cash += (notional - fee)
                total_fees = entry_fees + fee
                gross = (fill - entry_price) * shares
                net = gross - total_fees
                if pending_reason == "stop":
                    stop_exits += 1
                trades.append(Trade(
                    code=code, entry_date=idx[entry_i], exit_date=idx[i],
                    entry_price=entry_price, exit_price=fill, shares=shares,
                    gross_pnl=gross, fees=total_fees, net_pnl=net,
                    holding_days=i - entry_i,
                    ret=net / (entry_price * shares) if shares else 0.0,
                    exit_reason=pending_reason or "signal",
                ))
                shares = 0
                entry_i = -1
                entry_fees = 0.0
                pending_sell = False
                pending_reason = ""

        # ---- 2) 用 i 日信号/止损挂【下一根 bar】的单（无前视，收盘触发→次开成交）----
        if shares == 0 and entry_sig[i]:
            pending_buy = True
        elif shares > 0 and not pending_sell:
            if hi[i] > hwm_high:
                hwm_high = hi[i]
            # 止损位 = 硬止损 与 吊灯止损 取较高(较紧)者
            stop_level = -np.inf
            if risk.use_hard_stop:
                stop_level = max(stop_level, entry_price * (1.0 - risk.hard_stop_pct))
            if risk.use_atr_trailing and np.isfinite(atr_arr[i]):
                stop_level = max(stop_level, hwm_high - risk.atr_trail_mult * atr_arr[i])
            stop_hit = cl[i] <= stop_level
            if risk.use_time_stop and (i - entry_i) >= risk.max_holding_days:
                stop_hit = True
            if exit_sig[i]:
                pending_sell, pending_reason = True, "signal"
            elif stop_hit:
                pending_sell, pending_reason = True, "stop"

        # ---- 3) i 日盯市净值 ----
        pos_arr[i] = shares
        equity[i] = cash + shares * cl[i]

    # ---- 末尾若仍持仓：按最后一根 close 强平（仅用于交易统计；净值已盯市）----
    # 注意：若恰在最后一根 bar 才买入（entry_i==n-1），强平会得到 0 持仓日且当日买卖，
    # 违反 T+1 的“次日才可卖”。这种持仓不计入交易日志（净值已盯市反映其浮盈亏），仅标记为未平仓。
    open_pos = shares > 0
    if open_pos and entry_i >= n - 1:
        pass  # 最后一根才买入：保留为未平仓，不生成 0 日交易
    elif open_pos:
        fill = cl[-1]
        notional = shares * fill
        fee = _sell_fees(notional, cost)
        total_fees = entry_fees + fee
        gross = (fill - entry_price) * shares
        net = gross - total_fees
        trades.append(Trade(
            code=code, entry_date=idx[entry_i], exit_date=idx[-1],
            entry_price=entry_price, exit_price=fill, shares=shares,
            gross_pnl=gross, fees=total_fees, net_pnl=net,
            holding_days=(n - 1) - entry_i,
            ret=net / (entry_price * shares) if shares else 0.0,
            exit_reason="eod",
        ))

    return StockResult(
        code=code,
        equity=pd.Series(equity, index=idx, name=code),
        position=pd.Series(pos_arr, index=idx, name=code),
        trades=trades, sleeve=sleeve,
        skipped_buys=skipped_buys, deferred_sells=deferred_sells,
        stop_exits=stop_exits, open_position=open_pos,
    )


# --------------------------------------------------------------------------- #
# 组合回测
# --------------------------------------------------------------------------- #
def run_backtest(factor_panel: Dict[str, pd.DataFrame],
                 signals_panel: Dict[str, pd.DataFrame],
                 init_capital: float = C.INIT_CAPITAL,
                 cost: C.CostModel = C.COST) -> BacktestResult:
    """整池等权回测。每股分配 init_capital/N 的独立小账户。"""
    codes = list(factor_panel.keys())
    if not codes:
        raise ValueError("空股池，无法回测。")
    sleeve = init_capital / len(codes)

    per_stock: Dict[str, StockResult] = {}
    for code in codes:
        per_stock[code] = backtest_stock(code, factor_panel[code], signals_panel[code], sleeve, cost)

    # 统一交易日历 = 各股 bar 日期并集
    cal = None
    for r in per_stock.values():
        cal = r.equity.index if cal is None else cal.union(r.equity.index)

    # 组合净值：每股净值对齐到日历（停牌/未上市前用 ffill；首个 bar 之前 = 初始 sleeve 现金）
    parts = []
    for r in per_stock.values():
        e = r.equity.reindex(cal).ffill()
        e = e.fillna(sleeve)        # 该股上市前，sleeve 资金闲置
        parts.append(e)
    portfolio = pd.concat(parts, axis=1).sum(axis=1)
    portfolio.name = "portfolio"

    return BacktestResult(per_stock=per_stock, portfolio_equity=portfolio,
                          calendar=cal, init_capital=init_capital)


# --------------------------------------------------------------------------- #
# 交易日志导出
# --------------------------------------------------------------------------- #
def trades_to_frame(result: BacktestResult) -> pd.DataFrame:
    """把所有逐笔交易拍平成一张表。"""
    rows = []
    for code, r in result.per_stock.items():
        for t in r.trades:
            rows.append({
                "code": t.code,
                "entry_date": t.entry_date.date(),
                "exit_date": t.exit_date.date(),
                "holding_days": t.holding_days,
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4),
                "shares": t.shares,
                "gross_pnl": round(t.gross_pnl, 2),
                "fees": round(t.fees, 2),
                "net_pnl": round(t.net_pnl, 2),
                "ret": round(t.ret, 4),
                "exit_reason": t.exit_reason,
            })
    cols = ["code", "entry_date", "exit_date", "holding_days", "entry_price", "exit_price",
            "shares", "gross_pnl", "fees", "net_pnl", "ret", "exit_reason"]
    return pd.DataFrame(rows, columns=cols)
