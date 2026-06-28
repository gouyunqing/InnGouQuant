"""
strategy.py — M3 策略层（可插拔）

接口约定（V2 换 LightGBM 时只需实现同样的 generate_signals）：
    generate_signals(factor_df, params) -> DataFrame[index=date, cols=entry, exit, target_pos]

规则策略（多头、T+1）：
    买入(entry) = flip_up AND vol_break
    卖出(exit)  = flip_down
    target_pos  = t 日收盘后的“理想目标仓位”(0/1)，仅供参考；
                  真实成交受 A股约束（T+1 / 涨跌停 / 停牌）影响，由 backtest 引擎在 t+1 开盘执行。

为什么输出 entry/exit 而非直接成交：把“信号”与“受约束的成交”解耦，
回测引擎据此施加无前视、T+1、一字板等规则——这也让未来 ML 策略可无缝替换。
"""
from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

import config as C


class Strategy(Protocol):
    """策略接口。任何实现此方法的对象都可被回测引擎驱动。"""

    def generate_signals(self, factor_df: pd.DataFrame, params: C.StrategyParams) -> pd.DataFrame:
        ...


def _target_position_from_events(entry: pd.Series, exit_: pd.Series) -> pd.Series:
    """由 entry/exit 事件推出“理想”目标仓位序列（不含约束）。entry 优先于同日 exit。"""
    pos = np.zeros(len(entry), dtype=int)
    held = 0
    e = entry.to_numpy()
    x = exit_.to_numpy()
    for i in range(len(pos)):
        if held == 0 and e[i]:
            held = 1
        elif held == 1 and x[i]:
            held = 0
        pos[i] = held
    return pd.Series(pos, index=entry.index)


class RuleStrategy:
    """SuperTrend 翻多 + 量能突破 的规则策略。"""

    def generate_signals(self, factor_df: pd.DataFrame, params: C.StrategyParams) -> pd.DataFrame:
        entry = (factor_df["flip_up"] & factor_df["vol_break"]).fillna(False)
        exit_ = factor_df["flip_down"].fillna(False)
        target = _target_position_from_events(entry, exit_)
        out = pd.DataFrame(index=factor_df.index)
        out["entry"] = entry.astype(bool)
        out["exit"] = exit_.astype(bool)
        out["target_pos"] = target.astype(int)
        return out


# 默认策略实例
DEFAULT_STRATEGY: Strategy = RuleStrategy()


def generate_signals(factor_df: pd.DataFrame, params: C.StrategyParams,
                     strategy: Strategy = DEFAULT_STRATEGY) -> pd.DataFrame:
    """模块级便捷入口。"""
    return strategy.generate_signals(factor_df, params)
