"""
benchmark.py — 评估基准（指数）数据层

本地数据集只有个股、没有指数，故从东方财富公开行情接口抓取指数日线并缓存到本地，
供 report.py 计算机构级【基准相对指标】：β / Jensen's α / 超额收益 / 跟踪误差 / 信息比率 / 上下行捕获。

为什么需要基准：本策略只做多、长期持有个股，天生高 β（大盘涨它就涨）。不减掉 β，
就分不清 OOS 的正收益是“策略真本事(α)”还是“牛市搭便车(β)”。沪深300 是 A股最通用的市场基准。

数据口径：指数为【价格指数】（不含分红），直接用收盘价算日收益做 β/α —— 与业界惯例一致。
缓存为 benchmarks/{code}.csv（一次抓取，后续离线复用；删文件即可强制刷新）。
"""
from __future__ import annotations

import os
import json
import socket
import urllib.request
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

import config as C


def _cache_path(code: str) -> str:
    os.makedirs(C.BENCHMARK_CACHE_DIR, exist_ok=True)
    return os.path.join(C.BENCHMARK_CACHE_DIR, f"{code}.csv")


def _fetch_index_eastmoney(code: str, beg: str = "19900101", end: str = "20991231",
                           timeout: int = 20) -> pd.DataFrame:
    """从东方财富 push2his 接口抓取指数日线（无需 token）。返回 [date, close]（日期为索引）。"""
    secid = C.BENCHMARK_SECIDS.get(code)
    if secid is None:
        raise ValueError(f"未知基准代码 {code}，请在 config.BENCHMARK_SECIDS 配置 secid。")
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get?"
           f"secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
           "&fields2=f51,f52,f53,f54,f55,f56,f57,f58"   # date,open,close,high,low,vol,amount,amplitude
           f"&klt=101&fqt=1&beg={beg}&end={end}&lmt=100000")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"})
    socket.setdefaulttimeout(timeout)
    raw = urllib.request.urlopen(req).read().decode("utf-8")
    data = (json.loads(raw).get("data") or {})
    klines = data.get("klines") or []
    if not klines:
        raise RuntimeError(f"基准 {code} 抓取为空（接口无数据）。")
    rows = []
    for line in klines:
        parts = line.split(",")
        rows.append((parts[0], float(parts[2])))   # date, close
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def load_benchmark(code: Optional[str] = None, force_refresh: bool = False,
                   verbose: bool = True) -> pd.DataFrame:
    """取基准指数日线（[close]，日期索引）。优先读缓存；缺失/强制时抓取并写缓存。"""
    code = code or C.BENCHMARK_CODE
    path = _cache_path(code)
    if not force_refresh and os.path.exists(path):
        df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"]).set_index("date")
        return df
    if verbose:
        print(f"  抓取基准指数 {code}（{C.BENCHMARK_NAMES.get(code, code)}）...")
    df = _fetch_index_eastmoney(code)
    df.to_csv(path, encoding="utf-8-sig")
    if verbose:
        print(f"  基准 {code}：{df.index.min().date()}~{df.index.max().date()}（{len(df)} 行）→ 缓存 {path}")
    return df


@lru_cache(maxsize=8)
def _benchmark_returns_cached(code: str) -> pd.Series:
    df = load_benchmark(code, verbose=False)
    r = df["close"].pct_change()
    r.name = f"bench_{code}"
    return r


def benchmark_returns(calendar: pd.DatetimeIndex, code: Optional[str] = None) -> pd.Series:
    """基准日收益序列，对齐到给定交易日历（缺口 ffill 后再算 pct_change，避免假跳空）。

    若 config.BENCHMARK_TOTAL_RETURN，则在价格指数日收益上加回估算股息（近似全收益指数），
    避免用价格指数当基准把策略 α 高估约一个股息率。"""
    code = code or C.BENCHMARK_CODE
    close = load_benchmark(code, verbose=False)["close"]
    close = close.reindex(close.index.union(calendar)).ffill().reindex(calendar)
    r = close.pct_change()
    if getattr(C, "BENCHMARK_TOTAL_RETURN", False):
        div_d = (1.0 + C.BENCHMARK_DIVIDEND_YIELD) ** (1.0 / C.TRADING_DAYS_PER_YEAR) - 1.0
        r = r + div_d                       # 每个交易日加回一份日度股息
    r.name = f"bench_{code}"
    return r


def benchmark_equity(calendar: pd.DatetimeIndex, code: Optional[str] = None,
                     init: float = 1.0) -> pd.Series:
    """基准归一净值（用于和组合净值同图对比）。"""
    r = benchmark_returns(calendar, code).fillna(0.0)
    eq = init * (1.0 + r).cumprod()
    eq.name = f"bench_{code}"
    return eq
