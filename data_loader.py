"""
data_loader.py — M1 数据加载层

职责：
  1. 从 DAILY_ZIP（按股票 CSV）读取池内每只股的原始日线 OHLCV（不解压整个 zip，只读需要的成员）。
  2. 读取复权因子（自动识别同花顺双列 / 涨跌幅单列两种格式），取【后复权因子】。
  3. 计算【前复权】OHLC：
         后复权价 = 原始价 × 后复权因子
         前复权价 = 原始价 × 后复权因子 / 最新(末日)后复权因子
     —— “最新因子”一律固定为【该股在本数据窗口末日】的后复权因子（防前视，且使末日前复权价=原始价）。
  4. 标注涨跌停价 / 一字板，供回测引擎判定可否成交。
  5. 数据体检：时间范围、停牌天数、复权前后抽样、异常值、因子非单调告警。

输出：标准化 per-stock DataFrame（索引=日期(datetime)，列见 STD_COLUMNS）。

【数据格式实测备忘】
  日线（zip 内 sh600703.csv 等）表头：
    股票代码,日期,开盘价,最高价,最低价,收盘价,昨收价,涨跌额,涨跌幅,成交量,成交额
    UTF-8 BOM；股票代码为 6 位裸码；日期形如 2026-03-23。
  因子-同花顺（复权因子(同花顺)/sh600703.csv）表头：日期,前复权因子,后复权因子
    —— 前复权因子列含负值不可用；只用后复权因子列。
  因子-涨跌幅（涨跌幅/全部复权因子/sh600703.csv）表头：股票代码,日期,复权因子(=后复权因子)
"""
from __future__ import annotations

import os
import glob
import zipfile
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config as C

# 标准化输出列
STD_COLUMNS = [
    "open", "high", "low", "close",          # 前复权 OHLC（回测/因子用这套）
    "raw_open", "raw_high", "raw_low", "raw_close",  # 原始未复权（参考/体检）
    "volume", "amount", "prev_close",         # 成交量/额（原始，不调整）+ 原始昨收
    "hfq_factor",                              # 当日后复权因子
    "limit_up", "limit_down",                 # 当日涨/跌停价（前复权口径）
    "one_word_up", "one_word_down",           # 一字涨停/跌停（开盘即贴停板）
    "limit_up_hit", "limit_down_hit",         # 收盘封涨/跌停
    "suspended",                              # 当日是否停牌（在对齐到统一日历后才有意义）
]


# --------------------------------------------------------------------------- #
# zip 成员索引（basename -> 内部成员名），只建一次
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _zip_basename_index(zip_path: str) -> Dict[str, str]:
    """建立 {去掉乱码目录前缀的文件名: zip内部成员全名} 索引。

    zip 内部目录名是中文“日k/”的乱码，故只按 basename 匹配。
    """
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
    return {os.path.basename(n): n for n in names}


# --------------------------------------------------------------------------- #
# 原始日线（从 zip 读单只）
# --------------------------------------------------------------------------- #
def load_raw_daily(code: str) -> Optional[pd.DataFrame]:
    """从 DAILY_ZIP 读取单只股票的原始日线。找不到返回 None。"""
    prefix = C.exchange_prefix(code)
    fname = f"{prefix}{code}.csv"
    idx = _zip_basename_index(C.DAILY_ZIP)
    member = idx.get(fname)
    if member is None:
        # 兜底：尝试裸代码或其它前缀
        cand = [b for b in idx if code in b]
        if not cand:
            return None
        member = idx[cand[0]]

    with zipfile.ZipFile(C.DAILY_ZIP) as z:
        with z.open(member) as fp:
            df = pd.read_csv(
                fp, encoding="utf-8-sig",
                dtype={"股票代码": str},
            )

    # 可选：用按日目录把 zip 末日之后的最新行补上
    if C.USE_BYDAY_EXTENSION:
        ext = _load_byday_extension(code, after=df["日期"].max())
        if ext is not None and len(ext):
            df = pd.concat([df, ext], ignore_index=True)

    df = df.rename(columns={
        "开盘价": "raw_open", "最高价": "raw_high", "最低价": "raw_low",
        "收盘价": "raw_close", "昨收价": "prev_close",
        "成交量": "volume", "成交额": "amount", "日期": "date",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")
    keep = ["raw_open", "raw_high", "raw_low", "raw_close", "prev_close", "volume", "amount"]
    df = df[keep].astype(float)
    return df


def _load_byday_extension(code: str, after: str) -> Optional[pd.DataFrame]:
    """扫描按日全市场目录，抽取该股在 `after` 之后的行（用于增量更新）。"""
    files = sorted(glob.glob(os.path.join(C.DAILY_BYDAY_DIR, "*.csv")))
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f, encoding="utf-8-sig", dtype={"股票代码": str})
        except Exception:
            continue
        d = d[d["股票代码"] == code]
        if len(d):
            rows.append(d)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out = out[out["日期"] > after]
    return out if len(out) else None


# --------------------------------------------------------------------------- #
# 复权因子（自动识别两种格式）
# --------------------------------------------------------------------------- #
def load_hfq_factor(code: str) -> Optional[pd.Series]:
    """加载单只股票的【后复权因子】序列（索引=日期）。自动识别单列/双列格式。"""
    prefix = C.exchange_prefix(code)
    primary = C.FACTOR_THS_DIR if C.FACTOR_SOURCE == "ths" else C.FACTOR_ZDF_DIR
    path = os.path.join(primary, f"{prefix}{code}.csv")
    # 不做跨源静默兜底：同花顺双列在本数据集已损坏，若主源缺失宁可返回 None 让该股被跳过。
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, encoding="utf-8-sig")
    cols = set(df.columns)
    if "后复权因子" in cols:                 # 同花顺双列：只取后复权，弃用不可靠的“前复权因子”列
        s = df.set_index("日期")["后复权因子"]
    elif "复权因子" in cols:                 # 涨跌幅单列：复权因子即后复权因子
        s = df.set_index("日期")["复权因子"]
    else:
        raise ValueError(f"无法识别的因子文件格式：{path} 列={list(df.columns)}")

    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index().astype(float)
    s.name = "hfq_factor"
    return s


# --------------------------------------------------------------------------- #
# 前复权 + 涨跌停标注 → 标准化 DataFrame
# --------------------------------------------------------------------------- #
def build_stock(code: str, verbose: bool = True) -> Optional[pd.DataFrame]:
    """组装单只股票标准化日线（前复权 OHLC + 原始量额 + 涨跌停标记）。"""
    raw = load_raw_daily(code)
    if raw is None or raw.empty:
        if verbose:
            print(f"  [告警] {code}：日线缺失，跳过。")
        return None
    fac = load_hfq_factor(code)
    if fac is None or fac.empty:
        if verbose:
            print(f"  [告警] {code}：复权因子缺失，跳过。")
        return None

    df = raw.copy()

    # 因子 join 到行情日期。复权因子是【阶梯函数】(除权日跳变、之间不变)，其记录日期未必精确
    # 落在交易日上；故先在 (因子日期 ∪ 行情日期) 上 ffill(每个交易日取≤它的最近因子)再对回行情日期。
    # 若直接 reindex(df.index) 可能因日期不重叠而全 NaN → 前复权价全 NaN(已修此 bug)。
    _f = fac.reindex(fac.index.union(df.index)).ffill().bfill()
    df["hfq_factor"] = _f.reindex(df.index)

    # 防御：极少数早年坏 tick（0/负价）。原始列(raw_*)保持不动供体检统计；
    # 复权计算用“清洗后”的价格基（前值填充），避免 0 价传染到前复权 OHLC / 指标。
    price_cols = ["raw_open", "raw_high", "raw_low", "raw_close", "prev_close"]
    clean = df[price_cols].mask(df[price_cols] <= 0).ffill().bfill()

    # —— 防前视核心：最新因子 = 本窗口末日（该股最后一根 K 线）的后复权因子 ——
    anchor = float(df["hfq_factor"].iloc[-1])
    if anchor <= 0:
        if verbose:
            print(f"  [告警] {code}：末日后复权因子非正({anchor})，跳过。")
        return None

    # 前复权价 = 原始价 × 后复权因子 / 末日后复权因子
    ratio = df["hfq_factor"] / anchor
    df["open"] = clean["raw_open"] * ratio
    df["high"] = clean["raw_high"] * ratio
    df["low"] = clean["raw_low"] * ratio
    df["close"] = clean["raw_close"] * ratio

    # 涨跌停（前复权口径）：用【原始昨收价】算原始停板价，再按当日 ratio 折算到前复权空间，
    # 与前复权 OHLC 同口径比较。（量/额保持原始，未调整——V1 已知简化。）
    limit_pct = _limit_pct_series(code, df.index)
    raw_prev = clean["prev_close"]
    # 原始停板价四舍五入到分（交易所规则），再折算到前复权
    raw_limit_up = (raw_prev * (1 + limit_pct)).round(2)
    raw_limit_down = (raw_prev * (1 - limit_pct)).round(2)
    df["limit_up"] = raw_limit_up * ratio
    df["limit_down"] = raw_limit_down * ratio

    tol = C.LIMIT_TOUCH_TOL
    # 一字涨停：开盘即≥涨停价（买不进）；一字跌停：开盘即≤跌停价（卖不出）
    df["one_word_up"] = df["open"] >= df["limit_up"] * (1 - tol)
    df["one_word_down"] = df["open"] <= df["limit_down"] * (1 + tol)
    # 收盘封板（仅作记录/体检）
    df["limit_up_hit"] = df["close"] >= df["limit_up"] * (1 - tol)
    df["limit_down_hit"] = df["close"] <= df["limit_down"] * (1 + tol)

    df["suspended"] = False  # 单股自身日历内无停牌；对齐统一日历后再标（见 align_calendar）
    return df[STD_COLUMNS]


def _limit_pct_series(code: str, index: pd.DatetimeIndex) -> pd.Series:
    """逐日涨跌停幅度（按板块；可选 ST 时间段覆盖为 ±5%）。"""
    base = C.LIMIT_PCT[C.board_of(code)]
    s = pd.Series(base, index=index)
    for (start, end) in C.ST_PERIODS.get(code, []):
        mask = (index >= pd.to_datetime(start)) & (index <= pd.to_datetime(end))
        s[mask] = C.LIMIT_PCT["st"]
    return s


# --------------------------------------------------------------------------- #
# 加载整个股池
# --------------------------------------------------------------------------- #
def load_pool(pool: Optional[List[str]] = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """加载并校验整个股池，返回 {code: 标准化 DataFrame}。非法/缺失代码跳过并告警。"""
    pool = pool if pool is not None else C.POOL
    out: Dict[str, pd.DataFrame] = {}
    for code in pool:
        if not C.is_valid_code(code):
            if verbose:
                print(f"  [告警] 非法代码 {code}（非合法 A 股板块前缀），跳过。")
            continue
        df = build_stock(code, verbose=verbose)
        if df is not None and len(df) > 0:
            out[code] = df
    if verbose:
        print(f"  成功加载 {len(out)}/{len(pool)} 只。")
    return out


# --------------------------------------------------------------------------- #
# 程序化股池（评估框架修复：大、可比、抗操纵的训练/测试样本）
# --------------------------------------------------------------------------- #
def _board_allowed(prefix: str, code: str) -> bool:
    """按 config 的板块开关判断该票是否纳入候选。主板恒入；688/300/bj 看开关。"""
    if prefix == "bj":
        return C.UNIVERSE_INCLUDE_BJ
    if code.startswith("688"):
        return C.UNIVERSE_INCLUDE_STAR
    if code.startswith("300"):
        return C.UNIVERSE_INCLUDE_GEM
    return code[:3] in C.VALID_SH_PREFIX or code[:3] in C.VALID_SZ_PREFIX


def _scan_universe_candidates(verbose: bool = True) -> pd.DataFrame:
    """扫描全市场候选：对每只有复权因子的票，读日线(日期,成交额)，
    算 上市日/末日/窗口内bar数/窗口内日均成交额。返回汇总表（一行一票）。"""
    win_start = pd.Timestamp(C.BACKTEST_START) if C.BACKTEST_START else None
    fac_files = glob.glob(os.path.join(C.FACTOR_ZDF_DIR, "*.csv"))
    cands = []
    for f in fac_files:
        b = os.path.basename(f)[:-4]          # sh600703
        prefix, code = b[:2], b[2:]
        if len(code) == 6 and code.isdigit() and _board_allowed(prefix, code):
            cands.append((prefix, code, b))
    if verbose:
        print(f"  候选（有因子 & 板块符合）：{len(cands)} 只，开始扫描流动性/上市日 ...")

    idx = _zip_basename_index(C.DAILY_ZIP)
    rows = []
    with zipfile.ZipFile(C.DAILY_ZIP) as z:
        for i, (prefix, code, b) in enumerate(cands, 1):
            member = idx.get(f"{b}.csv")
            if member is None:
                continue
            try:
                with z.open(member) as fp:
                    d = pd.read_csv(fp, encoding="utf-8-sig", usecols=["日期", "成交额"])
            except Exception:
                continue
            if d.empty:
                continue
            dt = pd.to_datetime(d["日期"])
            win = d[dt >= win_start] if win_start is not None else d
            rows.append({
                "code": code, "prefix": prefix,
                "first": dt.min(), "last": dt.max(),
                "nbars_win": int(len(win)),
                "med_amt": float(win["成交额"].median()) if len(win) else float("nan"),
            })
            if verbose and i % 800 == 0:
                print(f"    ...{i}/{len(cands)}")
    return pd.DataFrame(rows)


def build_universe(verbose: bool = True, write_cache: bool = True) -> List[str]:
    """构建程序化股池：板块过滤 → 上市/仍交易/流动性门槛 → 按成交额降序等距抽 SIZE 只。

    选股完全确定可复现（不依赖随机种子）。结果（含统计）写入 UNIVERSE_CACHE_FILE。
    """
    df = _scan_universe_candidates(verbose=verbose)
    list_before = pd.Timestamp(C.UNIVERSE_LIST_BEFORE)
    still_by = pd.Timestamp(C.UNIVERSE_STILL_TRADING_BY)
    min_amt = C.UNIVERSE_MIN_AMOUNT_YI * 1e8

    elig = df[(df["first"] <= list_before) & (df["last"] >= still_by)
              & (df["nbars_win"] >= C.UNIVERSE_MIN_WIN_BARS)
              & (df["med_amt"] >= min_amt)].copy()
    elig = elig.sort_values("med_amt", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"  通过门槛（上市<{list_before.date()} & 仍交易 & ≥{C.UNIVERSE_MIN_WIN_BARS}bar "
              f"& 日均额≥{C.UNIVERSE_MIN_AMOUNT_YI}亿）：{len(elig)} 只")

    n = min(C.UNIVERSE_SIZE, len(elig))
    if len(elig) > n > 0:
        # 在成交额降序表上等距取 n 个 → 覆盖“百亿到数千亿”的代表性
        sel_idx = np.linspace(0, len(elig) - 1, n).round().astype(int)
        sel_idx = sorted(set(sel_idx.tolist()))
        sel = elig.iloc[sel_idx].copy()
    else:
        sel = elig

    if write_cache:
        out = sel[["code", "prefix", "first", "last", "nbars_win", "med_amt"]].copy()
        out["med_amt_yi"] = (out["med_amt"] / 1e8).round(3)
        out.to_csv(C.UNIVERSE_CACHE_FILE, index=False, encoding="utf-8-sig")
        if verbose:
            print(f"  已写股池缓存：{C.UNIVERSE_CACHE_FILE}（{len(out)} 只）")
    return sel["code"].astype(str).tolist()


def get_universe(force_rebuild: bool = False, verbose: bool = True) -> List[str]:
    """取程序化股池：优先读缓存（快），缺失或强制时重建（约 1 分钟全市场扫描）。"""
    if not force_rebuild and os.path.exists(C.UNIVERSE_CACHE_FILE):
        codes = pd.read_csv(C.UNIVERSE_CACHE_FILE, encoding="utf-8-sig",
                            dtype={"code": str})["code"].tolist()
        if verbose:
            print(f"  载入股池缓存 {C.UNIVERSE_CACHE_FILE}：{len(codes)} 只"
                  f"（重建用 build_universe）")
        return codes
    return build_universe(verbose=verbose)


# --------------------------------------------------------------------------- #
# 回测损益窗口（因子在全历史因果计算，只在窗口内交易/计损益）
# --------------------------------------------------------------------------- #
def backtest_window():
    """返回 (start_ts, end_ts)，空配置对应 None（即不设边界）。"""
    start = pd.Timestamp(C.BACKTEST_START) if C.BACKTEST_START else None
    end = pd.Timestamp(C.BACKTEST_END) if C.BACKTEST_END else None
    return start, end


def trim_to_window(df: pd.DataFrame) -> pd.DataFrame:
    """把（已在全历史上因果算好的）因子/信号表裁剪到回测窗口。
    因子值在窗口首日已含完整历史 → 无 warmup 断层、无前视；只是不在窗口外交易。"""
    start, end = backtest_window()
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df


# --------------------------------------------------------------------------- #
# 统一交易日历（标停牌；前复权价穿越停牌持仓用）
# --------------------------------------------------------------------------- #
def union_calendar(stocks: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """以股池所有股票日期的并集作为代理交易日历（无独立日历文件时的近似）。"""
    idx = None
    for df in stocks.values():
        idx = df.index if idx is None else idx.union(df.index)
    return idx if idx is not None else pd.DatetimeIndex([])


# --------------------------------------------------------------------------- #
# 数据体检
# --------------------------------------------------------------------------- #
def health_check(stocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """打印并返回每只股的体检摘要：时间范围、交易日数、停牌天数、异常值、因子单调性、复权抽样。"""
    cal = union_calendar(stocks)
    rows = []
    print("\n========== 数据体检 ==========")
    print(f"代理交易日历：{cal.min().date()} ~ {cal.max().date()}，共 {len(cal)} 个交易日\n")
    for code, df in stocks.items():
        rng_days = cal[(cal >= df.index.min()) & (cal <= df.index.max())]
        suspended = len(rng_days) - len(df.index)   # 在自身上市区间内、却缺 bar 的天数=停牌
        # 异常值统计基于【原始】价（raw_*），复权前的真实坏 tick；前复权列已做防御清洗
        neg_price = int(((df[["raw_open", "raw_high", "raw_low", "raw_close"]] <= 0).any(axis=1)).sum())
        zero_vol = int((df["volume"] <= 0).sum())
        fac = df["hfq_factor"]
        non_monotonic = int((fac.diff().fillna(0) < -1e-9).sum())  # 后复权因子理应非递减，反例计数
        # 复权前后抽样：除权日（因子跳变）原始价跳空 vs 前复权连续
        sample = _adjust_sample(df)
        rows.append({
            "code": code,
            "start": df.index.min().date(),
            "end": df.index.max().date(),
            "bars": len(df),
            "suspended_days": suspended,
            "neg_price_rows": neg_price,
            "zero_vol_rows": zero_vol,
            "factor_drops": non_monotonic,
        })
        print(f"[{code}] {df.index.min().date()}~{df.index.max().date()}  "
              f"bars={len(df)}  停牌={suspended}  负/零价={neg_price}  零量={zero_vol}  "
              f"因子回撤={non_monotonic}")
        if sample:
            print(f"        除权抽样 {sample['date']}：原始收 {sample['raw_prev']:.2f}->{sample['raw_cur']:.2f} "
                  f"(跳 {sample['raw_jump']:+.1%}) | 前复权收 {sample['adj_prev']:.2f}->{sample['adj_cur']:.2f} "
                  f"(跳 {sample['adj_jump']:+.1%})  —— 前复权应明显更平滑")
    summary = pd.DataFrame(rows).set_index("code")
    print("==============================\n")
    return summary


def _adjust_sample(df: pd.DataFrame) -> Optional[dict]:
    """找因子变动最大的一天，对比原始 vs 前复权的隔夜跳空，验证除权日无假跳空。"""
    fac = df["hfq_factor"]
    chg = fac.pct_change().abs()
    if chg.notna().sum() == 0 or chg.max() < 1e-6:
        return None
    i = int(np.nanargmax(chg.values))
    if i == 0:
        return None
    prev, cur = df.iloc[i - 1], df.iloc[i]
    return {
        "date": str(df.index[i].date()),
        "raw_prev": prev["raw_close"], "raw_cur": cur["raw_close"],
        "raw_jump": cur["raw_close"] / prev["raw_close"] - 1,
        "adj_prev": prev["close"], "adj_cur": cur["close"],
        "adj_jump": cur["close"] / prev["close"] - 1,
    }


# --------------------------------------------------------------------------- #
# 分钟线（V1 不实现，留 TODO 桩）
# --------------------------------------------------------------------------- #
def load_minute(code: str, freq: str = "1min") -> None:
    """TODO(V2) 分钟线加载桩。

    实现路线：
      1) 用 `unar` 解压 MINUTE_RAR_DIR 下对应级别 rar（已装好 unar）。
      2) 复用与日线相同结构的 loader（同样 utf-8-sig、按股票 CSV）。
      3) 分钟 bar 的复权：用【当日】的后复权因子（同一天内因子不变）做前复权。
      4) 1 分钟可向上聚合出 5/15/30 分钟 bar（OHLC 用 first/max/min/last，量/额求和）。
    本版仅日线，分钟线接口预留，不实现。
    """
    raise NotImplementedError("分钟线为 V2 TODO；当前仅支持日线。见 load_minute docstring。")
