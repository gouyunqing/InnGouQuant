"""
config.py — 集中配置（数据路径 / 股池 / 参数网格 / 成本 / 涨跌停 / 切分 / 随机种子）

本系统是【纯离线】A股日线策略研究系统：前复权 → 因子 → 回测 → 寻优 → 报告。
不接实盘、不下单、不联网。所有数据路径都指向本地，**只读不改、不复制大文件夹**。

数据现状（已 inventory 实测，2026-06 实机）：
  行情：/Users/gouyunqing/Downloads/a股分钟线/
  复权：/Users/gouyunqing/Downloads/复权因子/
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------- #
# 0. 随机种子（可复现）
# --------------------------------------------------------------------------- #
RANDOM_SEED: int = 20260627

# --------------------------------------------------------------------------- #
# 1. 数据路径
# --------------------------------------------------------------------------- #
# 数据根。换机（如 Mac→Windows）时【不要改代码】，改设环境变量 INNGOU_DATA_ROOT 即可：
#   Windows(PowerShell): $env:INNGOU_DATA_ROOT="D:\Downloads"
#   Windows(cmd):        set INNGOU_DATA_ROOT=D:\Downloads
#   macOS/Linux:         export INNGOU_DATA_ROOT=/Users/you/Downloads
# 未设环境变量时回退到下面这台 Mac 的实机路径。
DATA_ROOT: str = os.environ.get("INNGOU_DATA_ROOT", "/Users/gouyunqing/Downloads")

# --- 行情（市场数据，全部在 a股分钟线/ 下）---
MARKET_ROOT: str = os.path.join(DATA_ROOT, "a股分钟线")

# 日线主数据源：一个 zip，内含【按股票】组织的全历史 CSV，文件名形如 sh600703.csv / sz000792.csv。
# 表头：股票代码,日期,开盘价,最高价,最低价,收盘价,昨收价,涨跌额,涨跌幅,成交量,成交额
# 历史覆盖至 2026-04-30。直接用 zipfile 读取需要的成员，不解压、不展开整个文件夹。
DAILY_ZIP: str = os.path.join(MARKET_ROOT, "日k", "全部日k.zip")

# 日线“按交易日一个文件、全市场”的补充目录（可选）。当前 2026/ 仅 2026-03-23~04-30，
# 与 DAILY_ZIP 完全重叠，默认不启用；保留以便将来增量更新。
DAILY_BYDAY_DIR: str = os.path.join(MARKET_ROOT, "日k", "2026")
USE_BYDAY_EXTENSION: bool = False  # True 时把 DAILY_BYDAY_DIR 里超出 zip 末日的行追加进来

# --- 复权因子（全部在 复权因子/ 下）---
FACTOR_ROOT: str = os.path.join(DATA_ROOT, "复权因子")

# 因子【主】数据源：涨跌幅【按股票】单列文件，文件名形如 sh600703.csv，表头：股票代码,日期,复权因子(=后复权因子)
# 实测验证：用它做后复权，复权后日收益与原始日收益仅在【除权日】不同（21/6868≈0.3%），
# 因子全程仅变动 26 次(=分红送转次数)，是干净的累计后复权因子。✔
FACTOR_ZDF_DIR: str = os.path.join(FACTOR_ROOT, "涨跌幅", "全部复权因子")

# 因子【弃用】数据源：同花顺【按股票】双列文件 日期,前复权因子,后复权因子。
# 实测：本数据集里这份文件两列都坏——“前复权因子”含负值；“后复权因子”逐日抖动(6545/6868天在变，
# 甚至出现 0.0)，复权后日收益与原始日收益在 42% 的交易日不一致。故【不采用】，仅留路径备查。
FACTOR_THS_DIR: str = os.path.join(FACTOR_ROOT, "复权因子(同花顺)")

# 因子来源选择：'zdf'（涨跌幅单列，干净，默认）| 'ths'（同花顺双列，本数据集已损坏，不建议）
FACTOR_SOURCE: str = "zdf"

# --- 分钟线（V1 不实现，留 TODO）---
MINUTE_RAR_DIR: str = os.path.join(MARKET_ROOT, "2000-2025（分k）")  # 15min.rar / 5min.rar ...（unar 解压）

# 输出目录
OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

# 缓存目录（可移植、跨平台）：放价格面板 pkl / 收益面板等【依赖本地行情、与机器绑定】的中间产物。
# 已 gitignore，不入库（换机重建）。研究脚本统一用 C.CACHE_DIR 而非硬编码 /tmp 路径。
CACHE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. 股池
# --------------------------------------------------------------------------- #
# 注：用户原写 660186 非法（无 660 开头），已确认改为 600186。
# 这 11 只是最初手挑的小样本，统计上不足以验证策略（见 README 局限）。
# 现默认改用【程序化全市场股池】(build_universe)，本列表仅作 USE_UNIVERSE=False 时的回退。
POOL: List[str] = [
    "600703", "605589", "300308", "600186", "600246",
    "600478", "600226", "000792", "601777", "002228", "600602",
]

# --- 程序化股池（M5 评估框架修复：用足够大、可比、抗操纵的样本做训练/测试）---
# 为什么不再用手挑 11 只：跨股中位数只在 ~10 只上算，统计无意义；且不同上市日被同一日期
# 一刀切会让晚上市的票“整段落到 OOS”。改为从全市场按规则筛出 ~120 只代表性票。
USE_UNIVERSE: bool = True
UNIVERSE_SIZE: int = 120                 # 目标股池大小（约）
# 板块：主板 + 创业板(300)，剔除 科创板(688) 与 北交所(bj)。
UNIVERSE_INCLUDE_GEM: bool = True        # 含创业板 300
UNIVERSE_INCLUDE_STAR: bool = False      # 含科创板 688（默认否）
UNIVERSE_INCLUDE_BJ: bool = False        # 含北交所 bj（默认否）
# 上市需早于此日，保证回测窗口内 holdout 两侧都有样本（避免“整段 OOS”）。
UNIVERSE_LIST_BEFORE: str = "2016-01-01"
# 末日需晚于此，剔除已退市 / 长期停牌的票。
UNIVERSE_STILL_TRADING_BY: str = "2025-06-01"
# 抗操纵/稳定性筛：窗口内【日均成交额】门槛（亿元）。本数据无股本/市值字段，
# 故用流动性做代理——能被主力“乱拉”的恰是成交额小的票。1.5亿 ≈ 百亿市值量级。
UNIVERSE_MIN_AMOUNT_YI: float = 1.5
UNIVERSE_MIN_WIN_BARS: int = 1000        # 窗口内最少 bar 数（确保有足够历史）
# 选股方式：在通过门槛的票里，按成交额降序【等距抽样】到 UNIVERSE_SIZE 只，
# 兼顾“百亿以上”与“从百亿到数千亿的代表性”，且完全确定可复现（不依赖随机种子）。
UNIVERSE_CACHE_FILE: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "universe.csv")

# --------------------------------------------------------------------------- #
# 3. 代码合法性 / 交易所映射
# --------------------------------------------------------------------------- #
# 沪：600/601/603/605/688 ；深：000/001/002/003/300
VALID_SH_PREFIX = ("600", "601", "603", "605", "688")
VALID_SZ_PREFIX = ("000", "001", "002", "003", "300")


def is_valid_code(code: str) -> bool:
    """A股代码合法性校验（6位数字 + 合法板块前缀）。非法者由 loader 跳过并告警。"""
    if not (isinstance(code, str) and len(code) == 6 and code.isdigit()):
        return False
    return code[:3] in VALID_SH_PREFIX or code[:3] in VALID_SZ_PREFIX


def exchange_prefix(code: str) -> str:
    """裸 6 位代码 → 'sh'/'sz' 前缀（用于对上 sh/sz 命名的因子/行情文件）。"""
    if code[:3] in VALID_SH_PREFIX:
        return "sh"
    if code[:3] in VALID_SZ_PREFIX:
        return "sz"
    raise ValueError(f"非法代码，无法映射交易所：{code}")


def board_of(code: str) -> str:
    """板块判定，决定涨跌停幅度。'gem'=创业板300, 'star'=科创688, 'main'=主板。"""
    if code.startswith("300"):
        return "gem"   # 创业板 ±20%
    if code.startswith("688"):
        return "star"  # 科创板 ±20%
    return "main"      # 主板 ±10%


# --------------------------------------------------------------------------- #
# 4. 涨跌停（用 昨收价 × (1±limit) 计算，按板块）
# --------------------------------------------------------------------------- #
LIMIT_PCT: Dict[str, float] = {
    "main": 0.10,
    "gem": 0.20,
    "star": 0.20,
    "st": 0.05,
}
# ST 在本数据中无可靠的历史 ST 时间轴，V1 不自动识别 ST（见 README 局限）。
# 如需手工指定某股某段为 ST，可在此填 {code: [(start,end), ...]}（默认空）。
ST_PERIODS: Dict[str, list] = {}

# 判定“一字板”用：开盘价是否已贴在涨跌停价（带一点容差，防浮点/四舍五入误差）。
LIMIT_TOUCH_TOL: float = 1e-4  # 相对容差


# --------------------------------------------------------------------------- #
# 5. 因子默认参数（基线回测用；寻优会覆盖）
# --------------------------------------------------------------------------- #
@dataclass
class StrategyParams:
    n: int = 10      # SuperTrend ATR 周期
    m: float = 3.0   # SuperTrend ATR 倍数
    w: int = 10      # 量能均线窗口
    k: float = 1.5   # 量能突破倍数 vol > k * SMA(vol, w)

    def key(self) -> str:
        return f"n{self.n}_m{self.m}_w{self.w}_k{self.k}"


DEFAULT_PARAMS = StrategyParams()

# --------------------------------------------------------------------------- #
# 6. 参数寻优网格
# --------------------------------------------------------------------------- #
PARAM_GRID: Dict[str, list] = {
    "n": [7, 10, 14],
    "m": [2.0, 2.5, 3.0, 3.5],
    "w": [5, 10],
    "k": [1.3, 1.5, 2.0],
}

# 防过拟合：训练集上最少成交次数门槛（每只股 + 组合）；低于则该参数组判无效。
MIN_TRADES_PER_STOCK: int = 3       # 单股训练集最少交易笔数（用于稳健性统计的“有效股”）
MIN_TRADES_TOTAL: int = 15          # 组合训练集最少总交易笔数

# 寻优目标：'calmar'（年化/最大回撤，默认）| 'sharpe'
OBJECTIVE: str = "calmar"

# 跨股稳健性聚合：'median'（默认，抗单股暴富）| 'mean'
ROBUST_AGG: str = "median"

# --------------------------------------------------------------------------- #
# 7. 时间序列切分（严禁随机打乱）
# --------------------------------------------------------------------------- #
# --- 回测损益窗口（评估框架修复核心）---
# 因子在【全历史】上因果计算（保留 warmup、无前视），但只在 [BACKTEST_START, END]
# 内交易/计损益 —— 把 1990s~2000s 完全不同制度/微观结构的远古行情踢出评估。
# 起点 2010：覆盖 2015泡沫+股灾 / 2018熊 / 2019-21牛 / 2022-24熊 / 2025复苏 多个完整周期，
# 让 walk-forward 能跨 regime 验证，最适合暴露过拟合。
BACKTEST_START: str = "2010-01-01"
BACKTEST_END: str = ""               # 空=用数据末日（~2026-04-30）

# 'holdout'：顺序切 train/test（默认 70/30）
# 'walkforward'：滚动窗口
SPLIT_MODE: str = "holdout"
TRAIN_RATIO: float = 0.70

# walk-forward 配置（按交易日计），SPLIT_MODE='walkforward' 时生效
WF_TRAIN_DAYS: int = 750     # 约 3 年
WF_TEST_DAYS: int = 250      # 约 1 年
WF_STEP_DAYS: int = 250      # 滚动步长

# Purge / Embargo（防训练-测试边界泄漏）：
#   purge  = OOS 只统计【完全落在测试窗内】的交易，剔除跨界（训练段买入、测试段才卖出）的交易；
#   embargo= 测试窗开头再跳过若干交易日，避免跨界持仓的净值尾巴/自相关污染 OOS。
EMBARGO_DAYS: int = 5

# PBO（回测过拟合概率，CSCV 法）：把时间切成 PBO_BLOCKS 块，组合式地分 IS/OOS，
# 看“样本内最优参数”在样本外排名中位以下的频率。块数需为偶数（C(S,S/2) 个组合）。
PBO_BLOCKS: int = 16

# --------------------------------------------------------------------------- #
# 8. 交易成本（A股，做成可配置）
# --------------------------------------------------------------------------- #
@dataclass
class CostModel:
    commission_rate: float = 0.00025   # 佣金，双边 0.025%
    commission_min: float = 5.0        # 佣金最低 5 元/笔
    stamp_rate: float = 0.0005         # 印花税，卖出单边 0.05%
    transfer_rate: float = 0.00001     # 过户费，双边 0.001%
    slippage: float = 0.0015           # 滑点 0.15%（0.1%~0.2% 区间中值）


COST = CostModel()

# --------------------------------------------------------------------------- #
# 8b. 风控止损（Tier 1）：在 SuperTrend 翻空之外，加更紧、独立的单笔风险止损
# --------------------------------------------------------------------------- #
# SuperTrend 本身已是一条 ATR 移动止损（价跌破 HL2-m*ATR 翻空）。这里再叠加：
#   · 硬止损：单笔从买入价回撤到 -hard_stop_pct 无条件离场（封死单笔灾难亏损）。
#   · 吊灯止损(Chandelier)：从持仓期【最高价】回落 atr_trail_mult*ATR 离场（比 ST 更紧地锁利润）。
# 执行口径：收盘触发 → 次日开盘成交（与买卖信号同一套无前视/T+1/一字板规则，不引入盘中乐观假设）。
# 参数先固定、不进网格寻优——隔离“止损本身”的效果，避免增加过拟合自由度（调参是 Tier 2）。
@dataclass
class RiskParams:
    use_hard_stop: bool = True
    hard_stop_pct: float = 0.08         # 单笔最大亏损 8%
    use_atr_trailing: bool = True
    atr_trail_mult: float = 3.0         # 吊灯：峰值回落 3×ATR 离场
    use_time_stop: bool = False         # 时间止损（默认关）
    max_holding_days: int = 60


RISK = RiskParams()

# --------------------------------------------------------------------------- #
# 9. 回测设置
# --------------------------------------------------------------------------- #
INIT_CAPITAL: float = 1_000_000.0   # 组合初始资金（等权分配到每只股）
LOT_SIZE: int = 100                 # A股一手=100股
TRADING_DAYS_PER_YEAR: int = 244    # 年化用
MAX_STOCK_PLOTS: int = 12           # 报告里逐股出图的上限（大股池下抽样展示）

# --------------------------------------------------------------------------- #
# 10. 评估基准与无风险利率（机构级指标：α/β/信息比率/Sortino/夏普）
# --------------------------------------------------------------------------- #
# 无风险利率（年化）：A股语境约 2%（10年国债 ~2.5% / 货基 ~1.5-2%）。夏普/Sortino/α 用它。
RISK_FREE_ANNUAL: float = 0.02

# 闲置现金计息（复盘修正①）：策略约一半时间空仓，那部分现金本应吃无风险利率，
# 否则净值/α 被系统性低估约 (空仓占比 × 2%)/年。True=按 RISK_FREE_ANNUAL 给在手现金计息。
CREDIT_CASH_INTEREST: bool = True

# 基准全收益（复盘修正②）：沪深300 是【价格指数】，不含分红(~2%/年)；直接拿来比会把策略 α 高估约2%。
# True=在价格指数日收益上加回估算股息率，近似【全收益指数】。可日后替换为真实全收益指数。
BENCHMARK_TOTAL_RETURN: bool = True
BENCHMARK_DIVIDEND_YIELD: float = 0.02   # 沪深300 历史股息率约 1.5~2.5%，取 2% 近似

# 主基准 = 沪深300（本地数据无指数，从东方财富公开行情接口抓取并缓存）。
# secid 规则：1.=上交所，0.=深交所。沪深300=1.000300，中证500=1.000905，创业板指=0.399006。
BENCHMARK_CODE: str = "000300"
BENCHMARK_NAME: str = "沪深300"
BENCHMARK_SECIDS: Dict[str, str] = {
    "000300": "1.000300",   # 沪深300（主基准）
    "000905": "1.000905",   # 中证500
    "399006": "0.399006",   # 创业板指
}
BENCHMARK_NAMES: Dict[str, str] = {
    "000300": "沪深300", "000905": "中证500", "399006": "创业板指",
}
BENCHMARK_CACHE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks")


def ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR
