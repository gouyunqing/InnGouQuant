# InnGouQuant — A股自选股策略研究与回测系统（V1 · 日线）

> 我自己的量化系统，逐步迭代中。

纯离线的 A 股规则策略研究系统：**前复权 → 因子 → 回测 → 参数寻优 → 报告**。
**不接实盘、不下单、不联网取数。** 只读本地数据，不复制/不展开大文件夹。

策略（V1）：**SuperTrend 翻多 + 量能突破**，多头、T+1。

> 📓 **研究历程 / 认知 / 当前最优方案 / 下一步 / 换机注意事项 → 见 [RESEARCH_LOG.md](RESEARCH_LOG.md)。**
> 当前研究主线已从「单股 SuperTrend 择时」推进到「**横截面相对强度趋势因子 + regime 闸 + 波动率缩放的市场中性策略**」（方案1，Sharpe~1.0/β≈0），下一步是 robustness 终审。

---

## 快速开始

```bash
pip install -r requirements.txt

# 数据根用环境变量指定（换机不改代码）：
#   macOS/Linux:  export INNGOU_DATA_ROOT=/Users/you/Downloads
#   Win PowerShell: $env:INNGOU_DATA_ROOT="D:\Downloads"; $env:PYTHONUTF8="1"
#   Win cmd:        set INNGOU_DATA_ROOT=D:\Downloads & set PYTHONUTF8=1

python main.py --no-optimize     # ① 最快：基线回测 + 报告，验证全链路
python main.py                   # ② 完整：基线 + holdout(70/30) 寻优 + IS/OOS 报告
python main.py --walkforward     # ③ 额外做 walk-forward 滚动寻优

# 研究脚本（趋势因子主线，见 RESEARCH_LOG.md §5）：
python test_trend_xs.py          # v1 横截面趋势因子
python test_trend_gated.py       # 指数 regime 闸（条件 α 证实）
python test_trend_v2.py          # 方案1：β中性+闸+vol-target（当前最优）
```

> **Windows 必读**：换机数据路径、控制台中文编码(`PYTHONUTF8=1`)、行业数据续拉等注意事项见 [RESEARCH_LOG.md §6](RESEARCH_LOG.md)。

产物在 `outputs/`：`report.md`（汇总）、`metrics.csv`、`trades.csv`、`health_check.csv`、
`grid_search.csv`、`portfolio_equity.png`、`stock_*.png`、`optimize_heatmap.png`。

---

## 数据适配（已按实机 inventory，不要假设）

数据全部在 `/Users/gouyunqing/Downloads` 下，**只读引用、不复制**：

| 用途 | 路径 | 格式 |
|---|---|---|
| 日线行情（主） | `a股分钟线/日k/全部日k.zip` | zip 内**按股票** CSV：`sh600703.csv`…；表头 `股票代码,日期,开盘价,最高价,最低价,收盘价,昨收价,涨跌额,涨跌幅,成交量,成交额`，UTF-8 BOM，全历史至 2026-04-30 |
| 日线行情（增量，可选） | `a股分钟线/日k/2026/YYYYMMDD.csv` | **按交易日**全市场（当前与 zip 重叠，默认关闭） |
| 复权因子（主） | `复权因子/复权因子(同花顺)/{sh,sz}{code}.csv` | 双列 `日期,前复权因子,后复权因子` |
| 复权因子（备） | `复权因子/涨跌幅/全部复权因子/{sh,sz}{code}.csv` | 单列 `股票代码,日期,复权因子`(=后复权因子) |
| 分钟线 | `a股分钟线/2000-2025（分k）/*.rar` | V1 不实现，留 TODO |

直接用 `zipfile` 读 zip 内需要的成员，**不解压整包**。loader 用 `encoding="utf-8-sig"` 处理 BOM。

### 复权口径（同花顺，务必照此）
- 后复权价 = 原始价 × 后复权因子
- 前复权价 = 原始价 × 后复权因子 / **最新(末日)后复权因子**
- **最新因子固定为【该股本数据窗口末日】的后复权因子**（防前视；使末日前复权价=原始价）。
- ⚠️ 实测同花顺双列文件的**“前复权因子”列不可靠（含负值）**，本系统**只用“后复权因子”列**自行重算前复权，
  不直接采用文件给的前复权因子列。
- **回测一律用前复权 OHLC**；成交量/成交额保持原始（V1 不调整量，量能突破是相对自身均量的比值，已知简化）。

### 代码合法性 / 交易所映射
- 合法前缀：沪 `600/601/603/605/688`，深 `000/001/002/003/300`。非法代码（如 `660186`）**跳过并告警**。
  > 注：用户原写 `660186` 非法，已确认改为 `600186`。
- `600/601/603/605/688→sh`，`000/001/002/003/300→sz`（对上 sh/sz 命名的因子/行情文件）。

---

## 模块

| 文件 | 角色 |
|---|---|
| `config.py` | 集中配置：数据路径、股池、参数网格、成本、涨跌停、切分、随机种子 |
| `data_loader.py` | M1 加载/复权/体检；涨跌停标注；`load_minute()` TODO 桩 |
| `factors.py` | M2 SuperTrend（Wilder ATR，自实现）+ 量能突破；因子矩阵 |
| `strategy.py` | M3 可插拔 `generate_signals`（输出 entry/exit），便于 V2 换 LightGBM |
| `backtest.py` | M4 A股约束回测引擎（无前视/T+1/涨跌停/停牌/成本） |
| `optimize.py` | M5 网格寻优；holdout & walk-forward；跨股稳健性；OOS 复核 |
| `report.py` | M6 指标/图/markdown（指标函数同时供 optimize 复用） |
| `main.py` | 端到端编排 |

---

## A股约束（回测引擎）

- **无前视**：t 日收盘信号 → t+1 日开盘成交（引擎显式把单挂到下一根 bar 的 open）。
- **T+1**：当日买入次日才可卖（卖出仅在持仓日严格晚于买入日时触发）。
- **涨跌停**：主板 ±10%，创业板(300)/科创(688) ±20%，ST ±5%（用 `昨收价 × (1±limit)` 折算到前复权口径）。
  一字涨停（开盘≥涨停价）→ 买不进、跳过该笔；一字跌停（开盘≤跌停价）→ 卖不出、顺延到下一可成交 bar。
- **停牌**：无 bar 的交易日视为停牌，持仓穿越（按股自身 bar 迭代，挂单自然顺延）。
- **成本**（可配 `config.CostModel`）：佣金双边 0.025%(最低 5 元)、印花税卖出 0.05%、过户费双边 0.001%、滑点 0.15%。
- 长仓 only；每股同一时间至多 1 个仓位；每股分配等额小账户独立交易，组合净值=各股净值在统一日历对齐求和。

---

## 防前视 / 防过拟合

- 信号→成交隔一根 bar；指标只用 t 及之前的数据；前复权“最新因子”固定为窗口末日，不取窗口外未来因子。
- 寻优**只在训练集**做网格搜索；目标默认 **Calmar**（年化/最大回撤），可切 Sharpe。
- 硬约束：① 最少交易次数门槛（训练集低于则该参数组无效）；② 跨股稳健性用**每股目标值中位数**选参（抗单股暴富），要求多数股票有效；③ 必报 **OOS**，并如实给出过拟合判定。
- 切分：默认 holdout 顺序 70/30；另有 walk-forward 滚动窗口。**严禁随机打乱。**

---

## SuperTrend 公式（自实现，见 `factors.py`）

```
TR_t      = max(H-L, |H-C_{t-1}|, |L-C_{t-1}|)
ATR_t     = (ATR_{t-1}*(n-1)+TR_t)/n        # 种子=前 n 根 TR 均值（Wilder）
HL2_t     = (H+L)/2
UB_t      = HL2_t + m*ATR_t ; LB_t = HL2_t - m*ATR_t
FinalUB_t = (UB_t<FinalUB_{t-1}) or (C_{t-1}>FinalUB_{t-1}) ? UB_t : FinalUB_{t-1}
FinalLB_t = (LB_t>FinalLB_{t-1}) or (C_{t-1}<FinalLB_{t-1}) ? LB_t : FinalLB_{t-1}
dir_t     = (C_t>FinalUB_{t-1})?+1 : (C_t<FinalLB_{t-1})?-1 : dir_{t-1}
ST_t      = (dir_t==+1)? FinalLB_t : FinalUB_t
flip_up   = dir_{t-1}==-1 and dir_t==+1 ;  flip_down = dir_{t-1}==+1 and dir_t==-1
vol_break = Volume_t > k * SMA(Volume,w)_t
买入 = flip_up and vol_break ; 卖出 = flip_down ; 成交在 t+1 开盘 ; 用前复权价
```

> dir 极性已用真实数据图肉眼核对（买卖点应落在趋势翻转处，见 `outputs/stock_*.png`）。

---

## 已知局限 / V2 TODO

- 分钟线未实现（`load_minute()` 桩）：`unar` 解压 rar → 同结构 loader → 分钟 bar 用当日复权因子 → 1min 可聚合 5/15/30min。
- 量/额未做复权调整（V1 简化）。
- ST 历史时间轴缺失，未自动识别（可在 `config.ST_PERIODS` 手工指定）。
- 组合为每股等额独立小账户，无跨股资金再平衡/调仓。
- 策略层已抽象为可插拔接口，V2 可替换为 LightGBM（吃同一张因子矩阵）。

## 技术选型说明
回测引擎为**自写向量化/事件驱动混合引擎**（非 vectorbt）。原因：A股的 T+1、一字板不可成交、停牌顺延、
最低佣金等约束高度有状态，自写引擎对这些规则的可读性与可验证性更好；指标也全部自实现，避免第三方默认参数黑盒。
