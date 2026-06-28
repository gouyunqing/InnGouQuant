# InnGouQuant 研究日志 & 跨机交接文档

> 本文件记录从「诊断 SuperTrend 过拟合」到「用业界正道做出第一个市场中性趋势策略」的**完整迭代、认知、当前结果、下一步及其原因**，并附 **Windows 换机注意事项**。
> 系统说明见 [README.md](README.md)；本文件是**研究历程 + 交接**。最后更新：2026-06-28。

---

## 0. TL;DR —— 现在在哪

起点：用户认为 SuperTrend 策略「严重过拟合」。一路下来：
1. **先修评估框架**（这才是根因之一）：窗口限 2010+、程序化 120 大盘股池、**walk-forward(WF) 升为主判据**、补齐机构级指标（β/α/Sortino/VaR/DSR/PBO）。
2. **一系列趋势/事件 edge 被逐个证伪**：裸 SuperTrend、regime 门控、领先探测器、低量反转、解禁反转——每个看着有 edge 的，在 size 中性 + WF + DSR/PBO 下都化掉了。
3. **回到「抓趋势」主线**，但换了范式：不再单股择时，改 **横截面相对强度因子**。纯因子版无条件 α≈0，**但发现趋势 α 是 regime 条件的**（指数趋势市 vs 震荡市，多空价差差 32 个点）。
4. **方案1（当前最好结果）**：把已证实的条件 α 用业界正道收获——**β 中性 + 指数 regime 闸 + 波动率缩放(vol-target)**。IF 对冲版 **净值年化 +8.3%、Sharpe ~1.0、回撤 −18%、β≈0.02、WF 69% 窗为正**，扣完线性成本仍正。

**下一步**：对方案1 跑 **robustness 终审**（PBO/DSR + 参数敏感性 + 子区间 + 成本/基差压力测试），确认 Sharpe~1.0 里有多少是真的。**这是开始 Windows 开发后的第一件事。**

---

## 1. 核心认知沉淀（最值钱，先读这个）

### A. 评估方法论
- **单次 holdout 会系统性地把策略夸好**（本项目里反复出现 holdout α 正、WF α 负/零）。**WF 是唯一诚实的判据**，必须看逐窗 OOS、正窗占比、参数跨窗稳定性。
- **DSR（去偏夏普，Bailey & López de Prado）** 按试验次数修正选择偏差；**PBO/CSCV** 测过拟合概率。二者互补：PBO 低 + DSR 不显著 = 「选哪组参数都差不多，但策略族整体没显著绝对 edge」。
- **「减大盘指数」≠「size 中性」**。事件研究只减沪深300（大盘）会把小盘超额误读成 α。**正确做法是减等权同伴 / 多空对冲**。解禁案就栽在这（见 §2.6）。
- **幸存者偏差不可修**：本地数据只含当前在市股，退市样本缺失，所有结果系统性偏乐观。

### B. 这个市场 / 股池的统计事实
- **A 股月度横截面动量是死的**，短期（1–6 月）反转主导，根因是散户高换手把「过度反应→修正」周期压短。集成多 lookback + skip 月 + 残差都救不回（v1 实测 IC≈0）。
- **但趋势是 regime 条件存在的**：指数自身处于上升趋势时，横截面趋势因子多空价差 +17%/年；震荡市 −15%/年。**用户最初「趋势工具只在趋势 regime 有效」的假设——在因子层被决定性证实**。
- **唯一真实的大收益源是 size 因子（小盘溢价 ~11%/年）**，但它①人尽皆知②风险巨大（小盘股灾/壳价值消退）③连它都被幸存者吹高。本股池/数据上**没有藏着的免费午餐**。
- **A 股惩罚追高、奖励安静**：确认型信号（量能突破/ATR 跳升/箱体突破/regime 投票）对 SuperTrend 的 flip_up 未来超额**一致削弱**；反转型（低量抄底）才一致为正。

### C. SuperTrend 这条线的最终结论
裸 SuperTrend flip_up 有**一丝真信号**（超额~2%、WF α+2.74%），但①全程不显著（PSR/DSR<95%）②最好裸着收，**任何过滤器（追高型 OR 反转型）都只降风险不增 α**③无法放大成稳健显著可交易 edge。单股时序择时这条路走到头。

### D. 范式级教训
**问题从来不是参数，是范式。** 把趋势工具用在「单股价格时序择时」上，在 A 股统计上不成立；趋势真正活在 **① regime 条件的横截面相对强度** 上，且必须配 **β 中性 + 波动率缩放** 才能收获——这正是 TSMOM/CTA 教科书（业界趋势策略一半的引擎是 vol-scaling，不是信号本身）。

---

## 2. 完整迭代时间线（做了什么 → 发现什么 → 为什么）

> 每个研究脚本都是一次受控实验，文件名见 §5。

### 2.0 起点诊断
表面「严重过拟合」，根因 = **(A) 评估框架破碎 + (B) 策略近零 edge 且无风控**。定下分层计划 Tier0(修尺子)→Tier1(风控)→Tier2/3(换 edge)。

### 2.1 Tier 0 修评估框架 ✅
窗口限 2010+；股池从手挑 11 只 → **程序化 120 大盘股**（主板+创业板、剔 688/北交所、**日均成交额≥1.5亿 作「剔百亿以下」代理**，因本地无市值字段）；WF 升主判据；补 purge+embargo、PBO/CSCV；补机构级指标（benchmark.py 抓沪深300、report.py 加 β/α/Sortino/VaR/DSR）。**认知**：尺子修正后，原策略 WF OOS α 中位 ≈0、DSR 60%（不显著）——不是「清晰无 edge」而是「弱、不显著、参数不稳、幸存者高估」。

### 2.2 Tier 1 风控止损 ✅
硬止损 8% + 吊灯 3×ATR 跟踪。**回撤砍半、Calmar 翻倍，但 α 仍≈0**。**认知**：止损只压风险、不创造 α。

### 2.3 纯因子检验 + HMA + regime 门控（证伪「波动 regime 是元凶」）
- 纯动量因子消融：1/3/6/12-1 月 IC 全不显著（短周期略负=反转味）。
- HMA200 过滤 + 因子式网格寻优：holdout 好看但 **WF α 负、PBO 从 0 飙到 53%**＝教科书级过拟合。
- ER/ADX/R² 三专家 2/3 投票门控：**α 不升反降**，门控是风险开关非 α 发生器。**锐利条件检验**（5138 次 flip_up 按入场 regime 分组比未来超额）：**趋势组反而更差、按票数单调递减**＝ER/ADX/R² 滞后确认→追在趋势尾巴吃反转。**证伪假设**。

### 2.4 领先探测器（regime 变化点而非水平）
量能扩张/VWAP 重夺/ATR 跳升/Donchian 箱体突破/ADX 上穿…8 个探测器对 flip_up 未来超额的增益：**无一显著为正，7/8 为负**。**认知**：A 股惩罚追高，确认型一律削弱 α。

### 2.5 反转线索（把探测器反过来）
**首个既正又显著**：`低量`flip_up（量<10日均量）exc20=4.38%、**t=2.22**。但**组合级终审 washout**（太稀疏 8%、β≈0 近现金、WF 逐窗噪声打回 0）。**认知**：事件级显著 ≠ 策略级可收；低量是不错的低风险工具但「低风险零 α」非「有 edge」。

### 2.6 结构性 edge 转向：解禁反转（最接近真 edge 的候选，最终也被证伪）
- 数据审计：**解禁=首选可行**（akshare `stock_restricted_release_detail_em`，2010+，缓存 `unlock_calendar*.csv`）；指数调仓被堵（无历史成分变更）；北向仅 2017+。
- 事件研究：解禁后 [0,+40] 市场调整超额 **t=3.41、剂量-反应单调、不衰减**——机制清晰（解禁股东被迫抛→超跌→反弹）。
- 全市场组合（12785 起大解禁）：表面 α 7.39% 但 **β=0.95**——变成满仓小盘多头。
- **多空对冲隔离（决定性）**：多=解禁后[+5,+25]股、空=等权全市场。**中性化后价差全负、剂量效应彻底反转**。**解禁反转 = size 因子伪装**。一减 size，t=3.41 化成负。**这就是 §1.A「减指数≠size 中性」教训的来源**。

### 2.7 回归抓趋势 · 横截面范式（当前主线）
用户收口结构 edge，回到「上涨趋势可判断、只是策略不到位」。调研业界（TSMOM/MOP2012、A 股动量反转文献、CTA vol-targeting）后复盘 SuperTrend **4 个范式错误**：①单名时序择时（该横截面相对强度）②挑单一参数（该集成多 lookback）③二元满仓（该 vol-scaling）④不做 size 中性。

- **v1 纯横截面趋势因子**（`test_trend_xs.py`）：综合分=52周高点贴近 + 中期动量(skip月,集成{63,126,252}) + 残差动量，截面 rank 等权，月频，5 档。**无条件 α≈0**：Rank-IC −0.003、Q5−Q1 −2.27%/年、非单调。**但 WF 暴露强 regime 依赖**（趋势市 +85% vs 震荡市 −43%）。
- **指数 regime 闸**（`test_trend_gated.py`，沪深300>200日线 Faber 同款）：**条件分裂校验=决定性证实**——Q5−Q1 **闸开 +17.0% / 闸关 −15.0%（差 32 点）**。但 long-only 收不干净（闸关仍赚 β，空仓=拿收益换风险，Sharpe 0.47→0.48 几乎不动）。**真 α 是多空性质、regime 切换**。
- **方案1 = β 中性 + 闸 + vol-target**（`test_trend_v2.py`，当前最好，详见 §3）。
- **行业动量腿（`test_sector_mom.py`）= 数据层卡死**：申万一级成分接口 `index_component_sw` 被 **IP 时间冷却**（每会话约 3 次后长封），全市场映射在 Mac 这台 IP 上拉不出。**留给 Windows（新 IP）跑 `fetch_sectors.py` 续传**，见 §6。

---

## 3. 当前结果 —— 方案1 v2（β 中性 + regime 闸 + vol-target）

`python test_trend_v2.py`，日频回测，2010+，β 是对沪深300：

| 变体 | 年化 | Sharpe | 回撤 | Calmar | β |
|---|---|---|---|---|---|
| A 未闸 Q5−Q1 多空(gross) | +7.1% | 0.31 | −64% | 0.11 | −0.05 |
| B 闸控 LS(gross,未杠杆) | +10.9% | 0.67 | −48% | 0.23 | +0.08 |
| C 闸控+voltgt LS(gross) | +6.1% | 0.74 | −25% | 0.24 | +0.04 |
| C 闸控+voltgt LS(**net 扣15bp**) | +5.8% | **0.70** | −25% | 0.23 | +0.04 |
| **D 闸控+voltgt IF对冲(net)** | **+8.3%** | **0.98** | **−18%** | **0.46** | **+0.02** |
| HS300 基准 | +4.1% | 0.19 | −47% | 0.09 | 1.00 |

- **逐件引擎贡献都看得见**：闸 Sharpe 翻倍（0.31→0.67）；vol-target 把回撤再砍半（−48%→−25%）；IF 对冲版 Sharpe~1.0、回撤 −18%、β≈0（真中性）、扣线性成本仍 +8.3%。成本几乎不咬（月频大盘换手低）。
- **WF（C net，最严判据）**：中位年化 +5.5%、Sharpe 1.06、**69% 的 OOS 窗为正**（v1 才 46%）。强窗 2017 Sharpe 2.4 / 2021–22 Sharpe 2.3；弱窗只剩 2014–15(股灾) 和 2024–25。

### ⚠️ 还不能信到底——4 条红线（这就是为什么要做 §4）
1. **做空现实性**：C(Q5−Q1) 要融券做空个股，A 股**融券贵且券源稀缺**→C 偏理想化。**D(空 IF 期货) 才真正可交易**，且 D 本就是最好的，**主推 D**。但 D 的 Sharpe 还没扣**期货基差/展期**和**市场冲击**。
2. **过拟合未验**：200日线、10% 目标波动、五分位都是标准值、没乱调，但**没跑 PBO/DSR**——Sharpe 1.0 可能含运气。
3. **幸存者偏差**：仍在，系统性高估（解禁就栽这）。
4. **参数敏感性未测**：换 SMA 窗(150/250)、目标波动(8/12%)、分位数，结果该稳；不稳就是过拟合。

---

## 4. 下一步计划（开始 Windows 开发后的第一件事）

**对方案1（主测 D，兼测 C）跑 robustness 终审**，原因 = §3 那 4 条红线必须扫掉才能信 Sharpe~1.0：

1. **PBO + DSR**：把 D/C 的夏普按试验次数去偏、测过拟合概率。要求 DSR>95%、PBO 低。
2. **参数敏感性网格**：SMA∈{150,200,250} × 目标波动∈{8%,10%,12%} × 分位数∈{quintile,decile} × vol 回看∈{40,60,90}。看 Sharpe 曲面是否**平台**（稳）还是**尖峰**（过拟合）。
3. **子区间稳定性**：2010–2015 / 2016–2020 / 2021–2026 三段分别看，β、Sharpe、α 是否一致。
4. **成本/基差压力测试**：D 加 IF 年化基差成本（保守 −2~4%/年）+ 冲击成本上调；C 加融券费（保守 8%/年）。看净 Sharpe 还剩多少。
5. （可选，待 Windows 拉到行业数据后）**把行业动量作为第四个因子折进综合分**再跑一版，看广度能否抬升 IC。

**判据**：D 扛过 PBO/DSR + 参数平台 + 子区间一致 + 成本压力后净 Sharpe 仍 >0.6 → 方案1 是真的，可推进到组合工程化；否则定位清楚水分在哪。

---

## 5. 代码地图

### 系统核心模块（V1 SuperTrend 链路，详见 README）
| 文件 | 角色 |
|---|---|
| `config.py` | 集中配置。**数据根走环境变量 `INNGOU_DATA_ROOT`**；`CACHE_DIR` 放机器专属 pkl |
| `data_loader.py` | 加载/前复权/体检；`build_universe()/get_universe()` 程序化 120 股池；`backtest_window()` |
| `factors.py` | SuperTrend(自实现 Wilder ATR) + 量能突破 + HMA/ER/ADX/R²/ATR跳升/Donchian/VWAP |
| `strategy.py` `backtest.py` | 信号接口 + A 股约束回测引擎(T+1/涨跌停/停牌/成本/止损) |
| `optimize.py` | 网格寻优；holdout & **walk-forward**；`walkforward_windows()` |
| `report.py` | 指标/图/markdown；`metrics_from_equity` `metrics_vs_benchmark` `deflated_sharpe_ratio` `probability_of_backtest_overfitting` |
| `benchmark.py` | 抓指数日线(东方财富)缓存 `benchmarks/` |
| `main.py` | 端到端编排 |

### 研究脚本（每个=一次受控实验；`python xxx.py` 跑）
| 脚本 | 干嘛 | 结论 |
|---|---|---|
| `factor_test.py` | 纯动量因子 IC 消融 | 全不显著 |
| `test_hma.py` `optimize_hma.py` | HMA200 过滤 + 因子式寻优 | 过滤降 α；寻优 PBO 53% 过拟合 |
| `test_gated.py` `test_conditional.py` | ER/ADX/R² 门控 + 锐利条件检验 | 证伪「震荡是元凶」，门控滞后吃反转 |
| `test_detectors.py` `test_contrarian.py` | 领先探测器 / 反转线索 | 确认型削弱 α；低量 t=2.22 但组合 washout |
| `test_lowvol.py` | 低量信号组合级终审 | WF washout |
| `study_unlock.py` `test_unlock.py` `test_unlock_market.py` `test_unlock_ls.py` | 解禁反转：事件研究→策略→全市场→多空隔离 | **决定性证伪=size 伪装** |
| **`test_trend_xs.py`** | **v1 横截面趋势因子** | 无条件 α≈0，但 regime 依赖强 |
| **`test_trend_gated.py`** | **指数 regime 闸** | 条件分裂 +17%/−15%，决定性证实 |
| **`test_trend_v2.py`** | **方案1：β中性+闸+vol-target** | **当前最好，Sharpe~1.0/β≈0** |
| `test_sector_mom.py` | 行业动量（待数据） | 申万接口限流，需 Windows 续拉 |
| `fetch_sectors.py` | 拉申万一级 code→行业 全市场映射（可续传） | Mac IP 被封，Windows 重跑 |

**复用资产**：`test_trend_xs.py` 的 `build_panel()/sig_*()/monthly_eval()/walkforward()/ann_from_monthly()` 被 gated/v2/sector 直接 import 复用。

---

## 6. ⚠️ Windows 换机注意事项（必读）

### 6.1 数据路径 —— 用环境变量，别改代码
`config.py` 的 `DATA_ROOT` 已改成读环境变量 `INNGOU_DATA_ROOT`。下载好数据后（目录结构同 README §数据适配），设：
```
PowerShell:  $env:INNGOU_DATA_ROOT="D:\你的下载目录"
cmd:         set INNGOU_DATA_ROOT=D:\你的下载目录
```
数据目录内部结构（`a股分钟线/日k/全部日k.zip`、`复权因子/...`）要和 README 一致，否则改 `config.py` 里对应子路径。

### 6.2 控制台中文编码 —— 必设，否则 print 中文报错
所有脚本大量 print 中文。Windows 默认 GBK 控制台会 `UnicodeEncodeError` 或乱码。**运行前设**：
```
PowerShell:  $env:PYTHONUTF8="1"        # 或 chcp 65001
cmd:         set PYTHONUTF8=1
```
建议直接设为系统环境变量一劳永逸。

### 6.3 python / 运行方式
- Windows 一般是 `python`（不是 `python3`）。
- 脚本都在仓库根、import 同级模块。**在仓库根目录运行**即可，通常不用设 `PYTHONPATH`；若 import 报错，设 `set PYTHONPATH=%CD%`（cmd）/ `$env:PYTHONPATH=$PWD`（PowerShell）。

### 6.4 依赖
`pip install -r requirements.txt`。注意：**本机无 scipy**（spearman 用 rank+pearson 自己算，见 `test_trend_xs.spearman`）；若 Windows 装了 scipy 也不冲突。pandas 用的 2.3.x。

### 6.5 缓存会重建（首次慢）
`cache/` 和 `*.pkl`（价格面板 `trend_xs_panel.pkl` 等）**未入库**，依赖本地行情，Windows 首次跑 `test_trend_*.py` 会**重建价格面板（加载 120 只，慢几十秒）**，之后走缓存。`universe.csv` 也未入库，首次 `get_universe()` 会从本地行情按流动性重建（你的数据内容不同→股池可能与 Mac 不同，结果会有差异，正常）。

### 6.6 行业数据要在 Windows 续拉
申万成分接口在 Mac 这台 IP 被时间冷却封死。**Windows（新 IP）上跑 `python fetch_sectors.py`**（已写成分块+可续传，跑失败重跑会跳过已完成行业续传）。拉满 ≥25 个行业后会写 `sector_map_full.csv`，再 `python test_sector_mom.py`。

### 6.7 已修的跨平台坑（FYI）
- `config.DATA_ROOT` → 环境变量。
- `test_trend_xs.py` / `test_unlock_ls.py` 的 pkl 路径：原硬编码 Mac 的 `/private/tmp/...`，已改 `C.CACHE_DIR`。
- `fetch_sectors.py` 输出路径：原绝对路径，已改相对脚本目录。

---

## 7. 数据与缓存入库策略

**入库（已 commit，Windows 直接离线复用）**——市场级参考数据，与本地行情无关、跨机一致：
- `benchmarks/`（沪深300 等指数日线）
- `unlock_calendar.csv` / `unlock_calendar_full.csv`（解禁日历，23527 起，akshare 多分钟拉取）

**不入库（gitignore，换机重建）**——依赖本地行情、机器绑定：
- `cache/` `*.pkl`（价格/收益面板）、`universe.csv`（程序化股池）、`sector_map*.csv`（拉取未完成）

---

*完整逐条实验记录另见 Claude 记忆 `project-strategy-overfit-plan.md`（与本文件互为详略）。*
