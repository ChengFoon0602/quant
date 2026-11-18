# quant — 轻量量化交易研究框架

个人量化策略研究与实验平台。目标：轻量、可审计、数学上严谨。

## 结构

```
quant/
├── CLAUDE.md                 # 方法论铁律 + 踩坑教训（已编译为测试的部分见 tests/）
├── CLAUDE_AGENT_SOP.md       # 因子自动挖掘 Agent 工作流（G-O-B-R 循环）
├── data/                     # 数据层：fetcher（baostock/akshare）+ 本地缓存 + manifest 快照
│   ├── index_membership.py   # PIT 指数成分股矩阵（沪深300/中证500，含退市股）
│   ├── price_adjust.py       # 复权因子 + 真 VWAP（build_true_vwap），Alpha191 唯一口径入口
│   ├── download_adjust_cache.py # 补齐 adj1(后复权)/adj3(不复权) 缓存（--index hs300|zz500）
│   ├── pit_auditor.py        # PIT 对齐离线审计（防未来函数自检）
│   └── cache/                # CSV 缓存（{symbol}_adj{1,3}.csv，不入 git，可增量更新）
├── signals/                  # 因子库：alpha191（191 因子 + 算子层）、基本面 factors、横截面正交化
├── backtest/                 # 向量化回测引擎 + metrics（CAGR/Sharpe 口径）+ invariants（契约断言）+ reconciliation（账本对账）
├── risk/                     # 组合构建（权重追踪法）+ cost_model 成本真源 + 约束编排（orchestrator）+ 拥挤度 + 交易可行性（tradability）+ 回撤控制（drawdown_control）
├── strategies/               # 策略研究：每子目录 = report.md + report.py + figures/
├── models/                   # ML 非线性合成（LightGBM）+ Walk-Forward + PIT-Select（去 selection bias）+ 方法论修正记录
├── tests/                    # 312 项测试函数（铁律编译、契约断言、自洽对账、Alpha191 口径守卫）
├── docs/                     # 口径论证、回测语义对照表、Alpha191 审计/修复/执行记录
├── run_reconciliation.py     # 真实数据账本对账（离线运行器 → reconciliation_report.txt）
├── run_tradability_impact.py # 交易约束代价量化（离线运行器 → tradability_report.txt）
├── run_drawdown_control.py   # 回撤控制参数网格 + 前沿对比（离线运行器 → drawdown_control_report.txt）
├── run_drawdown_walk_forward.py # 回撤控制 walk-forward 复核（离线运行器 → drawdown_wf_report.txt）
├── viz/                      # 可视化工具
└── bootstrap.py              # 首次数据拉取脚本
```

## 研究报告索引（按研究弧线阅读顺序）

> **研究弧线**：从「单因子/单票 → ML 合成 → 换指数 → 换信息源 → 市场结构 → 跨指数验证 →
> ETF 与大类资产 → 全天候配置」一路推进。
>
> **⚠️ 成本口径修正（2026-09-05）**：早期「A 股日线级截面 alpha 在扣除真实成本后不成立」的核心数字
> 曾被一个交易成本 bug（代码 0.3% vs 铁律 0.1%）压低。收口到铁律 0.1% 后，多空 alpha 在扣除真实
> 成本后**显著成立**（沪深300 LS 0.036→1.175、中证500 去 bias LS −0.188→+1.214）；多头增强转为
> 「方向转正、显著性边缘、容量 <0.5 亿」的中间态；做空端仍否定（真实融券成本 8-10% 主导）。
>
> **⚠️ Alpha191 口径修复（2026-10-24，影响面最大的一次）**：`signals/alpha191/` 曾有三类缺陷——
> ① 因子用**前复权**价计算（前复权因子由**未来**除权事件决定，含前视）；② `vwap = amount/volume`
> 是**未复权**均价，与复权 close 量纲不一致（实测 `vwap/close` 中位 1.31、最大 4.01），15 个因子
> 退化为常数；③ 算子层两个实现 bug（`REGBETA` ddof 不一致使 β 放大 (n−1)/n、`SMA` 初值路径与研报
> 不等价）。修复后**全部报告已重跑或标注作废**：因子池 16→15，`models` 新增「方法论修正 V/VI」，
> 沪深300 拥挤度两条结论反转、一条撤销。铁律见 `CLAUDE.md` §3.5.1；证据
> `docs/alpha191_复现审计_2026-10-24.md`，执行 `docs/alpha191_修复执行_2026-10-24.md`，
> 冻结版本 `git tag pre-alpha191-fix-20261024`。
>
> **⚠️ ETF 复权口径修复（2026-11-18，影响全部 ETF 类报告）**：`data/etf_fetcher.py` 原实现
> 调用 `akshare.fund_etf_hist_sina`（**该接口无复权参数**）却声称「前复权」，分红与份额折算
> 跳空直接进入收益序列。18 只 ETF 池实测 **8 处份额折算（幅度 53%~249%）+ 45 次分红**未计入：
> 510500 在 2015-04-15 有 **+248.6%** 折算跳空（该标的年化虚高 12.6pp）、512200 在 2024-08-12
> 有 **+170.1%**（对等权组合单日贡献 **+17.01%**）、512010 在 2021-06-28 有 **−73.9%**（年化虚低 12.6pp）。
> 现由 `data/etf_adjust.py` 构造后复权因子（分红 × 折算），`load_etf_daily()` 默认返回 `hfq`
> 口径。**`etf_momentum_crowding`、`all_weather_risk_parity`、`qvix_regime` 三份报告已全部重跑**，
> 前两份开头加了口径修正块；其中动量报告的「夏普由 0.471 降至 0.394」**结论反转**为
> 「0.459 升至 0.468，回撤降四成且夏普不降」。诊断与修法见
> `strategies/qvix_regime/report.md` §6.7。
>
> 报告索引中的每份报告都含「方法论修正」章节——其记录的真实踩坑往往比结论本身更有价值。
> 标 ⛔ 者为**数字已作废、保留正文以记录研究过程**，不要引用其数字。

| 阅读序 | 报告 | 阶段 | 核心结论 |
|--------|------|------|---------|
| 1 | [`ma_crossover`](strategies/ma_crossover/report.md) | 单票时序 | MA 交叉无显著超额——A 股有效性起点 |
| 2 | [`alpha001_trial`](strategies/alpha001_trial/report.md) | 单因子截面 | Alpha001 IC>0 但信息比率负，跑不赢基准；口径修复后逐项复核**数字不变**（对照组） |
| 3 | [`alpha055_trial`](strategies/alpha055_trial/report.md) | 单因子深度 | 新口径下**转负/剔除**：Rank IC 0.016、IC_IR 0.127、多空夏普 **−0.246**、分层不单调（首次成文） |
| 4 | [`multi_factor_trial/alpha012_alpha055_alpha191`](strategies/multi_factor_trial/alpha012_alpha055_alpha191/report.md) | 多因子合成 | ⛔ **不可修复**：核心结论依赖被污染因子值 + 事故 universe + 055 已转负，三重缺陷叠加 |
| 5 | [`factor_discovery`](strategies/factor_discovery/report.md) | 因子提纯 | ⛔ **不可修复**：141 是「纸老虎」（IC 高无分散度），055/001 价格绝对量受前复权污染 |
| 6 | [`feature_selection`](strategies/feature_selection/report.md) | 特征选择 | **191→15 因子**（新口径重筛 2026-10-24；三维通过 **30** = 口径修复时 29 + 算子修复新增 alpha149，后未进池；成分换 9），PIT 沪深300 |
| 7 | [`models/report.md`](models/report.md) | ML 非线性合成 | **方法论修正 I–VI**：夏普 8.75→0.08 的完整踩坑史 + 成本 bug + Alpha191 口径/算子修复；canonical WF 22.32%/1.582，**唯一可引用去 bias LS 年化 4.93%、夏普 0.524、回撤 −22.51%** |
| 8 | [`zz500_pit_trial`](strategies/zz500_pit_trial/report.md) | 路线② 量价×中证500 | 新口径全链重跑（1622 只）：**去 bias WF 夏普 1.350 / 年化 16.16% / 回撤 −20.91%**；含 bias 的 WF 2.521、LS 2.327、OOF IC 0.0863 不可单独引用。**P1/P2 仍为修复前口径** |
| 9 | [`zz500_fundamental_trial`](strategies/zz500_fundamental_trial/report.md) | 方向2 基本面 | 换源 akshare，20 因子三层检验全否定，弧线闭合（不受 Alpha191 口径修复影响） |
| 10 | [`zz500_crowding_trial`](strategies/zz500_crowding_trial/report.md) | 方向C 市场结构 | 量价延续 vs 基本面反转双面体 + 低拥挤择时可交易性初探（中证500 基线） |
| 11 | [`hs300_crowding_trial`](strategies/hs300_crowding_trial/report.md) | 方向C 延伸④ 跨指数 | **跨指数两指数同池重跑**：量价拥挤度**方向相反**（zz500 Q4 **−0.084%** t=−1.90 vs hs300 **+0.103%** t=2.31）；基本面同向但效应量差 4.4×；**撤销**旧「p=0.0002 确证负 alpha」（系 0.3% 成本 bug）；择时两边都不显著 |
| 12 | [`etf_momentum_crowding`](strategies/etf_momentum_crowding/report.md) | 阶段D ETF与大类资产 | ⚠️ **行业动量反转 + 股债金避险**：行业 ETF 截面动量显现反转（Rank IC −0.0385）；叠加 MA20 趋势与国债/黄金避险后，**最大回撤从 −46.48% 降至 −25.58%（降幅 45%）**，**夏普由 0.459 升至 0.468**（2026-11-18 后复权口径修正；旧「夏普同降 0.471→0.394」**结论已反转**）；双边 4 bps 超低摩擦 |
| 13 | [`all_weather_risk_parity`](strategies/all_weather_risk_parity/report.md) | 阶段E 全天候多资产配置 | **欧拉风险平价 (ERC)**：跨资产等风险贡献消除股票风险霸权，2015-2025 全周期 **SR=1.873（后复权修正，原 1.775）、MDD 仅 −4.02%（降低 90%）、Bootstrap p=0.0000**，奠定极稳底层资产配置底座 |
| 14 | [`qvix_regime`](strategies/qvix_regime/report.md) | 阶段F 波动率状态 | **期权隐含波动率闸门 2/4 关通过**：QVIX 预测波动优于历史 RV（$R^2$ 0.410 vs 0.359）但不预测收益（p=0.30）；同等暴露下夏普 0.665 vs MA20 0.492，代价是回撤 −40.49% vs −25.61%，优势 146% 来自 2021 单年。**不可替代 MA20；双闸门并联夏普 0.529、回撤 −25.51%，两项均优于单 MA20**。附 7 条方法论修正，含查出并修复 **ETF 缓存实为未复权**（8 处折算 + 45 次分红未计入，512200 单日虚假 +170% 污染组合 +17.01%） |
| 15 | [`pairs_trading`](strategies/pairs_trading/report.md) | 阶段G 统计套利 | ⛔ **否定结论**：zz500 同门类配对（1468 只 / 465 对 / 6142 回合）毛年化 **−36.53%、毛夏普 −5.198、日胜率 22.7%** —— 价差突破 ±2σ 后**继续扩大而非回归**（事件研究回归比例仅 55%~58%，协整跨期存活率仅 24%）。合成数据毛年化 **+37.46%** 验证实现无误，六组参数网格毛收益全部 −33%~−38%（非参数问题）。**A 股个股价差是动量的，与 ETF 层面反转相反**。另探测排除**股债配对**（历史转股价不可得 + 转债不能融券 + 比值半衰期中位 78 天） |

**方法论铁律**（贯穿全部报告，详见 `CLAUDE.md`）：
未来函数（`.shift(1)` / PIT 公告日 / 复权方式）、幸存者偏差（PIT 成分股）、Overlapping Returns 平滑陷阱、
selection bias（WF 年度重选池）、摩擦成本（双边 ≈ 0.1%，买 0.026%/卖 0.076% 方向分离）、
数据可复现性（manifest + 随机种子）、**Alpha191 因子口径（铁律 3.5.1：后复权 + 真 VWAP + 面板 RANK +
退化因子不修 + golden 只防回归）**、向量化优先。

**工程铁律**（管系统本身是否可靠，详见 `CLAUDE.md` E1–E4）：
确定性内核用断言不用告警（`backtest/invariants.py`，失败 `raise`）、四个时点显式化
（信号可用 / 订单生效 / 成本发生 / 收益归属）、回测↔实盘语义一致（`docs/回测语义对照表.md`）、
AI 干预深度 ∝ 1 ÷ 错误代价。账本三对账（仓位 / 流水 / 盈亏闭合）由 `tests/test_reconciliation.py`
以**独立重算**方式守护。

## 工程质量（2026-09-09 / 09-14 / **10-24** 收口）

方法论纪律不止写在 `CLAUDE.md`，**已编译为可执行测试与单一真源**：

| 层 | 内容 |
|---|---|
| **成本单一真源** | `risk/cost_model.py`（BUY_COST/SELL_COST/ROUND_TRIP）；全仓库调用点统一 import，无硬编码残留 |
| **铁律守护测试** | `tests/test_cost_model.py` 用 `inspect` 读取**真实默认值/显式 COST 常量**断言=真源；变异验证过（改回 0.003 立即失败） |
| **Alpha191 口径守护** | 铁律 3.5.1 编译为测试：`test_alpha191_health.py`（4 项：无全 NaN / 截面分散度 / VWAP 量纲 / alpha001 前视不变式）；`verify_operators_independent.py`（纯 numpy 双实现 **85/85 逐位一致**，TOL 1e-9）；`factor_golden_baseline.py`（**191/191 防回归**）；`docs/alpha191_公式语义表.csv`（191 行逐条注解）；`tests/alpha191_validation_report.md`（L1–L5 分层验证，**L5 外部对拍未完成**） |
| **指标规范口径** | `backtest/metrics.py`：年化统一为路径 CAGR；Sharpe 口径跨模块一致 |
| **输入契约断言** | `backtest/invariants.py`：价格面板 / 收益面板 / 权重矩阵 / 费率 / **结果合理性**，接入数据层（`data/fetcher.py`）、4 个回测入口与输出端，失败 `raise` 不 warn |
| **账本自洽对账** | `backtest/reconciliation.py`（单一真源）+ `tests/test_reconciliation*.py`：仓位 / 流水 / 盈亏闭合，**独立重算**比对三条组合实现；**覆盖层三层账本**（仓位/费用/现金）；**Walk-Forward 逐窗拼接对账**（防重叠/乱序/重复 + 缺口报告）；`run_reconciliation.py` 对**真实报告数据**（790 票）离线对账 |
| **时点契约** | `tests/test_time_point_contract.py`：手算场景同时钉住信号可用 / 订单生效 / 成本发生 / 收益归属四个时点 |
| **成交假设断言** | `tests/test_execution_assumptions.py`：涨跌停识别（主板/创业板/ST/北交所）+ **撮合层真拦单**；`test_price_adjust.py` 锁死三复权不共用文件 |
| **未实现参数守卫** | `tests/test_risk_constraints.py`：`max_turnover` 等未实现的约束必须 `raise`，不允许静默忽略；`max_leverage` 语义锁死为 gross（`Σ\|W\|`）|
| **滑点口径守卫** | `test_cost_model.py::test_slippage_default_is_zero_everywhere`：滑点默认必须 0.0（不进主口径），改动须先量化（`run_slippage_impact.py` 实测高换手链路夏普掉 ~38%）|
| **交易可行性度量** | `risk/tradability.py`：`build_trade_limits_from_cache()` **一行接入**涨跌停约束；`run_tradability_impact.py` **跨 4 条链路**量化代价（换手几乎不变但夏普相对掉 3%~12%，**下界**；结论方向均未改变）|
| **回撤控制（opt-in）** | `risk/drawdown_control.py` 两个变体：两态滞后带 + **连续映射（自适应形态）**，**输出即闭合账本**（仓位/费用/现金三层；净值与回撤按净收益路径推进）。`run_drawdown_control.py` 做匹配仓位前沿对比 + 边界扫描（最优 mca=0.02 为内部点）；`run_drawdown_walk_forward.py` 做 WF 复核（连续优于两态 7/11；夏普 1.48→1.99、回撤 −29.5%→−8.0%，**但 CAGR 近乎减半**）→ 风险-收益重配比，须**并列报告**。**不回改已发布报告** |
| **PIT 自检** | `data/pit_auditor.py` + `run_pit_audit.py` 离线审计基本面缓存日期约定 |
| **向量化引擎** | `build_portfolio` 权重构造向量化（27×，等价性测试逐位守护）；`TSRANK` 用 `sliding_window_view`（约 8×） |
| **测试规模** | **312 项测试函数**（`tests/`，26 个文件）：铁律 1/2/3 编译、契约断言、自洽对账、时点契约、成交假设、面板构建、约束守卫、清算三层账本、WF 拼接对账、滑点口径、容量重估、回撤控制、正交化、IC 锚定方向、7 算子、组合等价、AM-GM、**Alpha191 口径守卫** |

> 驱动这些收口的两轮评审发现：成本守护测试此前是「空转」（自声明常量测算术，从不 import 真实代码）、
> 基本面缓存存在 100× 尺度污染、Walk-Forward 隐藏调用点仍传 0.3% 成本——均已被修正并加守护。
>
> **2026-09-14 加固**：正确性保障从事后回归前移到**事前断言**。同时经独立重算对账发现一处真实口径差异——
> `min_stocks` 阈值 `risk/portfolio.py` 用 `base*2`、`models/portfolio_backtest.py` 用 `base*3`（默认 10 vs 15）；
> 除该阈值外两条实现逐位一致，差异已由 `TestKnownDivergence` 钉住，**未擅自统一**（会改变已发布报告持仓日集合）。
>
> **2026-09-15 残余收口**：断言下移到**数据层与结果层**；对账升级为**真实报告数据**（790 票全量逐位一致）；
> 成交假设与复权从人读清单变为自动断言；四时点收敛为一条手算契约。
> `min_stocks` 参数化为 `min_stocks_mult`（默认行为逐位不变）并实测：**全量 universe 下差异为 0**，
> 仅稀疏截面显著 → **决策不统一**，留作显式开关。
> 顺带修复 `trade_limits` 日循环从 `i=1` 起导致**首日持仓绕过涨跌停**的边界缺口（对已发布结果零影响）。
>
> **2026-10-24 Alpha191 收口**：因子口径（后复权 + 真 VWAP）与算子层（REGBETA ddof、SMA 初值）两个 bug
> 修复后，**全链重跑并归因**：A→B 口径修复 WF 夏普 0.434→1.285（**+196%**）、B→C 池换血 →1.427（+11%）、
> C→D 算子修复 →1.582（+11%）、D→D′ 去 selection bias →0.524（**−67%**）。
> **selection bias 是主导项，口径修复是最大正向项，算子精度是二阶项**（去偏值几乎不动：0.510→0.524）。
> 顺带发现并修复两个静默 bug：CSV 缓存 symbol 前导零被吃掉（深市全错、沪市对，且不报错）、
> 批量下载把服务端限流误记为「停牌或退市」（zz500 首轮 485 只假失败）。

## 快速开始

```bash
# 依赖（版本锁定见 requirements.txt：pandas/numpy/scipy/matplotlib/baostock/akshare/lightgbm/scikit-learn）
pip install -r requirements.txt

# ① 数据前置：Alpha191 需后复权(adj1)/不复权(adj3) 缓存
python data/download_adjust_cache.py --index hs300     # 或 --index zz500

# ② 首启数据（沪深 300 日线）
python bootstrap.py

# ③ 测试（务必指定 tests/，不要裸跑 pytest——竞赛材料/ 下另有测试会干扰收集）
pytest tests/ -q                                       # 312 项测试函数

# ④ Alpha191 口径自检（改因子代码前后各跑一次）
pytest tests/test_alpha191_health.py                   # 4 项守卫
python tests/verify_operators_independent.py           # 算子双实现互证（85 项）
python tests/factor_golden_baseline.py --verify        # 191 因子防回归

# ⑤ 跑报告
cd strategies/ma_crossover && python report.py

# ⑥ 离线运行器
python run_reconciliation.py           # 真实报告数据账本对账（790 票，约 10s）
python run_tradability_impact.py       # 交易约束代价量化（4 条链路，约 40s）
python run_slippage_impact.py          # 滑点敏感性网格（4 条链路，约 40s）
python run_capacity_reestimate.py      # 容量上限重估（约束+滑点纳入，约 60s）
python run_drawdown_control.py         # 回撤控制参数网格 + 前沿对比（790 票，约 25s）
python run_drawdown_walk_forward.py    # 回撤控制 walk-forward 复核（790 票，约 30s）
```

> 数据已缓存在 `data/cache/`（日线后复权/不复权 + 基本面 + ETF + 指数成分），fetcher 负责增量更新。
> 复现纪律：数据变更后运行 `python data/manifest.py` 重建 manifest 快照并提交。

## 技术栈

Python 3.12 · baostock · akshare · pandas · numpy · scipy · matplotlib · scikit-learn · lightgbm

## 声明

本项目仅用于量化研究学习，不构成任何投资建议。
