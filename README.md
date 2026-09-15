# quant — 轻量量化交易研究框架

个人量化策略研究与实验平台。目标：轻量、可审计、数学上严谨。

## 结构

```
quant/
├── CLAUDE.md                 # 方法论铁律 + 踩坑教训（已编译为测试的部分见 tests/）
├── CLAUDE_AGENT_SOP.md       # 因子自动挖掘 Agent 工作流（G-O-B-R 循环）
├── data/                     # 数据层：fetcher（baostock/akshare）+ 本地缓存 + manifest 快照
│   ├── index_membership.py   # PIT 指数成分股矩阵（沪深300/中证500，含退市股）
│   ├── pit_auditor.py        # PIT 对齐离线审计（防未来函数自检）
│   └── cache_*/              # CSV 缓存（不入 git，可增量更新）
├── signals/                  # 因子库：alpha191、基本面 factors、横截面正交化
├── backtest/                 # 向量化回测引擎 + metrics（CAGR/Sharpe 口径）+ invariants（契约断言）+ reconciliation（账本对账）
├── risk/                     # 组合构建（权重追踪法）+ cost_model 成本真源 + 约束编排（orchestrator）+ 拥挤度 + 交易可行性（tradability）+ 回撤控制（drawdown_control）
├── strategies/               # 策略研究：每子目录 = report.md + report.py + figures/
├── models/                   # ML 非线性合成（LightGBM）+ Walk-Forward + 方法论修正记录
├── tests/                    # 257 项回归测试（铁律编译、契约断言、自洽对账、变异验证守护）
├── docs/                     # 口径论证、回测语义对照表、实施计划
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
> **⚠️ 成本口径修正（2026-09-05）**：早期「A 股日线级截面 alpha 在扣除真实成本后不成立」的核心数字
> 曾被一个交易成本 bug（代码 0.3% vs 铁律 0.1%）压低。收口到铁律 0.1% 后，多空 alpha 在扣除真实
> 成本后**显著成立**（沪深300 LS 0.036→1.175、中证500 去bias LS −0.188→+1.214）；多头增强转为
> 「方向转正、显著性边缘、容量 <0.5 亿」的中间态；做空端仍否定（真实融券成本 8-10% 主导）。
> 报告索引中的每份报告都含「方法论修正」章节——其记录的真实踩坑往往比结论本身更有价值。

| 阅读序 | 报告 | 阶段 | 核心结论 |
|--------|------|------|---------|
| 1 | [`ma_crossover`](strategies/ma_crossover/report.md) | 单票时序 | MA 交叉无显著超额——A 股有效性起点 |
| 2 | [`alpha001_trial`](strategies/alpha001_trial/report.md) | 单因子截面 | Alpha001 IC>0 但信息比率负，跑不赢基准 |
| 3 | [`multi_factor_trial/alpha012_alpha055_alpha191`](strategies/multi_factor_trial/alpha012_alpha055_alpha191/report.md) | 多因子合成 | 三因子两个是噪声一个有效，合成不改善 |
| 4 | [`factor_discovery`](strategies/factor_discovery/report.md) | 因子提纯 | 191→106 候选，alpha141 是「纸老虎」（IC 高无分散度） |
| 5 | [`feature_selection`](strategies/feature_selection/report.md) | 特征选择 | 191→16 因子（含「事故 universe」更正记录） |
| 6 | [`models/report.md`](models/report.md) | ML 非线性合成 | **方法论修正 I/II/III/IV**：夏普 8.75→0.08 的完整踩坑史（overlapping returns 平滑 + 幸存者偏差 + 成本口径 bug） |
| 7 | [`zz500_pit_trial`](strategies/zz500_pit_trial/report.md) | 路线② 量价×中证500 | 信号翻倍（OOF 1.618）但 selection bias 消除后 0.295；**成本口径修正 IV 后去 bias LS −0.188→+1.214，多空 alpha 显著成立** |
| 8 | [`zz500_fundamental_trial`](strategies/zz500_fundamental_trial/report.md) | 方向2 基本面 | 换源 akshare，20 因子三层检验全否定，弧线闭合 |
| 9 | [`zz500_crowding_trial`](strategies/zz500_crowding_trial/report.md) | 方向C 市场结构 | 量价延续 vs 基本面反转双面体 + 低拥挤择时可交易性初探 |
| 10 | [`hs300_crowding_trial`](strategies/hs300_crowding_trial/report.md) | 方向C 延伸④ 跨指数 | **跨指数验证证伪**：沪深300 双面体不成立、择时 SR −1.69、基本面因子多空 p=0.0002 确证负 alpha——中证500 发现是小盘特例 |
| 11 | [`etf_momentum_crowding`](strategies/etf_momentum_crowding/report.md) | 阶段D ETF与大类资产 | **行业动量反转 + 股债金避险**：行业 ETF 截面动量显现反转（Rank IC −0.0385）；叠加 MA20 趋势与国债/黄金避险后，**最大回撤从 −53.18% 降至 −22.97%（回撤腰斩）**，双边 4 bps 超低摩擦 |
| 12 | [`all_weather_risk_parity`](strategies/all_weather_risk_parity/report.md) | 阶段E 全天候多资产配置 | **欧拉风险平价 (ERC)**：跨资产等风险贡献消除股票风险霸权，2015-2025 全周期 **SR=1.775、MDD 仅 −4.02% (降低 91%)、Bootstrap p=0.0000**，奠定极稳底层资产配置底座 |

**方法论铁律**（贯穿全部报告，详见 `CLAUDE.md`）：
未来函数（`.shift(1)` / PIT 公告日 / 复权方式）、幸存者偏差（PIT 成分股）、Overlapping Returns 平滑陷阱、
selection bias（WF 年度重选池）、摩擦成本（双边 ≈ 0.1%，买 0.026%/卖 0.076% 方向分离）、
数据可复现性（manifest + 随机种子）、向量化优先。

**工程铁律**（管系统本身是否可靠，详见 `CLAUDE.md` E1–E4）：
确定性内核用断言不用告警（`backtest/invariants.py`，失败 `raise`）、四个时点显式化
（信号可用 / 订单生效 / 成本发生 / 收益归属）、回测↔实盘语义一致（`docs/回测语义对照表.md`）、
AI 干预深度 ∝ 1 ÷ 错误代价。账本三对账（仓位 / 流水 / 盈亏闭合）由 `tests/test_reconciliation.py`
以**独立重算**方式守护。

## 工程质量（2026-09-09 / 2026-09-14 收口）

方法论纪律不止写在 `CLAUDE.md`，**已编译为可执行测试与单一真源**：

| 层 | 内容 |
|---|---|
| **成本单一真源** | `risk/cost_model.py`（BUY_COST/SELL_COST/ROUND_TRIP）；全仓库调用点统一 import，无硬编码残留 |
| **铁律守护测试** | `tests/test_cost_model.py` 用 `inspect` 读取**真实默认值/显式 COST 常量**断言=真源；变异验证过（改回 0.003 立即失败） |
| **指标规范口径** | `backtest/metrics.py`：年化统一为路径 CAGR；Sharpe 口径跨模块一致 |
| **输入契约断言** | `backtest/invariants.py`：价格面板 / 收益面板 / 权重矩阵 / 费率 / **结果合理性**，接入数据层（`data/fetcher.py`）、4 个回测入口与输出端，失败 `raise` 不 warn |
| **账本自洽对账** | `backtest/reconciliation.py`（单一真源）+ `tests/test_reconciliation*.py`：仓位 / 流水 / 盈亏闭合，**独立重算**比对三条组合实现；`run_reconciliation.py` 对**真实报告数据**（790 票）离线对账 |
| **时点契约** | `tests/test_time_point_contract.py`：手算场景同时钉住信号可用 / 订单生效 / 成本发生 / 收益归属四个时点 |
| **成交假设断言** | `tests/test_execution_assumptions.py`：涨跌停识别（主板/创业板/ST/北交所）+ **撮合层真拦单**；`test_price_adjust.py` 锁死三复权不共用文件 |
| **未实现参数守卫** | `tests/test_risk_constraints.py`：`max_turnover` 等未实现的约束必须 `raise`，不允许静默忽略；`max_leverage` 语义锁死为 gross（`Σ\|W\|`）|
| **交易可行性度量** | `risk/tradability.py` + `run_tradability_impact.py`：**跨 4 条链路**量化涨跌停约束的代价（换手几乎不变但夏普相对掉 3%~12%，**下界**；结论方向均未改变）|
| **回撤控制（opt-in）** | `risk/drawdown_control.py` 两个变体：两态滞后带 + **连续映射（自适应形态）**。`run_drawdown_control.py` 以**匹配仓位的前沿对比**证明连续在 4/4 档胜出；`run_drawdown_walk_forward.py` 进一步做 WF 复核：连续优于两态**样本外 8/11 成立**，夏普 1.48→2.15、回撤 −29.5%→−8.0%，**但 CAGR 腰斩（27.5%→13.4%）、敞口降至 1/3** → 是重配比而非免费保护。**未接入生产** |
| **PIT 自检** | `data/pit_auditor.py` + `run_pit_audit.py` 离线审计基本面缓存日期约定 |
| **向量化引擎** | `build_portfolio` 权重构造向量化（27×，等价性测试逐位守护） |
| **测试规模** | 257 项全绿：铁律 1/2/3 编译、契约断言、自洽对账、时点契约、成交假设、面板构建、约束守卫、回撤控制、正交化、IC 锚定方向、7 算子、组合等价、AM-GM |

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

## 快速开始

```bash
# 依赖（版本锁定见 requirements.txt：pandas/numpy/scipy/matplotlib/baostock/akshare/lightgbm/scikit-learn）
pip install -r requirements.txt

python bootstrap.py                    # 拉取沪深 300 全量日线（首启）
python -m unittest discover -s tests   # 257 项回归测试
cd strategies/ma_crossover
python report.py                       # 生成回测报告

python run_reconciliation.py           # 真实报告数据账本对账（790 票，约 10s）
python run_tradability_impact.py       # 交易约束代价量化（4 条链路，约 40s）
python run_drawdown_control.py         # 回撤控制参数网格 + 前沿对比（790 票，约 25s）
python run_drawdown_walk_forward.py    # 回撤控制 walk-forward 复核（790 票，约 30s）
```

> 数据已缓存在 `data/cache_*`（日线 + 基本面 + ETF + 指数成分），fetcher 负责增量更新。
> 复现纪律：数据变更后运行 `python data/manifest.py` 重建 manifest 快照并提交。

## 技术栈

Python 3.12 · baostock · akshare · pandas · numpy · scipy · matplotlib · scikit-learn · lightgbm

## 声明

本项目仅用于量化研究学习，不构成任何投资建议。
