# quant — 量化交易框架

个人量化研究与策略开发项目。目标：轻量、可审计、数学上严谨。

## 量化研究铁律

### 1. 绝对禁止未来函数 (Look-ahead Bias)

- **日线信号**：当日收盘信号 → 次日开盘执行。代码上用 `signal.shift(1)` 保证
- **财务数据**：必须用**实际发布日期 (Report Date / Announcement Date)** 对齐，禁止用报告期截止日 (End Date)。财报在 4 月才发布，Q4 数据不能出现在 1 月的交易信号里
- **成分股调整**：指数成分股变更以公告日为准，不以生效日偷跑

### 2. 消灭幸存者偏差 (Survivorship Bias)

- 回测 Universe 必须包含历史上被剔除/退市的股票，不能用"当前成分股"回看历史
- 每个交易日使用的成分股列表必须是 Point-in-Time (PIT) 的历史快照
- **PIT 基础设施已就位（2026-07）**：`data/index_membership.py` 提供沪深 300 历史成员矩阵（baostock 月末快照，790 只含退市股，缓存于 `data/cache_meta/hs300_membership.csv`）；特征矩阵用 `strategies/feature_selection/build_pit_matrix.py` 生成。新截面策略必须走这条链路
- **Universe 必须显式声明，禁止用缓存目录切片决定股票池**。曾因 `sorted(cache)[:300]` 在缓存扩容后漂移成 298 只纯深市股票，导致 models/ 全部结论建立在"事故 universe"上（LS 夏普 2.13 实为 0.08，见 `models/report.md`「方法论修正 II」）。股票池只能来自：显式名单文件 / 成员矩阵 / X_matrix 自身的股票集合

### 3. 摩擦成本 (Friction Costs)

- 每次换手扣除双边成本，底线标准：
  - 买入：0.026%（佣金万 2.5 + 过户费 0.001%）
  - 卖出：0.076%（佣金 + 印花税 0.05% + 过户费）
  - 合计双边 ≈ 0.1%。滑点另计，日线级别默认加 0.05% 滑点
- 当前 `backtest/engine.py` 和 `backtest/cross_section.py` 均已实现，新策略必须沿用默认费率参数
- **单一真源（2026-09-09 建立）**：`risk/cost_model.py` 定义 `BUY_COST` / `SELL_COST` / `ROUND_TRIP` /
  `CostModel`。**新回测代码必须 import 该模块，禁止再写字面量**。
  `tests/test_cost_model.py` 用 `inspect` 读取 8 处硬编码默认值并断言等于真源，任何漂移立即失败
- ⚠️ **历史教训**：`tests/test_methodology.py::TestFrictionCost` 是**空转测试**——它在函数体内
  自己声明 `buy_cost, sell_cost = 0.00026, 0.00076`，只验证了加法算术，从未 import 任何模块默认值。
  因此 `COST_BPS=0.003`（铁律 3 倍）能在测试全绿下存活并污染结论。**测试必须断言真实代码，不能断言自己写的常量**
- `risk/portfolio.py::build_weight_portfolio` 支持两种成本口径：`buy_cost`/`sell_cost`（方向分离，默认铁律标准）与 `cost`（双边合计，向后兼容 ETF 等低摩擦资产）。**优先用方向分离口径**，`cost=` 仅用于 ETF/低摩擦场景
- **收益锚定日约定（全仓库统一，2026-09 确立）**：`pct_change()` 把 t→t+1 收益记在 t+1 日；组合收益 = W[t-1] · daily_ret[t]（t-1 日持仓吃 t→t+1 收益）。任何新回测代码必须沿用此约定，禁止另起炉灶

### 3.5 复权方式选择 (Price Adjustment)

- 前复权（adjust="2"）：历史价随**最新除权事件**回溯调整，**只适合单票时序回测**，会污染截面因子与 ML 训练的时点可比性
- 后复权（adjust="1"）：历史价相对恒定，**截面比较与因子计算必须用后复权**
- 不复权（adjust="3"）：真实成交价，需配复权因子表才能算总回报
- `data/fetcher.py::_cache_path` 已按复权方式区分缓存文件（`{symbol}.csv` 前复权 / `{symbol}_adj{adjust}.csv` 其余），避免互相覆盖

### 3.6 数据可复现性 (Data Manifest)

- `data/manifest.py` 生成 `data/manifest.json`，记录缓存文件的行数/日期范围/抓取时间/sha256，作为数据层快照锚点（纳入 git）
- 数据增量更新后，运行 `python data/manifest.py` 重建清单并 commit，使「数据变了」可追溯
- 校验：`from data.manifest import verify_manifest; verify_manifest()` 返回 `drifted`/`missing`/`new` 列表

### 3.7 随机种子约定 (Reproducibility)

- 凡涉及随机性的代码（Bootstrap、合成数据、shuffle），必须显式传入 `seed`，禁止依赖全局 `np.random.seed()` 隐式状态
- 已固化种子：`risk/portfolio.py::bootstrap_sharpe_test` 默认 `seed=42`；`tests/factor_golden_baseline.py` 合成面板 `seed=42`

### 4. 因子合成：基线→正交化→非线性

- **基线**：等权合成（务必先跑，作为比较基准）
- **进阶**：ICIR 加权、正交化 (Gram-Schmidt / 回归取残差)
- **高阶**：LightGBM / XGBoost 非线性合成
- 等权基线不过关的因子组合，复杂合成大概率是过拟合

### 5. 防范过拟合

- 参数相图应呈现连续平滑区域，孤立的超高收益尖峰 = 过拟合信号
- 机器学习阶段：使用 Purged K-Fold 交叉验证（清除时序泄露），禁止依赖训练集 Sharpe 评估模型
- 策略收益率 Bootstrap 检验作为统计显著性底线

### 6. 警惕 Overlapping Returns 平滑陷阱

多日持有（hold_days > 1）的组合回测，**禁止**对"信号收益序列"做 `rolling(hold_days).mean()` 来构造每日组合收益——这是移动平均滤波，会把日波动人为压掉 √hold_days 倍，导致夏普虚高（曾在 `models/portfolio_backtest.py` 中把真实夏普 1.8~2.1 算成 8.75，虚高近 8 倍）。

- **正确构造**：基于权重追踪。每天生成目标持仓权重 `w[t]`，实际持仓 `W[t] = 过去 hold_days 天目标权重的平均`，组合收益 `= W[t-1] · daily_ret[t]`（所有 tranche 同一天经历同一市场波动，不做跨日平均）。成本按当日换手 `sum|ΔW|` 计。参考实现：`models/portfolio_backtest.py::build_portfolio`。
- **判断是否中招的三个信号**：① 夏普 / √hold_days 收敛到一个常数；② 组合日收益的 lag-1 自相关随 hold_days 上升趋近 1；③ Newey-West 调整夏普大幅低于报告夏普，但仍显著高于权重追踪法算出的真实值（说明不只是方差被低估，收益均值本身也被高估）。
- **案发与修复记录**：见 `models/report.md`「方法论修正」章节，完整踩坑过程可复用。

## 工程铁律（确定性内核）

> 铁律 1–6 管「研究结论是否可信」；这一节管「系统本身是否可靠」。
> 共同前提：**有一类系统不允许「差不多」——要么对，要么错，不存在第二种情况。**

### E1. 确定性内核用断言，不用告警

- 错误必须在**使用点**立刻中断（`raise`），不允许降级成 warn 日志——日志会被淹没，问题会静默通过。
  铁律 3 的 `COST_BPS=0.003` 正是在 109 项测试全绿的情况下静默存活，并把「多空 alpha 不成立」写进了三份报告。
- 实现：`backtest/invariants.py`（`assert_price_panel` / `assert_returns_panel` /
  `assert_weight_matrix` / `assert_cost_params` / `assert_backtest_inputs`），已接入四个回测入口：
  `backtest/engine.py::run`、`risk/portfolio.py::build_weight_portfolio`、
  `risk/orchestrator.py::run`、`models/portfolio_backtest.py::build_portfolio`。
- **不设全局关闭开关**：任何 `strict=False` 都会退化成第二个 warn 日志。个别场景确需容错时，
  由调用点显式 `try/except` 并记录，责任留在调用点。
- 用**显式 `raise`**，不用 `assert` 语句：`python -O` 会剥离 `assert` 语句，而契约检查不允许被优化掉。
- 只断言**必然成立**的性质（负价格、`r < -1`、索引乱序、权重 NaN……），不越界到「通常成立」的
  判断（如 `close > 0`），以免在真实数据上误报。
- 该模块**零项目内依赖**（只用 numpy/pandas）。若反向 import `risk.*` 会形成循环导入——
  `risk/__init__.py` 在最外层就 import 了 `risk.portfolio`。

### E2. 时点必须显式化，禁止用一个时间戳糊过去

一笔业务涉及**四个不同的时点**，写代码时必须说清是哪一个：

| 时点 | 含义 | 本仓库对应 |
|---|---|---|
| 信号可用时点 | 该时点才知道的信息才能进信号 | 铁律 1（`.shift(1)` / PIT 公告日） |
| 订单生效时点 | 信号何时变成持仓 | 铁律 1（次日开盘执行） |
| 成本发生时点 | 换手在何时被计费 | 铁律 3（当日换手 `Σ\|ΔW\|`） |
| 收益归属时点 | 收益记在哪一天 | 铁律 3 末（`pct_change()` → t→t+1 收益记 t+1） |

「确认」与「到账」是两个时点：**已确认的订单 T+1 才到账，当天不算**；而转账类操作当天生效。
回测里所有「信号 → 持仓 → 成本 → 收益」的对齐，都必须能指出落在上表哪一格。

### E3. 回测 ↔ 实盘语义一致是第一层，参数敏感性是第二层

- 四类语义（数据口径 / 成交假设 / 费用口径 / 时序假设）的逐项对照表见
  `docs/回测语义对照表.md`，每项锚定到具体代码位置，使口径漂移一眼可见。
- **推论**：语义错位时调参是在拟合一个错误的世界——**调得越细，偏离越远**。
  拿到不理想的结果，第一件事是核对语义，不是调参、换窗口、加过滤条件。
- **跨实现必须同账**：`tests/test_reconciliation.py` 对三条组合实现
  （`orchestrator.run` / `build_weight_portfolio` / `build_portfolio`）做独立重算对账并逐位比对。
  已发现并**钉住**的真实口径差异见该文件 `TestKnownDivergence`（`min_stocks` 阈值 ×2 vs ×3）。
- 观测不到的部分若恰好是结论所依赖的，仿真再漂亮也没有验证意义。本仓库红线「不做高频/tick 级」，
  因此**不做** L2/L3 订单簿仿真；但保留该判据作为审计问题。

### E4. AI 干预深度 ∝ 1 / 错误代价

- 错误代价越低的地方越可放手（重复性工作、探索脚本、样例代码）；**代价越高的地方越需要人逐行确认**
  （框架设计、交易关键路径、口径与费率）。
- 与既有「计划先行与用户审批纪律」一致：架构重构 / 模块抽取 / 跨文件改动，先出 Implementation Plan。

### 回测自洽性对账（账本一致性）

组合收益是一本账，必须能自证闭合。三个对账项由 `tests/test_reconciliation.py` 守护：

| 对账项 | 断言 |
|---|---|
| 仓位对账 | 持仓满足杠杆 / 单资产上限；列集合与价格矩阵一致 |
| 流水对账 | `turnover == Σ\|ΔW\|`；`cost == buy_turnover·BUY_COST + sell_turnover·SELL_COST` |
| 盈亏核对 | `port_ret == gross_ret - cost`（逐日闭合）；`cum == Π(1+port_ret)` |

⚠️ **对账方式必须是独立重算**（从 `W_held` 重新推一遍损益），不能用实现自己的中间量验证实现——
那是空转，与「测试必须断言真实代码，不能断言自己写的常量」是同一个教训。

## 目录结构

```
quant/
├── data/              # 数据获取 + 本地缓存（.gitignore）
│   └── cache/         # CSV 缓存，不入 git
├── signals/           # 因子库（alpha191 等）
├── backtest/          # 回测引擎（单票 + 截面）
├── strategies/        # 策略实例（每个子目录 = 一项独立策略研究）
│   ├── ma_crossover/           # 单票时序策略
│   ├── alpha001_trial/         # 单因子截面策略
│   └── multi_factor_trial/     # 多因子合成实验
│       └── alpha012_alpha055_alpha191/  # 以因子名命名的具体实验
├── risk/              # 仓位管理 / 风控约束
├── models/            # ML 模型（LightGBM 等，预留）
├── viz/               # 可视化：权益曲线、回撤图、参数相图
├── notebooks/         # 探索性分析 .ipynb
├── tests/             # 关键路径测试（防未来函数等边界用例）
└── bootstrap.py       # 首次数据拉取脚本（一次性）
```

## 约定

- **向量化优先**：回测和信号计算用 pandas/numpy 矩阵运算，禁止对逐行循环
- **数据永不过期**：fetcher 负责增量更新，不重复拉取已有数据
- **单文件策略**：一个策略 = 一个子目录，包含 `report.py`（生成指标+图表）+ `report.md`（分析报告）+ `figures/`
- **参数扫描**：策略参数用 list/dict 传入，回测引擎负责网格扫描
- **坐标轴标签中英混合**：title 用中文概括结论，axis label 用英文变量名
- **模型假设写在文件顶部注释**，不是分散在函数 docstring 里
- **核心函数加 Type Hints**：回测引擎、因子计算等公开 API 用类型标注；探索性脚本不强制
- **计划先行与用户审批纪律**：任何涉及架构重构、模块抽取、新策略开发或跨文件改动的任务，编写/更新 Implementation Plan 后必须先呈送用户审核确认，严禁未获批准直接执行代码变更


## 数据质量检查清单

每项新策略启动前，确认以下三项：

1. **时序对齐**：收盘价和因子值是否在同一日期索引？`signal.shift(1)` 是否正确应用？
2. **Universe 确认**：当前使用的股票列表是历史 PIT 成分股还是最新成分股？如果是后者，报告中标注"含幸存者偏差"
3. **停牌/涨跌停**：日线级别是否对停牌日的收益率做了处理？（默认填 0，报告中注明）

## 数值验证标准

- 回测结果必须可复现：给定相同数据和参数，两次运行输出完全一致
- 策略收益与 buy-and-hold 基准对比，夏普比率需在合理范围（**|SR| < 3 否则依次排查**：① 未来函数 ② 幸存者偏差 ③ **Overlapping Returns 平滑陷阱**（多日持有组合是否误用 `rolling(hold_days).mean()`，见铁律第 6 条））
- 参数相图应呈现连续平滑区域，孤立的超高收益尖峰 = 过拟合信号
- 多空收益必须扣除双边交易成本后再比较

## 新对话快速上手

数据已缓存在 `data/cache/`（300 只 CSI 300，2010–2025 日线），无需重新拉取。

```bash
cd D:/桌面文件/quant

# 验证环境
python -c "from data.fetcher import load_daily; d=load_daily('000001'); print(len(d))"

# 跑已有策略报告
cd strategies/ma_crossover && python report.py

# 新增单因子策略：在 strategies/ 下建新文件夹
mkdir strategies/my_factor
# 1. 参考 alpha001_trial/report.py 改策略逻辑
# 2. 跑 python report.py 生成报告和图表

# 新增多因子合成实验：在 strategies/multi_factor_trial/ 下以因子名建子目录
mkdir -p strategies/multi_factor_trial/alphaXXX_alphaYYY
# 1. 参考 alpha012_alpha055_alpha191/report.py
# 2. 修改 FACTOR_IDS 列表为你的因子
# 3. 跑 python report.py
```

## 标准分析流程

每项新策略依次执行：
1. **trial_run** — 单票单参数跑通端到端
2. **scan** — 参数相图网格扫描，画夏普热力图
3. **walk_forward** — 滚动窗口优化，检验参数稳定性
4. **out_of_sample** — 训练集选参 → 测试集验证
5. **capm** — OLS 回归分解 α/β
6. **bootstrap_mc** — 收益率重采样检验统计显著性
7. **cross_section** — 全市场截面检验 + Bonferroni 校正
8. **survivorship** — 幸存者偏差讨论
9. **tradability** — 涨跌停/停牌约束下的代价核验：`build_trade_limits` 造掩码 + `measure_restriction_impact` 出对照。**新策略默认执行**，报告中须给出约束前后的换手/年化/夏普/回撤对照（实测量级：夏普约 −7%、年化约 −1.6pp；⚠️ 只覆盖一字板，属**下界**）

每步都有对应的代码模式在 `strategies/ma_crossover/` 下可参考。第 9 步的独立运行器见 `run_tradability_impact.py`。

## 工程开发流程

上一节「标准分析流程」管**策略研究**；这一节管**框架演进**。两者是一条流程的两端。

| 步骤 | 内容 |
|---|---|
| ① 理解需求 | 先明确要解决的工程问题、性能目标与约束边界——不是先写代码 |
| ② 系统设计 | 抽象、拆模块、定接口与分层；权衡正确性 / 性能 / 扩展性。**抽象质量决定后续每一步的重构成本** |
| ③ 性能调优 | 测量 → 找瓶颈 → 定位根因 → 优化热路径 → 验证结果。**先测量，再优化**，禁止凭直觉 |
| ④ 上线观察 | 观察实际运行行为，与使用方对齐，持续完善 |

已实证的两次性能结论，避免重复劳动：`build_portfolio` 权重构造向量化 **27×**
（`tests/test_weight_vectorization.py` 逐位守护）；`risk/orchestrator.py::run` 保留逐日循环
**属合理**（时序依赖约束，非缺陷）。

### 动手前的三层理解

设计任何改动前，先确认自己看懂了这三层——**只懂自己负责的那一层，解决方案也会设计得不对**：

| 层次 | 内容 | 为什么重要 |
|---|---|---|
| **理解系统** | 数据怎么流动、瓶颈在哪、这段代码跑在哪条链路的哪个位置、上下游是谁 | 基本功 |
| **理解策略** | 策略需要什么数据、产出什么信号、哪些环节对时点敏感 | 量化的特有要求 |
| **理解业务** | 交易链路如何运作、有哪些约束、外部市场规则 | 规则不只是纸面的，**它会对结果产生实质影响** |

⚠️ 很多时候**问题的根因不在你负责的那一层**，而在上游的数据约束或下游的业务限制
（例：`min_stocks` 阈值差异、PIT 公告日的可得性——都不是回测引擎本身的问题）。

## 研究报告写作与提交流程

每项新策略的报告生成遵循以下标准流程。报告由 Claude Code 辅助生成，用户负责最终审核。

### 写作标准

对标 `strategies/ma_crossover/report.md` 的质量和完整度：

- **结论先行**：头部用 blockquote 给出核心结论，再展开分析
- **假设检验形式化**：$H_0$ / $H_1$ 表格
- **图表嵌入**：每张 png 用 `![描述](figures/XX_name.png)` 嵌入，路径相对 md 文件
- **图表数量底线**：每份 report.md 必须嵌入不少于 2 张图表；一张 PNG 含多个子图时按子图数量计算（如 2×2 子图 = 4 张）。报告产出后必须先用 `grep '!\[.*\](.*)' report.md` 检查图片已嵌入，未达标不得提交
- **公式 LaTeX**：`$$...$$` 块级公式，`$...$` 内联
- **检验框架表格**：每个检验维度 → 方法 → 控制的风险
- **数据概况表格**：来源、成分股、时间范围、样本量
- **最终判决总表**：检验汇总 → 判决，用 Unicode box-drawing 字符
- **附录**：代码结构 + 复现命令 + 依赖

### 报告执行流程

```bash
cd D:/桌面文件/quant/strategies/新策略名
python report.py    # 跑完整分析 → print 全部指标 + 保存图表到 figures/
```

`report.py` 职责：
1. 计算所有数值指标（IC、SR、回撤、p-value 等）
2. 生成独立 PNG 图表到 `figures/`，每张图有编号标题
3. print 全部关键数值（供 Claude Code 写入 report.md 时引用）

`report.md` 由 Claude Code 基于 report.py 的输出数值撰写，用户审核后提交。

### 提交流程

1. 确认 `report.md` + `report.py` + `figures/` 齐全
2. **Claude Code 负责 stage + commit**：
   ```bash
   git add 策略目录/ && git commit -m "描述性 message"
   ```
3. **用户手动在 VS Code 点 Sync / Push** 到 `origin/main`

### Alpha 191 因子策略报告（截面型）

截面因子策略与单票时序策略的分析维度有差异：

| 单票时序（MA crossover） | 多票截面（Alpha 191） |
|--------------------------|------------------------|
| 参数相图 (FAST × SLOW) | 因子 IC 分析（Rank IC + IC_IR） |
| 滚动窗口 Walk-Forward | 分层回测（5 分组等权） |
| CAPM 回归 α/β | Fama-MacBeth 截面回归 λ |
| 单票 Bootstrap MC | 策略收益率 Bootstrap |
| 单票全市场截面 | 单票 IC × Bonferroni |

截面报告的 `report.py` 参考 `strategies/alpha001_trial/report.py`，
时序报告的 `report.py` 参考 `strategies/ma_crossover/report.py`。

## 红线

- 不接实盘（该项目现阶段仅用于研究和模拟）
- 不做高频/tick 级数据（日线/周线为主）
- 策略逻辑不含未来函数——代码审查第一关就是 `.shift(1)` 和 PIT 对齐
- 回测不含幸存者偏差——至少要在报告中标注此风险
- 禁止为了拟合结果而调整回测参数——如果基线不 work，诚实报告
- 外部点评/审阅意见（含其他 AI 给出的"发现的问题"）必须先核实技术前提是否匹配实际代码——grep 关键字、读源码确认claim 成立，再决定要不要改数字或结论。曾出现点评声称"数据是周频、sqrt(252)用错"，实际全仓库查无此代码，前提完全不成立的情况（见 `strategies/factor_discovery/report.md` 第 7 节）

## 待论证议题（TODO）

### ✅ 已收口（2026-09-05，commits `63a2b27`/`22ecbad`/`cbc7e4e`）

1. ~~**`build_portfolio` 的错位收益**~~ ✅ **已解决**。
   定性：非未来函数，而是「信号延迟错配」（动量类 alpha 被低估约 10%，方向保守）。
   已统一为 `pct_change()`（t→t+1 收益记 t+1 日），论证文档见 `docs/收益成本口径统一论证.md`。
2. ~~**成本口径三套并存的收口**~~ ✅ **已解决**。
   `COST_BPS = 0.003`（铁律 3 倍）已收口为 0.00102，成本改为买/卖方向分离。
   真实数据对照（790 只 × 3886 日）：LS 夏普 0.036→**1.175**、LO 1.187→**1.613**；
   中证 500 去 bias LS −0.188→**+1.214**、LO 0.248→**0.600**。
   三份报告（models / zz500_pit_trial / zz500_fundamental_trial）结论已更新。
   **遗留**：`models/nn_trainer.py:236,258` 仍用 `cost=0.003`，但该脚本属已标注作废的
   旧 universe 遗留物（`models/report.md:754,862`），不污染结论。

### ⏳ 待实施（跨文件改动，需先呈送 Implementation Plan 审批）

3. ~~**年化口径三套不一致**~~ ✅ **已收口（2026-09-09）**：
   `models/portfolio_backtest.py::performance_metrics`（原 `(1+mean)**252-1`）与
   `risk/portfolio.py::calculate_metrics`（原 `daily_mean*252` 算术式）已统一为**路径 CAGR**，
   规范函数见 `backtest/metrics.py::annualize_cagr`。`engine.py` 本就是 CAGR，未动。
   Sharpe 口径未变（mean/std*sqrt(252)），仅「年化收益」展示列变化，不影响任何结论判断。
   算术年化保留在 `annual_arith` / `annual_return_arith` 键供 AM-GM 对照。
4. **成本参数仍有 8 处硬编码**：真源已建（`risk/cost_model.py`）+ 守护测试已就位
   （`tests/test_cost_model.py`），但 8 处调用点尚未改为 import 真源。
5. ~~**核心回测逐日 Python 循环**~~ ✅ **已收口（2026-09-09）**：
   `models/portfolio_backtest.py::build_portfolio` 与 `risk/portfolio.py::build_weight_portfolio`
   的 W_target 构造已向量化（27×，3886 日 × 1326 股 ~0.3s）。等价性由
   `tests/test_weight_vectorization.py`（9 项，冻结原始循环为参考）逐位守护。
   注意：`risk/portfolio.py` 的 trade_limits 分支是时序依赖约束，保留逐日循环属合理。
   `models/portfolio_backtest.py::build_portfolio_naive`（错误示范）保留原循环以作对照。
4. ~~**成本参数 8 处硬编码**~~ ✅ **已收口（2026-09-09）**：
   全部调用点已迁移到 `risk.cost_model` 单一真源（engine/cross_section/portfolio/
   orchestrator 默认值、evaluate 模块常量、portfolio_backtest 模块常量+签名）。
   ⚠️ **收口 63a2b27 曾遗漏「显式传 cost=」的隐藏调用点**（walk_forward /
   methodology_correction / execution_optimization 的 `COST = 0.003`）——
   本次已修正并加 `test_explicit_roundtrip_call_sites` 守护。教训：
   **默认值修正不覆盖显式传参，守护测试必须同时检查显式调用点。**
6. **可观测性（范围限定，延后）**：113 个 py 文件仅 1 个用 `logging`，全仓 1178 处 `print()`。
   决策：不做全量迁移；如需，仅给长耗时入口（`strategies/*/report.py`、`walk_forward.py`、
   `lgbm_trainer.py`、`data/fetch_*.py`）加 logging。
7. **昨日新增模块（部分已处理）**：
   - ✅ `ic_evaluator` / `alpha191_ext` 已补测试（`test_ic_evaluator.py` 11 项 +
     `test_alpha191_ext.py` 19 项）。二者设计为 SOP Agent 运行时调用，零引用属预期。
   - ✅ `pit_auditor` 已审计 `zz500_fundamental_trial` 链路（2026-09-09）并修复 4 个缓存
     的尺度污染（gpMargin ×100、roeAvg/npMargin/epsTTM 删异源行）。审计结论：方向 2
     「全否定」对该 bug 稳健（清理后 IC 未变强），无需全链重跑。详见报告「数据质量审计」。
8. **`alpha191_ext` 算子性能（推迟）**：`rolling().apply()` 是 Python 逐窗口回调，
   实测 1000×300 需 2.5s（ts_argmax），真实规模约 13s/算子。
   当前零引用，优化未使用的代码是过早优化——推迟到 Agent 实际调用时再做（已有测试兜底）。
9. ~~**`CLAUDE_AGENT_SOP.md` 阈值灰色地带 + 失效引用**~~ ✅ **已收口（2026-09-09）**：
   补 `[1.0, 1.5]` 观察档（默认按观察处理）；阈值注明为经验值；
   `auto_mined_alpha.py` 改为「不存在则创建」；补充 rolling.apply 同样禁止的说明。

### ✅ 已收口（2026-09-14，确定性内核加固）

10. ~~**确定性内核缺事前断言 + 回测结果无自洽对账**~~ ✅ **已解决**。
    - 新建 `backtest/invariants.py`（零项目内依赖），接入四个回测入口，失败 `raise` 不 warn。
    - 新建 `tests/test_invariants.py`（30 项）+ `tests/test_reconciliation.py`（15 项），
      全仓库回归从 109 → **154 项全绿**。
    - 新增 `docs/回测语义对照表.md` + 本文档「工程铁律 E1–E4」。
    - 对账发现并**钉住**一个真实口径差异：`min_stocks` 阈值
      `risk/portfolio.py` 用 `base*2`、`models/portfolio_backtest.py` 用 `base*3`（默认 10 vs 15）。
      同一输入下两条实现逐位一致（除该阈值外），差异被 `TestKnownDivergence` 钉住。
      **未修改**——统一阈值会改变已发布报告的持仓日集合，须先评估影响再决策。
11. **`risk/orchestrator.py` docstring 与实现不一致** ✅ **已收口（2026-09-15）**：
    `max_leverage` 的 docstring 原写「`|sum(W)|` 的绝对值上限」（净敞口），实现约束的是 `Σ|W|`（总杠杆）。
    已改为 gross leverage 表述，并补注「LS 组合须显式传 `max_leverage=2.0`，否则被静默缩放到半仓」。

### ✅ 已收口（2026-09-15，残余五项）

12. ~~**断言只覆盖回测入口**~~ ✅ 已下移到**数据层与结果层**：
    `data/fetcher.py::_assert_daily_frame`（必需列 + close 价格面板）接入 `load_daily`/`download_daily`
    全部返回点；新增 `assert_result_sane`（只断言必然错误；**不含 `|SR|>3`**——那是启发式，不是不变量）
    接入三条实现的输出端。已实测真实面板不误报（NaN 占比 50%、0 负价、0 全 NaN 列）。
13. ~~**对账只在合成数据上跑过**~~ ✅ 新增 `run_reconciliation.py`（真实 OOF + 真实价格，全量 790 票）
    + `tests/test_reconciliation_real_data.py`（50 票采样，秒级随套件运行）。
    真实数据对账**全部通过**，两条实现逐位一致。
14. ~~**成交假设 / 复权仍是人读清单**~~ ✅ 新增 `tests/test_execution_assumptions.py`
    （跌停 / ST / 北交所识别 + **撮合层真拦单**）与 `tests/test_price_adjust.py`
    （三复权不共用文件 + 读写同映射）。
    **顺带修复**：`build_weight_portfolio` 的 `trade_limits` 日循环原从 `i=1` 起，
    使**首日持仓绕过涨跌停**；已修。该缺口仅影响被丢弃的建仓爬坡期，对已发布结果零影响。
15. ~~**四时点无专门断言**~~ ✅ 新增 `tests/test_time_point_contract.py`：
    手算场景（价格 `[10,10,20,20]` + 信号 `[1,1,0,0]`）同时钉住四个时点，
    并对 `engine.run` 与面板参考账本**双双**验证锚定方向一致。
16. ~~**`min_stocks` 硬编码差异**~~ ✅ 已参数化为 `min_stocks_mult`（默认 2 / 3，**行为逐位不变**），
    并在真实数据上量化影响：**全量 790 票下差异为 0**（持仓日 3238/3238、夏普 1.175/1.175，
    两条实现逐位一致）；仅稀疏截面（<15 只/日）显著（抽样 60 票差 43.96% 持仓日）。
    **决策：不统一**——对已发布结论零影响，参数留作稀疏截面的显式开关。
17. **`models/` 依赖被 gitignore 的生成物（记录，不修）**：`load_data()` / `walk_forward.py` /
    `nn_trainer.py` / `train_final_model.py` / `linear_baseline.py` 均需
    `strategies/feature_selection/X_matrix.csv` —— `build_pit_matrix.py` 的生成物，被
    `.gitignore`「特征矩阵」段有意排除。已查清脉络（`git check-ignore` 确认，非误删）：
    **不是缺陷，是「大文件不入库」的设计取舍**。`load_data()` 报错信息已指明生成脚本与
    既存规避方案 `models/rerun_portfolio_backtest.py`。对账链路不依赖它。

### ⏳ 待定夺（2026-09-15 风控/清算覆盖核对中发现）

18. ~~**`orchestrator.max_turnover` 是死参数（缺陷）**~~ ✅ **已收口（2026-09-15）**：
    改为**守卫式失败** —— 传入非 None 立即 `raise NotImplementedError`，属性恒为 `None`，
    docstring 示例已移除该参数并注明未实现。理由：按 E1，不确定的行为必须响亮地失败，
    而不是静默放行（这正是铁律 3 成本 bug 的失效模式）。守护测试
    `tests/test_risk_constraints.py::TestUnimplementedParamsFailLoudly`。
19. **交易可行性风控三件套零生产调用** —— ⚠️ **接线路径已就绪，是否接入仍待你定夺**：
    全仓 grep 确认 `PortfolioOrchestrator` / `detect_limit_moves` / `apply_volatility_target`
    仅出现在 `risk/` 自身与 `tests/`，**从未约束过任何已发布结论**。
    （对比：`gate` 与 `position_scale` **确在生产** —— `etf_momentum_crowding` 的 MA20 避险、
    `zz500_pit_trial/bear_short.py` 熊市门、`zz500_fundamental_trial/backtest_monthly.py` 月度门。）

    实测量（`run_tradability_impact.py`，真实 790 票 / 2010-2025 / LS hold_days=5）：

    | 指标 | 无限制 | 加一字涨跌停约束 | 变化 |
    |---|---|---|---|
    | 换手合计 | 2188.80 | 2188.14 | −0.66（**−0.03%**）|
    | 成交日数 | 3238 | 3238 | 0 |
    | 累计收益 | 1376.13% | 1102.74% | −273.39 pp |
    | **年化 CAGR** | **19.10%** | **17.53%** | **−1.57 pp** |
    | **夏普** | **1.1753** | **1.0924** | **−0.083（约 −7%）** |
    | 最大回撤 | −37.17% | −38.79% | −1.62 pp |

    一字板发生率极低：一字涨停 0.1100%、一字跌停 0.0375%（占有效格数）。
    **判读**：约束代价**温和但不为零**——夏普 −7%、年化 −1.57pp，回撤略差。
    机理是「被拦住的恰是涨停追涨类交易」，对动量型 alpha 略有伤。

    **跨链路影响评估（2026-09-15，`run_tradability_impact.py`）**：
    各链路用**自己的 OOF 预测**复现 LS 组合，其余参数不变，只切换 `trade_limits`。
    ⚠️ 复现**未含**各报告的 PIT 成员掩码与 selection-bias 剔除 → 绝对值与报告**不可比**，
    **只看差值**（相对幅度比绝对值更可移植）。

    | 链路 | hold | 股票 | 一字涨停% | 换手Δ | Δ夏普 | **夏普相对Δ** | Δ回撤 |
    |---|---|---|---|---|---|---|---|
    | models（ML 合成） | 5 | 790 | 0.1100 | −0.03% | −0.083 | **−7.05%** | −1.62pp |
    | zz500_pit_trial（量价） | 5 | 1624 | 0.1329 | −0.07% | −0.159 | **−5.30%** | −1.16pp |
    | zz500 WF PIT-select | 5 | 1326 | 0.1466 | −0.05% | −0.150 | **−12.38%** | −0.74pp |
    | zz500_fundamental（月度） | 1 | 1624 | 0.1329 | −0.00% | −0.013 | **−2.76%** | −0.02pp |

    **判读**：
    - **换手几乎不动（≤0.07%）、夏普却掉 3%~12%** —— 再次印证「被拦住的恰是关键交易」
      （涨停追涨类），而不是量的问题；
    - 基本面（月度、hold=1）受影响最小 —— 月调仓撞上一字板的概率极低；
    - **没有任何一条链路的结论方向被改变** → 接线属「补齐现实性」，不是「修正错误」。

    **已就绪的接线路径（2026-09-15）**：
    - `data/fetcher.py::load_field_panel` 统一了「从 cache 拼多标的面板」（此前散在各 runner 里重复）；
    - `risk/tradability.py::build_trade_limits` 补上了此前无人推导的**前收盘价**环节；
    - 「标准分析流程」已新增第 9 步 **tradability**，新策略默认核验约束代价。
    **决策建议**：**不回改已发布报告**（代价温和，回改成本远超收益）；新策略默认开启。
    ⚠️ 上表是**下界** —— `detect_limit_moves` 只识别**一字板**（开盘即触限价且 high==low），
    盘中封板无法成交的情形未建模，真实代价更大。
20. **回撤控制** —— ⚙️ **组件已建 + 参数网格已出，接入待你定夺**：
    全仓 grep `stop_loss|止损|max_drawdown_limit|trailing_stop` 原为零命中。
    新增 `risk/drawdown_control.py::apply_drawdown_control`（滞后带状态机、**闭环** ——
    回撤在受控路径上计算，与实盘观察一致；决策只用截至前一日的回撤，无未来函数）
    + `run_drawdown_control.py` 参数网格 + `tests/test_drawdown_control.py`（13 项）。
    **不接入生产**（与 `gate` / `position_scale` 同属 opt-in）。

    实测（真实 790 票 / LS / 2010-2025，`recovery = threshold/2`）：

    | 配置 | 夏普 | CAGR | 最大回撤 | 平均仓位 |
    |---|---|---|---|---|
    | baseline（无控制） | 1.1753 | 19.10% | −37.17% | 1.000 |
    | thr=0.03 cut=0.3 | **1.4491** | 10.52% | **−14.66%** | 0.567 |
    | thr=0.05 cut=0.5 | 1.3058 | 14.06% | −22.44% | 0.759 |
    | thr=0.15 cut=0.4 | 0.9504 | 10.93% | −24.64% | 0.796 |

    **判读（关键，勿误读）**：
    - 夏普对**常数**杠杆不变 → 夏普提升（+0.27）来自**择时减仓**，可与基线直接比较；
    - 但 **CAGR 不可直接比**：受控组合平均仓位更低（0.567），收益下降含降杠杆成本；
    - 本网格单调：**减仓越深 → 夏普越高、回撤越浅、绝对收益越低**。
      这不是「最优解在角落」，而是**风险偏好问题**，继续外推只会更极端。
    - ⏳ 因此**接入与否、用哪组参数，是风险偏好决策**，不由数据单独决定。

    **「能否自适应」的结论（2026-09-15）**：模块现提供两个变体 ——
    `apply_drawdown_control`（**两态开关 + 滞后带**）与
    `apply_drawdown_scaling`（**连续线性映射** `scale = clip(1 + dd_prev/max_cut_at, floor, 1)`）。

    在**匹配平均仓位**下做前沿对比（4 个仓位档）：

    | 目标平均仓位 | 两态夏普 | 连续夏普 | 差 |
    |---|---|---|---|
    | 0.55 | 1.4491 | **1.7371** | +0.2880 |
    | 0.65 | 1.4152 | **1.5495** | +0.1343 |
    | 0.75 | 1.0536 | **1.3872** | +0.3336 |
    | 0.85 | 1.2475 | **1.3210** | +0.0734 |

    → **连续在 4/4 档全胜，平均 +0.21**。机理：两态开关在浅回撤时仍满仓
    （dd 从 0 掉到 −threshold 之间不动作），连续映射「早减、缓减」，能吃到**回撤初段** ——
    而回撤初段的减仓最有效。**响应形态本身带信息量**，不只是同一条前沿上的另一个点。

    ⚠️ 两点保留：
    1. 两族参数都在**同一段样本内**挑的，夏普含样本内选择效应；据此定稿须做
       walk-forward / out-of-sample 复核（铁律 5）。
    2. 「连续」只说明**响应**连续，**参数本身仍是固定的**。让阈值随波动率自适应
       （`threshold_t = k·σ_t`）会再引入 `k` 与 lookback 两个参数，样本内更易拟合出好看曲线。

    **walk-forward 复核（2026-09-15，`run_drawdown_walk_forward.py`）**：
    测试年 2015–2025（11 窗口），训练期 = 扩张窗口，各家族按训练期夏普选参、在测试期评估；
    每个候选参数只对全序列跑一次再按窗口切片（**状态自然延续**，避免「每年重置回撤」的失真）。
    连续族网格刻意下探到 `mca=0.01 / floor=0.2`。

    | 口径 | 无控制 | 两态 | 连续 |
    |---|---|---|---|
    | 测试期逐窗口平均夏普 | 1.5756 | 1.4788 | **1.9623** |
    | 平均仓位 | 1.000 | 0.491 | **0.342** |
    | 拼接测试期 CAGR | **27.54%** | 13.39% | 13.44% |
    | 拼接测试期夏普 | 1.4791 | 1.5769 | **2.1477** |
    | 拼接测试期最大回撤 | −29.50% | −14.04% | **−8.01%** |

    胜出窗口数：连续 > 两态 **8/11**；连续 > 无控制 **8/11**；两态 > 无控制 6/11。

    **两个结论**：
    1. ✅ **「连续优于两态」样本外成立（8/11）** —— 不是样本内选择效应，是真实的结构差异。
    2. ✅ **夏普与回撤的改善在样本外也是真的**（1.48 → 2.15；−29.5% → −8.0%），
       **但代价是 CAGR 从 27.54% 掉到 13.44%** —— 收益减半。这不是「免费的保护」，
       而是**一次风险-收益重配比**。

    **⚠️ 据此把上一轮的建议下调**：不要在报告里把它当**默认叠加层**。因为
    - 连续族的最优**每个窗口都贴下界**（`max_cut_at` 已下探到 0.01 仍贴边）、`floor` 也贴边
      → 该族「想要更极端」，网格没包住，外推前须补更大范围检验；
    - 两态 `threshold` 在 3 个取值间跳动 → **选择噪声大**，不支持「每期重选参数」；
    - 两族在 WF 口径下**敞口不可比**（本复核问的是「用同一套选参准则谁更好」，
      不是「同敞口下谁更好」）。
    **要用就当作独立的风险偏好选择**：固定一组参数、明确接受低敞口。
21. ~~**风控拦截缺少代价度量**~~ ✅ **已收口（2026-09-15）**：
    新增 `risk/tradability.py`（`build_trade_limits` 补上前收盘价推导 +
    `measure_restriction_impact` 两条账本对比）与 `run_tradability_impact.py`；
    守护测试 `tests/test_tradability.py`。**这是第 19 条决策的输入。**
