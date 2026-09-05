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

每步都有对应的代码模式在 `strategies/ma_crossover/` 下可参考。

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

> 以下议题已识别但**暂未改动代码**，需单独论证后才实施（涉及会改变历史报告结论的改动）：

1. **`models/portfolio_backtest.py::build_portfolio` 的错位收益**：
   该函数（被几乎所有截面/ML 策略脚本调用）使用 `daily_ret = close.shift(-2)/close.shift(-1)-1`，
   即「t+1→t+2 收益记在 t 日」，配合 `W_lag = W_held.shift(1)` 导致时序错配。
   需单独论证：这是作者有意的「信号日收盘→次日开盘买→隔日卖」单日持有近似，还是真实未来函数？
   论证文档见 `docs/收益成本口径统一论证.md`。在结论明确前，**不得**擅自修改该函数及重跑受影响报告。
2. **成本口径三套并存的收口**：`engine.py`（买/卖分离）、`cross_section.py`（买/卖分离）、
   `models/portfolio_backtest.py`（cost/2 对半）三处口径不统一，需随议题 1 一并收口。
