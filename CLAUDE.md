# quant — 量化交易框架

个人量化研究与策略开发项目。目标：轻量、可审计、数学上严谨。

## 目录结构

```
quant/
├── data/
│   ├── fetcher.py     # 数据获取（akshare/yfinance 封装）
│   └── cache/         # 本地 CSV 缓存，.gitignore
├── signals/           # 策略信号生成，每个策略一个文件
├── backtest/          # 回测引擎 + 绩效指标
├── risk/              # 仓位管理 / 风控约束
├── viz/               # 可视化：权益曲线、回撤图、参数相图
└── notebooks/         # 探索性分析 .ipynb
```

## 约定

- **向量化优先**：回测和信号计算用 pandas/numpy 矩阵运算，禁止对逐行循环
- **数据永不过期**：fetcher 负责增量更新，不重复拉取已有数据
- **单文件策略**：一个策略 = 一个 .py 文件，只暴露 `generate_signals(df) -> pd.Series` 接口
- **参数扫描**：策略参数用 list/dict 传入，回测引擎负责网格扫描
- **坐标轴标签中英混合**：title 用中文概括结论，axis label 用英文变量名
- **模型假设写在文件顶部注释**，不是分散在函数 docstring 里

## 数值验证标准

- 回测结果必须可复现：给定相同数据和参数，两次运行输出完全一致
- 策略收益与 buy-and-hold 基准对比，夏普比率需在合理范围（|SR| < 3 否则检查未来函数/幸存者偏差）
- 参数相图应呈现连续平滑区域，孤立的超高收益尖峰 = 过拟合信号

## 新对话快速上手

数据已缓存在 `data/cache/`（300 只 CSI 300，2010–2025 日线），无需重新拉取。

```bash
cd D:/桌面文件/quant

# 验证环境
python -c "from data.fetcher import load_daily; d=load_daily('000001'); print(len(d))"

# 跑已有策略报告
cd strategies/ma_crossover && python report.py

# 新增策略：在 strategies/ 下建新文件夹
mkdir strategies/my_strategy
# 1. 复制 report.py 改策略逻辑
# 2. 跑 python report.py 生成报告和图表
# 3. 运行期间自动存 report_output.txt
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
- 策略逻辑不含未来函数
