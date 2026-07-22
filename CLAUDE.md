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

## 红线

- 不接实盘（该项目现阶段仅用于研究和模拟）
- 不做高频/tick 级数据（日线/周线为主）
- 策略逻辑不含未来函数
