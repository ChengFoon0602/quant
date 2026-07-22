# quant — 轻量量化交易研究框架

个人量化策略研究与实验平台。目标：轻量、可审计、数学上严谨。

## 结构

```
quant/
├── data/fetcher.py            # 数据获取（baostock + akshare）
├── backtest/engine.py         # 向量化回测引擎
├── signals/                   # 策略信号（待扩展）
├── risk/                      # 风控模块（待扩展）
├── viz/                       # 可视化（待扩展）
├── strategies/                # 策略实验
│   └── ma_crossover/          # MA 交叉趋势跟踪完整回测
│       ├── report.md          # 图文报告
│       └── figures/           # 图表
└── bootstrap.py               # 首次数据拉取脚本
```

## 已完成研究

### MA 交叉趋势跟踪策略

- **结论: 策略在统计上不产生显著超额收益。**
- 方法: 参数相图 + 样本外检验 + Bootstrap MC + 全市场 Bonferroni 校正
- 详见 [`strategies/ma_crossover/report.md`](strategies/ma_crossover/report.md)

## 快速开始

```bash
pip install baostock akshare pandas numpy matplotlib scipy
python bootstrap.py    # 拉取沪深 300 全量日线
cd strategies/ma_crossover
python report.py       # 生成回测报告
```

## 技术栈

Python 3.12 · baostock · akshare · pandas · numpy · matplotlib · scipy

## 声明

本项目仅用于量化研究学习，不构成任何投资建议。
