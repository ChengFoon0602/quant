# MA 交叉趋势跟踪策略 — 完整回测分析

## 策略概述

双均线交叉（MA Crossover）——快线上穿慢线全仓买入，下穿平仓。日线级别，T+1 执行，无未来函数。

## 实验设计

| 维度 | 方法 |
|---|---|
| 参数优化 | 训练集 2010–2019，网格扫描 FAST×SLOW 共 544 组合 |
| 样本外检验 | 测试集 2020–2025，训练最优参数不变 |
| 稳健性 | Bootstrap 500 条重采样，95% CI |
| 截面检验 | 全市场 300 票 × 200 bootstrap，Bonferroni 校正 |

## 文件说明

```
ma_crossover/
├── report.py          # 完整回测报告（运行 python report.py）
├── trial_run.py       # 初始验证：单票单参数跑通
├── scan.py            # 参数相图扫描
├── mc_robustness.py   # 单票 bootstrap 蒙特卡洛
├── batch_mc.py        # 全市场截面 bootstrap
├── figures/           # 报告图表输出
└── README.md
```

## 核心结论

**MA 交叉趋势跟踪策略在 A 股市场不产生统计显著超额收益。**

- 训练集最优 MA2/45: SR=0.615, 年化 14.6%
- 样本外 MA2/45: SR=-0.17, 年化 -3.3% ← 过拟合
- Bootstrap p=0.17 ← 不显著
- Bonferroni 校正后全市场 0/297 通过

训练集正夏普可由牛市 beta + 数据挖掘解释。

## 依赖

项目根目录 `quant/` 下的:
- `data/fetcher.py` — 数据获取
- `backtest/engine.py` — 向量化回测引擎

## 运行

```bash
cd strategies/ma_crossover
python report.py       # 完整报告
python trial_run.py    # 快速验证
```
