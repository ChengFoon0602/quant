# quant — 轻量量化交易研究框架

个人量化策略研究与实验平台。目标：轻量、可审计、数学上严谨。

## 结构

```
quant/
├── data/                      # 数据获取（baostock + akshare）+ 缓存
├── signals/                   # 因子库（alpha191、基本面）
├── backtest/                  # 向量化回测引擎
├── strategies/                # 策略实验（每个子目录 = 一项独立研究）
│   └── <name>/report.md       # 图文报告 + figures/
├── models/                    # ML 模型 + 方法论修正记录
└── bootstrap.py               # 首次数据拉取脚本
```

## 研究报告索引（按研究弧线阅读顺序）

> **研究弧线**：从「单因子/单票 → ML 合成 → 换指数 → 换信息源 → 市场结构 → 跨指数验证」一路收敛，
> 最后结论是 **A 股日线级截面 alpha 在扣除真实成本后不成立**，转向理解市场结构。
> 每份报告的「方法论修正」章节往往是比结论更有价值的部分（记录了真实踩坑）。

| 阅读序 | 报告 | 阶段 | 核心结论 |
|--------|------|------|---------|
| 1 | [`ma_crossover`](strategies/ma_crossover/report.md) | 单票时序 | MA 交叉无显著超额——A 股有效性起点 |
| 2 | [`alpha001_trial`](strategies/alpha001_trial/report.md) | 单因子截面 | Alpha001 IC>0 但信息比率负，跑不赢基准 |
| 3 | [`multi_factor_trial/alpha012_alpha055_alpha191`](strategies/multi_factor_trial/alpha012_alpha055_alpha191/report.md) | 多因子合成 | 三因子两个是噪声一个有效，合成不改善 |
| 4 | [`factor_discovery`](strategies/factor_discovery/report.md) | 因子提纯 | 191→106 候选，alpha141 是「纸老虎」（IC 高无分散度） |
| 5 | [`feature_selection`](strategies/feature_selection/report.md) | 特征选择 | 191→16 因子（含「事故 universe」更正记录） |
| 6 | [`models/report.md`](models/report.md) | ML 非线性合成 | **方法论修正 I/II/III**：夏普 8.75→0.08 的完整踩坑史（overlapping returns 平滑 + 幸存者偏差） |
| 7 | [`zz500_pit_trial`](strategies/zz500_pit_trial/report.md) | 路线② 量价×中证500 | 信号翻倍（OOF 1.618）但 **selection bias 消除后 0.295**，三方向全否定 |
| 8 | [`zz500_fundamental_trial`](strategies/zz500_fundamental_trial/report.md) | 方向2 基本面 | 换源 akshare，20 因子三层检验全否定，弧线闭合 |
| 9 | [`zz500_crowding_trial`](strategies/zz500_crowding_trial/report.md) | 方向C 市场结构 | 量价延续 vs 基本面反转双面体 + 低拥挤择时可交易性初探 |
| 10 | [`hs300_crowding_trial`](strategies/hs300_crowding_trial/report.md) | 方向C 延伸④ 跨指数 | **跨指数验证证伪**：沪深300 双面体不成立、择时 SR −1.69、基本面因子多空 p=0.0002 确证负 alpha——中证500 发现是小盘特例 |
| 11 | [`etf_momentum_crowding`](strategies/etf_momentum_crowding/report.md) | 阶段D ETF与大类资产 | **行业动量反转 + 股债金避险**：行业 ETF 截面动量显现反转（Rank IC −0.0385）；叠加 MA20 趋势与国债/黄金避险后，**最大回撤从 −53.18% 降至 −22.97%（回撤腰斩）**，双边 4 bps 超低摩擦 |

**方法论铁律**（贯穿全部报告，详见 `CLAUDE.md`）：
未来函数（`.shift(1)` / PIT 公告日）、幸存者偏差（PIT 成分股）、Overlapping Returns 平滑陷阱、
selection bias（WF 年度重选池）、摩擦成本（双边 0.3%）。

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
