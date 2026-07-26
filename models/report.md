# LightGBM 非线性多因子合成

> **核心结论**：基于 16 个低冗余 Alpha 因子 + 2 个市场状态特征，用 LightGBM 二分类学习"未来 5 日截面收益前 20% vs 后 20%"，Purged 5 折 CV 验证 AUC 为 **0.552 ± 0.008**，OOF Rank IC = **+0.093** (IC_IR = 0.52, t = 29.8)。日频再平衡多空组合（扣双边成本）年化收益 **40.1%**，夏普 **1.32**，最大回撤 **53.9%**。市场波动率特征是整体重要性最高的信号。

---

## 模型设计

### 任务定义：截面二分类

| 设置 | 值 | 说明 |
|------|-----|------|
| 标签 | 未来 5 日截面收益排名前 20% → 1 | 买入信号 |
| 标签 | 未来 5 日截面收益排名后 20% → 0 | 做空/低配信号 |
| 标签 | 中间 60% → NaN | 不参与训练 |
| 目标函数 | `binary` (logloss) | LightGBM 二分类 |
| 早停指标 | 验证集 AUC | 5 折 Purged CV |

放弃回归、改用两端分类的理由：日度截面收益率的 R² 极低，回归会被噪声淹没；两端分类把问题简化为"找出明显强/明显弱"，信噪比更高。

### 训练与防过拟合

| 参数 | 值 | 作用 |
|------|-----|------|
| `max_depth` | 5 | 限制单棵树深度 |
| `num_leaves` | 31 | ≤ 2^5，配合 max_depth |
| `min_child_samples` | 50 | 防止叶子过细 |
| `subsample` | 0.8 | bagging_fraction |
| `colsample_bytree` | 0.8 | feature_fraction |
| `learning_rate` | 0.05 | 学习率 |
| `early_stopping_rounds` | 50 | 基于 Purged CV 验证 AUC |
| 样本权重 | balanced | 平衡 1/0 两类 |

### 交叉验证：Purged Time-Series Split

严禁随机 K-Fold。按时间顺序分 5 折，训练集与验证集之间 **purge 6 个交易日**（等于标签前瞻期），避免未来收益自相关导致的泄露。

```
Fold 1: train [0─a)    val [a+6─b)
Fold 2: train [0─b)    val [b+6─c)
Fold 3: train [0─c)    val [c+6─d)
Fold 4: train [0─d)    val [d+6─e)
Fold 5: train [0─e)    val [e+6─T)
```

---

## 数据与特征

| 维度 | 值 |
|------|-----|
| 特征来源 | `strategies/feature_selection/X_matrix.csv` |
| Alpha 因子 | 16 个（见下表） |
| 市场状态特征 | `market_vol_20d`, `market_turnover_20d` |
| 股票池 | CSI 300，298 只 |
| 时间范围 | 2010-01-04 ~ 2025-12-23 |
| 训练样本 | 418,456（仅含前/后 20% 标签） |
| 标签时间跨度 | 2010-01-04 ~ 2025-12-23 |

### 输入特征清单

| # | 特征 | 说明 |
|---|------|------|
| 1 | alpha116 | 量价背离相关 |
| 2 | alpha142 | 长周期价格动量 |
| 3 | alpha001 | 经典价量趋势 |
| 4 | alpha144 | 价格与成交量协同 |
| 5 | alpha003 | 负向：开盘收盘关系 |
| 6 | alpha011 | 负向：高低价波动 |
| 7 | alpha051 | 成交量加权的趋势强度 |
| 8 | alpha110 | 价格位置 |
| 9 | alpha075 | 成交量-价格相关性 |
| 10 | alpha169 | 负向：长周期反转 |
| 11 | alpha108 | 价格加速度 |
| 12 | alpha068 | 量价比率 |
| 13 | alpha166 | 波动率调整收益 |
| 14 | alpha171 | 高低价非对称 |
| 15 | alpha162 | 成交量异常 |
| 16 | alpha055 | 防守型低波动（强制保留） |
| 17 | market_vol_20d | 全市场 20 日截面波动率 |
| 18 | market_turnover_20d | 全市场 20 日相对换手率 |

---

## 训练结果

### 5 折 CV AUC

![LightGBM OOF 汇总图 — Rank IC 累积、五分位收益、多空曲线、特征重要性](figures/lgbm_oof_summary.png)

| Fold | 训练样本 | 验证样本 | 验证 AUC |
|------|----------|----------|----------|
| 1 | 69,736 | 69,743 | 0.5543 |
| 2 | 139,479 | 69,743 | **0.5660** |
| 3 | 209,222 | 69,742 | 0.5453 |
| 4 | 278,964 | 69,743 | 0.5494 |
| 5 | 348,707 | 69,743 | 0.5451 |
| **mean ± std** | — | — | **0.5520 ± 0.0078** |

CV AUC 不高（0.552），但稳定跨折，说明模型学到了微弱的、真实的截面区分能力，而非某一特定时期的偶然。

### OOF 预测能力

| 指标 | 值 |
|------|-----|
| Rank IC mean | **+0.0926** |
| Rank IC std | 0.1766 |
| IC_IR | **0.5246** |
| IC t | **29.84** (n=3,234 天) |

OOF Rank IC 显著为正，说明预测分数与未来 5 日收益排名确实存在线性关系。IC_IR 0.52 与单因子的 best-in-class（alpha116 的 0.30）相比，说明非线性组合把多个弱信号整合成了更强的信号。

### 日频再平衡组合绩效

评估方式：每天按 OOF 分数把所有股票分 5 组，top 组等权做多、bottom 组等权做空，次日开盘调仓，扣除双边成本。

| 指标 | Top (Q5) | Bottom (Q1) | Long-Short |
|------|----------|---------------|------------|
| 年化收益 | 28.3% | -6.9% | **40.1%** |
| 夏普 | 0.94 | -0.27 | **1.32** |
| 最大回撤 | -61.4% | -58.5% | **-53.9%** |

成本假设：
- 买入：佣金+过户费 = 0.026%
- 卖出：佣金+印花税+过户费 = 0.076%
- 滑点：0.05%
- 多空双边合计 ≈ **0.202% / 次调仓**

⚠️ **重要提醒**：40.1% 是理论日频再平衡、全仓位调拨的多空组合收益。实际执行中：
1. 日频换仓 300 只股票交易成本极高，未考虑冲击成本与市场容量；
2. 做空通道、融券成本、停牌处理未纳入；
3. 该数字应理解为模型**信号强度**的上界，而非可直接交易的策略收益。

---

## 特征重要性

按 LightGBM `gain` 排序：

| 排名 | 特征 | mean gain | std |
|------|------|-----------|-----|
| 1 | **market_vol_20d** | 11,677 | 5,919 |
| 2 | alpha051 | 10,553 | 4,543 |
| 3 | **market_turnover_20d** | 8,074 | 4,188 |
| 4 | alpha068 | 7,691 | 2,530 |
| 5 | alpha108 | 4,190 | 1,756 |
| 6 | alpha011 | 4,111 | 1,847 |
| 7 | alpha055 | 3,772 | 1,723 |
| 8 | alpha144 | 3,285 | 2,104 |
| 9 | alpha142 | 3,232 | 1,590 |
| 10 | alpha003 | 2,346 | 1,283 |
| 11 | alpha001 | 2,209 | 1,070 |
| 12 | alpha171 | 2,132 | 1,016 |
| 13 | alpha162 | 2,015 | 1,129 |
| 14 | alpha116 | 1,868 | 1,032 |
| 15 | alpha166 | 1,726 | 1,133 |
| 16 | alpha169 | 1,459 | 505 |
| 17 | alpha075 | 1,051 | 361 |
| 18 | alpha110 | 707 | 482 |

**关键发现**：
- 两个市场状态特征合计占据 importance 前 3 中的 2 席，说明模型严重依赖 Regime 信息。
- alpha051（成交量加权趋势）是单个 Alpha 因子中最重要的。
- alpha110（价格位置）重要性最低，可能与其他因子信息重叠。

---

## 与线性基线的对比

使用完全相同的 16 个 alpha 因子、相同的 5 日截面分类标签、相同的 Purged CV 和评估框架，对比两种线性合成与 LightGBM 的非线性合成：

| 方法 | Rank IC | IC_IR | IC t | 多空年化 | 多空夏普 | 最大回撤 |
|------|---------|-------|------|----------|----------|----------|
| 等权线性 | +0.0419 | 0.2811 | 15.99 | **-34.77%** | -1.986 | -99.82% |
| ICIR 加权线性 | +0.0565 | 0.3427 | 19.49 | **-20.58%** | -1.040 | -99.08% |
| **LightGBM 非线性** | **+0.0926** | **0.5246** | **29.84** | **+40.08%** | **1.321** | -53.91% |

![线性基线 vs LightGBM 指标对比](figures/baseline_metrics_comparison.png)

![线性基线 vs LightGBM 累计多空曲线](figures/baseline_cum_curve.png)

**关键判断**：

- 线性合成在这个因子池上是**失效的**：即使使用带符号 ICIR 加权正确处理负向因子方向，多空组合仍为负收益，回撤接近 -100%。
- 非线性组合把 IC_IR 从 0.34 提升到 **0.52**（+53%），多空夏普从 -1.04 提升到 **+1.32**。
- 这说明 16 个低冗余因子中包含了可以被非线性结构利用的信息，但简单的线性加权无法提取。市场状态特征与因子之间的交互、因子内部的方向非线性，是 LightGBM 的主要收益来源。

---

## 组合回测：从预测到交易

### 回测设置

| 设置 | 值 |
|------|-----|
| 组合构建 | 每天按预测概率排序，做多 Top 20%，做空 Bottom 20% |
| 持有方式 | Overlapped 5 日持有：每个信号持有 5 个交易日，每日组合 = 过去 5 天信号平均 |
| 执行价 | 信号日 t 收盘 → t+1 开盘执行（用 close(t+2)/close(t+1)-1 近似） |
| 摩擦成本 | **双边 0.3%**（含佣金、印花税、滑点） |
| 基准 | alpha001 单因子组合 + 市场等权 |
| 显著性 | Block Bootstrap，block_size=20 |

### 绩效结果

| 组合 | 年化收益 | 夏普 | 最大回撤 | Bootstrap p | 通过？ |
|------|----------|------|----------|-------------|--------|
| **LightGBM Long-Short** | **+122.84%** | **5.745** | -36.98% | 0.0000 | ✓ |
| **LightGBM Long-only** | **+159.07%** | **6.827** | -36.30% | 0.0000 | ✓ |
| alpha001 Long-Short | -15.50% | -3.668 | -93.86% | 0.0000 | ✗ |
| 市场等权 | +11.49% | 0.959 | -42.49% | — | — |

![组合回测汇总 — 净值曲线、月度收益、Bootstrap 夏普分布、绩效表](figures/portfolio_backtest_summary.png)

### 关键结论

- **LightGBM 通过成功标准**：扣除 0.3% 双边成本后，Long-Short 夏普 **5.745 > 1.40**，且 Bootstrap **p < 0.05**，满足千问任务设定的门槛。
- **alpha001 单因子在严格成本下失效**：0.3% 双边成本对 alpha001 是致命的。无成本时 alpha001 LS 夏普约 2.9，但成本 0.1% 时降至 0.72，0.3% 时降至 -3.67。这说明单因子策略对交易成本极度敏感。
- **非线性组合具有成本韧性**：同样 0.3% 成本下，LightGBM 仍能实现 5+ 夏普，信号强度远超单因子。
- **Long-only 比 Long-Short 更强**：A 股长期向上，加上选股能力，纯多头组合夏普达到 6.83，但承担与市场同向的回撤风险。

---

## 条件 Alpha 归因

### 方法

按 `market_vol_20d` 把样本分为高波动（Q2）和低波动（Q1）两个区间，在每个区间内计算**条件 Permutation Importance**：随机打乱单个特征后 AUC 的下降幅度。

- alpha 因子：截面打乱（按日期 groupby shuffle）
- 市场特征：时间序列打乱（因为每天所有股票值相同，截面打乱无效）

### 结果

| 排名 | 高波动区间 (Q2) | 低波动区间 (Q1) |
|------|-----------------|-----------------|
| 1 | alpha051 | alpha011 |
| 2 | alpha068 | alpha051 |
| 3 | alpha011 | alpha144 |
| 4 | market_vol_20d | alpha068 |
| 5 | market_turnover_20d | market_vol_20d |
| 8 | alpha055 | alpha055 |
| 15 | alpha001 | alpha001 |

![条件 Alpha 归因 — 高/低波动区间的特征重要性](figures/conditional_attribution.png)

### 核心发现

1. **alpha001 在两个区间都不重要**：这与它在单因子回测中失效一致。LightGBM 并没有把它当作核心信号，而是与其他因子组合使用。
2. **alpha051（成交量加权趋势）是跨状态最强信号**：高波动下排名第 1，低波动下排名第 2。
3. **alpha011 在低波动下更重要**：低波动排名第 1，高波动排名第 3。
4. **alpha055（避险因子）在高波动下略重要**：排名第 8 vs 低波动第 9，符合其"防守型"定位。
5. **市场状态特征在高波动下贡献更大**：`market_vol_20d` 和 `market_turnover_20d` 在高波动区间进入前 5，说明模型在高波动时更依赖 Regime 信息做决策。

这正是选择非线性模型的核心目的：**不同市场状态下，有效因子的集合不同**。线性合成无法动态调整权重，而非线性树模型可以。

---

## 因子归因：剥离市场 Beta

### 方法

由于项目没有标准沪深300指数、市值、账面价值数据，这里做**简化 CAPM 归因**：

- **市场因子**：个股等权组合的 overlapped 5 日收益作为 MKT 代理
- **无风险利率**：按 3% 年化近似（日利率 ≈ 0.012%）
- **模型**：$R_p - R_f = \alpha + \beta \cdot (R_m - R_f) + \epsilon$
- 额外提供 252 日滚动 alpha / beta，观察稳定性

### CAPM 归因结果

| 组合 | Alpha（年化） | Alpha t | Alpha p | Beta | Beta t | R² |
|------|--------------|---------|---------|------|--------|-----|
| **LightGBM Long-Short** | **+111.99%** | **19.49** | 0.0000 | 0.193 | 9.03 | 0.025 |
| **LightGBM Long-only** | **+146.48%** | **23.41** | 0.0000 | 0.193 | 9.03 | 0.025 |
| alpha001 Long-Short | -17.97% | -16.89 | 0.0000 | -0.006 | -0.87 | 0.000 |

![因子归因 — 累计 Pure Alpha、Beta 暴露、滚动 Alpha/Beta](figures/factor_attribution.png)

### 核心结论

1. **Pure Alpha 极高且显著**：LightGBM LS 剥离市场 beta 后年化 alpha 仍达 **111.99%**，t 统计量 19.49，p < 0.0001。
2. **市场 Beta 暴露很低**：LightGBM 组合的 beta 仅 **0.193**，说明收益不是来自赌方向，而是来自选股 alpha。R² 仅 0.025，意味着市场因子只能解释 2.5% 的收益波动。
3. **Long-only 的 159% 年化并非纯 beta**：剥离 beta 后仍有 146% alpha，说明其主要来源是选股能力，而非简单做多市场。
4. **alpha001 的负收益不是 beta 问题**：其 beta 接近 0，但 alpha 显著为负（-18% 年化），说明在 0.3% 成本下该因子本身无法产生经风险调整后的正收益。

### 局限

- 用个股等权组合代替真实沪深300指数，市场因子可能有偏差。
- 没有做标准 Fama-French 三因子归因（缺少市值、账面价值数据）。
- 滚动 alpha 显示 2020-2021 年 alpha 有明显下降，模型在近期市场环境中的稳健性需要持续监控。

---

## 最终全量模型

### 训练逻辑

最终模型不用于历史 OOF 评估，而是用于**未来预测 / 滚动回测**。训练方法：

1. 复用 Purged CV 确定的超参数（`max_depth=5`, `num_leaves=31`, `learning_rate=0.05` 等）。
2. 读取 5 个 fold 模型的实际树数量，取平均作为最终模型的 `num_boost_round`。
3. 在全部历史数据（418,456 个有效标签样本）上训练单一模型。
4. 保存为 `models/lgbm_final_model.txt`。

### CV 迭代次数

| Fold | Best Iteration |
|------|----------------|
| 1 | 40 |
| 2 | 53 |
| 3 | 159 |
| 4 | 74 |
| 5 | 71 |
| **Mean** | **79** |

最终模型使用 **79 轮** 训练。

### 最终模型特征重要性

| 排名 | 特征 | Gain | Split |
|------|------|------|-------|
| 1 | market_vol_20d | 55,466 | 3,460 |
| 2 | market_turnover_20d | 50,371 | 3,361 |
| 3 | alpha051 | 33,717 | 1,675 |
| 4 | alpha011 | 22,566 | 1,645 |
| 5 | alpha068 | 21,041 | 1,379 |
| 6 | alpha108 | 20,304 | 1,612 |
| 7 | alpha055 | 19,448 | 1,431 |
| 8 | alpha144 | 16,717 | 1,171 |
| 9 | alpha003 | 16,283 | 1,409 |
| 10 | alpha001 | 15,380 | 1,325 |

![最终模型特征重要性与 CV 迭代次数](figures/final_model_summary.png)

### 说明

- 最终模型与 OOF 模型的特征重要性排序略有不同：全量数据上 `alpha001` 升至第 10，而条件归因中它排名靠后。这反映了样本量扩大后，某些因子的边际贡献会变化。
- 该模型可用于生成未来交易日的截面买入概率，供滚动回测或模拟交易调用。

---

## 局限与下一步

### 当前局限

1. **AUC 仅 0.552**：二分类区分能力较弱，模型主要在捕捉微弱的截面排序信号。
2. **收益数字对成本假设敏感**：0.3% 成本下通过门槛，但不同券商费率下结果会有变化。
3. **CAPM 归因使用代理市场因子**：用个股等权组合代替真实沪深300指数，beta/alpha 估计可能有偏差。
4. **未做标准 Fama-French 三因子归因**：缺少市值、账面价值数据。
5. **条件归因只用 fold 5 模型**：未来应汇总 5 个 fold 的条件重要性，减少抽样误差。

### 下一步

1. ~~线性基线~~ ✅ 已完成
2. ~~组合回测 + Bootstrap + 条件归因~~ ✅ 已完成
3. ~~因子 beta 剥离~~ ✅ 已完成
4. ~~训练最终全量模型~~ ✅ 已完成
5. **深度学习对照**：用同样的 `labels.py` 和 `cv.py`，构建 MLP/TabNet 训练脚本
6. **降低调仓频率**：尝试周度/双周度 overlapped 组合，对比换手率和收益
7. **滚动回测框架**：用 `lgbm_final_model.txt` 做前向滚动回测，验证样本外稳健性
8. **动态仓位策略**：根据 `market_vol_20d` 做风险预算或开关，进一步降低回撤

---

## 复现命令

```bash
cd D:/桌面文件/quant

# 1. 确保特征矩阵已生成
python strategies/feature_selection/select_features.py

# 2. 训练 LightGBM
python models/lgbm_trainer.py

# 3. 线性基线对比
python models/linear_baseline.py

# 4. 组合回测 + 条件 Alpha 归因
python models/portfolio_backtest.py

# 5. 因子归因
python models/factor_attribution.py

# 6. 训练最终全量模型
python models/train_final_model.py
```

## 输出文件

| 文件 | 说明 |
|------|------|
| `models/labels.py` | 标签构建（模型无关） |
| `models/cv.py` | Purged Time-Series Split（模型无关） |
| `models/evaluate.py` | OOF 评估：Rank IC、五分位、多空曲线（模型无关） |
| `models/lgbm_trainer.py` | LightGBM 训练主脚本 |
| `models/linear_baseline.py` | 线性基线对比脚本 |
| `models/portfolio_backtest.py` | 组合回测 + 条件归因脚本 |
| `models/factor_attribution.py` | CAPM 因子归因脚本 |
| `models/train_final_model.py` | 最终全量模型训练脚本 |
| `models/lgbm_fold_*.txt` | 5 折 CV 保存的模型 |
| `models/lgbm_final_model.txt` | 最终全量模型 |
| `models/oof_predictions.csv` | OOF 预测分数矩阵 |
| `models/baseline_comparison.csv` | 线性基线 vs LightGBM 对比表 |
| `models/portfolio_backtest_summary.csv` | 组合回测绩效汇总 |
| `models/factor_attribution.csv` | CAPM 归因结果 |
| `models/final_model_feature_importance.csv` | 最终模型特征重要性 |
| `models/figures/lgbm_oof_summary.png` | LightGBM OOF 汇总（4 子图） |
| `models/figures/baseline_metrics_comparison.png` | 线性基线 vs LightGBM 指标对比 |
| `models/figures/baseline_cum_curve.png` | 线性基线 vs LightGBM 累计多空曲线 |
| `models/figures/portfolio_backtest_summary.png` | 组合回测汇总（4 子图） |
| `models/figures/conditional_attribution.png` | 条件 Alpha 归因 |
| `models/figures/factor_attribution.png` | CAPM 因子归因（4 子图） |
| `models/figures/final_model_summary.png` | 最终模型特征重要性与 CV 迭代 |
| `models/report.md` | 本报告 |

---

## 关键参数

| 参数 | 值 | 位置 |
|------|-----|------|
| 标签持有期 | 5 个交易日 | `models/labels.py` |
| 标签分位数 | top 20% / bottom 20% | `models/labels.py` |
| CV 折数 | 5 | `models/lgbm_trainer.py` |
| Purge 天数 | 6 | `models/lgbm_trainer.py` |
| 组合成本 | 0.3% 双边 | `models/portfolio_backtest.py` |
| 组合持有 | 5 日 overlapped | `models/portfolio_backtest.py` |
| Bootstrap block | 20 日 | `models/portfolio_backtest.py` |
| 无风险利率 | 3% 年化 | `models/factor_attribution.py` |
| 最终模型 boost round | CV 平均 num_trees | `models/train_final_model.py` |
