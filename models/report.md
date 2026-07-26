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

作为后续深度学习/非线性研究的基准，当前 LightGBM 的结果可作为参照：

| 模型 | IC_IR | 多空年化 | 多空夏普 |
|------|-------|----------|----------|
| 等权线性（16 因子） | 待计算 | 待计算 | 待计算 |
| ICIR 加权线性 | 待计算 | 待计算 | 待计算 |
| **LightGBM 非线性** | **0.52** | **40.1%** | **1.32** |

> 下一步：补做等权/ICIR 加权线性合成，用完全相同的标签和评估框架，量化 LightGBM 的非线性增益。

---

## 局限与下一步

### 当前局限

1. **AUC 仅 0.552**：二分类区分能力较弱，模型主要在捕捉微弱的截面排序信号。
2. **回撤较大（53.9%）**：多空组合在某些年份（如 2015、2022）可能出现大幅回撤，需要 Regime 过滤。
3. **日频再平衡不现实**：理论收益高，但换仓频率和交易成本在实际中不可忽略。
4. **未做 Bootstrap 显著性检验**：OOF 收益的统计显著性尚未用重采样验证。

### 下一步

1. **线性基线**：等权 / ICIR 加权 vs LightGBM，验证非线性增益。
2. **深度学习对照**：用同样的 `labels.py` 和 `cv.py`，构建 MLP/TabNet 训练脚本。
3. **换仓可行性**：降低调仓频率（周度/双周度），或加入换手率惩罚。
4. **Bootstrap 检验**：对 OOF 日收益做重采样，确认 SR 的置信区间。
5. **Regime 策略**：根据 `market_vol_20d` 做动态仓位/开关，降低回撤。

---

## 输出文件

| 文件 | 说明 |
|------|------|
| `models/labels.py` | 标签构建（模型无关） |
| `models/cv.py` | Purged Time-Series Split（模型无关） |
| `models/evaluate.py` | OOF 评估：Rank IC、五分位、多空曲线（模型无关） |
| `models/lgbm_trainer.py` | LightGBM 训练主脚本 |
| `models/lgbm_fold_*.txt` | 5 折 CV 保存的模型 |
| `models/oof_predictions.csv` | OOF 预测分数矩阵 (date × stocks) |
| `models/feature_importance.csv` | 每折 + 平均特征重要性 |
| `models/summary.json` | 关键指标 JSON |
| `models/figures/lgbm_oof_summary.png` | 4 张 OOF 汇总图 |
| `models/report.md` | 本报告 |

---

## 复现命令

```bash
cd D:/桌面文件/quant

# 1. 确保特征矩阵已生成
python strategies/feature_selection/select_features.py

# 2. 训练 LightGBM
python models/lgbm_trainer.py
```

---

## 关键参数

| 参数 | 值 | 位置 |
|------|-----|------|
| 标签持有期 | 5 个交易日 | `models/labels.py` |
| 标签分位数 | top 20% / bottom 20% | `models/labels.py` |
| CV 折数 | 5 | `models/lgbm_trainer.py` |
| Purge 天数 | 6 | `models/lgbm_trainer.py` |
| max_depth | 5 | `models/lgbm_trainer.py` |
| num_leaves | 31 | `models/lgbm_trainer.py` |
| min_child_samples | 50 | `models/lgbm_trainer.py` |
| 学习率 | 0.05 | `models/lgbm_trainer.py` |
| early_stopping | 50 rounds | `models/lgbm_trainer.py` |
| 多空成本 | 0.202% / 次 | `models/evaluate.py` |
