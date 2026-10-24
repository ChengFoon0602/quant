# Alpha191 实现验证报告（替代 `alpha191_diff_report.md`）

生成日期：**2026-10-24**　最后一次全量复跑：**2026-10-24 18:30**

> **旧报告 `tests/alpha191_diff_report.md` 已作废（2026-10-24）**
>
> 它的参考引擎 `tests/_aurumq_stub` 已丢失、无法复跑，且 191 个因子里 162 个被判
> `sign_diff` 却**从未给出解释**——那 162 个「差异」多数是 `RANK` 语义架构差异
> （参考引擎截面排名 vs 本实现时序排名），不是公式错误。把未解释的分歧留在报告里，
> 比没有报告更糟。本报告取代它。

---

## 一、这份报告能证明什么、不能证明什么

先把话说清楚，避免把「防回归」当成「正确」：

| 层级 | 验证手段 | 能证明 | **不能**证明 |
|---|---|---|---|
| L1 防回归 | golden 基线哈希（191 因子） | 改动 operators/factors 后数值不漂移 | 实现与研报公式一致 |
| L2 实现正确性 | 纯 numpy 独立双实现互证（16 算子 + 5 因子） | 窗口边界 / NaN 传导 / 并列处理 / 自由度 / 浮点路径无 bug | 公式转写本身的理解错误——两套实现可能犯同一个错 |
| L3 口径健康 | 健康度守卫 4 项 | 无全 NaN、截面有效、VWAP 量纲正确、alpha001 前视不变式成立 | 191 个因子的经济含义 |
| L4 公式一致性 | `docs/alpha191_公式语义表.csv`（191 行逐条注解） | 每个算子的语义、偏离研报处有据可查 | 语法级等价（靠人工审计） |
| ~~L5 外部对拍~~ | 聚宽 / 米筐 / BRAIN 真值 | **未做**——沙箱网络阻断，见 §四 | — |

**结论**：当前实现达到 L1–L4，**未达 L5**。可以称为「口径可辩护、防回归、有审计记录的复现」，
不能称为「已与第三方实现逐位对齐的复现」。

---

## 二、L2：算子层独立双实现互证（本次新增）

### 为什么是双实现而不是外部对拍

原计划（修复计划 §D3）抽 20 个非 VWAP 因子对独立数据源。实测不可行：沙箱网络只放行代理白名单，
`akshare.stock_zh_a_hist` 抛 `ProxyError`（`push2his.eastmoney.com` 不可达），禁用代理直连也被断。

替代做法：用**纯 numpy 另写一套** 16 个算子 + 5 个因子，与 `signals/alpha191/operators.py`
的 pandas 实现逐位比较。两套实现走完全不同的代码路径（手写滑动窗口 vs pandas `rolling`），
因此浮点路径、NaN 传导、窗口边界都是独立事件。

- 脚本：`tests/verify_operators_independent.py`
- 明细：`tests/operator_dual_impl_report.csv`（85 行）
- 判据：有限值相对误差 < 1e-9，**且 NaN 位置完全一致**（NaN 位置比数值更容易出错，也更致命）
- 数据：000001 / 000002 / 600000 / 600036 / 601318，2015-01-01 ~ 2020-12-31

### 结果（2026-10-24 18:30 复跑）

```
✅ 全部 85 项逐位一致（NaN 位置 + 有限值，相对误差 < 1e-09）
   16 算子 × 5 只 = 80 项：SUM STD MAX MIN DELTA DELAY RANK TSRANK
                            CORR SMA REGBETA REGRESI COUNT LOG ABS SIGN
   5 因子 × 5 只 = 25 项：001 / 003 / 011 / 018 / 055
```

5 个端到端因子的选择是有意的，覆盖四类构造：`RANK+CORR`（001）、条件求和（003）、
`RANK` 乘积（011）、`DELTA` 组合（018/055）。两套实现在这 5 个因子上一致，
说明「公式转写 → 算子组合」这一段没有实现层 bug。

### 这次互证抓到的两个真 bug

**① `REGBETA` 自由度不一致（严重）**

```python
# 修复前
cov = y.rolling(n).cov(x)        # pandas 默认 ddof=1 → 样本协方差
var = x.rolling(n).var(ddof=0)   # 总体方差
return cov / var                 # β 被系统性放大 (n−1)/n 倍
```

分子分母自由度不同，n=5 时 β **偏大 20%（实测相对误差 0.200）**。
影响 alpha142–149 共 8 个因子，其中 **142 在新因子池内**。
修复：`cov = y.rolling(n).cov(x, ddof=0)`。

**② `SMA` 初值路径与研报不等价（中等）**

原实现用 `ewm(alpha=m/n, adjust=False)` 的值填充 `t≥n`，只在 `t=n−1` 处覆盖为前 n 日均值。
但 ewm 的递归从 `x[0]` 起步——在 `n−1` 打个补丁之后仍走 ewm 自己的历史路径，
与研报「均值初值 → 纯递归」的语义不等价（实测偏差 0.13%~0.46%）。

影响 alpha009、076–093（其中 **087 是新池里 IC_IR 最高的因子**）、173、174。
修复：改为显式递归，初值取前 n 日均值，NaN 处沿用上一值（与 `ignore_na=True` 一致）。

**③ 顺带的向量化（非 bug）**：`TSRANK` 改用 `np.lib.stride_tricks.sliding_window_view`，
数值 0 差异、提速约 8×。

---

## 三、L1 + L3：回归基线与健康度守卫（2026-10-24 18:30 复跑）

| 检查 | 命令 | 结果 |
|---|---|---|
| golden 基线（v2 面板模式，191 因子） | `.venv/Scripts/python.exe tests/factor_golden_baseline.py --verify` | ✅ 191/191 一致，无数值回归 |
| 无全 NaN 因子 | `pytest tests/test_alpha191_health.py` | ✅ 4 passed |
| 截面有效（CS_eff） | 同上 | ✅ |
| VWAP 量纲 `(amount/volume)/close_adj3` 中位 ≈ 1 | 同上 | ✅ 实测 0.9999 |
| alpha001 前视不变式 | 同上 | ✅ |

golden 基线的定位必须强调：它锁的是**当前实现**的数值，只用于防回归。
把错误实现锁进基线，基线只会忠诚地保护这个错误——所以 L2 和 L4 不可省。

---

## 四、L5 外部对拍：未完成，及为什么

| 数据源 | 状态 | 原因 |
|---|---|---|
| akshare（东财） | ❌ | `ProxyError`，沙箱网络仅放行代理白名单 |
| 聚宽 / 米筐 | ❌ | 需账号 token，且同样走外网 |
| WorldQuant BRAIN | ❌ | 只接受 alpha 表达式与 IC 反馈，不返回逐日因子值，无法做数值对拍 |

**遗留待办**（网络条件允许时补做）：抽 20 个非 VWAP 因子，取第三方因子值落盘到
`tests/golden/external_truth/*.csv`，与本地实现逐日比较。只有这一步做完，
L5 才能从「内部一致」升级为「外部一致」。

---

## 五、复现命令

```bash
cd D:/桌面文件/quant
.venv/Scripts/python.exe tests/verify_operators_independent.py   # L2 双实现互证，明细落 CSV
.venv/Scripts/python.exe tests/factor_golden_baseline.py --verify # L1 防回归
.venv/Scripts/python.exe -m pytest tests/test_alpha191_health.py -q  # L3 口径守卫
.venv/Scripts/python.exe tests/audit_alpha191_semantics.py       # L4 语义表 → docs/alpha191_公式语义表.csv
```

相关文档：`docs/alpha191_复现审计_2026-10-24.md`（缺陷清单）、
`docs/alpha191_修复计划_2026-10-24.md`、`docs/alpha191_修复执行_2026-10-24.md`（§九 双实现互证）。
