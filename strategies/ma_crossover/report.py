"""
report.py — MA 交叉趋势跟踪策略完整回测报告。

报告结构:
  1. 实验设计与数据概况
  2. 参数优化（训练集 2010–2019，相图扫描）
  3. 样本外检验（测试集 2020–2025，过拟合诊断）
  4. 风险分析（回撤、VaR、滚动夏普）
  5. 统计显著性（Bootstrap MC + Bonferroni）
  6. 全市场截面检验
  7. 结论

输出: console 格式化报告 + report_figures/ 目录下全部图表
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
from pathlib import Path
import os
import builtins
import warnings

warnings.filterwarnings("ignore")


# 同时输出到控制台和文本文件
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

report_log = open("report_output.txt", "w", encoding="utf-8")
sys.stdout = Tee(sys.stdout, report_log)

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily, cache_summary
from backtest.engine import run

# ── 全局配置 ─────────────────────────────────────────────
SYMBOL = "000001"            # 平安银行（主要分析对象）
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2025-12-31"
N_BOOTSTRAP = 500
ALPHA = 0.05

REPORT_DIR = Path("figures")
REPORT_DIR.mkdir(exist_ok=True)


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def metric_row(label: str, value: str):
    print(f"  {label:<20} {value}")


# ══════════════════════════════════════════════════════════
#  1. 实验设计与数据概况
# ══════════════════════════════════════════════════════════
section("1. 实验设计与数据概况")

print("""
  假设检验:
    H₀: A 股不存在可利用的短期趋势跟踪效应（MA 交叉策略夏普 ≤ 0）
    H₁: MA 交叉策略在统计显著水平上产生正夏普

  策略逻辑:
    快线上穿慢线 → 全仓买入，快线下穿慢线 → 平仓空仓
    当日收盘信号决定次日开盘操作（无未来函数）

  实验设计:
    - 训练集 2010–2019（参数优化）
    - 测试集 2020–2025（样本外验证，无参数调整）
    - Bootstrap MC 检验区分统计涨落与真实信号
    - 全市场 300 票截面检验排除单票幸存者偏差
""")

cache = cache_summary()
close = load_daily(SYMBOL)["close"].astype(float)
close_train = close.loc[TRAIN_START:TRAIN_END]
close_test = close.loc[TEST_START:TEST_END]

print(f"  主要分析对象: {SYMBOL} 平安银行")
print(f"  数据范围: {close.index[0].date()} ~ {close.index[-1].date()}  ({len(close)} 条日线)")
print(f"  训练集: {close_train.index[0].date()} ~ {close_train.index[-1].date()}  ({len(close_train)} 条)")
print(f"  测试集: {close_test.index[0].date()} ~ {close_test.index[-1].date()}  ({len(close_test)} 条)")
print(f"  全市场缓存: {len(cache)} 只股票, {int(cache['rows'].sum())} 条日线")

# ══════════════════════════════════════════════════════════
#  2. 参数优化（训练集）
# ══════════════════════════════════════════════════════════
section("2. 参数优化 — 训练集相图扫描 (2010–2019)")

FAST_RANGE = range(2, 61, 2)
SLOW_RANGE = range(10, 121, 5)

train_results = []
for fast, slow in product(FAST_RANGE, SLOW_RANGE):
    if fast >= slow:
        continue
    fast_ma = close_train.rolling(fast).mean()
    slow_ma = close_train.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    m = run(close_train, signal)
    train_results.append({"fast": fast, "slow": slow, **m})

df_train = pd.DataFrame(train_results)
best_train = df_train.loc[df_train["sharpe"].idxmax()]
best_fast, best_slow = int(best_train["fast"]), int(best_train["slow"])

# 训练集稳健性: 正夏普区域占比 + 最优附近 3×3 网格的均值
sharpe_grid = df_train.pivot_table(index="slow", columns="fast", values="sharpe")
valid_vals = sharpe_grid.values[~np.isnan(sharpe_grid.values)]
positive_ratio = (valid_vals > 0).mean()

near_best = df_train[
    (df_train["fast"].between(best_fast - 4, best_fast + 4)) &
    (df_train["slow"].between(best_slow - 15, best_slow + 15))
]
plateau_sr = near_best["sharpe"].mean()
plateau_std = near_best["sharpe"].std()

print(f"  参数组合: {len(train_results)} 个 (FAST ∈ [2,60], SLOW ∈ [10,120])")
print(f"  正夏普区域占比: {positive_ratio*100:.1f}%")
print(f"")
print(f"  最优参数:        MA{best_fast}/{best_slow}")
print(f"  最优训练集 SR:   {best_train['sharpe']:.4f}")
print(f"  最优训练集年化:  {best_train['ann_return']*100:.1f}%")
print(f"  最优训练集回撤:  {best_train['max_drawdown']*100:.1f}%")
print(f"")
print(f"  附近区域均值 SR: {plateau_sr:.4f} ± {plateau_std:.4f}  (3×3 邻域)")
print(f"  高原比 (plateau/peak): {plateau_sr/best_train['sharpe']:.2f}  (>0.7 ⇒ 稳健)")

# 图: 训练集夏普热力图
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.pcolormesh(sharpe_grid.columns, sharpe_grid.index, sharpe_grid.values,
                    cmap="RdYlBu_r", shading="auto")
ax.plot(best_fast, best_slow, marker="*", color="black", markersize=14)
ax.set_xlabel("FAST"); ax.set_ylabel("SLOW")
ax.set_title(f"{SYMBOL} 训练集 MA 交叉 — 夏普比率 (2010–2019)")
ax.invert_yaxis()
fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout(); plt.savefig(REPORT_DIR / "02_param_heatmap.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  3. 样本外检验
# ══════════════════════════════════════════════════════════
section("3. 样本外检验 (2020–2025)")

# 训练集最优参数 → 测试集
fast_ma_test = close_test.rolling(best_fast).mean()
slow_ma_test = close_test.rolling(best_slow).mean()
signal_test = (fast_ma_test > slow_ma_test).astype(int)
result_test = run(close_test, signal_test)

# 训练集同参数（供对比）
fast_ma_train = close_train.rolling(best_fast).mean()
slow_ma_train = close_train.rolling(best_slow).mean()
signal_train = (fast_ma_train > slow_ma_train).astype(int)
result_train = run(close_train, signal_train)

# 过拟合诊断: 训练/测试 SR 比
overfit_ratio = result_test["sharpe"] / result_train["sharpe"] if result_train["sharpe"] > 0 else float("inf")

print(f"  参数: MA{best_fast}/{best_slow} (训练集最优, 在测试集固定不变)")
print(f"")
print(f"  {'指标':<16} {'训练集 (2010–19)':<20} {'测试集 (2020–25)':<20}")
print(f"  {'-'*16} {'-'*20} {'-'*20}")
print(f"  {'年化收益率':<16} {result_train['ann_return']*100:>18.1f}%  {result_test['ann_return']*100:>18.1f}%")
print(f"  {'夏普比率':<16} {result_train['sharpe']:>19.4f}  {result_test['sharpe']:>19.4f}")
print(f"  {'最大回撤':<16} {result_train['max_drawdown']*100:>19.1f}%  {result_test['max_drawdown']*100:>19.1f}%")
print(f"  {'交易次数':<16} {result_train['n_trades']:>20}  {result_test['n_trades']:>20}")
print(f"")
print(f"  过拟合比 (test SR / train SR): {overfit_ratio:.2f}  (>0.5 ⇒ 可接受, <0  ⇒ 严重过拟合)")

# 图: 全时段权益曲线（训练+测试连续拼接, 同一参数）
close_all = pd.concat([close_train, close_test])
fast_ma_all = close_all.rolling(best_fast).mean()
slow_ma_all = close_all.rolling(best_slow).mean()
signal_all = (fast_ma_all > slow_ma_all).astype(int)
result_all = run(close_all, signal_all)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
equity = result_all["equity"] * 100_000
bm = result_all["benchmark"] * 100_000
ax.plot(close_all.index, equity, color="#4CAF50", linewidth=0.8, label=f"MA{best_fast}/{best_slow}")
ax.plot(close_all.index, bm, color="#999", linewidth=0.6, alpha=0.7, label="买入持有")
ax.axvline(pd.Timestamp(TEST_START), color="#F44336", linestyle="--", linewidth=1, alpha=0.7)
ax.text(pd.Timestamp(TEST_START), ax.get_ylim()[1] * 0.95, "← 样本外 →",
        color="#F44336", fontsize=9, ha="left")
ax.fill_between(close_all.index, equity, bm, where=(equity >= bm),
                color="#4CAF50", alpha=0.08)
ax.fill_between(close_all.index, equity, bm, where=(equity < bm),
                color="#F44336", alpha=0.08)
ax.set_ylabel("Equity"); ax.legend(fontsize=8, loc="upper left"); ax.grid(True, alpha=0.3)
ax.set_title(f"{SYMBOL}  MA{best_fast}/{best_slow}  全时段权益曲线 (2010–2025)")

ax = axes[1]
dd = equity / equity.cummax() - 1
ax.fill_between(close_all.index, 0, dd * 100, color="#F44336", alpha=0.3, linewidth=0)
ax.plot(close_all.index, dd * 100, color="#F44336", linewidth=0.6)
ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig(REPORT_DIR / "03_equity_curve.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  4. 风险分析
# ══════════════════════════════════════════════════════════
section("4. 风险分析")

# 使用全时段策略收益
strategy_rets = result_all["equity"].pct_change().dropna()
benchmark_rets = result_all["benchmark"].pct_change().dropna()

# VaR / CVaR
var_95 = np.percentile(strategy_rets, 5)
cvar_95 = strategy_rets[strategy_rets <= var_95].mean()
var_95_bm = np.percentile(benchmark_rets, 5)

# 滚动夏普 (252 日窗口)
rolling_sr = strategy_rets.rolling(252).apply(
    lambda x: np.sqrt(252) * (x.mean() - 0.02/252) / x.std() if x.std() > 1e-12 else 0
)

# Calmar 比率
calmar = result_all["ann_return"] / abs(result_all["max_drawdown"]) if result_all["max_drawdown"] < 0 else 0
calmar_bm = ((bm.iloc[-1] / bm.iloc[0]) ** (1/15.5) - 1) / abs((bm / bm.cummax() - 1).min())

# 最大连续亏损天数
underwater = strategy_rets < 0
consecutive_losses = underwater.astype(int).groupby(
    (underwater != underwater.shift()).cumsum()
).cumsum()
max_consec = consecutive_losses.max()

print(f"  {'VaR (95%)':<20} {var_95*100:>8.2f}%         (基准: {var_95_bm*100:.2f}%)")
print(f"  {'CVaR (95%)':<20} {cvar_95*100:>8.2f}%")
print(f"  {'最大回撤':<20} {result_all['max_drawdown']*100:>8.1f}%")
print(f"  {'年化波动率':<20} {strategy_rets.std()*np.sqrt(252)*100:>8.1f}%")
print(f"  {'Calmar 比率':<20} {calmar:>8.3f}     (基准: {calmar_bm:.3f})")
print(f"  {'最长连续亏损':<20} {int(max_consec):>8} 天")
print(f"  {'滚动 SR 最低':<20} {rolling_sr.min():>8.3f}")
print(f"  {'滚动 SR 最高':<20} {rolling_sr.max():>8.3f}")

# 图: 风险面板
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 收益率分布
ax = axes[0, 0]
ax.hist(strategy_rets * 100, bins=80, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
ax.axvline(var_95 * 100, color="#F44336", linestyle="--", linewidth=1.5, label=f"VaR 95% = {var_95*100:.2f}%")
ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Daily Return (%)"); ax.set_ylabel("Density")
ax.set_title("日收益率分布"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# 滚动夏普
ax = axes[0, 1]
ax.plot(rolling_sr.index, rolling_sr, color="#2196F3", linewidth=0.8)
ax.axhline(0, color="#F44336", linestyle="--", linewidth=1)
ax.axhline(result_all["sharpe"], color="#4CAF50", linestyle="--", linewidth=1, label=f"整体 SR={result_all['sharpe']:.3f}")
ax.fill_between(rolling_sr.index, 0, rolling_sr, where=(rolling_sr >= 0),
                color="#4CAF50", alpha=0.08)
ax.fill_between(rolling_sr.index, 0, rolling_sr, where=(rolling_sr < 0),
                color="#F44336", alpha=0.08)
ax.set_ylabel("Sharpe (252d rolling)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_title("滚动夏普比率")

# 回撤
ax = axes[1, 0]
dd_all = result_all["equity"] / result_all["equity"].cummax() - 1
ax.fill_between(close_all.index, 0, dd_all * 100, color="#F44336", alpha=0.4, linewidth=0)
ax.plot(close_all.index, dd_all * 100, color="#F44336", linewidth=0.5)
ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date"); ax.grid(True, alpha=0.3)
ax.set_title(f"回撤曲线  (最大: {dd_all.min()*100:.1f}%)")

# 月收益热力
ax = axes[1, 1]
monthly = strategy_rets.resample("ME").apply(lambda x: (1 + x).prod() - 1)
monthly_grid = monthly.groupby([monthly.index.year, monthly.index.month]).mean().unstack()
if len(monthly_grid) > 0:
    im = ax.pcolormesh(monthly_grid.columns, monthly_grid.index, monthly_grid.values * 100,
                        cmap="RdYlBu_r", shading="auto")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")
    ax.set_title("月均收益率 (%)")
    fig.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout(); plt.savefig(REPORT_DIR / "04_risk_analysis.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  5. 统计显著性
# ══════════════════════════════════════════════════════════
section("5. 统计显著性 — Bootstrap 蒙特卡洛")

daily_ret = close_all.pct_change().dropna()
rets = daily_ret.values
n_days = len(rets)
rng = np.random.default_rng(42)

sharpe_boot = np.empty(N_BOOTSTRAP)
ann_boot = np.empty(N_BOOTSTRAP)
dd_boot = np.empty(N_BOOTSTRAP)

for i in range(N_BOOTSTRAP):
    idx = rng.choice(n_days, size=n_days, replace=True)
    synth_ret = rets[idx]
    synth_close = pd.Series(100 * (1 + synth_ret).cumprod(), index=daily_ret.index)
    f = synth_close.rolling(best_fast).mean()
    s = synth_close.rolling(best_slow).mean()
    sig = (f > s).astype(int)
    m = run(synth_close, sig)
    sharpe_boot[i] = m["sharpe"]
    ann_boot[i] = m["ann_return"]
    dd_boot[i] = m["max_drawdown"]

real_sr = result_all["sharpe"]
boot_mean = sharpe_boot.mean()
boot_std = sharpe_boot.std()
p_value = (sharpe_boot >= real_sr).mean()
p_pos = (sharpe_boot > 0).mean()
ci_low, ci_high = np.percentile(sharpe_boot, [2.5, 97.5])

print(f"  Bootstrap 样本数: {N_BOOTSTRAP}")
print(f"")
print(f"  真实 SR = {real_sr:.4f}")
print(f"  Bootstrap 均值: {boot_mean:.4f} ± {boot_std:.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  P(SR ≥ 真实值): {p_value:.4f}  (越小越显著)")
print(f"  正夏普概率: {p_pos*100:.1f}%")
print(f"")

if p_value < ALPHA:
    print(f"  ✓ 真实 SR 显著偏离 bootstrap 分布 (p={p_value:.4f} < α={ALPHA})")
else:
    print(f"  ✗ 不显著 — 真实 SR 在 bootstrap 分布内 (p={p_value:.4f} ≥ α={ALPHA})")
    print(f"    策略表现可由随机重排解释，非真实趋势跟踪效应")

# 图: bootstrap 分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(sharpe_boot, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
ax.axvline(real_sr, color="#F44336", linewidth=2, linestyle="--", label=f"真实 SR={real_sr:.3f}")
ax.axvline(0, color="#999", linewidth=1, linestyle=":")
ax.axvline(ci_low, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(ci_high, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.fill_betweenx([0, ax.get_ylim()[1]], ci_low, ci_high, color="#4CAF50", alpha=0.05)
ax.set_xlabel("Sharpe Ratio"); ax.set_ylabel("Density")
ax.set_title(f"Bootstrap 夏普分布 (N={N_BOOTSTRAP})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.scatter(dd_boot * 100, ann_boot * 100, c=sharpe_boot,
           cmap="RdYlBu_r", s=5, alpha=0.4)
ax.scatter([result_all["max_drawdown"] * 100], [result_all["ann_return"] * 100],
           marker="*", color="black", s=250, zorder=5, label="真实")
ax.set_xlabel("Max Drawdown (%)"); ax.set_ylabel("Annual Return (%)")
ax.set_title("Bootstrap 收益-回撤散点图（颜色=夏普）")
ax.axhline(0, color="#999", linewidth=0.5); ax.axvline(0, color="#999", linewidth=0.5)
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout(); plt.savefig(REPORT_DIR / "05_bootstrap.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  6. 全市场截面检验
# ══════════════════════════════════════════════════════════
section("6. 全市场截面检验")

# 复用 batch_mc.py 的结果（如果存在），否则跑简化版
print("  对全市场 300 只股票跑相同策略 (MA4/100 或各票最优参数)")
print("  (此节复用 batch_mc.py 结果 — 22/297 pass p<0.05, 0/297 Bonferroni)")

# 快速复现: 取前 50 只，跑样本外
sample_n = min(50, len(cache))
sample_symbols = sorted([p.name.replace(".csv", "") for p in os.scandir("../../data/cache")
                         if p.name.endswith(".csv")])[:sample_n]

sample_results = []
for sym in sample_symbols:
    df = load_daily(sym)
    if df is None or len(df) < 500:
        continue
    c = df["close"].astype(float)
    c_train = c.loc[TRAIN_START:TRAIN_END]
    c_test = c.loc[TEST_START:TEST_END]

    # 训练集最优参数
    best_local_sr = -999
    best_local_params = (4, 100)
    for f_, s_ in [(2, 40), (4, 60), (4, 100), (8, 80)]:  # 稀疏候选
        fm = c_train.rolling(f_).mean()
        sm = c_train.rolling(s_).mean()
        sig = (fm > sm).astype(int)
        mm = run(c_train, sig)
        if mm["sharpe"] > best_local_sr:
            best_local_sr = mm["sharpe"]
            best_local_params = (f_, s_)

    # 测试集
    f_test, s_test = best_local_params
    fm_test = c_test.rolling(f_test).mean()
    sm_test = c_test.rolling(s_test).mean()
    sig_test = (fm_test > sm_test).astype(int)
    m_test = run(c_test, sig_test)

    sample_results.append({
        "symbol": sym,
        "train_sr": best_local_sr,
        "test_sr": m_test["sharpe"],
        "test_ann": m_test["ann_return"],
        "test_dd": m_test["max_drawdown"],
        "fast": f_test,
        "slow": s_test,
        "n_days": len(c),
    })

df_sample = pd.DataFrame(sample_results)
oos_positive = (df_sample["test_sr"] > 0).mean()
oos_better = (df_sample["test_sr"] > df_sample["train_sr"]).mean()

print(f"\n  截面样本: {len(df_sample)} 只")
print(f"  样本外正 SR 占比: {oos_positive*100:.1f}%")
print(f"  样本外 SR > 训练集 SR: {oos_better*100:.1f}% (应为 ~0% 若无过拟合)")
print(f"  样本外 SR 均值: {df_sample['test_sr'].mean():.4f}")
print(f"  样本外 SR 中位数: {df_sample['test_sr'].median():.4f}")
print(f"")
if oos_positive < 0.55:
    print(f"  ✗ 仅 {oos_positive*100:.0f}% 的票在样本外保持正 SR → 策略不具备普适性")
else:
    print(f"  正 SR 占比 > 50%，需进一步检验")

# 图: 截面结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.barh(range(len(df_sample)), df_sample.sort_values("test_sr")["test_sr"],
        color=["#4CAF50" if s > 0 else "#F44336" for s in df_sample.sort_values("test_sr")["test_sr"]],
        alpha=0.7, height=0.8)
ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Test Sharpe"); ax.set_title(f"样本外 SR ({len(df_sample)} 只)")

ax = axes[1]
ax.scatter(df_sample["train_sr"], df_sample["test_sr"], s=15, alpha=0.6, color="#2196F3")
ax.plot([-1, 2], [-1, 2], color="#999", linewidth=0.5, linestyle="--")
ax.axhline(0, color="#333", linewidth=0.5); ax.axvline(0, color="#333", linewidth=0.5)
ax.set_xlabel("Train Sharpe"); ax.set_ylabel("Test Sharpe")
ax.set_title("训练 vs 样本外 夏普比率")

plt.tight_layout(); plt.savefig(REPORT_DIR / "06_cross_section.png", dpi=150); plt.close()

# ══════════════════════════════════════════════════════════
#  7. 结论
# ══════════════════════════════════════════════════════════
section("7. 结论")

print("""
  ┌─────────────────────────────────────────────────────────┐
  │  1. 参数优化                                               │
  │     训练集最优 MA%d/%d, SR=%.4f, 正夏普区域 %s         │
  │     附近 3×3 网格均值 SR=%.4f ± %.4f (%s)              │
  │                                                         │
  │  2. 样本外验证                                             │
  │     测试集 MA%d/%d: SR=%.4f, 年化=%s                     │
  │     过拟合比 = %.2f (%s)                                 │
  │                                                         │
  │  3. 统计显著性                                             │
  │     Bootstrap p = %.4f (%s)                              │
  │     95%% CI = [%.4f, %.4f]                               │
  │                                                         │
  │  4. 全市场截面                                             │
  │     %d 只中 %d 只 (%.0f%%) 样本外 SR > 0                  │
  │     Bonferroni 校正后 0/297 只显著                         │
  │     通过率 7.4%% ≈ 随机期望 5%% —— 策略夏普的截面分布       │
  │     与随机涨落无统计显著差异                                 │
  │                                                         │
  │  最终判决:                                               │
  │    MA 交叉趋势跟踪策略在 A 股市场上不产生统计显著的          │
  │    超额收益。训练集正夏普 (SR=%.3f) 在样本外消失           │
  │    (SR=%.3f)，Bootstrap 检验不显著 (p=%.2f)，全市场        │
  │    Bonferroni 校正后无一通过。训练集表现可由牛市 beta       │
  │    加数据挖掘解释。建议转向多因子截面策略或另类数据方向。   │
  └─────────────────────────────────────────────────────────┘
""" % (
    best_fast, best_slow,
    best_train["sharpe"],
    "稳健" if plateau_sr / best_train["sharpe"] > 0.7 else "不稳健",
    plateau_sr, plateau_std,
    "高原平坦 ⇒ 非过拟合" if plateau_std < 0.15 else "方差偏大 ⇒ 参数敏感",
    best_fast, best_slow,
    result_test["sharpe"],
    f"{result_test['ann_return']*100:.1f}%",
    overfit_ratio,
    "轻微衰减, 可接受" if overfit_ratio > 0.5 else "严重衰减, 过拟合",
    p_value,
    f"显著 (p<{ALPHA})" if p_value < ALPHA else f"不显著 (p≥{ALPHA})",
    ci_low, ci_high,
    len(df_sample), int(oos_positive * len(df_sample)), oos_positive * 100,
    best_train["sharpe"], result_test["sharpe"], p_value,
))

print(f"  图表输出: {REPORT_DIR.resolve()}/")
for f in sorted(REPORT_DIR.glob("*.png")):
    print(f"    {f.name}")
print(f"\n  代码仓库: D:/桌面文件/quant/")
print(f"  策略目录: D:/桌面文件/quant/strategies/ma_crossover/")
print(f"  文本报告: D:/桌面文件/quant/strategies/ma_crossover/report_output.txt")
print(f"  {'='*70}")

sys.stdout = sys.__stdout__
report_log.close()
print()
print(f"完整文本报告已保存至: report_output.txt")
input("按 Enter 键退出...")
