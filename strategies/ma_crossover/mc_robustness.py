"""
mc_robustness.py — 蒙特卡洛稳健性检验：对最优策略做 bootstrap 重采样，
生成合成价格路径，评估夏普比率在扰动下的分布。

方法：
- 对历史日收益率做有放回重采样（保持同期相关性）
- 生成 N 条合成价格路径
- 每条路径独立回测 MA4/100 策略
- 得到夏普分布 → 判断策略是否统计显著

物理类比：统计力学里的系综平均——真实路径只是一个微观态，
bootstrap 采样构造的合成路径是同一哈密顿量下的其他微观态。
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily
from backtest.engine import run

# ── 参数 ─────────────────────────────────────────────────
SYMBOL = "000001"
FAST, SLOW = 4, 100          # 扫描阶段的最优参数
N_PATHS = 1000                # bootstrap 样本数
INIT_PRICE = 100              # 归一化初始价格

# ── 1. 加载真实数据 & 基准策略 ───────────────────────────
close = load_daily(SYMBOL)["close"].astype(float)
daily_ret = close.pct_change().dropna()

# 真实路径回测
fast_ma = close.rolling(FAST).mean()
slow_ma = close.rolling(SLOW).mean()
signal = (fast_ma > slow_ma).astype(int)
real_result = run(close, signal)
real_sharpe = real_result["sharpe"]
real_ann = real_result["ann_return"]

print(f"品种: {SYMBOL}  |  MA{FAST}/{SLOW}")
print(f"真实路径: SR={real_sharpe:.3f}, 年化={real_ann*100:.1f}%")
print(f"Bootstrap: {N_PATHS} 条合成路径\n")

# ── 2. Bootstrap 重采样 ──────────────────────────────────
rets = daily_ret.values
n_days = len(rets)
rng = np.random.default_rng(42)

sharpe_samples = np.empty(N_PATHS)
ann_ret_samples = np.empty(N_PATHS)
max_dd_samples = np.empty(N_PATHS)

for i in range(N_PATHS):
    # 有放回采样日收益率
    idx = rng.integers(0, n_days, size=n_days)
    synth_rets = rets[idx]

    # 合成价格路径
    synth_close = pd.Series(
        INIT_PRICE * (1 + synth_rets).cumprod(),
        index=daily_ret.index,
    )

    # 策略信号
    f = synth_close.rolling(FAST).mean()
    s = synth_close.rolling(SLOW).mean()
    sig = (f > s).astype(int)

    m = run(synth_close, sig)
    sharpe_samples[i] = m["sharpe"]
    ann_ret_samples[i] = m["ann_return"]
    max_dd_samples[i] = m["max_drawdown"]

# ── 3. 统计分析 ──────────────────────────────────────────
# 夏普 > 0 的比例（即策略在合成路径上盈利的概率）
p_positive = (sharpe_samples > 0).mean()

# 真实夏普在分布中的分位数
percentile = stats.percentileofscore(sharpe_samples, real_sharpe)

# 95% 置信区间
ci_low, ci_high = np.percentile(sharpe_samples, [2.5, 97.5])

print(f"── 夏普比率分布 ──")
print(f"均值: {sharpe_samples.mean():.4f}")
print(f"中位数: {np.median(sharpe_samples):.4f}")
print(f"标准差: {sharpe_samples.std():.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
print(f"真实 SR = {real_sharpe:.3f}  (分位数: {percentile:.1f}%)")
print(f"正夏普概率: {p_positive*100:.1f}%")

# 统计检验：零假设 H0 = 夏普 ≤ 0（策略无效）
# bootstrap 下 SR > 0 的概率即 p-value 的补
print(f"p-value (H0: SR≤0): {1 - p_positive:.4f}")

# ── 4. 可视化 ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 夏普分布直方图
ax = axes[0, 0]
ax.hist(sharpe_samples, bins=50, color="#2196F3", alpha=0.7, edgecolor="white", density=True)
ax.axvline(real_sharpe, color="#F44336", linewidth=2, linestyle="--", label=f"真实 SR={real_sharpe:.3f}")
ax.axvline(0, color="#999", linewidth=1, linestyle=":")
ax.axvline(ci_low, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(ci_high, color="#4CAF50", linewidth=1, linestyle="--", alpha=0.7)
ax.fill_betweenx([0, ax.get_ylim()[1]], ci_low, ci_high, color="#4CAF50", alpha=0.05)
ax.set_xlabel("Sharpe Ratio")
ax.set_ylabel("Density")
ax.set_title(f"Bootstrap 夏普分布 (N={N_PATHS})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 图2: 合成路径 vs 真实路径
ax = axes[0, 1]
# 随机选 50 条合成路径, 半透明
sample_paths = rng.choice(N_PATHS, size=50, replace=False)
for idx in sample_paths:
    synth_rets = rets[rng.integers(0, n_days, size=n_days)]
    synth_eq = (1 + synth_rets).cumprod()
    ax.plot(daily_ret.index, synth_eq, color="#2196F3", linewidth=0.2, alpha=0.3)
# 真实路径
real_eq = (1 + daily_ret.values).cumprod()
ax.plot(daily_ret.index, real_eq, color="#F44336", linewidth=1.2, label="真实路径")
ax.set_ylabel("Cumulative Return")
ax.set_title("合成路径 vs 真实路径")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 图3: 年化收益 vs 最大回撤 散点图（每点一条合成路径+策略）
ax = axes[1, 0]
sc = ax.scatter(max_dd_samples * 100, ann_ret_samples * 100,
                c=sharpe_samples, cmap="RdYlBu_r", s=3, alpha=0.5)
ax.scatter([real_result["max_drawdown"] * 100], [real_result["ann_return"] * 100],
           marker="*", color="black", s=200, zorder=5, label="真实")
ax.set_xlabel("Max Drawdown (%)")
ax.set_ylabel("Annual Return (%)")
ax.set_title("收益-回撤散点图（颜色=夏普）")
ax.axhline(0, color="#999", linewidth=0.5)
ax.axvline(0, color="#999", linewidth=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.colorbar(sc, ax=ax, shrink=0.8)

# 图4: 真实路径策略权益 vs benchmark
ax = axes[1, 1]
equity = real_result["equity"] * 100_000
benchmark = real_result["benchmark"] * 100_000
ax.plot(close.index, equity, color="#4CAF50", linewidth=0.8, label="策略")
ax.plot(close.index, benchmark, color="#999", linewidth=0.6, alpha=0.7, label="买入持有")
# 回撤填充
dd = equity / equity.cummax() - 1
ax2 = ax.twinx()
ax2.fill_between(close.index, 0, dd * 100, color="#F44336", alpha=0.15, linewidth=0)
ax2.set_ylabel("Drawdown (%)", color="#F44336")
ax2.tick_params(axis="y", labelcolor="#F44336")
ax.set_ylabel("Equity (¥)")
ax.set_title(f"{SYMBOL}  MA{FAST}/{SLOW}  权益曲线")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mc_robustness.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n图表已保存: mc_robustness.png")
