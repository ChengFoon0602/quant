"""
scan.py — 参数相图扫描：对均线交叉策略的 (FAST, SLOW) 网格做回测，
画出夏普比率热力图，寻找稳健参数区域。

物理类比：策略参数空间 = 相空间，夏普比率 = order parameter，
平坦高原 = 稳健相，孤立尖峰 = 过拟合假信号。
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import product

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import load_daily
from backtest.engine import run

# ── 参数空间 ─────────────────────────────────────────────
SYMBOL = "000001"
FAST_RANGE = range(2, 61, 2)     # 2, 4, 6, ..., 60
SLOW_RANGE = range(10, 121, 5)   # 10, 15, 20, ..., 120

# ── 扫描 ─────────────────────────────────────────────────
close = load_daily(SYMBOL)["close"].astype(float)
print(f"{SYMBOL}  |  {close.index[0].date()} ~ {close.index[-1].date()}  |  {len(close)} 条")

results = []
for fast, slow in product(FAST_RANGE, SLOW_RANGE):
    if fast >= slow:
        continue
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    signal = (fast_ma > slow_ma).astype(int)
    m = run(close, signal)
    results.append({"fast": fast, "slow": slow, **m})

df = pd.DataFrame(results)
print(f"参数组合: {len(df)} 个")

# ── 相图矩阵 ─────────────────────────────────────────────
sharpe_grid = df.pivot_table(index="slow", columns="fast", values="sharpe")
ann_ret_grid = df.pivot_table(index="slow", columns="fast", values="ann_return")

# ── 可视化 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 图1: 夏普比热力图
im = axes[0].pcolormesh(
    sharpe_grid.columns, sharpe_grid.index, sharpe_grid.values,
    cmap="RdYlBu_r", shading="auto",
    vmin=-0.5, vmax=0.5,
)
axes[0].set_xlabel("FAST")
axes[0].set_ylabel("SLOW")
axes[0].set_title(f"{SYMBOL}  MA 交叉 — 夏普比率")
axes[0].invert_yaxis()
# 标注最优
best = df.loc[df["sharpe"].idxmax()]
axes[0].plot(best["fast"], best["slow"], marker="*", color="black", markersize=12)
fig.colorbar(im, ax=axes[0], shrink=0.8)

# 图2: 年化收益率热力图
im2 = axes[1].pcolormesh(
    ann_ret_grid.columns, ann_ret_grid.index, ann_ret_grid.values * 100,
    cmap="RdYlBu_r", shading="auto",
)
axes[1].set_xlabel("FAST")
axes[1].set_ylabel("SLOW")
axes[1].set_title("年化收益率 (%)")
axes[1].invert_yaxis()
fig.colorbar(im2, ax=axes[1], shrink=0.8)

# 图3: 最优参数权益曲线
best_fast, best_slow = int(best["fast"]), int(best["slow"])
fast_ma = close.rolling(best_fast).mean()
slow_ma = close.rolling(best_slow).mean()
signal = (fast_ma > slow_ma).astype(int)
m = run(close, signal)
equity = m["equity"] * 100_000
benchmark = m["benchmark"] * 100_000

axes[2].plot(close.index, equity, color="#4CAF50", linewidth=0.8,
             label=f"MA{best_fast}/{best_slow}")
axes[2].plot(close.index, benchmark, color="#999", linewidth=0.6, alpha=0.7,
             label="买入持有")
axes[2].fill_between(close.index, equity, benchmark,
                     where=(equity >= benchmark), color="#4CAF50", alpha=0.08)
axes[2].fill_between(close.index, equity, benchmark,
                     where=(equity < benchmark), color="#F44336", alpha=0.08)
axes[2].set_ylabel("Equity (¥)")
axes[2].set_title(f"最优 MA{best_fast}/{best_slow}  |  "
                  f"SR={best['sharpe']:.3f}  |  "
                  f"年化={best['ann_return']*100:.1f}%  |  "
                  f"回撤={best['max_drawdown']*100:.1f}%")
axes[2].legend(fontsize=8)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("scan_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 稳健性分析 ───────────────────────────────────────────
# 找出夏普 > 0 且平坦的区域（> 50% 的相邻网格都有正夏普 → 稳健相）
sharpe_vals = sharpe_grid.values
# 有效区域（非 NaN）
valid = ~np.isnan(sharpe_vals)
# 正夏普区域占比
positive_ratio = (sharpe_vals[valid] > 0).mean()
# Top-5 参数
top5 = df.nlargest(5, "sharpe")[["fast", "slow", "sharpe", "ann_return", "max_drawdown", "n_trades"]]

print(f"\n── 相图分析 ──")
print(f"正夏普区域占比: {positive_ratio*100:.1f}%")
print(f"最优夏普: {best['sharpe']:.3f}  (MA{best_fast}/{best_slow})")
print(f"\nTop-5 参数组合:")
for _, r in top5.iterrows():
    print(f"  MA{int(r['fast']):>2}/{int(r['slow']):>3}  "
          f"SR={r['sharpe']:.3f}  "
          f"年化={r['ann_return']*100:>6.1f}%  "
          f"回撤={r['max_drawdown']*100:>5.1f}%  "
          f"交易={int(r['n_trades']):>3}次")

print(f"\n图表已保存: scan_heatmap.png")
