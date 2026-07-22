"""
batch_mc.py — 全市场蒙特卡洛检验：对沪深 300 每只股票跑 bootstrap，
判断 MA 交叉策略的夏普比率是否统计显著。

输出：
- 每只股票的 p-value（H0: SR ≤ 0）
- p-value 分布直方图（若策略无效 → 均匀分布 U[0,1]）
- 通过检验（p < 0.05）的股票列表
"""

import sys
sys.path.insert(0, "../..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import time
import os

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

from data.fetcher import cache_summary, load_daily
from backtest.engine import run

# ── 参数 ─────────────────────────────────────────────────
FAST, SLOW = 4, 100
N_PATHS = 200               # 每只股票的 bootstrap 次数
ALPHA = 0.05                # 显著性阈值

# ── 单只检验函数 ─────────────────────────────────────────
def test_stock(symbol: str) -> dict | None:
    """对单只股票跑 bootstrap 检验，返回统计量字典。"""
    df = load_daily(symbol)
    if df is None or len(df) < 500:
        return None
    close = df["close"].astype(float)
    daily_ret = close.pct_change().dropna()
    if len(daily_ret) < 200:
        return None
    rets = daily_ret.values
    n_days = len(rets)

    # 真实路径
    fast_ma = close.rolling(FAST).mean()
    slow_ma = close.rolling(SLOW).mean()
    signal = (fast_ma > slow_ma).astype(int)
    m = run(close, signal)
    real_sharpe = m["sharpe"]

    # Bootstrap
    rng = np.random.default_rng()
    idx_pool = np.arange(n_days)
    sharpe_boot = np.empty(N_PATHS)

    for i in range(N_PATHS):
        idx = rng.choice(idx_pool, size=n_days, replace=True)
        synth_rets = rets[idx]
        synth_close = pd.Series(
            100 * (1 + synth_rets).cumprod(),
            index=daily_ret.index,
        )
        f = synth_close.rolling(FAST).mean()
        s = synth_close.rolling(SLOW).mean()
        sig = (f > s).astype(int)
        bm = run(synth_close, sig)
        sharpe_boot[i] = bm["sharpe"]

    # 统计
    boot_mean = sharpe_boot.mean()
    boot_std = sharpe_boot.std()
    percentile = stats.percentileofscore(sharpe_boot, real_sharpe)
    p_value = 1 - (sharpe_boot > 0).mean()  # H0: SR ≤ 0
    ci_low, ci_high = np.percentile(sharpe_boot, [2.5, 97.5])

    return {
        "symbol": symbol,
        "real_sr": real_sharpe,
        "boot_mean": boot_mean,
        "boot_std": boot_std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "percentile": percentile,
        "p_value": p_value,
        "n_days": n_days,
        "real_ann": m["ann_return"],
        "real_dd": m["max_drawdown"],
        "n_trades": m["n_trades"],
    }


# ── 批量运行 ─────────────────────────────────────────────
symbols = sorted([
    p.name.replace(".csv", "") for p in os.scandir("../../data/cache") if p.name.endswith(".csv")
])
print(f"待检验: {len(symbols)} 只股票")
print(f"参数: MA{FAST}/{SLOW}, bootstrap N={N_PATHS}\n")

results = []
t_start = time.time()
for i, sym in enumerate(symbols):
    t0 = time.time()
    r = test_stock(sym)
    elapsed = time.time() - t0
    if r is None:
        print(f"[{i+1}/{len(symbols)}] {sym} 跳过 (数据不足)", flush=True)
        continue
    results.append(r)
    sig_mark = "**" if r["p_value"] < ALPHA else ""
    print(f"[{i+1}/{len(symbols)}] {sym}  "
          f"SR={r['real_sr']:+.3f}  "
          f"boot_mean={r['boot_mean']:+.3f}  "
          f"p={r['p_value']:.3f}  "
          f"{sig_mark}  "
          f"({elapsed:.1f}s)", flush=True)

elapsed = time.time() - t_start
df = pd.DataFrame(results)
n_sig = (df["p_value"] < ALPHA).sum()

print(f"\n── 汇总 ──")
print(f"完成: {len(results)} 只, 耗时: {elapsed:.0f}s ({elapsed/len(results):.1f}s/只)")
print(f"通过检验 (p<{ALPHA}): {n_sig}/{len(results)} ({n_sig/len(results)*100:.1f}%)")
print(f"预期随机通过率: {ALPHA*100:.0f}% (Bonferroni 校正后: {ALPHA/len(results):.4f})")

# Bonferroni 校正
bf_alpha = ALPHA / len(results)
n_bf = (df["p_value"] < bf_alpha).sum()
print(f"Bonferroni 通过 (p<{bf_alpha:.4f}): {n_bf}/{len(results)}")

# ── 可视化 ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: p-value 分布直方图（策略无效 → 均匀分布）
ax = axes[0, 0]
ax.hist(df["p_value"], bins=20, range=(0, 1), color="#2196F3", alpha=0.7, edgecolor="white")
ax.axhline(len(df) / 20, color="#999", linestyle="--", linewidth=1, label="均匀分布期望")
ax.axvline(ALPHA, color="#F44336", linestyle="--", linewidth=1, label=f"α={ALPHA}")
ax.set_xlabel("p-value")
ax.set_ylabel("Count")
ax.set_title(f"p-value 分布 (N={len(df)})")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 图2: 真实 SR vs bootstrap 均值
ax = axes[0, 1]
ax.scatter(df["boot_mean"], df["real_sr"], c=df["p_value"],
           cmap="RdYlBu_r", s=8, alpha=0.6, vmin=0, vmax=0.5)
ax.plot([-0.5, 1], [-0.5, 1], color="#999", linewidth=0.5, linestyle="--")
ax.set_xlabel("Bootstrap 均值 SR")
ax.set_ylabel("真实 SR")
ax.set_title("真实夏普 vs Bootstrap 夏普（颜色=p值）")
ax.grid(True, alpha=0.3)
ax.axhline(0, color="#333", linewidth=0.5)
ax.axvline(0, color="#333", linewidth=0.5)
fig.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(0, 0.5), cmap="RdYlBu_r"),
    ax=ax, shrink=0.8, label="p-value",
)

# 图3: 显著股票的 SR 条形图
ax = axes[1, 0]
sig = df[df["p_value"] < ALPHA].sort_values("real_sr", ascending=True)
if len(sig) > 0:
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in sig["real_sr"]]
    ax.barh(range(len(sig)), sig["real_sr"], color=colors, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(sig["symbol"], fontsize=7)
    ax.axvline(0, color="#333", linewidth=0.5)
    ax.set_xlabel("Sharpe Ratio")
    ax.set_title(f"显著股票 (p<{ALPHA}, n={len(sig)})")
else:
    ax.text(0.5, 0.5, "无显著股票", transform=ax.transAxes, ha="center", fontsize=14)
ax.grid(True, alpha=0.3, axis="x")

# 图4: p-value 排序 (Manhattan-like plot)
ax = axes[1, 1]
df_sorted = df.sort_values("p_value")
colors = ["#F44336" if p < ALPHA else "#2196F3" for p in df_sorted["p_value"]]
ax.bar(range(len(df_sorted)), df_sorted["p_value"], color=colors, alpha=0.7, width=1)
ax.axhline(ALPHA, color="#F44336", linestyle="--", linewidth=1, label=f"α={ALPHA}")
ax.axhline(bf_alpha, color="#FF9800", linestyle="--", linewidth=0.8, label=f"Bonferroni")
ax.set_xlabel("Stock Rank")
ax.set_ylabel("p-value")
ax.set_title("p-value 排序 (Manhattan)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("batch_mc.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 输出显著列表 ─────────────────────────────────────────
print(f"\n显著股票 (p<{ALPHA}):")
if len(sig) > 0:
    for _, r in sig.iterrows():
        print(f"  {r['symbol']:>6}  SR={r['real_sr']:+.3f}  "
              f"p={r['p_value']:.4f}  "
              f"年化={r['real_ann']*100:+.1f}%  "
              f"回撤={r['real_dd']*100:.1f}%")
print(f"\n图表已保存: batch_mc.png")
