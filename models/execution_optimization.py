"""
models/execution_optimization.py — 调仓频率 + 动态仓位优化（修正版）。

全部基于修正后的 build_portfolio（权重追踪真实重叠组合）。

1. 调仓频率：hold_days = 5/8/10/15/20，看夏普/回撤/年化随持有期变化。
2. 动态仓位：按 market_vol_20d 中位数分高/低波动，高波动降仓（100→70→50→30%）。

用法:
    cd D:/桌面文件/quant
    python models/execution_optimization.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.portfolio_backtest import load_data, build_portfolio, performance_metrics

MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
COST = 0.003


def main():
    print("=" * 70)
    print("调仓频率 + 动态仓位优化（修正版：权重追踪真实重叠组合）")
    print("=" * 70)

    pred_lgb, _, close_matrix, _, market_vol, _ = load_data()

    # ── 1. 调仓频率 ──
    hold_grid = [5, 8, 10, 15, 20]
    freq_rows = []
    print(f"\n{'hold_days':>9} | {'LS夏普':>7} {'LS年化':>8} {'LS回撤':>8} | {'LO夏普':>7} {'LO年化':>8} {'LO回撤':>8}")
    print("-" * 70)
    freq_curves = {}
    for hd in hold_grid:
        ls = build_portfolio(pred_lgb, close_matrix, long_only=False, cost=COST, hold_days=hd)
        lo = build_portfolio(pred_lgb, close_matrix, long_only=True, cost=COST, hold_days=hd)
        mls = performance_metrics(ls["port_ret"])
        mlo = performance_metrics(lo["port_ret"])
        freq_rows.append({"hold_days": hd,
                          "ls_sharpe": mls["sharpe"], "ls_annual": mls["annual"], "ls_mdd": mls["mdd"],
                          "lo_sharpe": mlo["sharpe"], "lo_annual": mlo["annual"], "lo_mdd": mlo["mdd"]})
        freq_curves[hd] = ls["cum"]
        print(f"{hd:>9} | {mls['sharpe']:>7.3f} {mls['annual']:>+7.1%} {mls['mdd']:>+7.1%} | "
              f"{mlo['sharpe']:>7.3f} {mlo['annual']:>+7.1%} {mlo['mdd']:>+7.1%}")
    freq_df = pd.DataFrame(freq_rows)

    # ── 2. 动态仓位（hold_days=10）──
    HD = 10
    vol_median = market_vol.median()
    # 高波动(>中位数)降仓，低波动满仓
    print(f"\n动态仓位（hold_days={HD}，market_vol 中位数={vol_median:.4f}）:")
    print(f"{'规则':>18} | {'夏普':>7} {'年化':>8} {'回撤':>8}")
    print("-" * 48)
    dyn_rows = []
    dyn_curves = {}
    for high_scale in [1.0, 0.7, 0.5, 0.3]:
        scale = pd.Series(1.0, index=market_vol.index)
        scale[market_vol > vol_median] = high_scale
        port = build_portfolio(pred_lgb, close_matrix, long_only=False, cost=COST,
                               hold_days=HD, position_scale=scale)
        m = performance_metrics(port["port_ret"])
        label = "满仓(基线)" if high_scale == 1.0 else f"高波动降至{high_scale:.0%}"
        dyn_rows.append({"rule": label, "high_scale": high_scale,
                         "sharpe": m["sharpe"], "annual": m["annual"], "mdd": m["mdd"]})
        dyn_curves[label] = port["cum"]
        print(f"{label:>18} | {m['sharpe']:>7.3f} {m['annual']:>+7.1%} {m['mdd']:>+7.1%}")
    dyn_df = pd.DataFrame(dyn_rows)

    # ── 图表（2×2）──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 夏普 vs hold_days
    ax = axes[0, 0]
    ax.plot(freq_df["hold_days"], freq_df["ls_sharpe"], "o-", color="#2ca02c", label="Long-Short")
    ax.plot(freq_df["hold_days"], freq_df["lo_sharpe"], "s-", color="#1f77b4", label="Long-only")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="SR=1.0")
    ax.set_xlabel("hold_days"); ax.set_ylabel("夏普比率")
    ax.set_title("调仓频率 → 夏普（真实权重追踪）")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. 回撤 vs hold_days
    ax = axes[0, 1]
    ax.plot(freq_df["hold_days"], [m * 100 for m in freq_df["ls_mdd"]], "o-", color="#2ca02c", label="Long-Short")
    ax.plot(freq_df["hold_days"], [m * 100 for m in freq_df["lo_mdd"]], "s-", color="#1f77b4", label="Long-only")
    ax.set_xlabel("hold_days"); ax.set_ylabel("最大回撤 (%)")
    ax.set_title("调仓频率 → 最大回撤")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 3. 动态仓位净值曲线
    ax = axes[1, 0]
    for label, cum in dyn_curves.items():
        ax.plot(cum.index, cum.values, linewidth=1.2, label=label)
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_title(f"动态仓位净值（hold_days={HD}，Long-Short）")
    ax.set_ylabel("净值"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. 动态仓位夏普/回撤柱状
    ax = axes[1, 1]
    x = np.arange(len(dyn_df))
    w = 0.35
    ax.bar(x - w/2, dyn_df["sharpe"], w, color="#2ca02c", label="夏普")
    ax.set_ylabel("夏普", color="#2ca02c")
    ax2 = ax.twinx()
    ax2.bar(x + w/2, [-m * 100 for m in dyn_df["mdd"]], w, color="#d62728", label="回撤%")
    ax2.set_ylabel("最大回撤 (%)", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(dyn_df["rule"], rotation=20, ha="right", fontsize=8)
    ax.set_title("动态仓位：夏普 vs 回撤")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "execution_optimization.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n图表保存: {fig_path}")

    freq_df.to_csv(MODEL_DIR / "execution_optimization_freq.csv", index=False)
    dyn_df.to_csv(MODEL_DIR / "execution_optimization_dyn.csv", index=False)
    print("结果保存: execution_optimization_freq.csv / _dyn.csv")


if __name__ == "__main__":
    main()
