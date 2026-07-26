"""
models/methodology_correction.py — overlapping returns 平滑陷阱的诊断与修正。

生成「方法论修正」章节的数据与图表：
  1. 错误法(build_portfolio_naive) vs 正确法(build_portfolio) across hold_days
  2. 夏普 / sqrt(hold_days) 收敛 → 证实平滑放大
  3. 日收益 lag-1 自相关：错误法趋近 1，正确法保持低位
  4. Newey-West 修正夏普

用法:
    cd D:/桌面文件/quant
    python models/methodology_correction.py
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

from models.portfolio_backtest import (
    load_data, build_portfolio, build_portfolio_naive, performance_metrics,
)

MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
COST = 0.003


def newey_west_sharpe(ret: pd.Series, lags: int) -> float:
    r = ret.dropna().values
    n = len(r)
    mu = r.mean()
    d = r - mu
    gamma0 = (d @ d) / n
    var = gamma0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * (d[lag:] @ d[:-lag]) / n
    return mu / np.sqrt(var) * np.sqrt(252)


def main():
    print("=" * 70)
    print("方法论修正：overlapping returns 平滑陷阱")
    print("=" * 70)

    pred_lgb, _, close_matrix, _, _, _ = load_data()

    hold_grid = [1, 2, 5, 8, 10, 15, 20]
    rows = []
    naive_curves, correct_curves = {}, {}
    print(f"\n{'hold':>4} | {'错误夏普':>8} {'/sqrt(h)':>9} {'错误lag1':>8} | "
          f"{'正确夏普':>8} {'正确lag1':>8} {'NW修正':>7}")
    print("-" * 66)
    for hd in hold_grid:
        naive = build_portfolio_naive(pred_lgb, close_matrix, long_only=False, cost=COST, hold_days=hd)
        correct = build_portfolio(pred_lgb, close_matrix, long_only=False, cost=COST, hold_days=hd)
        mn = performance_metrics(naive["port_ret"])
        mc = performance_metrics(correct["port_ret"])
        ac_n = naive["port_ret"].autocorr(lag=1)
        ac_c = correct["port_ret"].autocorr(lag=1)
        nw = newey_west_sharpe(naive["port_ret"], max(hd - 1, 1))
        rows.append({"hold_days": hd, "naive_sr": mn["sharpe"], "naive_ann": mn["annual"],
                     "naive_lag1": ac_n, "correct_sr": mc["sharpe"], "correct_ann": mc["annual"],
                     "correct_lag1": ac_c, "nw_sr": nw})
        naive_curves[hd] = naive["cum"]
        correct_curves[hd] = correct["cum"]
        print(f"{hd:>4} | {mn['sharpe']:>8.3f} {mn['sharpe']/np.sqrt(hd):>9.3f} {ac_n:>8.3f} | "
              f"{mc['sharpe']:>8.3f} {ac_c:>8.3f} {nw:>7.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(MODEL_DIR / "methodology_correction.csv", index=False)

    # ── 图表（2×2）──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 错误 vs 正确夏普
    ax = axes[0, 0]
    ax.plot(df["hold_days"], df["naive_sr"], "o-", color="#d62728", label="错误法(移动平均)")
    ax.plot(df["hold_days"], df["correct_sr"], "s-", color="#2ca02c", label="正确法(权重追踪)")
    ax.plot(df["hold_days"], df["nw_sr"], "^--", color="#ff7f0e", alpha=0.7, label="错误法+NW修正")
    ax.axhline(3.0, color="gray", linestyle=":", alpha=0.6, label="CLAUDE.md |SR|<3 红线")
    ax.set_xlabel("hold_days"); ax.set_ylabel("夏普比率")
    ax.set_title("夏普：错误法虚高 ~8x，正确法回到合理区间")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 2. sqrt(hold_days) 收敛
    ax = axes[0, 1]
    ax.plot(df["hold_days"], df["naive_sr"] / np.sqrt(df["hold_days"]), "o-", color="#d62728")
    ax.set_xlabel("hold_days"); ax.set_ylabel("错误夏普 / √hold_days")
    ax.set_title("错误夏普/√hold_days 收敛 → 证实机械平滑放大")
    ax.grid(True, alpha=0.3)

    # 3. lag-1 自相关
    ax = axes[1, 0]
    ax.plot(df["hold_days"], df["naive_lag1"], "o-", color="#d62728", label="错误法")
    ax.plot(df["hold_days"], df["correct_lag1"], "s-", color="#2ca02c", label="正确法")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("hold_days"); ax.set_ylabel("日收益 lag-1 自相关")
    ax.set_title("错误法 lag-1→1（移动平均特征），正确法保持低位")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # 4. 累计净值对比（hold_days=10，对数）
    ax = axes[1, 1]
    ax.plot(naive_curves[10].index, naive_curves[10].values, color="#d62728",
            linewidth=1.2, label=f"错误法 (SR={df[df.hold_days==10]['naive_sr'].iloc[0]:.2f})")
    ax.plot(correct_curves[10].index, correct_curves[10].values, color="#2ca02c",
            linewidth=1.2, label=f"正确法 (SR={df[df.hold_days==10]['correct_sr'].iloc[0]:.2f})")
    ax.axhline(1, color="black", linewidth=0.5)
    ax.set_yscale("log")
    ax.set_title("累计净值（hold_days=10，对数坐标）")
    ax.set_ylabel("净值 (log)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "methodology_correction.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\n图表保存: {fig_path}")
    print("结果保存: models/methodology_correction.csv")


if __name__ == "__main__":
    main()
