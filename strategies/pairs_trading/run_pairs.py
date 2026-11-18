"""
strategies/pairs_trading/run_pairs.py — 配对交易全量回测。

产出：控制台指标 + `figures/01_pairs_equity.png`、`figures/02_coint_count.png`。
执行：`python strategies/pairs_trading/run_pairs.py`（约 2 分钟，1468 只）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.fetcher import load_daily
from risk.cost_model import ROUND_TRIP
from risk.portfolio import calculate_metrics
from strategies.pairs_trading.pairs import (
    backtest_pairs,
    build_candidate_pairs,
    rolling_cointegration,
    summarize_trades,
)
from viz.plotting import plot_equity_curve

FIG = Path(__file__).parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_universe() -> pd.DataFrame:
    mem = pd.read_csv(ROOT / "data/cache_meta/zz500_membership.csv")
    syms = [str(c) for c in mem.columns if c != "Unnamed: 0"]
    px = {}
    for s in syms:
        d = load_daily(s, adjust="1")
        if d is not None and len(d) > 1500:
            px[s] = d["close"]
    return pd.DataFrame(px).sort_index()


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("  A 股配对交易可行性：全量回测")
    print("=" * 78)

    close = load_universe()
    print(f"[数据] {close.shape[1]} 只  {close.index.min().date()} ~ {close.index.max().date()}"
          f"  ({time.time()-t0:.0f}s)")

    ind = pd.read_csv(ROOT / "data/cache_meta/industry.csv",
                      dtype={"symbol": str}).set_index("symbol")["industry"]
    pairs = build_candidate_pairs(close, ind, min_corr=0.60, max_pairs_per_symbol=6)
    print(f"[配对] 候选 {len(pairs)} 对  ({time.time()-t0:.0f}s)")

    ok, alpha, beta, mu, sd = rolling_cointegration(
        close, pairs, window=250, step=20)
    print(f"[协整] 完成  ({time.time()-t0:.0f}s)")

    res, trades = backtest_pairs(
        close, pairs, ok, beta, alpha, mu, sd,
        z_entry=2.0, max_hold=40, round_trip_cost=ROUND_TRIP, return_trades=True)
    print(f"[回测] 完成  ({time.time()-t0:.0f}s)")

    td = summarize_trades(trades)
    m, g = calculate_metrics(res["port_ret"]), calculate_metrics(res["gross_ret"])

    print("\n" + "-" * 78)
    print(f"  完整回合 {len(td)}   平均持有 {td['days'].mean():.1f} 天   "
          f"平均同时持仓 {res['n_active'].mean():.1f} 对")
    print(f"  平仓原因: {dict(td['reason'].value_counts())}")
    print(f"  毛年化 {g['annual_return']:+.2%}   毛夏普 {g['sharpe']:+.3f}")
    print(f"  净年化 {m['annual_return']:+.2%}   净夏普 {m['sharpe']:+.3f}   "
          f"回撤 {m['max_drawdown']:+.2%}")
    print(f"  日胜率 {(res['port_ret'] > 0).mean():.2%}   "
          f"成本累计 {res['cost'].sum():.2f}")
    print("-" * 78)

    verdict = "否定" if g["sharpe"] < 0 else "存疑"
    print(f"\n  判决：{verdict}（毛夏普 {g['sharpe']:+.3f}，"
          f"价差{'不' if g['sharpe'] < 0 else ''}具备可交易的回归特性）")

    # ── 图 1：毛/净净值 ──
    plot_equity_curve(
        {"净（双边 0.1%）": res["cum"],
         "毛（未扣成本）": (1.0 + res["gross_ret"]).cumprod()},
        title="A 股配对交易：毛 vs 净累计净值（zz500，2010-2025）",
        save_path=FIG / "01_pairs_equity.png",
    )

    # ── 图 2：协整通过数与活跃持仓 ──
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(ok.index, ok.sum(axis=1), lw=0.9, color="#4C72B0", label="协整通过配对数")
    ax2 = ax.twinx()
    ax2.plot(res.index, res["n_active"], lw=0.6, color="#C44E52", alpha=0.7,
             label="活跃持仓对数")
    ax.set_ylabel("协整通过配对（个）")
    ax2.set_ylabel("活跃持仓（对）")
    ax.set_title("协整关系数量与持仓规模：关系频繁断裂导致持仓被动清空")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "02_coint_count.png", dpi=150)
    plt.close(fig)

    print(f"\n图表已保存至 {FIG}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
