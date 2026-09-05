"""
models/rerun_portfolio_backtest.py — 用 hs300 X_matrix 重跑组合回测（口径收口后）。

背景：
  models/portfolio_backtest.py::load_data 依赖 strategies/feature_selection/X_matrix.csv
  （已因 gitignore 不在工作区），但等价数据在 strategies/hs300_crowding_trial/X_matrix_hs300.csv
  （同为 CSI300 PIT 特征矩阵：790 只 × 16 alpha + 2 market 特征）。

本脚本显式加载 hs300 特征矩阵，复用 portfolio_backtest 的组合构建/绩效/绘图函数，
重跑口径收口后的组合回测，产出更新后的 summary + 图表。

用法: python models/rerun_portfolio_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetcher import load_daily
from models.labels import align_X_y, build_labels
from models.portfolio_backtest import (
    build_portfolio,
    performance_metrics,
    block_bootstrap_sharpe,
    plot_results,
    TOP_Q, BOTTOM_Q, FWD_DAYS, N_BOOT, BLOCK_SIZE, MARKET_COLS,
)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = Path(__file__).parent
FIGURES_DIR = MODEL_DIR / "figures"
HS300_X = PROJECT_ROOT / "strategies" / "hs300_crowding_trial" / "X_matrix_hs300.csv"


def load_data():
    """加载 OOF 预测 + hs300 X_matrix + close_matrix。"""
    pred_path = MODEL_DIR / "oof_predictions.csv"
    pred_lgb = pd.read_csv(pred_path, index_col=0, parse_dates=True)

    X_raw = pd.read_csv(HS300_X, dtype=str)
    X_raw["date"] = pd.to_datetime(X_raw["date"])
    stock_col = X_raw.columns[1]
    X_long = X_raw.set_index(["date", stock_col]).astype(float)

    alpha001 = X_long["alpha001"].unstack(level=1)
    market_vol = X_long["market_vol_20d"].unstack(level=1).iloc[:, 0]

    symbols = sorted(X_long.index.get_level_values(1).unique())
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= "2010-01-01") & (df.index <= "2025-12-31"), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    close_matrix = pd.DataFrame(close_data).sort_index()

    common_stocks = pred_lgb.columns.intersection(close_matrix.columns).intersection(alpha001.columns)
    pred_lgb = pred_lgb[common_stocks]
    alpha001 = alpha001[common_stocks]
    close_matrix = close_matrix[common_stocks]

    member_mask = pd.Series(True, index=X_long.index).unstack(fill_value=False)
    daily_ret = close_matrix.pct_change()
    mask_aligned = member_mask.reindex_like(daily_ret).fillna(False).astype(bool)
    market_ret = daily_ret.where(mask_aligned).mean(axis=1).dropna()

    return pred_lgb, alpha001, close_matrix, market_ret, market_vol, X_long


def main():
    print("=" * 70)
    print("组合回测重跑（口径收口后：pct_change + 铁律 0.1% 成本）")
    print("=" * 70)

    pred_lgb, alpha001, close_matrix, market_ret, market_vol, X_long = load_data()

    print("\n构建组合（铁律成本：买 0.026% / 卖 0.076%）...")
    lgb_ls = build_portfolio(pred_lgb, close_matrix, long_only=False)
    lgb_lo = build_portfolio(pred_lgb, close_matrix, long_only=True)
    a001_ls = build_portfolio(alpha001, close_matrix, long_only=False)

    market_cum = (1 + market_ret.fillna(0)).cumprod()
    results = {
        "LightGBM LS": lgb_ls,
        "LightGBM Long-only": lgb_lo,
        "alpha001 LS": a001_ls,
    }

    print("\n绩效汇总:")
    boot_dist = {}
    for name, df in results.items():
        m = performance_metrics(df["port_ret"])
        dist, p = block_bootstrap_sharpe(df["port_ret"])
        boot_dist[name] = (dist, p)
        status = "✓" if (m["sharpe"] > 1.40 and p < 0.05) else "✗"
        print(f"  {name:25s} 年化={m['annual']:+.2%}  夏普={m['sharpe']:.3f}  "
              f"最大回撤={m['mdd']:.2%}  Bootstrap p={p:.4f}  {status}")

    m_mkt = performance_metrics(market_ret)
    print(f"  {'市场等权':25s} 年化={m_mkt['annual']:+.2%}  夏普={m_mkt['sharpe']:.3f}  "
          f"最大回撤={m_mkt['mdd']:.2%}")

    # 作图
    plot_results(results, boot_dist)

    # 保存结果
    summary = []
    for name, df in results.items():
        m = performance_metrics(df["port_ret"])
        _, p = boot_dist[name]
        summary.append({
            "portfolio": name,
            "annual_return": m["annual"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["mdd"],
            "bootstrap_p": p,
            "pass": m["sharpe"] > 1.40 and p < 0.05,
        })
    summary.append({
        "portfolio": "市场等权",
        "annual_return": m_mkt["annual"],
        "sharpe": m_mkt["sharpe"],
        "max_drawdown": m_mkt["mdd"],
        "bootstrap_p": None,
        "pass": False,
    })
    pd.DataFrame(summary).to_csv(MODEL_DIR / "portfolio_backtest_summary.csv", index=False)
    print("\n结果保存: models/portfolio_backtest_summary.csv")


if __name__ == "__main__":
    main()
