"""
_compare_old_new.py — 一次性对照脚本：验证收益/成本口径收口对真实 ML 组合的影响。

目的：
  在真实 OOF 预测 + 缓存价格数据上，对比「旧口径（错位收益 shift(-2)/shift(-1)-1
  + cost/2 对半 0.003）」与「新口径（pct_change + 买/卖方向分离 0.1%）」的绩效差异，
  验证「+10% 低估」与「成本下降 2/3」的估计，确认核心结论是否反转。

用法: python _compare_old_new.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from data.fetcher import load_daily
from models.portfolio_backtest import build_portfolio, performance_metrics

PRED_PATH = Path("models/oof_predictions.csv")


def load_close_for(symbols, start="2010-01-01", end="2025-12-31"):
    """加载一篮子股票的收盘价矩阵。"""
    close_data = {}
    for sym in symbols:
        df = load_daily(sym)
        if df is not None and len(df) >= 100:
            s = df.loc[(df.index >= start) & (df.index <= end), "close"]
            if len(s) >= 100:
                close_data[sym] = s
    return pd.DataFrame(close_data).sort_index()


def build_portfolio_old(
    pred_df, close_matrix, long_only=False, short_only=False,
    top_q=0.20, bottom_q=0.20, cost=0.003, hold_days=5,
):
    """旧口径：错位收益 + cost/2 对半。"""
    daily_ret = close_matrix.shift(-2) / close_matrix.shift(-1) - 1
    common_dates = pred_df.index.intersection(daily_ret.index)
    common_cols = pred_df.columns.intersection(daily_ret.columns)
    p = pred_df.loc[common_dates, common_cols]
    r = daily_ret.loc[common_dates, common_cols]

    W_target = pd.DataFrame(0.0, index=common_dates, columns=common_cols)
    for d in common_dates:
        pv = p.loc[d]
        mask = pv.notna()
        if mask.sum() < max(int(1/top_q), int(1/bottom_q)) * 3:
            continue
        valid_p = pv[mask]
        top_thr = valid_p.quantile(1 - top_q)
        bottom_thr = valid_p.quantile(bottom_q)
        top = valid_p[valid_p >= top_thr].index
        bottom = valid_p[valid_p <= bottom_thr].index
        if not short_only and len(top):
            W_target.loc[d, top] = 1.0 / len(top)
        if not long_only and len(bottom):
            W_target.loc[d, bottom] = -1.0 / len(bottom)

    W_held = W_target.rolling(hold_days, min_periods=1).mean()
    W_lag = W_held.shift(1)
    port_gross = (W_lag * r).sum(axis=1, min_count=1)
    turnover = (W_held - W_held.shift(1)).abs().sum(axis=1)
    port_ret = port_gross - turnover * (cost / 2.0)
    port_ret = port_ret.iloc[hold_days:].dropna()
    return port_ret


def main():
    pred = pd.read_csv(PRED_PATH, index_col=0, parse_dates=True)
    symbols = list(pred.columns)
    print(f"加载 {len(symbols)} 只股票价格...")
    close = load_close_for(symbols)
    common = pred.columns.intersection(close.columns)
    pred = pred[common]
    close = close[common]
    print(f"  交集: {len(common)} 只, 日期 {close.index.min().date()} -> {close.index.max().date()}")

    print("\n" + "=" * 70)
    print("新旧口径对照（真实 OOF 预测，hold_days=5，top/bottom 20%）")
    print("=" * 70)

    rows = []
    for long_only in [False, True]:
        name = "LS" if not long_only else "LO"
        old_ret = build_portfolio_old(pred, close, long_only=long_only, cost=0.003, hold_days=5)
        new_ret = build_portfolio(pred, close, long_only=long_only, hold_days=5)
        m_old = performance_metrics(old_ret)
        m_new = performance_metrics(new_ret["port_ret"])
        rows.append({
            "组合": name,
            "旧口径_年化": m_old["annual"],
            "旧口径_夏普": m_old["sharpe"],
            "旧口径_回撤": m_old["mdd"],
            "新口径_年化": m_new["annual"],
            "新口径_夏普": m_new["sharpe"],
            "新口径_回撤": m_new["mdd"],
        })
        print(f"\n[{name}] 旧口径: 年化={m_old['annual']:+.2%} 夏普={m_old['sharpe']:+.3f} 回撤={m_old['mdd']:+.2%}")
        print(f"[{name}] 新口径: 年化={m_new['annual']:+.2%} 夏普={m_new['sharpe']:+.3f} 回撤={m_new['mdd']:+.2%}")
        print(f"[{name}] 夏普变化: {m_new['sharpe']-m_old['sharpe']:+.3f} ({m_new['sharpe']/m_old['sharpe']-1:+.1%})")

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("汇总对照表")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))


if __name__ == "__main__":
    main()
