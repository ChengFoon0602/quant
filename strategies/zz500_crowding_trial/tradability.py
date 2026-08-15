"""
tradability.py — 基本面低拥挤可交易性测试（方向C 延伸③）。

把「低拥挤 → 持有基本面因子多空」做成回测，回答「能不能交易」。

信号与收益对齐（无未来函数）：
  - rank_pct[t] = 基本面池综合拥挤度截至 t 的滚动分位（expanding min_periods=36）
  - signal[t] = 1 if rank_pct[t] < 0.25 else 0
  - mean_ret[t] = 20 因子月末多空收益（factor_monthly_returns，t 信号 → t+1..t+22 收益）
  - 信号与收益同 t 对齐（t 月末信号可用 → 收益 t+1 起），符合铁律 1

策略（二值状态机）：
  strat_ret[t] = signal[t] * mean_ret[t] - cost * |signal[t] - signal[t-1]|
  低拥挤月持有因子多空，否则空仓；切换扣双边成本。

对照：无条件基准（始终持有因子多空，扣换手 0.3%/月）+ bootstrap 显著性。
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, FEATURE_SEL_DIR

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(FEATURE_SEL_DIR))

from models.portfolio_backtest import performance_metrics, block_bootstrap_sharpe


def rolling_quantile(comp: pd.Series, min_periods: int = 36) -> pd.Series:
    """滚动分位（无前瞻）：rank_pct[t] = 截至 t 的历史分位。"""
    return comp.expanding(min_periods=min_periods).rank(pct=True)


def low_crowding_signal(rank_pct: pd.Series, threshold: float = 0.25) -> pd.Series:
    """signal = 1 if rank_pct < threshold else 0（低拥挤持有）。"""
    return (rank_pct < threshold).astype(int)


def state_machine_ret(signal: pd.Series, mean_ret: pd.Series,
                      cost: float = 0.003) -> pd.Series:
    """二值状态机：strat_ret = signal*mean_ret - cost*|Δsignal|（切换扣双边成本）。"""
    sig = signal.reindex(mean_ret.index).fillna(0).astype(int)
    switch = sig.diff().abs().fillna(0)   # 状态切换
    return sig * mean_ret - cost * switch


def capacity_stats(fr_f: pd.DataFrame, signal: pd.Series) -> dict:
    """容量/换手统计：持仓月份数、信号切换次数、名义多空股数。"""
    n_hold = int(signal.sum())
    n_switch = int(signal.diff().abs().fillna(0).sum())
    n_hold_frac = n_hold / len(signal) if len(signal) else 0
    # 名义容量：top/bottom 20% × 中证500 ~500 只
    return {
        "n_months": len(signal),
        "n_hold_months": n_hold,
        "hold_fraction": round(n_hold_frac, 3),
        "n_switches": n_switch,
        "n_top_stocks": int(500 * 0.20),   # 做多股数
        "n_bottom_stocks": int(500 * 0.20),  # 做空股数
        "nominal_capacity_mid": "中证500 top/bottom 20% ≈ 100+100 只",
    }


def run_tradability(cost_bps: float = 0.003) -> dict:
    """主流程：基本面池拥挤度 → 低拥挤信号 → 状态机回测 → 对照 + bootstrap。"""
    from fundamental_crowding import load_fundamental_long, FUNDAMENTAL_COLS
    from crowding import compute_all, factor_monthly_returns, month_end_dates
    from event_study import composite_crowding
    from build_pit_matrix import load_pit_panel, build_market_features_pit

    X_fund, close_f = load_fundamental_long()
    _, vol_f, mem_f = load_pit_panel("zz500")
    mkt_f = build_market_features_pit(close_f, vol_f, mem_f)
    crowd_f = compute_all(X_fund, close_f, mkt_f["market_turnover_20d"],
                          factor_cols=FUNDAMENTAL_COLS)
    med_f = month_end_dates(X_fund.index.get_level_values(0).unique())
    fr_f = factor_monthly_returns(X_fund, med_f, close_f, factor_cols=FUNDAMENTAL_COLS)
    mean_ret = fr_f.mean(axis=1).dropna()

    comp = composite_crowding(crowd_f)
    common = comp.index.intersection(mean_ret.index)
    comp, mean_ret = comp.loc[common], mean_ret.loc[common]

    rank_pct = rolling_quantile(comp)
    signal = low_crowding_signal(rank_pct)
    strat = state_machine_ret(signal, mean_ret, cost=cost_bps)

    # 无条件基准：始终持有因子多空，扣月换手 0.3%
    base = mean_ret - cost_bps

    m_s = performance_metrics(strat)
    m_b = performance_metrics(base)
    _, p_s = block_bootstrap_sharpe(strat)
    _, p_b = block_bootstrap_sharpe(base)

    # 成本敏感性
    sens = {}
    for c in [0.001, 0.003, 0.005]:
        s = state_machine_ret(signal, mean_ret, cost=c)
        sens[c] = round(performance_metrics(s)["sharpe"], 3)

    cap = capacity_stats(fr_f, signal)

    return {
        "strategy": {"sharpe": m_s["sharpe"], "annual": m_s["annual"],
                     "mdd": m_s["mdd"], "bootstrap_p": p_s},
        "baseline": {"sharpe": m_b["sharpe"], "annual": m_b["annual"],
                     "mdd": m_b["mdd"], "bootstrap_p": p_b},
        "cost_sensitivity": sens,
        "capacity": cap,
        "strategy_ret": strat,
        "baseline_ret": base,
    }


if __name__ == "__main__":
    r = run_tradability()
    print("=== 基本面低拥挤可交易性 ===")
    print(f"策略: SR={r['strategy']['sharpe']:+.3f} 年化={r['strategy']['annual']:+.2%} "
          f"回撤={r['strategy']['mdd']:.2f} bootstrap_p={r['strategy']['bootstrap_p']:.4f}")
    print(f"基准: SR={r['baseline']['sharpe']:+.3f} 年化={r['baseline']['annual']:+.2%} "
          f"回撤={r['baseline']['mdd']:.2f} bootstrap_p={r['baseline']['bootstrap_p']:.4f}")
    print(f"成本敏感性: " + " ".join(f"{c*100:.1f}%→SR={v:+.3f}" for c, v in r["cost_sensitivity"].items()))
    print(f"容量: {r['capacity']}")
