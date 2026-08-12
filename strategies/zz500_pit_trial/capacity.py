"""
capacity.py — 成交额加权容量检验（冲击成本模拟）。

P1 多头增强的容量验证：中小盘流动性差的 ZZ500，固定 0.3% 双边成本假设可能
严重低估真实交易成本。本模块对给定 AUM（管理规模）模拟逐日逐股冲击成本，
看 LO 组合夏普随规模上升的衰减曲线，找到容量上限。

冲击成本模型（平方根法则, sqrt law）:
    participation[s,d] = 交易金额 / 当日成交额 = |ΔW[s,d]| × AUM / amount[s,d]
    impact_bps[s,d]   = k × σ_20d[s,d] × sqrt(participation[s,d])
    k = 0.3（保守）/ 1.0（激进）两档敏感性

参与率 cap 到 0.3：sqrt 法则在参与率 >1 时发散失真（少数极端日成交额极低）。

用法:
    from strategies.zz500_pit_trial.capacity import run_capacity_sweep
    cap_df = run_capacity_sweep(pred_neutral, close, amount, aum_grid=[5e8, ...])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.portfolio_backtest import build_portfolio, performance_metrics

PARTICIPATION_CAP = 0.3


def _stock_vol_20d(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """个股日收益 20 日滚动标准差（年化前），用于 sqrt 法则的波动率项。"""
    daily_ret = close_matrix.pct_change()
    return daily_ret.rolling(20).std().reindex(index=close_matrix.index, columns=close_matrix.columns)


def impact_returns(
    flows: pd.DataFrame,
    amount_matrix: pd.DataFrame,
    sigma_20d: pd.DataFrame,
    aum: float,
    k: float = 0.5,
) -> pd.Series:
    """每日组合级冲击成本（权重单位，可直接从 port_ret 扣除）。

    cost_ret[d] = Σ_s k·σ[s,d]·sqrt(cap(participation[s,d])) · |ΔW[s,d]|
    """
    amount = amount_matrix.reindex(index=flows.index, columns=flows.columns)
    sig = sigma_20d.reindex(index=flows.index, columns=flows.columns)

    participation = flows.mul(aum).div(amount.replace(0, np.nan))
    participation = participation.clip(upper=PARTICIPATION_CAP).fillna(0)

    impact = k * sig.mul(np.sqrt(participation)).fillna(0)
    cost_ret = (impact * flows).sum(axis=1, min_count=1)
    return cost_ret


def run_capacity_sweep(
    pred_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    amount_matrix: pd.DataFrame,
    aum_grid: list[float],
    k: float = 0.5,
    hold_days: int = 5,
    cost: float = 0.003,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 AUM 网格跑容量检验，返回 (容量衰减表, 基准组合)。

    基准组合：build_portfolio(return_flows=True) 拿 flows + port_ret（已扣固定成本 cost）。
    每个 AUM 在基准上叠加冲击成本重算夏普。flows 与 cost 无关（cost 只影响收益）。

    aum_grid 单位：元（如 5e8 = 5 亿）。
    """
    df, flows = build_portfolio(pred_matrix, close_matrix, long_only=True,
                                cost=cost, hold_days=hold_days, return_flows=True)
    sigma_20d = _stock_vol_20d(close_matrix)

    base = performance_metrics(df["port_ret"])
    rows = []
    for aum in aum_grid:
        impact = impact_returns(flows, amount_matrix, sigma_20d, aum, k=k)
        net_ret = df["port_ret"] - impact
        m = performance_metrics(net_ret)
        rows.append({
            "aum_yi": aum / 1e8,
            "aum": aum,
            "annual": m["annual"],
            "sharpe": m["sharpe"],
            "mdd": m["mdd"],
            "avg_impact_bps": impact.mean() * 1e4,
            "max_impact_bps": impact.max() * 1e4,
        })
    cap_df = pd.DataFrame(rows)
    return cap_df, df
