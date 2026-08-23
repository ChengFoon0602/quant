"""
strategies/etf_momentum_crowding/signals.py — ETF 动量因子族与拥挤度/避险择时信号。
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def simple_momentum(close_matrix: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """简单区间收益动量：close(t) / close(t - window) - 1。"""
    return close_matrix / close_matrix.shift(window) - 1.0


def sharpe_momentum(close_matrix: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """波动调整动量（Sharpe 动量）：过去 window 天日收益率均值 / 日收益率标准差。

    能够有效惩罚高波动假突破，奖励低波动平稳上行趋势。
    """
    daily_rets = close_matrix.pct_change(fill_method=None)
    mean_ret = daily_rets.rolling(window, min_periods=max(5, window // 2)).mean()
    std_ret = daily_rets.rolling(window, min_periods=max(5, window // 2)).std().replace(0, np.nan)
    return (mean_ret / std_ret) * np.sqrt(252.0)


def ma_distance_momentum(close_matrix: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """均线偏离动量：close(t) / MA_window(t) - 1。"""
    ma = close_matrix.rolling(window, min_periods=max(5, window // 2)).mean()
    return close_matrix / ma - 1.0


def compute_volume_share_crowding(amount_matrix: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """行业/资产成交额集中度 Z-score。

    成交额占比 = amount_i / sum(amount)
    拥挤度 = (当前占比 - 过去 lookback 天均值) / 过去 lookback 天标准差。
    """
    total_amount = amount_matrix.sum(axis=1).replace(0, np.nan)
    vol_share = amount_matrix.div(total_amount, axis=0)

    exp_mean = vol_share.rolling(lookback, min_periods=max(10, lookback // 2)).mean()
    exp_std = vol_share.rolling(lookback, min_periods=max(10, lookback // 2)).std().replace(0, np.nan)

    crowding_z = (vol_share - exp_mean) / exp_std
    return crowding_z


def market_trend_gate(
    close_matrix: pd.DataFrame,
    benchmark_symbol: str = "510300",
    ma_window: int = 20,
) -> pd.Series:
    """市场趋势避险闸门。

    若基准 ETF (510300 沪深300) > MA20，则 gate=1.0（正常开仓权益动量）；
    若基准 ETF <= MA20，则 gate=0.0（触发避险，权益清仓，转入国债/黄金）。
    """
    if benchmark_symbol in close_matrix.columns:
        bench_close = close_matrix[benchmark_symbol]
    else:
        bench_close = close_matrix.mean(axis=1)

    ma = bench_close.rolling(ma_window, min_periods=max(5, ma_window // 2)).mean()
    gate = (bench_close > ma).astype(float)
    return gate
