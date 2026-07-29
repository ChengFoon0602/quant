"""
signals/alternative/factors.py — 非 Alpha 191 的另类因子。

全部基于现有日线 OHLCV 数据计算，无需新数据源。

因子:
  - residual_momentum: 动量剥离市场 Beta 后的残差累计（Blitz et al. 2013）
  - f52_week_high: 当前价 / 52 周最高价（George & Hwang 2004）

输出格式与 Alpha 191 一致: DataFrame(index=date, columns=symbols)
"""

import numpy as np
import pandas as pd


def compute_residual_momentum(
    close_matrix: pd.DataFrame,
    window: int = 252,
    momentum_period: int = 60,
    min_periods: int = 120,
) -> pd.DataFrame:
    """残差动量 — 剥离市场 Beta 后的累计超额收益。

    1. 市场收益 = 截面等权均值
    2. 滚动回归: r_i = alpha + beta * r_m + e_i (window=252)
    3. 残差动量 = sum(e_i) over momentum_period (60 天)

    Parameters
        close_matrix: date × symbol 收盘价矩阵
        window: Beta 估计窗口（默认 252 天）
        momentum_period: 残差累计周期（默认 60 天）

    Returns
        factor_df: date × symbol, 残差动量值
    """
    ret = close_matrix.pct_change().fillna(0)
    mkt_ret = ret.mean(axis=1)  # 等权市场收益

    # 滚动 Beta: cov(r_i, r_m) / var(r_m)
    cov = ret.rolling(window, min_periods=min_periods).cov(
        mkt_ret, pairwise=False
    )
    var_mkt = mkt_ret.rolling(window, min_periods=min_periods).var()
    beta = cov.div(var_mkt, axis=0)

    # 残差收益: r_i - beta * r_m
    residual = ret - beta.mul(mkt_ret, axis=0)

    # 残差动量: rolling sum of residuals
    factor = residual.rolling(momentum_period, min_periods=min(momentum_period, min_periods)).sum()

    return factor


def compute_52w_high(
    close_matrix: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60,
) -> pd.DataFrame:
    """52 周最高价比率 — 当前价 / 过去 252 天最高价。

    锚定效应: 投资者以 52 周高点为心理锚定。
    比率越高 → 越接近新高 → 实证预测方向不定（既可能是动量延续也可能是反转）。

    Parameters
        close_matrix: date × symbol 收盘价矩阵
        window: 最高价窗口（默认 252 天）

    Returns
        factor_df: date × symbol, close / 52w_high（0~1 之间的值）
    """
    high_52w = close_matrix.rolling(window, min_periods=min_periods).max()
    factor = close_matrix / high_52w.replace(0, np.nan)
    return factor
