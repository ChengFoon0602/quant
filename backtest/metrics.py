"""
backtest/metrics.py — 绩效指标规范函数（单一真源）。

背景（2026-09-09 框架评审，CLAUDE.md TODO #3）：
  年化收益曾有三套口径并存，跨模块比较失真：

    | 位置 | 公式 | 类型 |
    |---|---|---|
    | backtest/engine.py:54 | (1+total)**(1/n_years)-1 | CAGR（日历日） |
    | models/portfolio_backtest.py::performance_metrics | (1+mean)**252-1 | 复利式 |
    | risk/portfolio.py::calculate_metrics | daily_mean*252 | 算术式 |

  夏普比率口径本就统一（mean/std*sqrt(252)，仅在 rf 处理上有细微差异），
  因此本次统一只动「年化收益」：全部收敛到**路径 CAGR（几何）**——
  反映投资者真实获得的复利年化，对读者最诚实。

  算术年化（mean*252）与 CAGR 的关系满足 AM-GM 不等式：算术 ≥ 几何，
  差异随波动率上升而扩大（方差拖累 ≈ σ²/2）。

用法:
    from backtest.metrics import annualize_cagr, sharpe_ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualize_cagr(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """几何年化收益（CAGR）：从收益率序列算真实复利年化。

    Parameters
    ----------
    returns : pd.Series | np.ndarray
        日收益率序列（会 dropna）。
    periods_per_year : int, default 252
        年化周期数（日频 252，月频 12）。

    Returns
    -------
    float
        年化复利收益。序列为空或累计净值 ≤0 时返回 np.nan。
    """
    s = np.asarray(pd.Series(returns).dropna(), dtype=float)
    n = len(s)
    if n == 0:
        return np.nan
    total = np.prod(1.0 + s)
    if not np.isfinite(total) or total <= 0.0:
        # 累计净值非正 → 复利幂无实义
        return np.nan
    return float(total ** (periods_per_year / n) - 1.0)


def annualize_arithmetic(
    returns: pd.Series | np.ndarray,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """算术年化收益（mean * periods_per_year）。供 AM-GM 对照/Sharpe 分子。"""
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean() * periods_per_year)


def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    rf_annual: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """年化夏普比率：sqrt(PPY) * (mean - rf/PPY) / std。

    与既有各模块口径一致（mean/std*sqrt(252)），rf=0 时完全相同。
    """
    s = pd.Series(returns).dropna()
    if len(s) < 2:
        return np.nan
    std = s.std(ddof=1)
    if std < 1e-12:
        return 0.0
    daily_rf = rf_annual / periods_per_year
    return float((s.mean() - daily_rf) / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series | np.ndarray) -> float:
    """最大回撤（负值）。"""
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return np.nan
    cum = (1.0 + s).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    return float(dd.min())
