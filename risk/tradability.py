"""
risk/tradability.py — 交易可行性的度量（约束的代价）。

存在理由
--------
`trade_limits`（一字涨停禁买 / 一字跌停禁卖）此前**只被测试调用过**，没有任何研究
结论受它约束 —— 于是无法回答一个容量研究必需的问题：

> **涨跌停约束会让策略损失多少换手与收益？**

先把**度量**做出来，再谈要不要接线（接线会改变已发布报告的成交日集合，须先有数）。

同时补上一个缺失环节：`detect_limit_moves` 需要「前收盘价」，但全仓库没有任何地方
为它推导过 —— 本模块的 `build_trade_limits` 补上这一步。

用法
----
    from risk.tradability import build_trade_limits, measure_restriction_impact

    limits = build_trade_limits(open_m, high_m, low_m, close_m)
    table = measure_restriction_impact(pred, close_m, limits, hold_days=5)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backtest.metrics import annualize_cagr, max_drawdown, sharpe_ratio
from risk.portfolio import build_weight_portfolio, detect_limit_moves

__all__ = ["build_trade_limits", "measure_restriction_impact"]


def build_trade_limits(
    open_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame,
    low_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    st_symbols: Optional[set] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 OHLC 面板推导 `(一字涨停锁, 一字跌停锁)` 掩码。

    A 股涨跌停以**前收盘价**为基准。此处用 `close.shift(1)` 作代理。

    ⚠️ 已知近似：**未做除权调整**。除权除息日的前收盘价会被高估/低估，
    可能漏判或误判一字板。日频研究可接受，但报告引用结论时须注明。

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        `(is_limit_up_locked, is_limit_down_locked)`，布尔矩阵，与 `close_matrix` 同形。
    """
    pre_close = close_matrix.shift(1)
    return detect_limit_moves(
        open_matrix, high_matrix, low_matrix, pre_close, st_symbols=st_symbols)


def _summarize(res: pd.DataFrame) -> dict[str, float]:
    ret = res["port_ret"]
    cum = (1.0 + ret).prod()
    return {
        "total_turnover": float(res["turnover"].sum()),
        "n_trade_days": float((res["turnover"] > 0).sum()),
        "total_return": float(cum - 1.0) if cum > 0 else float("nan"),
        "cagr": annualize_cagr(ret),
        "sharpe": sharpe_ratio(ret),
        "max_drawdown": max_drawdown(ret),
    }


def measure_restriction_impact(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    trade_limits: tuple[pd.DataFrame, pd.DataFrame],
    hold_days: int = 5,
    **kwargs,
) -> pd.DataFrame:
    """对比「无交易限制」与「有交易限制」两条账本，量化约束的代价。

    两次回测**除 `trade_limits` 外参数完全相同**，因此差值可归因于约束本身。

    Returns
    -------
    pd.DataFrame
        index = `["unrestricted", "restricted", "delta"]`，列为
        `total_turnover` / `n_trade_days` / `total_return`（累计，小数）/
        `cagr` / `sharpe` / `max_drawdown`。
        `delta = restricted − unrestricted`（换手与成交日数应为负，即约束减少了交易）。
    """
    base = build_weight_portfolio(pred_df, close_matrix, hold_days=hold_days, **kwargs)
    limited = build_weight_portfolio(pred_df, close_matrix, hold_days=hold_days,
                                     trade_limits=trade_limits, **kwargs)

    rows = {"unrestricted": _summarize(base), "restricted": _summarize(limited)}
    rows["delta"] = {k: rows["restricted"][k] - rows["unrestricted"][k]
                     for k in rows["unrestricted"]}
    return pd.DataFrame(rows).T
