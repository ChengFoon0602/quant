"""
截面回测引擎 — 因子驱动的多票截面策略回测。

逻辑:
  1. 每个调仓日，按因子值对所有股票排序
  2. 选 top_pct 做多（或 top 做多 + bottom 做空）
  3. 等权配置，持有到下一调仓日
  4. 扣除交易成本

与 engine.py 互补：engine.py 做单票信号回测，cross_section.py 做多票截面回测。
"""

import numpy as np
import pandas as pd


def run_cross_section(
    close_matrix: pd.DataFrame,
    factor_df: pd.DataFrame,
    top_pct: float = 0.2,
    bottom_pct: float | None = None,
    rebalance: str = "monthly",
    buy_cost: float = 0.00026,
    sell_cost: float = 0.00076,
) -> dict:
    """多票截面因子回测。

    Parameters
        close_matrix: index=date, columns=symbols, 收盘价矩阵
        factor_df: index=date, columns=symbols, 因子值矩阵（越大越好）
        top_pct: 做多比例 (0~1)，选因子值最大的 top_pct 股票
        bottom_pct: 做空比例 (0~1)，None = 纯多。选因子值最小的 bottom_pct
        rebalance: 调仓频率 — "monthly"|"weekly"|"quarterly"
        buy_cost: 买入单边费率
        sell_cost: 卖出单边费率

    Returns
        dict: equity, benchmark, long_equity, short_equity, metrics
    """
    # 只保留两者都有的日期和股票
    common_dates = close_matrix.index.intersection(factor_df.index)
    common_symbols = close_matrix.columns.intersection(factor_df.columns)
    close = close_matrix.loc[common_dates, common_symbols]
    factor = factor_df.loc[common_dates, common_symbols]

    if close.empty or factor.empty:
        raise ValueError("close_matrix 和 factor_df 无共同日期/股票。")

    daily_ret = close.pct_change().fillna(0)
    n_symbols = len(common_symbols)

    # 确定调仓日期
    rebalance_dates = _get_rebalance_dates(close.index, rebalance)

    # 构建持仓权重矩阵: index=date, columns=symbols, value=weight
    weights = pd.DataFrame(0.0, index=close.index, columns=common_symbols)
    prev_weights = weights.copy()

    n_long = max(1, int(n_symbols * top_pct))
    n_short = 0 if bottom_pct is None else max(1, int(n_symbols * bottom_pct))

    for i, rb_date in enumerate(rebalance_dates):
        if rb_date not in factor.index:
            continue

        # 找到下一个调仓日（或最后一天）
        if i + 1 < len(rebalance_dates):
            next_rb = rebalance_dates[i + 1]
        else:
            next_rb = close.index[-1]

        period_mask = (weights.index >= rb_date) & (weights.index < next_rb)
        if not period_mask.any():
            continue

        # 当日因子值排序
        f_vals = factor.loc[rb_date].dropna()

        if len(f_vals) < n_long + n_short + 1:
            continue  # 数据不够，跳过这个调仓日

        # 选 top (多头) 和 bottom (空头)
        sorted_syms = f_vals.sort_values(ascending=False).index
        long_syms = sorted_syms[:n_long]
        short_syms = sorted_syms[-n_short:] if n_short > 0 else pd.Index([])

        long_weight = 1.0 / n_long
        short_weight = -1.0 / n_short if n_short > 0 else 0

        weights.loc[period_mask, long_syms] = long_weight
        if n_short > 0:
            weights.loc[period_mask, short_syms] = short_weight

    # 计算策略收益（扣除交易成本）
    # 策略日收益 = sum(w_i * r_i)
    strategy_ret = (weights * daily_ret).sum(axis=1)

    # 交易成本：只在调仓日扣除
    turnover_cost = pd.Series(0.0, index=close.index)
    for rb_date in rebalance_dates:
        if rb_date not in weights.index:
            continue
        w_curr = weights.loc[rb_date]
        # 找到前一天的权重
        idx_pos = weights.index.get_loc(rb_date)
        if idx_pos > 0:
            w_prev = weights.iloc[idx_pos - 1]
        else:
            w_prev = pd.Series(0.0, index=common_symbols)

        delta = (w_curr - w_prev).abs()
        # 买入成本
        buy_turnover = delta[delta > 0].sum() / 2  # 一半是买入
        sell_turnover = delta[delta > 0].sum() / 2  # 一半是卖出
        turnover_cost.loc[rb_date] = buy_turnover * buy_cost + sell_turnover * sell_cost

    strategy_net = strategy_ret - turnover_cost.fillna(0)

    # 基准: 等权全市场
    benchmark_ret = daily_ret.mean(axis=1)

    equity = (1 + strategy_net).cumprod()
    benchmark = (1 + benchmark_ret).cumprod()

    # 单独计算多空端
    long_weights = weights.clip(lower=0)
    short_weights = (-weights).clip(lower=0)
    long_ret = (long_weights * daily_ret).sum(axis=1).div(long_weights.sum(axis=1).replace(0, np.nan)).fillna(0)
    short_ret = (short_weights * daily_ret).sum(axis=1).div(short_weights.sum(axis=1).replace(0, np.nan)).fillna(0)

    # 绩效指标
    n_years = (close.index[-1] - close.index[0]).days / 365.25
    total_ret = equity.iloc[-1] - 1
    bm_ret = benchmark.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    excess = strategy_net - 0.02 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 1e-12 else 0

    # 信息比率 (vs 等权基准)
    active = strategy_net - benchmark_ret
    ir = np.sqrt(252) * active.mean() / active.std() if active.std() > 1e-12 else 0

    # 换手率
    turnover = np.mean([(weights.iloc[i] - weights.iloc[i-1]).abs().sum()
                        for i in range(1, len(weights))])

    return {
        "equity": equity,
        "benchmark": benchmark,
        "long_equity": (1 + long_ret).cumprod() if n_short > 0 else None,
        "short_equity": (1 + short_ret).cumprod() if n_short > 0 else None,
        "strategy_net": strategy_net,
        "weights": weights,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "information_ratio": ir,
        "turnover": turnover,
        "bm_return": bm_ret,
    }


def _get_rebalance_dates(dates_index, freq: str) -> pd.DatetimeIndex:
    """从日期索引中提取调仓日。"""
    if freq == "monthly":
        # 每月最后一个交易日
        months = dates_index.to_series().groupby([
            dates_index.year, dates_index.month
        ])
        return pd.DatetimeIndex(months.last().sort_index().values)
    elif freq == "weekly":
        # 每周最后一天
        weeks = dates_index.to_series().groupby([
            dates_index.isocalendar().year.values,
            dates_index.isocalendar().week.values,
        ])
        return pd.DatetimeIndex(weeks.last().sort_index().values)
    elif freq == "quarterly":
        months = dates_index.to_series().groupby([
            dates_index.year, dates_index.quarter
        ])
        return pd.DatetimeIndex(months.last().sort_index().values)
    else:
        raise ValueError(f"不支持的调仓频率: {freq}")
