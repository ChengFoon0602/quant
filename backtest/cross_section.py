"""
截面回测引擎 — 因子驱动的多票截面策略回测。

逻辑:
  1. 每个调仓日收盘，按因子值对所有股票排序
  2. 选 top_pct 做多（或 top 做多 + bottom 做空）
  3. 次日开盘等权配置，持有到下一调仓日收盘
  4. 扣除交易成本（在执行日计提）
  5. 支持 PIT Universe mask 过滤 + 退市股清算

无未来函数保证: 调仓日 rb_date 收盘产生信号 → rb_date 次日开盘执行。
rb_date 当天仍持有旧仓位，新权重从 > rb_date 开始生效。

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
    universe: pd.DataFrame | None = None,
    delist_info: dict | None = None,
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
        universe: index=date, columns=symbols, bool — True=可选股。
            None 表示全部可选（向后兼容）。
        delist_info: {'dates': {sym: Timestamp}, 'prices': {sym: float}}
            退市日与清算价。退市日在 weights 中强制清零。

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

    # ── 退市处理 ──
    delist_dates: dict = {}
    delist_prices: dict = {}
    if delist_info:
        delist_dates = delist_info.get("dates", {})
        delist_prices = delist_info.get("prices", {})

    # 退市日在收益矩阵中设为 NaN（该日之后无法交易）
    n_delisted = 0
    for sym, d_date in delist_dates.items():
        if sym not in close.columns or pd.isna(d_date):
            continue
        if d_date in close.index:
            # 清算价替代当日收盘价
            liq_price = delist_prices.get(sym, close.loc[d_date, sym] * 0.9)
            close.loc[d_date, sym] = liq_price
            daily_ret.loc[d_date, sym] = liq_price / close.iloc[close.index.get_loc(d_date) - 1][sym] - 1
            # 退市日后设为 NaN
            dates_after = close.index[close.index > d_date]
            close.loc[dates_after, sym] = np.nan
            daily_ret.loc[dates_after, sym] = np.nan
            n_delisted += 1

    # 重新计算收益（退市处理后的 close 变了）
    daily_ret = daily_ret.fillna(0)

    n_symbols = len(common_symbols)

    # 确定调仓日期
    rebalance_dates = _get_rebalance_dates(close.index, rebalance)

    # 构建持仓权重矩阵: index=date, columns=symbols, value=weight
    # 调仓逻辑：rb_date 收盘产生信号 → 次日开盘执行
    # rb_date 当天仍持有旧权重，新权重从 rb_date 后第一天生效
    weights = pd.DataFrame(0.0, index=close.index, columns=common_symbols)

    for i, rb_date in enumerate(rebalance_dates):
        if rb_date not in factor.index:
            continue

        # 找到下一个调仓日（或最后一天）
        if i + 1 < len(rebalance_dates):
            next_rb = rebalance_dates[i + 1]
        else:
            next_rb = close.index[-1]

        # 新权重从 rb_date 之后第一天生效，持续到 next_rb（含）
        period_mask = (weights.index > rb_date) & (weights.index <= next_rb)
        if not period_mask.any():
            continue

        # 当日因子值排序（应用 universe mask）
        f_vals = factor.loc[rb_date]
        if universe is not None and rb_date in universe.index:
            eligible = universe.loc[rb_date].reindex(f_vals.index, fill_value=False)
            f_vals = f_vals[eligible]
        # 剔除退市股：退市日后不可选
        for sym in delist_dates:
            if sym in f_vals.index and pd.notna(delist_dates[sym]) and rb_date >= delist_dates[sym]:
                f_vals = f_vals.drop(sym, errors="ignore")
        f_vals = f_vals.dropna()

        # 动态计算持仓数量（universe 过滤后股票数可能不同）
        n_effective = len(f_vals)
        n_long = max(1, int(n_effective * top_pct))
        n_short = 0 if bottom_pct is None else max(1, int(n_effective * bottom_pct))

        if n_effective < n_long + n_short + 1:
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

    # 交易成本：在实际执行日扣除（调仓日次日）
    turnover_cost = pd.Series(0.0, index=close.index)
    for rb_date in rebalance_dates:
        if rb_date not in weights.index:
            continue
        idx_pos = weights.index.get_loc(rb_date)
        if idx_pos + 1 >= len(weights.index):
            continue  # 最后一个调仓日无执行日
        exec_date = weights.index[idx_pos + 1]  # 实际执行日（次日开盘）
        w_new = weights.loc[exec_date]            # 新权重
        w_old = weights.loc[rb_date]              # 旧权重（rb_date 当天仍持旧仓）

        delta = (w_new - w_old).abs()
        buy_turnover = delta[delta > 0].sum() / 2   # 一半是买入
        sell_turnover = delta[delta > 0].sum() / 2  # 一半是卖出
        turnover_cost.loc[exec_date] = buy_turnover * buy_cost + sell_turnover * sell_cost

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
