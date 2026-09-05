"""
risk/portfolio.py — 基于权重追踪的真实重叠持仓、交易限制与组合管理引擎。

核心方法论（防范 Overlapping Returns 平滑陷阱与微观交易限制）：
  1. 严禁对信号收益率做 rolling(H).mean() 构造组合收益（该方法人为压低波动 sqrt(H) 倍导致夏普虚高）。
  2. 本模块基于每日目标权重向量 w[t]，追踪实际持有权重 W[t] = mean(w[t-H+1]..w[t])。
  3. 支持实盘微观交易限制拦截（一字涨停禁买、一字跌停禁卖）。
  4. 支持目标波动率风控（Volatility Targeting）与自适应杠杆控制。
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def detect_limit_moves(
    open_matrix: pd.DataFrame,
    high_matrix: pd.DataFrame,
    low_matrix: pd.DataFrame,
    pre_close_matrix: pd.DataFrame,
    st_symbols: Optional[set] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """根据开高低收和昨收价，检测全市场标的是否处于一字涨停（禁买）或一字跌停（禁卖）状态。

    A 股涨跌停规则适配：
      - 30 / 68 开头（创业板/科创板）：20% 涨跌停
      - 8 / 4 开头（北交所）：30% 涨跌停
      - ST / *ST 股（通过 st_symbols 传入）：5% 涨跌停
      - 其余主板：10% 涨跌停

    ⚠️ 已知简化（未建模）：
      - 创业板/科创板新股上市前 5 个交易日无涨跌停限制；
      - 北交所新股上市首日无涨跌停限制。
      这两类需要在更高层用「上市日 + 前 5 日」元数据单独豁免。

    Parameters
    ----------
    st_symbols : Optional[set]
        ST / *ST 股票的 symbol 集合。传入后这些股票按 5% 涨跌停处理。

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (is_limit_up_locked, is_limit_down_locked) 布尔掩码矩阵。
    """
    limit_pct = pd.Series(0.10, index=open_matrix.columns)
    for col in open_matrix.columns:
        s = str(col)
        if s.startswith(("30", "68")):
            limit_pct[col] = 0.20
        elif s.startswith(("8", "4")):
            limit_pct[col] = 0.30
    if st_symbols:
        for sym in st_symbols:
            if sym in limit_pct.index:
                limit_pct[sym] = 0.05

    # 近似涨跌停价（A 股四舍五入到分位）
    limit_up = (pre_close_matrix * (1.0 + limit_pct)).round(2)
    limit_down = (pre_close_matrix * (1.0 - limit_pct)).round(2)

    # 一字涨停：开盘价 >= 涨停价 且 最低价 == 最高价
    is_limit_up_locked = (open_matrix >= limit_up - 0.01) & (high_matrix == low_matrix)
    # 一字跌停：开盘价 <= 跌停价 且 最低价 == 最高价
    is_limit_down_locked = (open_matrix <= limit_down + 0.01) & (high_matrix == low_matrix)

    return is_limit_up_locked.fillna(False), is_limit_down_locked.fillna(False)


def build_weight_portfolio(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    long_only: bool = False,
    short_only: bool = False,
    top_q: float = 0.20,
    bottom_q: float = 0.20,
    cost: Optional[float] = None,
    buy_cost: float = 0.00026,
    sell_cost: float = 0.00076,
    hold_days: int = 5,
    position_scale: Optional[pd.Series] = None,
    gate: Optional[pd.Series] = None,
    trade_limits: Optional[Tuple[pd.DataFrame, pd.DataFrame]] = None,
    return_weights: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """构建真实重叠组合回测（权重追踪法）。

    Parameters
    ----------
    pred_df : pd.DataFrame
        截面预测信号矩阵（index=date, columns=symbols）。
    close_matrix : pd.DataFrame
        收盘价矩阵（index=date, columns=symbols）。
    long_only : bool, default False
        仅构建多头组合（Top 分位等权）。
    short_only : bool, default False
        仅构建空头组合（Bottom 分位等权做空）。与 long_only 互斥。
    top_q : float, default 0.20
        多头分位数阈值（默认前 20%）。
    bottom_q : float, default 0.20
        空头分位数阈值（默认后 20%）。
    cost : Optional[float], default None
        双边合计成本（买入+卖出），向后兼容 ETF 等低摩擦资产场景。
        若显式传入，则 buy_cost = sell_cost = cost / 2（对半拆）。
        为 None 时使用下方 buy_cost/sell_cost 方向分离口径。
    buy_cost : float, default 0.00026
        买入单边费率（佣金万 2.5 + 过户费），与铁律第 3 条一致。
    sell_cost : float, default 0.00076
        卖出单边费率（佣金 + 印花税 0.05% + 过户费），与铁律第 3 条一致。
    hold_days : int, default 5
        持有天数（tranche 数量）。
    position_scale : Optional[pd.Series], default None
        逐日仓位系数（如市场高波动时降低仓位），后乘于实际持仓 W 上。
    gate : Optional[pd.Series], default None
        市场状态闸门（0 或 1），前乘于目标权重 w 上（闸门关闭期间不建立新仓位）。
    trade_limits : Optional[Tuple[pd.DataFrame, pd.DataFrame]], default None
        (is_limit_up_locked, is_limit_down_locked) 交易不可执行限制掩码。
    return_weights : bool, default False
        是否同时返回每日实际持仓权重矩阵 W_held。
    """
    if long_only and short_only:
        raise ValueError("long_only 与 short_only 互斥，不能同时为 True")

    # 成本口径归一：cost 为 None 时用方向分离；否则对半拆（向后兼容 ETF 等低摩擦场景）
    if cost is not None:
        buy_cost = cost / 2.0
        sell_cost = cost / 2.0

    # 收益锚定日约定（全仓库统一）：t→t+1 收益记在 t+1 日。
    # 因此 daily_ret[t] = close[t] / close[t-1] - 1，与 engine.py / cross_section.py / labels.py 的 pct_change() 语义一致。
    daily_ret = close_matrix.pct_change()

    common_dates = pred_df.index.intersection(daily_ret.index).sort_values()
    common_cols = pred_df.columns.intersection(daily_ret.columns).sort_values()

    p = pred_df.loc[common_dates, common_cols]
    r = daily_ret.loc[common_dates, common_cols]

    # 1. 每日生成目标权重向量 w[t]
    W_target = pd.DataFrame(0.0, index=common_dates, columns=common_cols)
    min_stocks = max(int(1.0 / top_q), int(1.0 / bottom_q)) * 2

    for d in common_dates:
        pv = p.loc[d]
        mask = pv.notna()
        if mask.sum() < min_stocks:
            continue
        valid_p = pv[mask]
        top_thr = valid_p.quantile(1.0 - top_q)
        bot_thr = valid_p.quantile(bottom_q)

        top_stocks = valid_p[valid_p >= top_thr].index
        bot_stocks = valid_p[valid_p <= bot_thr].index

        if not short_only and len(top_stocks) > 0:
            W_target.loc[d, top_stocks] = 1.0 / len(top_stocks)
        if not long_only and len(bot_stocks) > 0:
            W_target.loc[d, bot_stocks] = -1.0 / len(bot_stocks)

    # 状态闸门
    if gate is not None:
        g = gate.reindex(common_dates).fillna(1.0)
        W_target = W_target.mul(g, axis=0)

    # 2. 实际持仓 W[t] = 过去 hold_days 天目标权重的平均
    W_held = W_target.rolling(hold_days, min_periods=1).mean()

    # 应用微观交易限制（若一字涨停无法买入增仓；一字跌停无法卖出减仓）
    if trade_limits is not None:
        lim_up, lim_down = trade_limits
        lim_up_aligned = lim_up.reindex(index=common_dates, columns=common_cols).fillna(False)
        lim_down_aligned = lim_down.reindex(index=common_dates, columns=common_cols).fillna(False)

        W_held_constrained = W_held.copy()
        for i in range(1, len(common_dates)):
            d_curr = common_dates[i]
            d_prev = common_dates[i - 1]

            target_w = W_held.loc[d_curr]
            prev_w = W_held_constrained.loc[d_prev]

            # 试图买入 (w > prev_w) 但一字涨停 -> 无法买入，维持 prev_w
            cant_buy = (target_w > prev_w) & lim_up_aligned.loc[d_curr]
            target_w = target_w.mask(cant_buy, prev_w)

            # 试图卖出 (w < prev_w) 但一字跌停 -> 无法卖出，维持 prev_w
            cant_sell = (target_w < prev_w) & lim_down_aligned.loc[d_curr]
            target_w = target_w.mask(cant_sell, prev_w)

            W_held_constrained.loc[d_curr] = target_w

        W_held = W_held_constrained

    # 仓位缩放
    if position_scale is not None:
        ps = position_scale.reindex(common_dates).fillna(1.0)
        W_held = W_held.mul(ps, axis=0)

    # 3. 计算组合收益：第 t 天收益 = W[t-1] · daily_ret[t]
    W_lag = W_held.shift(1).fillna(0.0)
    gross_ret = (W_lag * r).sum(axis=1)

    # 4. 换手成本：区分买入/卖出方向，与铁律第 3 条（买 0.026% / 卖 0.076%）一致
    delta_w = W_held - W_held.shift(1).fillna(0.0)
    turnover = delta_w.abs().sum(axis=1)
    # 买入换手 = 增仓部分，卖出换手 = 减仓部分（多空对称，各占 |ΔW| 的一半）
    buy_turnover = delta_w.clip(lower=0.0).sum(axis=1)
    sell_turnover = (-delta_w).clip(lower=0.0).sum(axis=1)
    cost_deduction = buy_turnover * buy_cost + sell_turnover * sell_cost
    port_ret = gross_ret - cost_deduction

    # 丢弃建仓爬坡期
    if len(port_ret) > hold_days:
        port_ret = port_ret.iloc[hold_days:].dropna()
        gross_ret = gross_ret.reindex(port_ret.index)
        turnover = turnover.reindex(port_ret.index)
        cost_deduction = cost_deduction.reindex(port_ret.index)
    else:
        port_ret = port_ret.dropna()
        gross_ret = gross_ret.reindex(port_ret.index)
        turnover = turnover.reindex(port_ret.index)
        cost_deduction = cost_deduction.reindex(port_ret.index)

    cum = (1.0 + port_ret).cumprod()

    result_df = pd.DataFrame({
        "gross_ret": gross_ret,
        "cost": cost_deduction,
        "turnover": turnover,
        "port_ret": port_ret,
        "cum": cum,
    })

    if return_weights:
        return result_df, W_held.loc[port_ret.index]
    return result_df


def apply_volatility_target(
    port_ret_series: pd.Series,
    target_vol: float = 0.08,
    max_leverage: float = 2.0,
    lookback: int = 20,
    borrow_rate: float = 0.025,
) -> pd.DataFrame:
    """对收益率序列应用动态目标波动率机制（Volatility Targeting）。

    杠杆系数 lambda_t = min( target_vol / rolling_vol_t, max_leverage )
    若 lambda_t > 1.0，扣除杠杆资金借贷成本 (lambda_t - 1) * (borrow_rate / 252)。

    Parameters
    ----------
    port_ret_series : pd.Series
        原始组合日收益率。
    target_vol : float, default 0.08
        目标年化波动率（例如 8%）。
    max_leverage : float, default 2.0
        最大允许杠杆倍数。
    lookback : int, default 20
        波动率滚动估计窗口。
    borrow_rate : float, default 0.025
        年化借贷资金成本（例如 2.5%）。

    Returns
    -------
    pd.DataFrame
        包含 leverage, raw_ret, targeted_ret, cum。
    """
    s = port_ret_series.dropna()
    rolling_std = s.rolling(lookback, min_periods=max(5, lookback // 2)).std() * np.sqrt(252.0)
    rolling_std = rolling_std.replace(0, np.nan).fillna(target_vol)

    # 动态杠杆系数 (滞后 1 期使用，避免未来函数)
    raw_leverage = target_vol / rolling_std
    leverage = raw_leverage.clip(lower=0.1, upper=max_leverage).shift(1).fillna(1.0)

    # 借贷成本
    borrow_cost = np.maximum(0.0, leverage - 1.0) * (borrow_rate / 252.0)

    targeted_ret = (s * leverage) - borrow_cost
    cum = (1.0 + targeted_ret).cumprod()

    return pd.DataFrame({
        "leverage": leverage,
        "raw_ret": s,
        "borrow_cost": borrow_cost,
        "targeted_ret": targeted_ret,
        "cum": cum,
    })


def calculate_metrics(ret_series: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """计算标准投资组合绩效指标。"""
    s = ret_series.dropna()
    if len(s) < 2:
        return {
            "annual_return": 0.0,
            "annual_vol": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_loss_ratio": 0.0,
            "autocorr_lag1": 0.0,
            "n_days": len(s),
        }

    daily_mean = s.mean()
    daily_std = s.std()

    annual_return = daily_mean * 252.0
    annual_vol = daily_std * np.sqrt(252.0)
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 1e-8 else 0.0

    cum = (1.0 + s).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_drawdown = float(drawdown.min())

    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-8 else 0.0

    wins = s[s > 0]
    losses = s[s < 0]
    win_rate = len(wins) / len(s) if len(s) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-8 else 0.0
    ac1 = float(s.autocorr(lag=1)) if len(s) > 2 else 0.0

    return {
        "annual_return": float(annual_return),
        "annual_vol": float(annual_vol),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_loss_ratio": float(profit_loss_ratio),
        "autocorr_lag1": ac1,
        "n_days": len(s),
    }


def bootstrap_sharpe_test(
    ret_series: pd.Series,
    n_boot: int = 10000,
    block_size: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    """使用 Moving Block Bootstrap 检验夏普比率的统计显著性。"""
    rng = np.random.default_rng(seed)
    s = ret_series.dropna().values
    n = len(s)
    if n < block_size * 2:
        obs = calculate_metrics(ret_series)["sharpe"]
        return {
            "observed_sharpe": obs,
            "p_value": 1.0,
            "ci_95_low": obs,
            "ci_95_high": obs,
            "boot_sharpes": np.array([obs]),
        }

    obs_sharpe = float(np.mean(s) / np.std(s, ddof=1) * np.sqrt(252)) if np.std(s, ddof=1) > 1e-8 else 0.0

    n_blocks = n - block_size + 1
    blocks = np.lib.stride_tricks.sliding_window_view(s, window_shape=block_size)

    k = int(np.ceil(n / block_size))
    boot_sharpes = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n_blocks, size=k)
        sample = blocks[idx].ravel()[:n]
        std_samp = np.std(sample, ddof=1)
        if std_samp > 1e-8:
            boot_sharpes[i] = np.mean(sample) / std_samp * np.sqrt(252)
        else:
            boot_sharpes[i] = 0.0

    p_value = float(np.mean(boot_sharpes <= 0.0))
    ci_low = float(np.percentile(boot_sharpes, 2.5))
    ci_high = float(np.percentile(boot_sharpes, 97.5))

    return {
        "observed_sharpe": obs_sharpe,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "boot_sharpes": boot_sharpes,
    }
