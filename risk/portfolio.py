"""
risk/portfolio.py — 基于权重追踪的真实重叠持仓与组合管理引擎。

核心方法论（防范 Overlapping Returns 平滑陷阱）：
  严禁对信号收益率做 rolling(H).mean() 构造组合收益（该方法人为压低波动 sqrt(H) 倍导致夏普虚高）。
  本模块基于每日目标权重向量 w[t]，追踪实际持有权重 W[t] = mean(w[t-H+1]..w[t])。
  第 t 日组合收益 = W[t-1] · daily_ret[t]，所有持有批次（tranche）在同一天经历真实市场波动。
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def build_weight_portfolio(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    long_only: bool = False,
    short_only: bool = False,
    top_q: float = 0.20,
    bottom_q: float = 0.20,
    cost: float = 0.003,
    hold_days: int = 5,
    position_scale: Optional[pd.Series] = None,
    gate: Optional[pd.Series] = None,
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
    cost : float, default 0.003
        双边交易成本与滑点（例如 0.003 表示买入+卖出合计千分之三）。
    hold_days : int, default 5
        持有天数（tranche 数量）。
    position_scale : Optional[pd.Series], default None
        逐日仓位系数（如市场高波动时降低仓位），后乘于实际持仓 W 上。
    gate : Optional[pd.Series], default None
        市场状态闸门（0 或 1），前乘于目标权重 w 上（闸门关闭期间不建立新仓位）。
    return_weights : bool, default False
        是否同时返回每日实际持仓权重矩阵 W_held。

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
        包含 port_ret (扣费后组合日收益), gross_ret (扣费前组合日收益),
        turnover (换手率绝对值之和), cum (累计净值) 的 DataFrame。
        若 return_weights=True，额外返回 W_held。
    """
    if long_only and short_only:
        raise ValueError("long_only 与 short_only 互斥，不能同时为 True")

    # 无未来函数的次日开盘至次次日开盘/收盘收益：close(t+2)/close(t+1) - 1
    daily_ret = close_matrix.shift(-2) / close_matrix.shift(-1) - 1

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

    # 若应用状态闸门 (gate)
    if gate is not None:
        g = gate.reindex(common_dates).fillna(1.0)
        W_target = W_target.mul(g, axis=0)

    # 2. 实际持仓 W[t] = 过去 hold_days 天目标权重的平均
    W_held = W_target.rolling(hold_days, min_periods=1).mean()

    # 若应用仓位缩放 (position_scale)
    if position_scale is not None:
        ps = position_scale.reindex(common_dates).fillna(1.0)
        W_held = W_held.mul(ps, axis=0)

    # 3. 计算组合收益：第 t 天收益 = W[t-1] · daily_ret[t]
    W_lag = W_held.shift(1).fillna(0.0)
    gross_ret = (W_lag * r).sum(axis=1)

    # 4. 换手成本：每日换手 = sum(|W[t] - W[t-1]|)，成本 = 换手 * (cost / 2)
    turnover = (W_held - W_held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_deduction = turnover * (cost / 2.0)
    port_ret = gross_ret - cost_deduction

    # 丢弃前 hold_days 天建仓爬坡期
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


def calculate_metrics(ret_series: pd.Series, rf: float = 0.0) -> Dict[str, float]:
    """计算标准投资组合绩效指标。

    Parameters
    ----------
    ret_series : pd.Series
        日收益率序列。
    rf : float, default 0.0
        年化无风险利率。

    Returns
    -------
    Dict[str, float]
        包含年化收益、年化波动、夏普比率、最大回撤、卡玛比率、胜率、盈亏比、lag-1自相关。
    """
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

    # 年化指标
    annual_return = daily_mean * 252.0
    annual_vol = daily_std * np.sqrt(252.0)
    sharpe = (annual_return - rf) / annual_vol if annual_vol > 1e-8 else 0.0

    # 累计净值与最大回撤
    cum = (1.0 + s).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_drawdown = float(drawdown.min())

    calmar = annual_return / abs(max_drawdown) if abs(max_drawdown) > 1e-8 else 0.0

    # 胜率与盈亏比
    wins = s[s > 0]
    losses = s[s < 0]
    win_rate = len(wins) / len(s) if len(s) > 0 else 0.0
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 1e-8 else 0.0

    # 滞后 1 阶自相关（用于诊断是否有平滑虚高）
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
    """使用 Moving Block Bootstrap 检验夏普比率的统计显著性。

    Parameters
    ----------
    ret_series : pd.Series
        日收益率序列。
    n_boot : int, default 10000
        Bootstrap 抽样次数。
    block_size : int, default 20
        块大小（用于保留时序自相关结构）。
    seed : int, default 42
        随机种子。

    Returns
    -------
    Dict[str, Any]
        包含 observed_sharpe, p_value, ci_95_low, ci_95_high, boot_sharpes。
    """
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

    # 构造重叠 blocks
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
