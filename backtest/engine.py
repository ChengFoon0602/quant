"""
向量化回测引擎 — 单票、信号驱动的日线回测。

A 股交易成本（2026 年标准）：
- 买入：佣金 0.025%（万2.5）+ 过户费 0.001%
- 卖出：佣金 0.025% + 印花税 0.05% + 过户费 0.001%
- 合计：买入 ~0.026%，卖出 ~0.076%

核心假设：
- 当日收盘信号决定次日开盘操作（无未来函数）
- 全仓进出：position ∈ {0, 1}
- 无涨跌停/停牌处理（日线级别回测可忽略）
"""

import numpy as np
import pandas as pd


def run(
    close: pd.Series,
    signal: pd.Series,
    buy_cost: float = 0.00026,
    sell_cost: float = 0.00076,
) -> dict:
    """向量化单票回测。

    Parameters
        close: 收盘价序列
        signal: 策略信号（>=1 → 全仓买入，<=0 → 空仓），长度与 close 一致
        buy_cost: 买入单边费率（默认 A 股万2.6）
        sell_cost: 卖岀单边费率（默认 A 股万7.6，含印花税）

    Returns
        dict: equity, benchmark, strategy_net, metrics
    """
    position = signal.shift(1).fillna(0).clip(0, 1)

    daily_ret = close.pct_change().fillna(0)
    strategy_ret = position * daily_ret

    # 交易成本：区分买入和卖出
    turnover = position.diff()
    turnover_buy = turnover.clip(lower=0)   # +1 = 买入日
    turnover_sell = (-turnover).clip(lower=0)  # +1 = 卖出日
    strategy_net = strategy_ret - turnover_buy * buy_cost - turnover_sell * sell_cost

    equity = (1 + strategy_net).cumprod()
    benchmark = (1 + daily_ret).cumprod()

    n_years = (close.index[-1] - close.index[0]).days / 365.25

    total_ret = equity.iloc[-1] - 1
    bm_ret = benchmark.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    excess = strategy_net - 0.02 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 1e-12 else 0

    n_trades = int((turnover_buy + turnover_sell).sum())

    return {
        "equity": equity,
        "benchmark": benchmark,
        "strategy_net": strategy_net,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
    }
