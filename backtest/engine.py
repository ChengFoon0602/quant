"""
向量化回测引擎 — 单票、信号驱动的日线回测。

核心假设：
- 当日收盘信号决定次日开盘操作（无未来函数）
- 全仓进出：position ∈ {0, 1}
- 交易成本从收益中扣除
"""

import numpy as np
import pandas as pd


def run(close: pd.Series, signal: pd.Series, commission: float = 0.0003) -> dict:
    """向量化单票回测。

    Parameters
        close: 收盘价序列
        signal: 策略信号（>=1 → 全仓买入，<=0 → 空仓），长度与 close 一致
        commission: 单边费率

    Returns
        dict: equity, benchmark, metrics
    """
    # 持仓信号前移一天：今天收盘的信号决定明天操作
    position = signal.shift(1).fillna(0).clip(0, 1)

    daily_ret = close.pct_change().fillna(0)
    strategy_ret = position * daily_ret

    # 交易成本
    turnover = position.diff().abs()
    strategy_net = strategy_ret - turnover * commission

    equity = (1 + strategy_net).cumprod()
    benchmark = (1 + daily_ret).cumprod()

    # 绩效指标
    n = len(close)
    n_years = (close.index[-1] - close.index[0]).days / 365.25

    total_ret = equity.iloc[-1] - 1
    bm_ret = benchmark.iloc[-1] - 1
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    dd = equity / equity.cummax() - 1
    max_dd = dd.min()

    excess = strategy_net - 0.02 / 252
    sharpe = np.sqrt(252) * excess.mean() / excess.std() if excess.std() > 1e-12 else 0

    n_trades = int(turnover.sum())

    return {
        "equity": equity,
        "benchmark": benchmark,
        "total_return": total_ret,
        "ann_return": ann_ret,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
    }
