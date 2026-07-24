"""
test_no_future_leak.py — 未来函数边界测试骨架。

验证回测引擎的信号偏移 (signal.shift(1)) 和
财务数据 PIT 对齐是否正确。

用法: python -m unittest tests.test_no_future_leak -v
     或: pytest tests/ -v
"""

import unittest
import numpy as np
import pandas as pd

try:
    import pytest
    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False


class TestNoFutureLeak(unittest.TestCase):
    """防未来函数 + 交易成本边界测试。"""

    def test_signal_shift_prevents_look_ahead(self):
        """当日收盘产生的信号不能用于当日交易。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        close = pd.Series([10, 11, 12, 11, 13], index=dates)
        signal = pd.Series([0, 0, 1, 0, 0], index=dates)

        position = signal.shift(1).fillna(0)
        daily_ret = close.pct_change().fillna(0)
        strategy_ret = position * daily_ret  # noqa: F841

        signal_date = dates[2]
        self.assertEqual(
            position.loc[signal_date], 0,
            f"未来函数：信号在 {signal_date} 当天被用于交易"
        )
        next_date = dates[3]
        self.assertEqual(
            position.loc[next_date], 1,
            f"信号偏移失败：{next_date} 应有仓位"
        )

    def test_financial_data_pit_alignment(self):
        """财务数据必须用 Report Date 对齐，不能用 End Date。"""
        end_date = pd.Timestamp("2023-12-31")
        report_date = pd.Timestamp("2024-04-15")
        backtest_date = pd.Timestamp("2024-01-15")

        use_by_end_date = end_date <= backtest_date
        use_by_report_date = report_date <= backtest_date

        self.assertNotEqual(
            use_by_end_date, use_by_report_date,
            "End Date 和 Report Date 应产生不同结果"
        )
        self.assertFalse(
            use_by_report_date,
            f"PIT 错误：{backtest_date.date()} 不应访问截止 "
            f"{end_date.date()} 的财务数据，该数据在 {report_date.date()} 才发布"
        )

    def test_turnover_cost_applied(self):
        """验证换手成本被正确扣除。"""
        n_days = 252
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        np.random.seed(42)
        ret = pd.Series(np.random.randn(n_days) * 0.02, index=dates)

        turnover = pd.Series(1.0, index=dates)
        buy_cost, sell_cost = 0.00026, 0.00076
        cost = turnover * (buy_cost + sell_cost)

        net_ret = ret - cost
        gross_equity = (1 + ret).cumprod()
        net_equity = (1 + net_ret).cumprod()

        self.assertLess(
            net_equity.iloc[-1], gross_equity.iloc[-1] * 0.9,
            f"换手成本扣除不足：net={net_equity.iloc[-1]:.2f}, "
            f"gross={gross_equity.iloc[-1]:.2f}"
        )
