"""
test_risk_portfolio.py — risk/portfolio 模块单元测试。
"""

import unittest
import numpy as np
import pandas as pd

from risk.portfolio import (
    build_weight_portfolio,
    calculate_metrics,
    bootstrap_sharpe_test,
)


class TestRiskPortfolio(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_days = 200
        n_stocks = 20
        self.dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
        self.symbols = [f"stock_{i:03d}" for i in range(n_stocks)]

        # 模拟预测信号与价格
        signals = np.random.randn(n_days, n_stocks)
        self.pred_df = pd.DataFrame(signals, index=self.dates, columns=self.symbols)

        prices = 100.0 * np.exp(np.cumsum(np.random.randn(n_days, n_stocks) * 0.02, axis=0))
        self.close_matrix = pd.DataFrame(prices, index=self.dates, columns=self.symbols)

    def test_build_weight_portfolio_columns_and_shape(self):
        res = build_weight_portfolio(
            self.pred_df,
            self.close_matrix,
            long_only=False,
            buy_cost=0.00026,
            sell_cost=0.00076,
            hold_days=5,
        )
        self.assertIsInstance(res, pd.DataFrame)
        for col in ["gross_ret", "cost", "turnover", "port_ret", "cum"]:
            self.assertIn(col, res.columns)

        # 检查 cost_deduction = buy_turnover*buy_cost + sell_turnover*sell_cost（方向分离）
        delta_w = res["turnover"]  # 此处 turnover 列存的是 |ΔW| 之和，非方向分离
        # 方向分离后的成本无法仅从 turnover 列反推，故只验证 port_ret = gross_ret - cost 恒等
        np.testing.assert_allclose(res["port_ret"].values, (res["gross_ret"] - res["cost"]).values, rtol=1e-5)
        # 成本应非负
        self.assertTrue((res["cost"].values >= 0).all())

    def test_long_only_and_short_only_mutual_exclusion(self):
        with self.assertRaises(ValueError):
            build_weight_portfolio(self.pred_df, self.close_matrix, long_only=True, short_only=True)

    def test_calculate_metrics(self):
        ret = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01], index=self.dates[:5])
        metrics = calculate_metrics(ret)
        self.assertIn("annual_return", metrics)
        self.assertIn("annual_vol", metrics)
        self.assertIn("sharpe", metrics)
        self.assertIn("max_drawdown", metrics)
        self.assertIn("win_rate", metrics)
        self.assertEqual(metrics["win_rate"], 3.0 / 5.0)

    def test_bootstrap_sharpe_test(self):
        ret = pd.Series(np.random.randn(100) * 0.01 + 0.001, index=self.dates[:100])
        res = bootstrap_sharpe_test(ret, n_boot=500, block_size=10, seed=42)
        self.assertIn("observed_sharpe", res)
        self.assertIn("p_value", res)
        self.assertIn("ci_95_low", res)
        self.assertIn("ci_95_high", res)
        self.assertLessEqual(res["ci_95_low"], res["ci_95_high"])


if __name__ == "__main__":
    unittest.main()
