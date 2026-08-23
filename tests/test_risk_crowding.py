"""
test_risk_crowding.py — risk/crowding 模块单元测试。
"""

import unittest
import numpy as np
import pandas as pd

from risk.crowding import (
    month_end_dates,
    wide_to_long,
    align_direction,
    factor_exposure_extreme_ratio,
    factor_monthly_returns,
    style_homogeneity,
    turnover_crowding,
    factor_return_spike,
    compute_crowding_indicators,
    compute_composite_crowding,
    detect_extreme_events,
)


class TestRiskCrowding(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # 生成两个月的每日数据，40 只股票，3 个因子
        dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
        self.dates = dates
        self.symbols = [f"{i:06d}" for i in range(40)]
        self.factor_names = ["alpha01", "alpha02", "alpha03"]

        # 构造 X_long: MultiIndex (date, symbol)
        idx = pd.MultiIndex.from_product([dates, self.symbols], names=["date", "symbol"])
        vals = np.random.randn(len(idx), len(self.factor_names))
        self.X_long = pd.DataFrame(vals, index=idx, columns=self.factor_names)

        # 构造 close_matrix
        prices = 10.0 * np.exp(np.cumsum(np.random.randn(len(dates), len(self.symbols)) * 0.02, axis=0))
        self.close_matrix = pd.DataFrame(prices, index=dates, columns=self.symbols)

        # 构造 turnover
        self.turnover = pd.Series(np.random.uniform(0.01, 0.05, len(dates)), index=dates)

    def test_month_end_dates(self):
        med = month_end_dates(self.dates)
        self.assertEqual(len(med), 6)  # 1月到6月共6个月末

    def test_wide_to_long_and_align_direction(self):
        tensor = {
            "f1": pd.DataFrame(np.ones((5, 5)), index=self.dates[:5], columns=self.symbols[:5]),
            "f2": pd.DataFrame(np.full((5, 5), 2.0), index=self.dates[:5], columns=self.symbols[:5]),
        }
        aligned = align_direction(tensor, {"f1": "+", "f2": "-"})
        self.assertEqual(aligned["f1"].iloc[0, 0], 1.0)
        self.assertEqual(aligned["f2"].iloc[0, 0], -2.0)

        long_df = wide_to_long(aligned)
        self.assertEqual(long_df.shape, (25, 2))
        self.assertIn("f1", long_df.columns)
        self.assertIn("f2", long_df.columns)

    def test_compute_crowding_indicators_and_composite(self):
        ind_df = compute_crowding_indicators(
            self.X_long,
            self.close_matrix,
            self.turnover,
            factor_cols=self.factor_names,
            min_stocks=10,
        )
        self.assertIn("C1_extreme_exposure", ind_df.columns)
        self.assertIn("C2_style_homogeneity", ind_df.columns)
        self.assertIn("C3_turnover_crowding", ind_df.columns)
        self.assertIn("C4_return_spike", ind_df.columns)

        comp_df = compute_composite_crowding(ind_df, expanding_min=2)
        self.assertIn("composite_z", comp_df.columns)

    def test_detect_extreme_events(self):
        dates = pd.date_range("2020-01-01", periods=20, freq="ME")
        z_scores = pd.Series(np.linspace(0, 3, 20), index=dates)
        events = detect_extreme_events(z_scores, quantile_thresh=0.85, merge_window_months=2)
        self.assertIsInstance(events, list)
        self.assertGreater(len(events), 0)


if __name__ == "__main__":
    unittest.main()
