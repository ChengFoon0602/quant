"""
test_viz_plotting.py — viz/plotting 模块单元测试。
"""

import unittest
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz.plotting import (
    plot_equity_curve,
    plot_crowding_timeseries,
    plot_stratified_returns,
    plot_bootstrap_distribution,
)

matplotlib.use("Agg")  # 无显示环境运行测试


class TestVizPlotting(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dates = pd.date_range("2023-01-01", periods=100, freq="B")

    def tearDown(self):
        plt.close("all")

    def test_plot_equity_curve(self):
        s1 = pd.Series((1 + np.random.randn(100) * 0.01).cumprod(), index=self.dates)
        s2 = pd.Series((1 + np.random.randn(100) * 0.01).cumprod(), index=self.dates)
        fig = plot_equity_curve({"策略A": s1, "基准": s2}, title="测试净值曲线")
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_crowding_timeseries(self):
        df = pd.DataFrame({
            "C1_extreme_exposure": np.random.uniform(0.2, 0.4, 100),
            "C2_style_homogeneity": np.random.uniform(0.1, 0.5, 100),
            "C3_turnover_crowding": np.random.uniform(-0.2, 0.5, 100),
            "C4_return_spike": np.random.uniform(0.01, 0.05, 100),
            "composite_z": np.random.randn(100),
        }, index=self.dates)
        fig = plot_crowding_timeseries(df, events=[(self.dates[50], 1.5)])
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_stratified_returns(self):
        rets = {"Q1(低)": 0.34, "Q2": 0.15, "Q3": 0.08, "Q4(高)": -0.12}
        t_stats = {"Q1(低)": 7.95, "Q2": 1.88, "Q3": 0.77, "Q4(高)": -1.50}
        n_obs = {"Q1(低)": 45, "Q2": 45, "Q3": 44, "Q4(高)": 45}
        fig = plot_stratified_returns(rets, t_stats=t_stats, n_obs=n_obs)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_bootstrap_distribution(self):
        boot_sharpes = np.random.randn(1000) * 0.5 + 0.8
        fig = plot_bootstrap_distribution(boot_sharpes, observed_sharpe=0.83, p_value=0.10, ci_low=-0.2, ci_high=1.8)
        self.assertIsInstance(fig, plt.Figure)


if __name__ == "__main__":
    unittest.main()
