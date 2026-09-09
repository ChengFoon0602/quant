"""
test_metrics.py — 年化口径统一（CLAUDE.md TODO #3）的回归测试。

守护目标：
  1. annualize_cagr = 真实路径复利（几何）；
  2. 算术年化 ≥ 几何年化（AM-GM 不等式），差异随波动放大；
  3. Sharpe 口径不变（mean/std*sqrt(252)）——统一只动「年化收益」，不动夏普；
  4. 既有入口（models/portfolio_backtest.performance_metrics、
     risk/portfolio.calculate_metrics）的 annual 已切换到 CAGR。

用法: python -m unittest tests.test_metrics -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.metrics import (
    annualize_arithmetic,
    annualize_cagr,
    max_drawdown,
    sharpe_ratio,
)


def _const_series(daily: float, n: int = 630) -> pd.Series:
    return pd.Series(daily, index=pd.date_range("2020-01-01", periods=n, freq="B"))


class TestAnnualizeCagr(unittest.TestCase):
    """几何年化 = 真实复利。"""

    def test_constant_return_compounds(self):
        """常数日收益下 CAGR = (1+d)^252-1（复利，而非 d*252）。"""
        daily = 0.0005
        s = _const_series(daily)
        cagr = annualize_cagr(s)
        expected = (1 + daily) ** 252 - 1
        self.assertAlmostEqual(cagr, expected, places=10)
        # 复利年化 > 算术年化（d*252）
        self.assertGreater(cagr, annualize_arithmetic(s))

    def test_matches_path_product(self):
        """与 (1+r).prod()^(252/n)-1 完全一致（真实路径复利）。"""
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0004, 0.01, size=500))
        expected = float(np.prod(1 + r) ** (252 / 500) - 1)
        self.assertAlmostEqual(annualize_cagr(r), expected, places=10)

    def test_empty_returns_nan(self):
        self.assertTrue(np.isnan(annualize_cagr(pd.Series([], dtype=float))))

    def test_geometric_leq_arithmetic_amgm(self):
        """AM-GM：几何年化 ≤ 算术年化，且波动越大差距越大。"""
        rng = np.random.default_rng(1)
        for sigma in [0.005, 0.02, 0.05]:
            r = pd.Series(rng.normal(0.0005, sigma, size=1000))
            cagr = annualize_cagr(r)
            arith = annualize_arithmetic(r)
            self.assertLessEqual(cagr, arith + 1e-12,
                                 f"σ={sigma}: CAGR 应 ≤ 算术年化")
            self.assertAlmostEqual(cagr, arith, delta=0.5 * sigma**2 * 252 + 0.02,
                                   msg=f"σ={sigma}: 差距应约等于方差拖累量级")

    def test_all_negative_path_returns_nan(self):
        """累计净值非正（出现 1+r<0）→ nan（复利幂无实义）。"""
        r = pd.Series([-1.5, 0.1, 0.1])  # prod(1+r) = (-0.5)*1.1*1.1 < 0
        self.assertTrue(np.isnan(annualize_cagr(r)))


class TestSharpeAndDrawdown(unittest.TestCase):
    """Sharpe/MDD 口径稳定。"""

    def test_sharpe_is_mean_over_std_sqrt252(self):
        rng = np.random.default_rng(2)
        r = pd.Series(rng.normal(0.0004, 0.012, size=800))
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        self.assertAlmostEqual(sharpe_ratio(r), expected, places=12)

    def test_sharpe_rf_equivalent(self):
        """rf>0 时 = (mean*252-rf)/(std*sqrt252)，与旧口径等价。"""
        rng = np.random.default_rng(3)
        r = pd.Series(rng.normal(0.0005, 0.01, size=800))
        rf = 0.02
        old = (r.mean() * 252 - rf) / (r.std(ddof=1) * np.sqrt(252))
        self.assertAlmostEqual(sharpe_ratio(r, rf_annual=rf), old, places=12)

    def test_max_drawdown_negative(self):
        cum_ret = pd.Series([0.1, 0.2, -0.3, 0.1])  # 从 0.2 跌到 -0.3
        # 净值路径: 1.1, 1.32, 0.924, 1.0164 → 回撤 = 0.924/1.32-1 = -0.30
        md = max_drawdown(cum_ret)
        self.assertAlmostEqual(md, -0.3, places=6)


class TestEntrypointsUnified(unittest.TestCase):
    """既有入口的 annual 已是 CAGR（CLAUDE.md TODO #3 收口）。"""

    def test_performance_metrics_annual_is_cagr(self):
        from models.portfolio_backtest import performance_metrics

        rng = np.random.default_rng(4)
        r = pd.Series(rng.normal(0.0004, 0.015, size=900))
        m = performance_metrics(r)
        self.assertAlmostEqual(m["annual"], annualize_cagr(r), places=12)
        # Sharpe 口径未变
        self.assertAlmostEqual(
            m["sharpe"], r.mean() / r.std() * np.sqrt(252), places=12)

    def test_calculate_metrics_annual_is_cagr_and_sharpe_stable(self):
        from risk.portfolio import calculate_metrics

        rng = np.random.default_rng(5)
        r = pd.Series(rng.normal(0.0004, 0.012, size=700))
        m = calculate_metrics(r)
        self.assertAlmostEqual(m["annual_return"], annualize_cagr(r), places=12)
        self.assertAlmostEqual(
            m["sharpe"], r.mean() / r.std(ddof=1) * np.sqrt(252), places=12)
        self.assertIn("annual_return_arith", m)

    def test_cagr_differs_when_volatile(self):
        """高波动下 CAGR 明显低于算术年化（否则两口径等同，无统一意义）。"""
        rng = np.random.default_rng(6)
        r = pd.Series(rng.normal(0.001, 0.05, size=1200))  # 高波动 ~25% 年化 vol
        diff = annualize_arithmetic(r) - annualize_cagr(r)
        self.assertGreater(diff, 0.005,
                           "高波动下 CAGR 与算术年化应有可感知差距")


if __name__ == "__main__":
    unittest.main()
