"""
test_invariants.py — 回测输入契约断言（backtest/invariants.py）的回归测试。

守护目标：
  1. 坏输入必须 raise InvariantViolation —— 不是 warn、不是静默通过；
  2. 合法输入不得误报（特别是真实数据里普遍存在的 NaN 形态）；
  3. 成本防错上界能拦住 0.003 事故值，同时放行 ETF 0.0004。

用法: python -m unittest tests.test_invariants -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.invariants import (
    MAX_ROUND_TRIP_COST,
    InvariantViolation,
    assert_backtest_inputs,
    assert_cost_params,
    assert_price_panel,
    assert_returns_panel,
    assert_weight_matrix,
)


def _panel(n_days: int = 10, n_cols: int = 3, seed: int = 0) -> pd.DataFrame:
    """构造形态合法的价格面板。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    cols = [f"{600000 + i:06d}" for i in range(n_cols)]
    return pd.DataFrame(100.0 + rng.normal(0, 1, (n_days, n_cols)),
                        index=dates, columns=cols)


class TestPricePanel(unittest.TestCase):
    def test_valid_panel_passes(self):
        assert_price_panel(_panel())

    def test_series_accepted(self):
        """单票序列（engine.py 的入参形态）必须接受。"""
        assert_price_panel(_panel()["600000"])

    def test_negative_price_raises(self):
        df = _panel()
        df.iloc[3, 1] = -12.5
        with self.assertRaises(InvariantViolation):
            assert_price_panel(df)

    def test_unsorted_index_raises(self):
        df = _panel().iloc[::-1]
        with self.assertRaises(InvariantViolation) as cm:
            assert_price_panel(df)
        self.assertIn("单调", str(cm.exception))

    def test_duplicate_index_raises(self):
        df = _panel()
        df.index = list(df.index[:-1]) + [df.index[1]]
        with self.assertRaises(InvariantViolation) as cm:
            assert_price_panel(df)
        self.assertIn("重复", str(cm.exception))

    def test_all_nan_raises(self):
        df = _panel()
        df.iloc[:, :] = np.nan
        with self.assertRaises(InvariantViolation):
            assert_price_panel(df)

    def test_all_nan_column_raises(self):
        df = _panel()
        df.iloc[:, 0] = np.nan
        with self.assertRaises(InvariantViolation):
            assert_price_panel(df)

    def test_isolated_nan_is_allowed(self):
        """个股未上市/停牌导致的 NaN 是真实数据形态，不得误报。"""
        df = _panel()
        df.iloc[0, 0] = np.nan
        df.iloc[1, 0] = np.nan
        assert_price_panel(df)

    def test_zero_price_is_allowed(self):
        """本模块只拦「必然错误」：非负条件放行 0，不越界到「通常为真」的判断。"""
        df = _panel()
        df.iloc[2, 1] = 0.0
        assert_price_panel(df)

    def test_non_frame_raises(self):
        with self.assertRaises(InvariantViolation):
            assert_price_panel([1, 2, 3])


class TestReturnsPanel(unittest.TestCase):
    def test_valid_passes(self):
        assert_returns_panel(_panel().pct_change())

    def test_below_minus_one_raises(self):
        r = _panel().pct_change()
        r.iloc[2, 0] = -1.5
        with self.assertRaises(InvariantViolation):
            assert_returns_panel(r)

    def test_inf_raises(self):
        r = _panel().pct_change()
        r.iloc[2, 0] = np.inf
        with self.assertRaises(InvariantViolation):
            assert_returns_panel(r)

    def test_exactly_minus_one_allowed(self):
        """r = -1（价格归零）在数学上合法，不得误报。"""
        r = _panel().pct_change()
        r.iloc[2, 0] = -1.0
        assert_returns_panel(r)


class TestWeightMatrix(unittest.TestCase):
    @staticmethod
    def _w() -> pd.DataFrame:
        dates = pd.bdate_range("2021-01-01", periods=5)
        return pd.DataFrame(
            [[0.5, -0.5], [0.3, -0.3], [0.0, 0.0], [0.6, -0.6], [0.4, -0.4]],
            index=dates, columns=["A", "B"],
        )

    def test_valid_passes(self):
        assert_weight_matrix(self._w(), max_gross=2.0)

    def test_ls_net_zero_gross_two_passes(self):
        """多空对冲：净敞口 0、总杠杆 Σ|w|=2 —— 必须放行，否则 LS 策略全被拦。"""
        w = pd.DataFrame([[0.5, -0.5]], columns=["A", "B"])
        assert_weight_matrix(w, max_gross=2.0)

    def test_nan_raises(self):
        w = self._w()
        w.iloc[0, 0] = np.nan
        with self.assertRaises(InvariantViolation):
            assert_weight_matrix(w)

    def test_gross_breach_raises(self):
        w = self._w().copy()
        w.iloc[0] = [1.2, -1.2]  # Σ|w| = 2.4 > 2
        with self.assertRaises(InvariantViolation):
            assert_weight_matrix(w, max_gross=2.0)

    def test_single_asset_breach_raises(self):
        w = self._w().copy()
        w.iloc[0, 0] = 0.9
        with self.assertRaises(InvariantViolation):
            assert_weight_matrix(w, max_weight_per_asset=0.5)

    def test_no_bound_skips_bound_checks(self):
        """未给上限时只做结构检查（NaN），不擅自加约束。"""
        w = self._w().copy()
        w.iloc[0, 0] = 5.0
        assert_weight_matrix(w)


class TestCostParams(unittest.TestCase):
    def test_truth_passes(self):
        assert_cost_params(0.00026, 0.00076)

    def test_etf_passes(self):
        assert_cost_params(0.0002, 0.0002)  # 双边合计 0.0004

    def test_legacy_bug_raises(self):
        """0.003（铁律 3 倍）必须被拦下 —— 这是本模块存在的首要理由。"""
        with self.assertRaises(InvariantViolation):
            assert_cost_params(0.0015, 0.0015)  # 合计 0.003

    def test_negative_raises(self):
        with self.assertRaises(InvariantViolation):
            assert_cost_params(-0.0001, 0.00076)

    def test_nan_raises(self):
        with self.assertRaises(InvariantViolation):
            assert_cost_params(np.nan, 0.00076)

    def test_ceiling_sits_between_truth_and_incident(self):
        """上界必须高于真值、低于事故值，否则形同虚设或误伤。"""
        self.assertGreater(MAX_ROUND_TRIP_COST, 0.00102)
        self.assertLess(MAX_ROUND_TRIP_COST, 0.003)


class TestBacktestInputs(unittest.TestCase):
    def test_combined_passes(self):
        assert_backtest_inputs(_panel(), buy_cost=0.00026, sell_cost=0.00076)

    def test_partial_cost_raises(self):
        with self.assertRaises(InvariantViolation):
            assert_backtest_inputs(_panel(), buy_cost=0.00026)

    def test_bad_cost_raises(self):
        with self.assertRaises(InvariantViolation):
            assert_backtest_inputs(_panel(), buy_cost=0.0015, sell_cost=0.0015)

    def test_bad_price_raises(self):
        df = _panel()
        df.iloc[1, 1] = -1.0
        with self.assertRaises(InvariantViolation):
            assert_backtest_inputs(df, buy_cost=0.00026, sell_cost=0.00076)


if __name__ == "__main__":
    unittest.main()
