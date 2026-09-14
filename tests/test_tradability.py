"""
test_tradability.py — 交易可行性度量的回归测试（risk/tradability.py）。

守护：
  1. `build_trade_limits` 用 `close.shift(1)` 作前收盘 —— 首日无前收盘不得判锁；
  2. `measure_restriction_impact` 的账本对比结构正确，且「无限制」时差值恒为 0
     （否则差值不可归因于约束本身）。

用法: python -m unittest tests.test_tradability -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risk.tradability import build_trade_limits, measure_restriction_impact

HOLD_DAYS = 5


def _synthetic_panel(n_days: int = 120, n_syms: int = 20, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    syms = [f"{600000 + i:06d}" for i in range(n_syms)]
    rets = rng.normal(0.0003, 0.02, size=(n_days, n_syms))
    close = pd.DataFrame(100.0 * np.cumprod(1 + rets, axis=0), index=dates, columns=syms)
    pred = pd.DataFrame(rng.normal(size=(n_days, n_syms)), index=dates, columns=syms)
    return close, pred


def _mask(close: pd.DataFrame, value: bool) -> pd.DataFrame:
    return pd.DataFrame(value, index=close.index, columns=close.columns)


class TestBuildTradeLimits(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2021-01-04", periods=3)

    def _px(self, values):
        return pd.DataFrame({"600000": values}, index=self.dates)

    def test_previous_close_is_the_reference(self):
        """前收盘 = close.shift(1)：第 2 日相对第 1 日 -10% 一字 → 跌停锁。"""
        px = self._px([10.0, 9.0, 9.0])
        up, down = build_trade_limits(px, px, px, px)
        self.assertFalse(bool(down.iloc[0, 0]), "首日无前收盘，不得判锁")
        self.assertTrue(bool(down.iloc[1, 0]), "-10% 一字应判跌停锁")
        self.assertFalse(bool(down.iloc[2, 0]), "第 3 日相对第 2 日无变化，不应判锁")

    def test_limit_up_via_shift(self):
        px = self._px([10.0, 11.0, 11.0])
        up, down = build_trade_limits(px, px, px, px)
        self.assertTrue(bool(up.iloc[1, 0]), "+10% 一字应判涨停锁")
        self.assertFalse(bool(down.iloc[1, 0]))

    def test_first_row_never_locked(self):
        """首行前收盘为 NaN → 整个首行不可能被判锁。"""
        px = self._px([10.0, 20.0, 5.0])
        up, down = build_trade_limits(px, px, px, px)
        self.assertFalse(bool(up.iloc[0, 0]))
        self.assertFalse(bool(down.iloc[0, 0]))

    def test_shape_matches_input(self):
        close, _ = _synthetic_panel()
        up, down = build_trade_limits(close, close, close, close)
        self.assertEqual(up.shape, close.shape)
        self.assertEqual(down.shape, close.shape)


class TestMeasureRestrictionImpact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.close, cls.pred = _synthetic_panel()

    def test_table_structure(self):
        table = measure_restriction_impact(
            self.pred, self.close, (_mask(self.close, False), _mask(self.close, False)),
            hold_days=HOLD_DAYS)
        self.assertEqual(list(table.index), ["unrestricted", "restricted", "delta"])
        for col in ("total_turnover", "n_trade_days", "total_return", "cagr",
                    "sharpe", "max_drawdown"):
            self.assertIn(col, table.columns)

    def test_no_restriction_means_zero_delta(self):
        """掩码全 False = 无限制 → 差值恒为 0（否则差值不可归因于约束）。"""
        table = measure_restriction_impact(
            self.pred, self.close, (_mask(self.close, False), _mask(self.close, False)),
            hold_days=HOLD_DAYS)
        for col in ("total_turnover", "total_return", "sharpe", "max_drawdown"):
            self.assertAlmostEqual(float(table.loc["delta", col]), 0.0, places=9,
                                   msg=f"{col} 在无限制下差值应为 0")

    def test_all_limit_up_reduces_long_only_turnover(self):
        """全市场一字涨停 → 多头无法建仓 → 换手必然下降。"""
        table = measure_restriction_impact(
            self.pred, self.close, (_mask(self.close, True), _mask(self.close, False)),
            hold_days=HOLD_DAYS, long_only=True)
        self.assertLess(float(table.loc["delta", "total_turnover"]), 0.0)
        self.assertLess(float(table.loc["delta", "n_trade_days"]), 0.0)

    def test_delta_is_restricted_minus_unrestricted(self):
        table = measure_restriction_impact(
            self.pred, self.close, (_mask(self.close, True), _mask(self.close, False)),
            hold_days=HOLD_DAYS, long_only=True)
        expected = (float(table.loc["restricted", "total_turnover"])
                    - float(table.loc["unrestricted", "total_turnover"]))
        self.assertAlmostEqual(float(table.loc["delta", "total_turnover"]), expected, places=9)


if __name__ == "__main__":
    unittest.main()
