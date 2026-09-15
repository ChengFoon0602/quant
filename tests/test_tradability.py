"""
test_tradability.py — 交易可行性度量的回归测试（risk/tradability.py）。

守护：
  1. `build_trade_limits` 用 `close.shift(1)` 作前收盘 —— 首日无前收盘不得判锁；
  2. `measure_restriction_impact` 的账本对比结构正确，且「无限制」时差值恒为 0
     （否则差值不可归因于约束本身）。

用法: python -m unittest tests.test_tradability -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from backtest.reconciliation import assert_closed
from data import fetcher
from risk.portfolio import build_weight_portfolio
from risk.tradability import (
    build_trade_limits,
    build_trade_limits_from_cache,
    measure_restriction_impact,
)

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


class TestBuildTradeLimitsFromCache(unittest.TestCase):
    """一步到位的接入路径：新策略一次调用即可拿到 trade_limits。"""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = Path(self.td.name)
        # 110 个交易日 ≥ load_field_panel 默认 min_rows=100
        dates = pd.bdate_range("2021-01-01", periods=110)
        close = pd.Series(100.0, index=dates)
        close.iloc[10:] = 90.0  # 第 10 个交易日起跌到 90 → 相对前收盘 −10%
        self.dates = dates
        px = pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                           "volume": 1e6, "amount": 1e8}, index=dates)
        px.reset_index(names="date").to_csv(self.dir / "600000.csv", index=False)

        patcher = mock.patch.object(
            fetcher, "_cache_path",
            side_effect=lambda s, adjust="2": self.dir / f"{s}.csv")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_matches_the_two_step_call(self):
        """一步版必须与「先 load_field_panel 再 build_trade_limits」逐位一致。"""
        up, down = build_trade_limits_from_cache(["600000"])
        panel = fetcher.load_field_panel(
            ["600000"], fields=("open", "high", "low", "close"))
        up2, down2 = build_trade_limits(
            panel["open"], panel["high"], panel["low"], panel["close"])
        pd.testing.assert_frame_equal(up, up2)
        pd.testing.assert_frame_equal(down, down2)

    def test_detects_limit_down_via_shift(self):
        _, down = build_trade_limits_from_cache(["600000"])
        self.assertTrue(bool(down.iloc[10, 0]), "相对前收盘 −10% 一字 → 跌停锁")
        self.assertFalse(bool(down.iloc[9, 0]), "跌停前一日不应被判锁")

    def test_shape_follows_the_shared_symbol_set(self):
        up, down = build_trade_limits_from_cache(["600000", "999999"])
        self.assertEqual(list(up.columns), ["600000"])
        self.assertEqual(up.shape, down.shape)


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


class TestRestrictedBookStillCloses(unittest.TestCase):
    """加涨跌停约束后账本仍须闭合 —— 覆盖层不能破坏清算恒等。"""

    @classmethod
    def setUpClass(cls):
        cls.close, cls.pred = _synthetic_panel()

    def test_restricted_book_closes(self):
        up = _mask(self.close, False)
        up.iloc[:, ::3] = True      # 1/3 标的一字涨停（禁买）
        down = _mask(self.close, False)
        down.iloc[:, 1::3] = True    # 另 1/3 一字跌停（禁卖）
        res, w = build_weight_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS,
            trade_limits=(up, down), return_weights=True)
        gaps = assert_closed(res, w, self.close)
        self.assertLess(max(gaps.values()), 1e-9)

    def test_restricted_book_differs_from_unrestricted(self):
        """确认掩码确实生效（否则上面的闭合是空转）。"""
        up = _mask(self.close, True)
        down = _mask(self.close, False)
        _, w_free = build_weight_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS, return_weights=True)
        _, w_lock = build_weight_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS,
            trade_limits=(up, down), return_weights=True)
        self.assertFalse(np.allclose(w_free.values, w_lock.values))


if __name__ == "__main__":
    unittest.main()
