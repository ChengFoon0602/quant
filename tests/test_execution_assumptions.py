"""
test_execution_assumptions.py — 成交假设的自动断言（`docs/回测语义对照表.md` §2）。

两部分：

  **A 涨跌停识别**  补齐既有覆盖缺口：一字跌停、ST 5%、北交所 30%。
     （既有 `test_signals_preprocess.py::test_detect_limit_moves` 只覆盖主板涨停 + 创业板限额）

  **B 撮合层真正拦单**  这才是「成交假设」的实质 —— 掩码生成了不等于成交被拦住。
     断言掩码**方向正确**（涨停禁买 / 跌停禁卖）且**不是全局冻结**。

用法: python -m unittest tests.test_execution_assumptions -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from risk.orchestrator import PortfolioOrchestrator
from risk.portfolio import build_weight_portfolio, detect_limit_moves

HOLD_DAYS = 5
D0 = pd.Timestamp("2021-03-01")


def _one(symbol: str, price: float) -> pd.DataFrame:
    return pd.DataFrame([[price]], index=[D0], columns=[symbol])


def _pre_close(symbol: str, price: float = 10.0) -> pd.DataFrame:
    return _one(symbol, price)


class TestLimitMoveDetection(unittest.TestCase):
    """A：涨跌停识别（一字板 = open 触及限价 且 high == low）。"""

    def test_main_board_limit_up(self):
        px = _one("600000", 11.0)  # +10%
        up, down = detect_limit_moves(px, px, px, _pre_close("600000"))
        self.assertTrue(up.loc[D0, "600000"])
        self.assertFalse(down.loc[D0, "600000"])

    def test_main_board_limit_down(self):
        px = _one("600000", 9.0)  # -10%
        up, down = detect_limit_moves(px, px, px, _pre_close("600000"))
        self.assertTrue(down.loc[D0, "600000"])
        self.assertFalse(up.loc[D0, "600000"])

    def test_st_five_percent_band(self):
        px = _one("600000", 9.5)  # -5%
        _, down_st = detect_limit_moves(px, px, px, _pre_close("600000"),
                                        st_symbols={"600000"})
        self.assertTrue(down_st.loc[D0, "600000"], "ST 股应按 5% 带宽识别跌停")
        # 同样的 -5% 在非 ST（10% 带宽）下不是跌停
        _, down_plain = detect_limit_moves(px, px, px, _pre_close("600000"))
        self.assertFalse(down_plain.loc[D0, "600000"])

    def test_bse_thirty_percent_band(self):
        px30 = _one("830001", 7.0)  # -30%
        _, down30 = detect_limit_moves(px30, px30, px30, _pre_close("830001"))
        self.assertTrue(down30.loc[D0, "830001"], "北交所应按 30% 带宽识别跌停")
        # -20% 对北交所不算跌停
        px20 = _one("830001", 8.0)
        _, down20 = detect_limit_moves(px20, px20, px20, _pre_close("830001"))
        self.assertFalse(down20.loc[D0, "830001"])

    def test_star_market_twenty_percent_band(self):
        px_up = _one("688001", 12.0)  # +20%
        up, _ = detect_limit_moves(px_up, px_up, px_up, _pre_close("688001"))
        self.assertTrue(up.loc[D0, "688001"])

    def test_not_locked_when_high_differs_from_low(self):
        """非一字板（盘中有成交区间）不应被当作锁死。"""
        open_m = _one("600000", 11.0)
        high_m = _one("600000", 11.0)
        low_m = _one("600000", 10.5)
        up, _ = detect_limit_moves(open_m, high_m, low_m, _pre_close("600000"))
        self.assertFalse(up.loc[D0, "600000"])


class TestExecutionRespectsMask(unittest.TestCase):
    """B：掩码必须真的拦住成交，且方向正确。"""

    @staticmethod
    def _panel(n_days: int = 120, n_syms: int = 20, seed: int = 5):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        syms = [f"{600000 + i:06d}" for i in range(n_syms)]
        rets = rng.normal(0.0003, 0.02, size=(n_days, n_syms))
        close = pd.DataFrame(100.0 * np.cumprod(1 + rets, axis=0), index=dates, columns=syms)
        pred = pd.DataFrame(rng.normal(size=(n_days, n_syms)), index=dates, columns=syms)
        return close, pred

    @staticmethod
    def _mask(close: pd.DataFrame, value: bool) -> pd.DataFrame:
        return pd.DataFrame(value, index=close.index, columns=close.columns)

    def test_orchestrator_execute_keeps_prev_when_untradeable(self):
        orch = PortfolioOrchestrator(rebalance="monthly", max_leverage=2.0)
        prev = pd.Series({"A": 0.5, "B": -0.5})
        target = pd.Series({"A": 0.1, "B": -0.9})
        got = orch.execute(target, prev, pd.Series({"A": True, "B": False}))
        self.assertAlmostEqual(got["A"], 0.5, msg="不可交易资产应维持 prev_w")
        self.assertAlmostEqual(got["B"], -0.9, msg="可交易资产应取目标权重")

    def test_orchestrator_without_mask_takes_target(self):
        orch = PortfolioOrchestrator(rebalance="monthly", max_leverage=2.0)
        prev = pd.Series({"A": 0.5})
        got = orch.execute(pd.Series({"A": 0.9}), prev, None)
        self.assertAlmostEqual(got["A"], 0.9)

    def test_all_limit_up_blocks_long_entry(self):
        """全市场一字涨停 → 多头无法建仓 → 持仓应恒为 0。"""
        close, pred = self._panel()
        up = self._mask(close, True)
        down = self._mask(close, False)
        _, W = build_weight_portfolio(
            pred, close, long_only=True, hold_days=HOLD_DAYS,
            trade_limits=(up, down), return_weights=True)
        self.assertLessEqual(float(W.abs().to_numpy().max()), 1e-12,
                             "一字涨停未拦住买入：多头持仓应恒为 0")

    def test_all_limit_down_blocks_short_entry(self):
        """全市场一字跌停 → 空头无法建仓 → 持仓应恒为 0。"""
        close, pred = self._panel()
        up = self._mask(close, False)
        down = self._mask(close, True)
        _, W = build_weight_portfolio(
            pred, close, short_only=True, hold_days=HOLD_DAYS,
            trade_limits=(up, down), return_weights=True)
        self.assertLessEqual(float(W.abs().to_numpy().max()), 1e-12,
                             "一字跌停未拦住卖出：空头持仓应恒为 0")

    def test_partial_mask_changes_holdings_without_freezing_all(self):
        """掩码是选择性的：半数标的一字涨停时，持仓改变但未被整体冻结。"""
        close, pred = self._panel()
        _, W0 = build_weight_portfolio(
            pred, close, long_only=True, hold_days=HOLD_DAYS, return_weights=True)

        up = self._mask(close, False)
        up.iloc[:, ::2] = True  # 一半标的一字涨停
        down = self._mask(close, False)
        _, W1 = build_weight_portfolio(
            pred, close, long_only=True, hold_days=HOLD_DAYS,
            trade_limits=(up, down), return_weights=True)

        self.assertFalse(np.allclose(W0.values, W1.values),
                         "掩码未生效：受限持仓与无限制持仓完全相同")
        self.assertGreater(float(W1.abs().to_numpy().max()), 0.0,
                           "掩码不应冻结全部持仓")

    def test_masks_not_provided_means_unconstrained(self):
        close, pred = self._panel()
        _, W_none = build_weight_portfolio(
            pred, close, long_only=True, hold_days=HOLD_DAYS, return_weights=True)
        up = self._mask(close, False)  # 全 False = 无限制
        down = self._mask(close, False)
        _, W_all_false = build_weight_portfolio(
            pred, close, long_only=True, hold_days=HOLD_DAYS,
            trade_limits=(up, down), return_weights=True)
        np.testing.assert_allclose(W_none.values, W_all_false.values, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
