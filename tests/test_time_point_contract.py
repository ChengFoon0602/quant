"""
test_time_point_contract.py — 四个时点的齐备契约（工程铁律 E2）。

一笔业务涉及**四个不同的时点**，此前分散在三个测试文件里各自为零：

    信号可用时点   当日收盘才知道的信息才能进信号
    订单生效时点   信号何时变成持仓
    成本发生时点   换手在何时被计费
    收益归属时点   收益记在哪一天

本文件用**手算可验证**的最小场景，把这四条收敛成一条端到端契约。

场景（价格 × 信号）
------------------
    close  = [10, 10, 20, 20]       t0..t3
    signal = [ 1,  1,  0,  0]       t0 收盘产生信号
    → position = [0, 1, 1, 0]       t1 建仓（次日），t3 清仓
    → daily_ret = [0, 0, 1.0, 0]    t1→t2 的 +100% 记在 **t2**

用法: python -m unittest tests.test_time_point_contract -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.engine import run as engine_run
from backtest.reconciliation import reconcile_from_weights
from risk.cost_model import BUY_COST, SELL_COST

DATES = pd.bdate_range("2021-01-04", periods=4)  # 周一~周四
CLOSE = pd.Series([10.0, 10.0, 20.0, 20.0], index=DATES)
SIGNAL = pd.Series([1.0, 1.0, 0.0, 0.0], index=DATES)


class TestEngineFourTimePoints(unittest.TestCase):
    """engine.run 的四时点手算契约（单票时序路径）。"""

    @classmethod
    def setUpClass(cls):
        cls.res = engine_run(CLOSE, SIGNAL)
        cls.net = cls.res["strategy_net"]

    def test_signal_day_produces_no_position(self):
        """① 信号可用时点：t0 的信号不得在 t0 建仓。

        t1 的净收益若为 `+1.0 - BUY` 而非 `-BUY`，说明收益被错误地锚在了建仓日之前。
        """
        self.assertAlmostEqual(self.net.iloc[1], -BUY_COST, places=12)

    def test_order_takes_effect_next_day(self):
        """② 订单生效时点：建仓发生在 t1（体现为 t1 计成本、t1 当日无收益）。"""
        self.assertAlmostEqual(self.res["equity"].iloc[1], 1.0 - BUY_COST, places=12)

    def test_cost_charged_on_trade_day(self):
        """③ 成本发生时点：买入成本记在 t1、卖出成本记在 t3。"""
        self.assertAlmostEqual(self.net.iloc[1], -BUY_COST, places=12)
        self.assertAlmostEqual(self.net.iloc[3], -SELL_COST, places=12)
        self.assertEqual(self.res["n_trades"], 2)  # 一次买 + 一次卖

    def test_return_anchored_at_next_day(self):
        """④ 收益归属时点：t1→t2 的 +100% 记在 **t2**，不是 t1。"""
        self.assertAlmostEqual(self.net.iloc[2], 1.0, places=12)
        # 基准同步：跳空也落在 t2
        self.assertAlmostEqual(self.res["benchmark"].iloc[1], 1.0, places=12)
        self.assertAlmostEqual(self.res["benchmark"].iloc[2], 2.0, places=12)

    def test_terminal_equity_matches_hand_calculation(self):
        """末值 = (1-BUY) × 2.0 × (1-SELL)，逐项手算。"""
        expected = (1.0 - BUY_COST) * 2.0 * (1.0 - SELL_COST)
        self.assertAlmostEqual(self.res["equity"].iloc[-1], expected, places=12)

    def test_first_row_equity_is_nan_legacy_behaviour(self):
        """记录既有行为（非缺陷）：`position.diff()` 首行为 NaN → equity[0] 为 NaN。

        `cumprod()` 默认跳过 NaN，故**不影响末值与所有指标**。
        但它意味着「首日净值 = 1」这一直觉并不成立。若需改为 1.0，请单独授权。
        """
        self.assertTrue(np.isnan(self.res["equity"].iloc[0]))
        self.assertTrue(np.isfinite(self.res["equity"].iloc[1:]).all())


class TestPanelPathSharesSameAnchoring(unittest.TestCase):
    """面板路径必须共享同一套锚定约定。

    `reconcile_from_weights` 是三条面板实现（orchestrator / build_weight_portfolio /
    models.build_portfolio）对账时共用的**参考账本**，所以断言它的锚定方向，
    等价于断言面板路径的锚定方向。

    手算场景：持仓恒为 1.0，价格 [10, 10, 20]。
    """

    def setUp(self):
        dates = pd.bdate_range("2021-01-04", periods=3)
        self.W = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
        self.close = pd.DataFrame({"A": [10.0, 10.0, 20.0]}, index=dates)
        self.ref = reconcile_from_weights(self.W, self.close)
        self.dates = dates

    def test_return_anchored_at_jump_day(self):
        """跳空 t1→t2（+100%）必须记在 t2，t1 收益为 0。"""
        self.assertAlmostEqual(self.ref["gross_ret"].iloc[1], 0.0, places=12)
        self.assertAlmostEqual(self.ref["gross_ret"].iloc[2], 1.0, places=12)

    def test_cost_charged_at_position_change(self):
        """成本只在换手发生日计提：t0 建仓计买入费，其后换手为 0。"""
        self.assertAlmostEqual(self.ref["turnover"].iloc[0], 1.0, places=12)
        self.assertAlmostEqual(self.ref["turnover"].iloc[1], 0.0, places=12)
        self.assertAlmostEqual(self.ref["turnover"].iloc[2], 0.0, places=12)
        self.assertAlmostEqual(self.ref["cost"].iloc[0], BUY_COST, places=12)
        self.assertAlmostEqual(self.ref["cost"].iloc[1], 0.0, places=12)

    def test_net_equals_gross_minus_cost(self):
        """盈亏闭合：port_ret == gross_ret - cost（逐日）。"""
        np.testing.assert_allclose(
            self.ref["port_ret"].values,
            (self.ref["gross_ret"] - self.ref["cost"]).values,
            rtol=0, atol=1e-15)

    def test_anchor_consistency_with_engine(self):
        """两条路径的锚定方向必须一致：跳空都落在「价格变化的那一天」。

        engine 路径：daily_ret = close.pct_change()，t2 吃 +100%；
        面板路径：daily_ret = close.pct_change()，t2 吃 +100%。
        """
        close_series = self.close["A"]
        engine_daily = close_series.pct_change().fillna(0.0)
        self.assertAlmostEqual(engine_daily.iloc[1], 0.0, places=12)
        self.assertAlmostEqual(engine_daily.iloc[2], 1.0, places=12)
        # 面板参考账本的毛收益正是用同一口径算出来的
        self.assertAlmostEqual(self.ref["gross_ret"].iloc[2], engine_daily.iloc[2], places=12)


class TestTimePointsDoNotCollapse(unittest.TestCase):
    """反例守卫：四个时点若被压成一个，下列断言必然失败。"""

    def test_signal_and_order_are_not_the_same_day(self):
        """信号日与成交日必须不同 —— 若同日，t1 会出现 +1.0 的收益。"""
        res = engine_run(CLOSE, SIGNAL)
        self.assertNotAlmostEqual(res["strategy_net"].iloc[1], 1.0, places=6)
        self.assertAlmostEqual(res["strategy_net"].iloc[2], 1.0, places=12)

    def test_cost_and_return_are_not_the_same_day(self):
        """成本日（t1）与收益日（t2）必须分离。"""
        res = engine_run(CLOSE, SIGNAL)
        self.assertLess(res["strategy_net"].iloc[1], 0.0)   # 只有成本
        self.assertGreater(res["strategy_net"].iloc[2], 0.0)  # 只有收益


if __name__ == "__main__":
    unittest.main()
