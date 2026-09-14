"""
test_reconciliation.py — 回测结果自洽性对账（账本一致性）。

思路
----
把清算的「对账」迁移到回测：组合收益是一本账，必须能自证闭合。三个对账项：

    仓位对账   实际持仓满足杠杆/单资产上限，且列集合与价格矩阵一致
    流水对账   turnover == Σ|ΔW|；cost == buy_turnover·BUY + sell_turnover·SELL
    盈亏核对   port_ret == gross_ret - cost（逐日闭合）；cum == Π(1+port_ret)

对账方式是**独立重算**：从实际持仓矩阵出发重新推一遍损益，与实现给出的
结果逐位比对。这与「用实现自己的中间量验证实现」有本质区别 —— 后者是空转。

覆盖三条独立实现：
    risk/portfolio.py::build_weight_portfolio
    risk/orchestrator.py::PortfolioOrchestrator.run
    models/portfolio_backtest.py::build_portfolio

除对账外，还做一次跨实现语义比对，并把一个已核实的真实口径差异**钉住**
（见 TestKnownDivergence）。

用法: python -m unittest tests.test_reconciliation -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.invariants import InvariantViolation, assert_weight_matrix
from risk.cost_model import BUY_COST, SELL_COST
from risk.orchestrator import PortfolioOrchestrator
from risk.portfolio import build_weight_portfolio

try:
    from models.portfolio_backtest import build_portfolio as _models_build_portfolio
except Exception:  # noqa: BLE001  —— 可选依赖缺失时跳过该组用例
    _models_build_portfolio = None

HOLD_DAYS = 5
TOP_Q = 0.20
BOTTOM_Q = 0.20
_TOL = 1e-10


def synthetic_panel(n_days: int = 400, n_syms: int = 30, seed: int = 7):
    """几何随机游走价格面板 + 截面预测信号（固定种子，完全可复现）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    syms = [f"{600000 + i:06d}" for i in range(n_syms)]
    rets = rng.normal(0.0003, 0.02, size=(n_days, n_syms))
    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0), index=dates, columns=syms)
    pred = pd.DataFrame(rng.normal(size=(n_days, n_syms)), index=dates, columns=syms)
    return close, pred


def reconcile_from_weights(W_held, close, buy_cost=BUY_COST, sell_cost=SELL_COST):
    """从实际持仓矩阵独立重算 (gross, cost, port_ret, turnover) —— 对账参考账本。"""
    cols = W_held.columns.intersection(close.columns)
    daily_ret = close.loc[W_held.index, cols].pct_change()
    W = W_held[cols]

    gross = (W.shift(1).fillna(0.0) * daily_ret).sum(axis=1)

    dW = W - W.shift(1).fillna(0.0)
    buy_to = dW.clip(lower=0.0).sum(axis=1)
    sell_to = (-dW).clip(lower=0.0).sum(axis=1)
    cost = buy_to * buy_cost + sell_to * sell_cost

    return gross, cost, gross - cost, dW.abs().sum(axis=1)


def assert_closed(res, W_held, close, buy_cost=BUY_COST, sell_cost=SELL_COST):
    """三类对账 + 净值路径一致性。第一行 shift 无前值，两侧一致跳过。"""
    # ── 仓位对账 ──
    assert_weight_matrix(W_held, name="W_held")
    if set(W_held.columns) != set(close.columns):
        raise AssertionError("仓位对账失败：持仓列集合与价格矩阵不一致")

    gross, cost, port, turnover = reconcile_from_weights(W_held, close, buy_cost, sell_cost)
    idx = gross.index[1:]  # 第一行缺少前一日持仓，参考账本无法复现

    # ── 流水对账 ──
    np.testing.assert_allclose(res["turnover"].reindex(idx).values,
                               turnover.reindex(idx).values, rtol=0, atol=_TOL)

    # ── 盈亏核对 ──
    np.testing.assert_allclose(res["port_ret"].reindex(idx).values,
                               port.reindex(idx).values, rtol=0, atol=_TOL)
    if "gross_ret" in res and "cost" in res:
        np.testing.assert_allclose(res["gross_ret"].reindex(idx).values,
                                   gross.reindex(idx).values, rtol=0, atol=_TOL)
        np.testing.assert_allclose(res["cost"].reindex(idx).values,
                                   cost.reindex(idx).values, rtol=0, atol=_TOL)
        # 逐日闭合：毛收益 - 成本 = 净收益
        np.testing.assert_allclose(res["port_ret"].values,
                                   (res["gross_ret"] - res["cost"]).values,
                                   rtol=0, atol=_TOL)

    # ── 净值由净收益唯一确定 ──
    np.testing.assert_allclose(res["cum"].values,
                               (1.0 + res["port_ret"]).cumprod().values,
                               rtol=1e-12, atol=_TOL)


class TestReconciliationRiskPortfolio(unittest.TestCase):
    """risk/portfolio.py::build_weight_portfolio。"""

    def setUp(self):
        self.close, self.pred = synthetic_panel()

    def test_long_short_book_closes(self):
        res, W = build_weight_portfolio(
            self.pred, self.close, top_q=TOP_Q, bottom_q=BOTTOM_Q,
            hold_days=HOLD_DAYS, return_weights=True,
        )
        assert_closed(res, W, self.close)

    def test_long_only_book_closes(self):
        res, W = build_weight_portfolio(
            self.pred, self.close, long_only=True,
            hold_days=HOLD_DAYS, return_weights=True,
        )
        assert_closed(res, W, self.close)

    def test_short_only_book_closes(self):
        res, W = build_weight_portfolio(
            self.pred, self.close, short_only=True,
            hold_days=HOLD_DAYS, return_weights=True,
        )
        assert_closed(res, W, self.close)

    def test_etf_cost_halved_is_consistent(self):
        """cost=0.0004 对半拆为买/卖各 0.0002 —— 对账必须用同一口径重算。"""
        res, W = build_weight_portfolio(
            self.pred, self.close, cost=0.0004,
            hold_days=HOLD_DAYS, return_weights=True,
        )
        assert_closed(res, W, self.close, buy_cost=0.0002, sell_cost=0.0002)


class TestReconciliationOrchestrator(unittest.TestCase):
    """risk/orchestrator.py::PortfolioOrchestrator.run（调仓日历 + 约束 + 撮合）。"""

    def setUp(self):
        self.close, self.pred = synthetic_panel()

    def _ls_target_fn(self):
        pred = self.pred

        def fn(d: pd.Timestamp) -> pd.Series:
            if d not in pred.index:
                return pd.Series(0.0, index=pred.columns)
            row = pred.loc[d].dropna()
            n_top = max(1, int(len(row) * TOP_Q))
            n_bot = max(1, int(len(row) * BOTTOM_Q))
            w = pd.Series(0.0, index=pred.columns)
            w[row.nlargest(n_top).index] = 1.0 / n_top
            w[row.nsmallest(n_bot).index] = -1.0 / n_bot
            return w

        return fn

    def test_book_closes_and_respects_leverage(self):
        orch = PortfolioOrchestrator(rebalance="monthly", max_leverage=2.0)
        res, W = orch.run(self._ls_target_fn(), self.close, return_weights=True)
        assert_closed(res, W, self.close)
        assert_weight_matrix(W, max_gross=2.0, name="W_held")

    def test_constraint_actually_binds(self):
        """max_leverage=1.0 对总杠杆为 2 的多空书必须产生缩放（约束不是摆设）。"""
        orch = PortfolioOrchestrator(rebalance="monthly", max_leverage=1.0)
        _, W = orch.run(self._ls_target_fn(), self.close, return_weights=True)
        assert_weight_matrix(W, max_gross=1.0, name="W_held")
        self.assertLessEqual(float(W.abs().sum(axis=1).max()), 1.0 + 1e-9)


@unittest.skipIf(_models_build_portfolio is None, "models.portfolio_backtest 不可导入")
class TestReconciliationModels(unittest.TestCase):
    """models/portfolio_backtest.py::build_portfolio（ML 链路主口径）。"""

    def setUp(self):
        self.close, self.pred = synthetic_panel()

    def test_book_closes(self):
        res, W = _models_build_portfolio(
            self.pred, self.close, top_q=TOP_Q, bottom_q=BOTTOM_Q,
            hold_days=HOLD_DAYS, return_weights=True,
        )
        assert_closed(res, W, self.close)

    def test_long_only_book_closes(self):
        res, W = _models_build_portfolio(
            self.pred, self.close, long_only=True,
            hold_days=HOLD_DAYS, return_weights=True,
        )
        assert_closed(res, W, self.close)


@unittest.skipIf(_models_build_portfolio is None, "models.portfolio_backtest 不可导入")
class TestCrossImplementationSemantics(unittest.TestCase):
    """两条独立实现必须对同一份输入给出同一本账 —— 语义一致性。

    构造参数刻意避开已知差异项（见 TestKnownDivergence），使阈值不成为约束，
    从而可以逐位比对。
    """

    def test_risk_and_models_agree_when_threshold_does_not_bind(self):
        close, pred = synthetic_panel(n_days=300, n_syms=40, seed=11)
        r1 = build_weight_portfolio(
            pred, close, top_q=TOP_Q, bottom_q=BOTTOM_Q, hold_days=HOLD_DAYS)
        r2 = _models_build_portfolio(
            pred, close, top_q=TOP_Q, bottom_q=BOTTOM_Q, hold_days=HOLD_DAYS)
        self.assertEqual(len(r1), len(r2))
        np.testing.assert_allclose(r1["port_ret"].values, r2["port_ret"].values,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(r1["turnover"].values, r2["turnover"].values,
                                   rtol=0, atol=1e-12)


@unittest.skipIf(_models_build_portfolio is None, "models.portfolio_backtest 不可导入")
class TestKnownDivergence(unittest.TestCase):
    """钉住一个已核实的真实口径差异（2026-09-14 对账建设中发现的）。

    差异项：开仓所需的最小有效股数

        risk/portfolio.py::build_weight_portfolio       min_stocks = base * 2
        models/portfolio_backtest.py::build_portfolio   min_stocks = base * 3

    其中 base = max(int(1/top_q), int(1/bottom_q))，默认 top_q=bottom_q=0.2 → base = 5。
    即阈值 10 vs 15。在 12 只股票的截面上：前者 12 ≥ 10 开仓，后者 12 < 15 空仓。

    这**不是本次改动引入的问题**，也不在本次授权范围内修改 —— 统一阈值会改变
    已发布报告的持仓日集合，须先评估影响再决策。本测试把差异钉住：任何一方
    发生改动都会立即失败，迫使做一次有意识的决策，而不是悄悄漂移。
    """

    def test_min_stocks_threshold_divergence_is_pinned(self):
        close, pred = synthetic_panel(n_days=200, n_syms=12, seed=23)

        r_risk = build_weight_portfolio(
            pred, close, top_q=TOP_Q, bottom_q=BOTTOM_Q, hold_days=HOLD_DAYS)
        self.assertGreater(float(r_risk["turnover"].sum()), 0.0,
                           "risk/portfolio 阈值更宽，12 只股票时应开仓")

        r_models = _models_build_portfolio(
            pred, close, top_q=TOP_Q, bottom_q=BOTTOM_Q, hold_days=HOLD_DAYS)
        self.assertEqual(float(r_models["turnover"].sum()), 0.0,
                         "models 阈值更严，12 只股票时应空仓")


class TestAssertionsWiredIntoBacktest(unittest.TestCase):
    """断言层必须真的接在回测入口上 —— 不是摆着好看的孤立模块。"""

    def test_bad_price_blocks_build_weight_portfolio(self):
        close, pred = synthetic_panel(n_days=60, n_syms=20, seed=3)
        close = close.copy()
        close.iloc[5, 2] = -1.0
        with self.assertRaises(InvariantViolation):
            build_weight_portfolio(pred, close, hold_days=HOLD_DAYS)

    def test_legacy_cost_blocks_build_weight_portfolio(self):
        """cost=0.003（铁律 3 倍）必须在入口被拦，而不是算出一个错的结果。"""
        close, pred = synthetic_panel(n_days=60, n_syms=20, seed=4)
        with self.assertRaises(InvariantViolation):
            build_weight_portfolio(pred, close, cost=0.003, hold_days=HOLD_DAYS)

    def test_bad_price_blocks_orchestrator(self):
        close, pred = synthetic_panel(n_days=60, n_syms=20, seed=5)
        close = close.copy()
        close.iloc[5, 2] = -1.0
        orch = PortfolioOrchestrator(rebalance="monthly", max_leverage=2.0)
        with self.assertRaises(InvariantViolation):
            orch.run(lambda d: pd.Series(0.0, index=pred.columns), close)

    def test_legacy_cost_blocks_engine(self):
        from backtest.engine import run as engine_run

        close, _ = synthetic_panel(n_days=60, n_syms=1, seed=6)
        s = close.iloc[:, 0]
        with self.assertRaises(InvariantViolation):
            engine_run(s, (s > s.mean()).astype(float),
                       buy_cost=0.0015, sell_cost=0.0015)

    @unittest.skipIf(_models_build_portfolio is None, "models.portfolio_backtest 不可导入")
    def test_legacy_cost_blocks_models(self):
        close, pred = synthetic_panel(n_days=60, n_syms=20, seed=7)
        with self.assertRaises(InvariantViolation):
            _models_build_portfolio(pred, close, cost=0.003, hold_days=HOLD_DAYS)


if __name__ == "__main__":
    unittest.main()
