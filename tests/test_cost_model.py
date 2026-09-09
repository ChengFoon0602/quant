"""
test_cost_model.py — 铁律第 3 条（摩擦成本）的真实守护测试。

为什么需要这个文件
------------------
`tests/test_methodology.py::TestFrictionCost` 已经存在，但它是**空转测试**：
它在函数体内自己声明 `buy_cost, sell_cost = 0.00026, 0.00076`（第 98、116 行），
只验证了「0.00026 + 0.00076 == 0.00102」这个算术恒等式，**从未 import 任何模块的默认值**。

因此 2026-09 的 `COST_BPS = 0.003`（0.3%，铁律 3 倍）事故能在测试全绿的情况下存活，
并污染了「多空 alpha 不成立」的核心结论（收口后 LS 0.036 → 1.175）。

本测试的做法：**用 inspect 读取真实模块的默认参数值**，断言其等于 `risk.cost_model` 的
规范常量。任何一处硬编码漂移都会立即失败，并指出精确的 file:line。

用法: python -m unittest tests.test_cost_model -v
"""

from __future__ import annotations

import dataclasses
import inspect
import unittest

from risk.cost_model import BUY_COST, ROUND_TRIP, SELL_COST, CostModel


def _default_of(func, param_name: str):
    """读取函数某参数的默认值；参数不存在时返回 _MISSING。"""
    sig = inspect.signature(func)
    if param_name not in sig.parameters:
        return inspect.Parameter.empty
    return sig.parameters[param_name].default


class TestCostSingleSource(unittest.TestCase):
    """铁律 3：所有回测入口的默认费率必须等于 risk.cost_model 规范值。"""

    def test_canonical_constants(self):
        """规范常量本身符合 A 股 2026 费率口径。"""
        self.assertAlmostEqual(BUY_COST, 0.00026, places=8, msg="买入单边应为 0.026%")
        self.assertAlmostEqual(SELL_COST, 0.00076, places=8, msg="卖出单边应为 0.076%")
        self.assertAlmostEqual(ROUND_TRIP, 0.00102, places=8, msg="双边合计应约 0.102%")
        self.assertGreater(SELL_COST, BUY_COST, "卖出须含印花税，应高于买入")

    def test_engine_defaults(self):
        """backtest/engine.py:22-23"""
        from backtest.engine import run

        self.assertEqual(_default_of(run, "buy_cost"), BUY_COST, "backtest/engine.py buy_cost 漂移")
        self.assertEqual(_default_of(run, "sell_cost"), SELL_COST, "backtest/engine.py sell_cost 漂移")

    def test_cross_section_defaults(self):
        """backtest/cross_section.py:27-28"""
        from backtest.cross_section import run_cross_section

        self.assertEqual(
            _default_of(run_cross_section, "buy_cost"), BUY_COST,
            "backtest/cross_section.py buy_cost 漂移",
        )
        self.assertEqual(
            _default_of(run_cross_section, "sell_cost"), SELL_COST,
            "backtest/cross_section.py sell_cost 漂移",
        )

    def test_risk_portfolio_defaults(self):
        """risk/portfolio.py:80-81"""
        from risk.portfolio import build_weight_portfolio

        self.assertEqual(
            _default_of(build_weight_portfolio, "buy_cost"), BUY_COST,
            "risk/portfolio.py buy_cost 漂移",
        )
        self.assertEqual(
            _default_of(build_weight_portfolio, "sell_cost"), SELL_COST,
            "risk/portfolio.py sell_cost 漂移",
        )

    def test_orchestrator_defaults(self):
        """risk/orchestrator.py:65-66"""
        from risk.orchestrator import PortfolioOrchestrator

        init = PortfolioOrchestrator.__init__
        self.assertEqual(
            _default_of(init, "buy_cost"), BUY_COST,
            "risk/orchestrator.py buy_cost 漂移",
        )
        self.assertEqual(
            _default_of(init, "sell_cost"), SELL_COST,
            "risk/orchestrator.py sell_cost 漂移",
        )

    def test_portfolio_backtest_defaults(self):
        """models/portfolio_backtest.py:117-118 —— 事故原发地，重点守护。"""
        from models.portfolio_backtest import build_portfolio

        self.assertEqual(
            _default_of(build_portfolio, "buy_cost"), BUY_COST,
            "models/portfolio_backtest.py buy_cost 漂移（曾是 0.3% bug 现场）",
        )
        self.assertEqual(
            _default_of(build_portfolio, "sell_cost"), SELL_COST,
            "models/portfolio_backtest.py sell_cost 漂移（曾是 0.3% bug 现场）",
        )

    def test_portfolio_backtest_module_constants(self):
        """models/portfolio_backtest.py:50-52 模块级常量。"""
        from models import portfolio_backtest as pb

        self.assertAlmostEqual(
            pb.COST_BPS, ROUND_TRIP, places=8,
            msg="models/portfolio_backtest.py COST_BPS 不等于双边合计 0.102%"
                "（历史上此处曾为 0.003，是铁律的 3 倍）",
        )
        self.assertAlmostEqual(pb.BUY_COST, BUY_COST, places=8)
        self.assertAlmostEqual(pb.SELL_COST, SELL_COST, places=8)

    def test_evaluate_module_constants(self):
        """models/evaluate.py:25-28"""
        from models import evaluate

        self.assertAlmostEqual(evaluate.COST_LONG_OPEN, BUY_COST, places=8)
        self.assertAlmostEqual(evaluate.COST_LONG_CLOSE, SELL_COST, places=8)
        self.assertAlmostEqual(evaluate.COST_SHORT_OPEN, BUY_COST, places=8)
        self.assertAlmostEqual(evaluate.COST_SHORT_CLOSE, SELL_COST, places=8)


class TestCostModelDataclass(unittest.TestCase):
    """CostModel 数据类自身行为。"""

    def test_default_round_trip(self):
        cm = CostModel()
        self.assertAlmostEqual(cm.round_trip, ROUND_TRIP, places=8)

    def test_deduction_is_directional(self):
        """买入/卖出按不同费率计提，不能对半拆。"""
        cm = CostModel()
        # 全部为买入换手
        self.assertAlmostEqual(cm.deduction(1.0, 0.0), BUY_COST, places=8)
        # 全部为卖出换手（含印花税，更高）
        self.assertAlmostEqual(cm.deduction(0.0, 1.0), SELL_COST, places=8)
        # 多空各一端
        self.assertAlmostEqual(cm.deduction(1.0, 1.0), ROUND_TRIP, places=8)

    def test_slippage_separate(self):
        """滑点另计，不并入双边合计。"""
        cm = CostModel()
        self.assertAlmostEqual(cm.with_slippage(1.0), 0.0005, places=8)
        self.assertAlmostEqual(cm.round_trip, ROUND_TRIP, places=8)

    def test_frozen(self):
        """成本模型不可变，防止运行中被意外篡改。"""
        cm = CostModel()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cm.buy_cost = 0.003  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
