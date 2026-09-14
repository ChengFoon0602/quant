"""
test_risk_constraints.py — 约束的 API 契约（CLAUDE.md TODO 18）。

守护两件事：

  1. **未实现的参数必须响亮地失败**，不允许静默忽略 —— `max_turnover` 此前
     声明了、写进 docstring 示例、存为 `self.max_turnover`，却在
     `apply_constraints` / `run` 中从未被读取；传了等于没传且毫无提示。
     与铁律 3 的成本 bug 是同一类失效模式（看着有防护，实际没有，且不报错）。
  2. 已实现的约束确实裁到目标上：`max_weight_per_asset` 裁剪、`max_leverage`
     按**总杠杆 Σ|W|**（gross）缩放。

用法: python -m unittest tests.test_risk_constraints -v
"""

from __future__ import annotations

import unittest

import pandas as pd

from risk.orchestrator import PortfolioOrchestrator


class TestUnimplementedParamsFailLoudly(unittest.TestCase):
    def test_max_turnover_raises_not_implemented(self):
        """传 max_turnover 必须立刻 raise，而不是静默忽略。"""
        with self.assertRaises(NotImplementedError):
            PortfolioOrchestrator(rebalance="monthly", max_turnover=0.50)

    def test_max_turnover_none_is_accepted(self):
        orch = PortfolioOrchestrator(rebalance="monthly", max_turnover=None)
        self.assertIsNone(orch.max_turnover)

    def test_max_turnover_attribute_stays_none(self):
        """属性恒为 None —— 避免有人以为它生效了。"""
        orch = PortfolioOrchestrator()
        self.assertIsNone(orch.max_turnover)


class TestImplementedConstraints(unittest.TestCase):
    def test_max_weight_per_asset_clips(self):
        orch = PortfolioOrchestrator(max_weight_per_asset=0.10, max_leverage=1.0)
        got = orch.apply_constraints(pd.Series({"A": 0.5, "B": 0.5}))
        self.assertLessEqual(float(got.abs().max()), 0.10 + 1e-12)

    def test_max_leverage_scales_down_gross(self):
        orch = PortfolioOrchestrator(max_leverage=1.0)
        got = orch.apply_constraints(pd.Series({"A": 1.0, "B": -1.0}))  # gross = 2.0
        self.assertAlmostEqual(float(got.abs().sum()), 1.0, places=12)

    def test_within_limits_is_untouched(self):
        orch = PortfolioOrchestrator(max_leverage=2.0)
        w = pd.Series({"A": 0.5, "B": -0.5})
        got = orch.apply_constraints(w)
        pd.testing.assert_series_equal(got, w)

    def test_max_leverage_is_gross_not_net(self):
        """语义锁死：约束的是 `Σ|W|`（gross），不是 `|ΣW|`（净敞口）。

        多空对冲 net = 0 但 gross = 2 —— 若按「净敞口」理解会误以为不受限，
        实际会被缩放到半仓。这正是 docstring 修正（TODO 11）钉住的点。
        """
        orch = PortfolioOrchestrator(max_leverage=1.0)
        got = orch.apply_constraints(pd.Series({"A": 1.0, "B": -1.0}))
        self.assertAlmostEqual(abs(float(got.sum())), 0.0, places=12)      # 净敞口仍为 0
        self.assertAlmostEqual(float(got.abs().sum()), 1.0, places=12)     # 总杠杆被裁到 1


if __name__ == "__main__":
    unittest.main()
