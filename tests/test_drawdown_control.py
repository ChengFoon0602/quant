"""
test_drawdown_control.py — 回撤控制的回归测试（risk/drawdown_control.py）。

守护四件事：
  1. **无未来函数**：缩放系数只用截至前一日的回撤（截断不变性）；
  2. **滞后带语义**：降仓只发生在 dd_prev ≤ −threshold，恢复只发生在 dd_prev ≥ −recovery；
  3. **退化即恒等**：cut=1.0 或阈值永不触及 → 受控收益 === 原始收益；
  4. 参数校验（recovery ≥ threshold、cut ∉ (0,1]、空序列）必须 raise。

用法: python -m unittest tests.test_drawdown_control -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.metrics import max_drawdown, sharpe_ratio
from risk.drawdown_control import apply_drawdown_control, apply_drawdown_scaling

THRESHOLD = 0.10
CUT = 0.5
RECOVERY = 0.05


def _crash_series(n_up: int = 30, n_down: int = 15, n_rec: int = 150) -> pd.Series:
    """先涨 → 急跌 → 恢复，制造一次明确的深回撤。

    恢复段要足够长：降仓状态下收益同样被缩放（×cut），所以回撤收窄得比直觉更慢，
    太短的恢复段不会等到滞后带退出（这正是需要测的行为）。
    """
    r = np.concatenate([
        np.full(n_up, 0.005),
        np.full(n_down, -0.02),
        np.full(n_rec, 0.006),
    ])
    idx = pd.bdate_range("2020-01-01", periods=len(r))
    return pd.Series(r, index=idx)


class TestNoFutureLeak(unittest.TestCase):
    def test_scale_uses_prior_day_drawdown(self):
        """截断不变性：只给前 N 天数据，前 N 天的 scale 必须与全量一致。"""
        r = _crash_series()
        full = apply_drawdown_control(r, threshold=THRESHOLD, cut=CUT,
                                      recovery=RECOVERY)["scale"]
        prefix = apply_drawdown_control(r.iloc[:40], threshold=THRESHOLD, cut=CUT,
                                        recovery=RECOVERY)["scale"]
        np.testing.assert_allclose(full.iloc[:40].values, prefix.values, rtol=0, atol=1e-15)

    def test_first_day_is_never_cut(self):
        """首日无历史回撤 → 不可能被降仓。"""
        out = apply_drawdown_control(_crash_series(), cut=CUT)
        self.assertAlmostEqual(float(out["scale"].iloc[0]), 1.0)

    def test_drawdown_prev_is_shift_of_drawdown(self):
        out = apply_drawdown_control(_crash_series(), cut=CUT)
        expected = out["drawdown"].shift(1).fillna(0.0)
        np.testing.assert_allclose(out["drawdown_prev"].values, expected.values,
                                   rtol=0, atol=1e-15)


class TestHysteresisSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = apply_drawdown_control(_crash_series(), threshold=THRESHOLD,
                                         cut=CUT, recovery=RECOVERY)

    def test_scale_takes_only_two_values(self):
        self.assertTrue(set(self.out["scale"].unique()) <= {1.0, CUT})

    def test_cut_only_when_drawdown_beyond_threshold(self):
        scale = self.out["scale"]
        prev_scale = scale.shift(1).fillna(1.0)
        downs = (prev_scale == 1.0) & (scale == CUT)
        self.assertTrue(bool(downs.any()), "该序列应至少触发一次降仓")
        self.assertTrue((self.out["drawdown_prev"][downs] <= -THRESHOLD + 1e-12).all())

    def test_recover_only_inside_recovery_band(self):
        scale = self.out["scale"]
        prev_scale = scale.shift(1).fillna(1.0)
        ups = (prev_scale != 1.0) & (scale == 1.0)
        self.assertTrue(bool(ups.any()), "该序列应至少恢复一次")
        self.assertTrue((self.out["drawdown_prev"][ups] >= -RECOVERY - 1e-12).all())

    def test_no_chatter_inside_band(self):
        """滞后带内不得反复开关：状态切换次数应远小于样本长度。"""
        scale = self.out["scale"]
        switches = int((scale != scale.shift(1).fillna(1.0)).sum())
        self.assertLessEqual(switches, 4, "滞后带失效，出现抖动")


class TestDegenerateCases(unittest.TestCase):
    def test_cut_one_is_identity(self):
        out = apply_drawdown_control(_crash_series(), threshold=THRESHOLD, cut=1.0)
        np.testing.assert_allclose(out["controlled_ret"].values, out["raw_ret"].values,
                                   rtol=0, atol=1e-15)

    def test_unreachable_threshold_is_identity(self):
        r = pd.Series([0.001] * 50, index=pd.bdate_range("2020-01-01", periods=50))
        out = apply_drawdown_control(r, threshold=0.10, cut=0.5)
        self.assertTrue((out["scale"] == 1.0).all())

    def test_deeper_cut_shallows_drawdown(self):
        r = _crash_series()
        deep = apply_drawdown_control(r, threshold=THRESHOLD, cut=0.3)["controlled_ret"]
        mild = apply_drawdown_control(r, threshold=THRESHOLD, cut=0.8)["controlled_ret"]
        self.assertGreater(max_drawdown(deep), max_drawdown(mild),
                           "更深的减仓应给出更浅的最大回撤")

    def test_controlled_equals_raw_times_scale(self):
        out = apply_drawdown_control(_crash_series(), cut=CUT)
        np.testing.assert_allclose(out["controlled_ret"].values,
                                   (out["raw_ret"] * out["scale"]).values,
                                   rtol=0, atol=1e-15)

    def test_metrics_are_computable_on_output(self):
        out = apply_drawdown_control(_crash_series(), cut=CUT)
        self.assertTrue(np.isfinite(sharpe_ratio(out["controlled_ret"])))


class TestAdaptiveContinuousScaling(unittest.TestCase):
    """连续映射变体（apply_drawdown_scaling）—— 「自适应」形态。"""

    def test_scale_within_bounds(self):
        out = apply_drawdown_scaling(_crash_series(), max_cut_at=0.10, floor=0.4)
        s = out["scale"]
        self.assertGreaterEqual(float(s.min()), 0.4 - 1e-12)
        self.assertLessEqual(float(s.max()), 1.0 + 1e-12)

    def test_first_day_is_full(self):
        out = apply_drawdown_scaling(_crash_series())
        self.assertAlmostEqual(float(out["scale"].iloc[0]), 1.0)

    def test_mapping_is_linear_in_drawdown(self):
        """逐点核对文档给出的映射：scale = clip(1 + dd_prev/max_cut_at, floor, 1)。"""
        out = apply_drawdown_scaling(_crash_series(), max_cut_at=0.20, floor=0.20)
        expected = (1.0 + out["drawdown_prev"] / 0.20).clip(lower=0.20, upper=1.0)
        np.testing.assert_allclose(out["scale"].values, expected.values, rtol=0, atol=1e-12)

    def test_response_is_continuous_not_two_state(self):
        """连续映射应产生大量不同取值 —— 这正是它与两态开关的本质差别。"""
        r = _crash_series()
        n_cont = int(apply_drawdown_scaling(r, max_cut_at=0.10, floor=0.4)["scale"].nunique())
        n_step = int(apply_drawdown_control(r, threshold=0.10, cut=0.5)["scale"].nunique())
        self.assertGreater(n_cont, 10)
        self.assertLessEqual(n_step, 2)

    def test_no_future_leak_truncation_invariance(self):
        r = _crash_series()
        full = apply_drawdown_scaling(r, max_cut_at=0.10, floor=0.4)["scale"]
        prefix = apply_drawdown_scaling(r.iloc[:40], max_cut_at=0.10, floor=0.4)["scale"]
        np.testing.assert_allclose(full.iloc[:40].values, prefix.values, rtol=0, atol=1e-15)

    def test_floor_one_is_identity(self):
        out = apply_drawdown_scaling(_crash_series(), max_cut_at=0.10, floor=1.0)
        np.testing.assert_allclose(out["controlled_ret"].values, out["raw_ret"].values,
                                   rtol=0, atol=1e-15)

    def test_larger_max_cut_at_means_less_aggressive(self):
        """max_cut_at 越大 → 同一回撤下减仓越少 → 平均仓位越高。"""
        r = _crash_series()
        tight = apply_drawdown_scaling(r, max_cut_at=0.05, floor=0.3)["scale"].mean()
        loose = apply_drawdown_scaling(r, max_cut_at=0.30, floor=0.3)["scale"].mean()
        self.assertGreater(float(loose), float(tight))

    def test_controlled_equals_raw_times_scale(self):
        out = apply_drawdown_scaling(_crash_series())
        np.testing.assert_allclose(out["controlled_ret"].values,
                                   (out["raw_ret"] * out["scale"]).values, rtol=0, atol=1e-15)

    def test_shallower_drawdown_than_step_rule(self):
        """连续映射会「早减、缓减」，因此最大回撤通常不深于不控制。"""
        r = _crash_series()
        out = apply_drawdown_scaling(r, max_cut_at=0.10, floor=0.4)
        self.assertGreater(max_drawdown(out["controlled_ret"]), max_drawdown(r))

    def test_parameter_validation(self):
        r = _crash_series(n_up=5, n_down=5, n_rec=5)
        with self.assertRaises(ValueError):
            apply_drawdown_scaling(r, max_cut_at=0.0)
        with self.assertRaises(ValueError):
            apply_drawdown_scaling(r, floor=0.0)
        with self.assertRaises(ValueError):
            apply_drawdown_scaling(r, floor=1.2)
        with self.assertRaises(ValueError):
            apply_drawdown_scaling(pd.Series([], dtype=float))


class TestParameterValidation(unittest.TestCase):
    def setUp(self):
        self.r = _crash_series(n_up=5, n_down=5, n_rec=5)

    def test_recovery_must_be_below_threshold(self):
        with self.assertRaises(ValueError):
            apply_drawdown_control(self.r, threshold=0.05, recovery=0.05)
        with self.assertRaises(ValueError):
            apply_drawdown_control(self.r, threshold=0.05, recovery=0.10)

    def test_cut_must_be_in_unit_interval(self):
        with self.assertRaises(ValueError):
            apply_drawdown_control(self.r, cut=0.0)
        with self.assertRaises(ValueError):
            apply_drawdown_control(self.r, cut=1.5)

    def test_empty_returns_raises(self):
        with self.assertRaises(ValueError):
            apply_drawdown_control(pd.Series([], dtype=float))


if __name__ == "__main__":
    unittest.main()
