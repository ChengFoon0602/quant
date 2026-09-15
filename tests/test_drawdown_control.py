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
from backtest.reconciliation import assert_overlay_closed
from risk.cost_model import BUY_COST, SELL_COST
from risk.drawdown_control import (
    apply_drawdown_control,
    apply_drawdown_scaling,
    relevering_cost,
)

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

    def test_controlled_equals_raw_times_scale_when_scale_is_one(self):
        """scale ≡ 1 时无调杠杆成本 → 受控 == 原始。"""
        out = apply_drawdown_control(_crash_series(), cut=1.0)
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

    def test_controlled_equals_raw_times_scale_when_unreachable(self):
        """阈值永不触及 → scale ≡ 1 → 受控 == 原始（无费用账）。"""
        out = apply_drawdown_scaling(_crash_series(n_up=200, n_down=0, n_rec=1),
                                     floor=1.0)
        np.testing.assert_allclose(out["controlled_ret"].values,
                                   (out["raw_ret"] * out["scale"]).values,
                                   rtol=0, atol=1e-15)

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


class TestReleveringCost(unittest.TestCase):
    """调杠杆成本（relevering_cost）—— 控制效果的「隐含成本」敏感性检验。"""

    def test_constant_scale_is_free(self):
        s = pd.Series([0.4] * 10, index=pd.bdate_range("2021-01-01", periods=10))
        self.assertTrue((relevering_cost(s) == 0.0).all())

    def test_increase_charged_at_buy_rate(self):
        s = pd.Series([1.0, 1.5], index=pd.bdate_range("2021-01-01", periods=2))
        got = relevering_cost(s, gross=2.0)
        self.assertAlmostEqual(float(got.iloc[1]), 0.5 * 2.0 * BUY_COST, places=15)

    def test_decrease_charged_at_sell_rate(self):
        s = pd.Series([1.0, 0.5], index=pd.bdate_range("2021-01-01", periods=2))
        got = relevering_cost(s, gross=2.0)
        self.assertAlmostEqual(float(got.iloc[1]), 0.5 * 2.0 * SELL_COST, places=15)

    def test_non_negative(self):
        r = _crash_series()
        ctl = apply_drawdown_scaling(r, max_cut_at=0.05, floor=0.3)
        self.assertGreaterEqual(float(relevering_cost(ctl["scale"]).min()), 0.0)

    def test_first_day_is_free(self):
        r = _crash_series()
        ctl = apply_drawdown_scaling(r, max_cut_at=0.05, floor=0.3)
        self.assertAlmostEqual(float(relevering_cost(ctl["scale"]).iloc[0]), 0.0)

    def test_cost_is_positive_for_varying_scale(self):
        r = _crash_series()
        varying = apply_drawdown_scaling(r, max_cut_at=0.30, floor=0.2)["scale"]
        self.assertGreater(float(relevering_cost(varying).sum()), 0.0)

    def test_tiny_mca_becomes_two_state_and_mean_scale_is_midway(self):
        """实测（与直觉相反）：mca 极小时 scale 只在 `{floor, 1}` 间取值、均值居中 ——
        既**不是**「长期贴在下限」，也**不是**「切换次数爆炸」。

        原因：回撤在**受控路径**上计算 —— 一旦降到 floor，受控回撤随即收窄，
        规则立刻恢复满仓，于是形成 `1.0 ↔ floor` 的双态振荡。
        所以「极端区」既不是简单的低敞口，也不是换手灾难（见边界扫描的 Σ|Δscale| 列）。
        """
        r = _crash_series()
        tiny = apply_drawdown_scaling(r, max_cut_at=0.002, floor=0.2)["scale"]
        wide = apply_drawdown_scaling(r, max_cut_at=0.30, floor=0.2)["scale"]
        self.assertLessEqual(int(tiny.nunique()), 3)          # 双态
        self.assertGreater(float(tiny.mean()), 0.25)          # 不贴下限
        self.assertLess(float(tiny.mean()), 0.95)             # 也不长期满仓
        self.assertLess(int(tiny.nunique()), int(wide.nunique()))


class TestClosedLedger(unittest.TestCase):
    """三层账本闭合 —— 这是「清算做闭合」的直接证明。"""

    @staticmethod
    def _raw() -> pd.Series:
        return _crash_series()

    def test_ledger_columns_present(self):
        out = apply_drawdown_scaling(self._raw(), max_cut_at=0.05, floor=0.4)
        for c in ("scale", "raw_ret", "position_ret", "relever_cost", "cash_ret",
                  "cash_weight", "net_ret", "controlled_ret"):
            self.assertIn(c, out.columns)

    def test_position_ledger_is_scale_times_raw(self):
        out = apply_drawdown_scaling(self._raw(), max_cut_at=0.05, floor=0.4)
        np.testing.assert_allclose(out["position_ret"].values,
                                   (out["raw_ret"] * out["scale"]).values,
                                   rtol=0, atol=1e-15)

    def test_three_layer_identity(self):
        """controlled_ret == 仓位账 − 费用账 + 现金账（逐位）。"""
        out = apply_drawdown_control(self._raw(), threshold=0.05, cut=0.4, recovery=0.02)
        np.testing.assert_allclose(
            out["controlled_ret"].values,
            (out["position_ret"] - out["relever_cost"] + out["cash_ret"]).values,
            rtol=0, atol=1e-15)

    def test_net_ret_equals_controlled_ret(self):
        out = apply_drawdown_scaling(self._raw(), max_cut_at=0.05, floor=0.4)
        np.testing.assert_allclose(out["net_ret"].values, out["controlled_ret"].values,
                                   rtol=0, atol=1e-15)

    def test_internal_cost_matches_independent_recomputation(self):
        """循环内累计的费用账必须与 `relevering_cost` 独立重算逐位一致。"""
        out = apply_drawdown_scaling(self._raw(), max_cut_at=0.05, floor=0.4, gross=2.0)
        recomputed = relevering_cost(out["scale"], gross=2.0)
        np.testing.assert_allclose(out["relever_cost"].values, recomputed.values,
                                   rtol=0, atol=1e-15)

    def test_first_day_is_full_so_no_relever_cost(self):
        """起始为满仓且两态/连续规则在 dd=0 时都给 1.0 → 首日无调杠杆成本。"""
        for out in (apply_drawdown_control(self._raw()),
                    apply_drawdown_scaling(self._raw())):
            self.assertAlmostEqual(float(out["scale"].iloc[0]), 1.0, places=15)
            self.assertAlmostEqual(float(out["relever_cost"].iloc[0]), 0.0, places=15)

    def test_assert_overlay_closed_passes_on_controlled_ret(self):
        """★ 核心：受控输出必须是**闭合账本** —— 直接送往对账即可通过。

        只断言判定项 `net_ret`；`position_only_gap` 是诊断项（按定义等于费用账，预期非零）。
        """
        raw = self._raw()
        for out in (apply_drawdown_control(raw, threshold=0.03, cut=0.3, recovery=0.015),
                    apply_drawdown_scaling(raw, max_cut_at=0.02, floor=0.2)):
            gaps = assert_overlay_closed(out, raw, out["scale"], gross=2.0)
            self.assertLess(gaps["net_ret"], 1e-12)
            # 诊断项：缺口 = 单日**最大**费用账，必然 > 0 且 ≤ 费用**合计**（逐日非负）
            self.assertGreater(gaps["position_only_gap"], 0.0)
            self.assertLessEqual(gaps["position_only_gap"],
                                 gaps["relever_cost_total"] + 1e-12)

    def test_position_only_output_would_fail_reconciliation(self):
        """反例守卫：只记仓位账（旧口径）送往对账必须失败。"""
        raw = self._raw()
        out = apply_drawdown_scaling(raw, max_cut_at=0.02, floor=0.2)
        position_only = pd.DataFrame({"port_ret": out["position_ret"]})
        with self.assertRaises(AssertionError):
            assert_overlay_closed(position_only, raw, out["scale"], gross=2.0)

    def test_include_relever_cost_false_reproduces_position_only(self):
        """诊断开关：关掉费用账即复现旧口径（= scale × raw_ret）。"""
        raw = self._raw()
        out = apply_drawdown_scaling(raw, max_cut_at=0.02, floor=0.2,
                                     include_relever_cost=False)
        self.assertAlmostEqual(float(out["relever_cost"].sum()), 0.0, places=15)
        np.testing.assert_allclose(out["controlled_ret"].values,
                                   (raw * out["scale"]).values, rtol=0, atol=1e-15)

    def test_cost_scales_with_gross(self):
        raw = self._raw()
        g1 = apply_drawdown_scaling(raw, max_cut_at=0.02, floor=0.2, gross=1.0)
        g2 = apply_drawdown_scaling(raw, max_cut_at=0.02, floor=0.2, gross=2.0)
        self.assertAlmostEqual(float(g2["relever_cost"].sum()),
                               2.0 * float(g1["relever_cost"].sum()), places=9)

    def test_non_positive_gross_raises(self):
        with self.assertRaises(ValueError):
            apply_drawdown_scaling(self._raw(), gross=0.0)
        with self.assertRaises(ValueError):
            apply_drawdown_control(self._raw(), gross=-1.0)


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
