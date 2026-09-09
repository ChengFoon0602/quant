"""
test_alpha191_ext.py — alpha191_ext.py 七个时序/截面算子的回归测试。

背景：signals/alpha191_ext.py 为 2026-09-08 新增（commit c725602），
`CLAUDE_AGENT_SOP.md` Step 1 要求 Agent 优先调用这些算子，但新增时**零测试、零引用**。
未测试的工具被 AI Agent 调用是风险 —— 算子语义错误会静默污染所有自动挖掘的因子。

用法: python -m unittest tests.test_alpha191_ext -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from signals.alpha191_ext import correlation, decay_linear, delay, delta, rank, scale, ts_argmax


def _panel(rows: int = 30, cols: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(size=(rows, cols)),
        index=pd.date_range("2023-01-01", periods=rows, freq="B"),
        columns=[f"S{i:03d}" for i in range(cols)],
    )


class TestRank(unittest.TestCase):
    """横截面分位排序。"""

    def test_values_in_open_unit_interval(self):
        """pct=True 时值域为 (0, 1]。"""
        df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0]])
        r = rank(df)
        vals = r.iloc[0].values
        self.assertTrue(np.all(vals > 0) and np.all(vals <= 1.0))
        self.assertEqual(vals[-1], 1.0, "最大值应为 1.0")

    def test_order_preserved_ascending(self):
        """升序分位：原值越大，rank 越高。"""
        df = pd.DataFrame([[3.0, 1.0, 4.0, 2.0]])
        r = rank(df)
        expected_order = np.argsort(df.iloc[0].values)
        got_order = np.argsort(r.iloc[0].values)
        np.testing.assert_array_equal(got_order, expected_order)

    def test_row_independent(self):
        """截面排序逐行独立，不受其他日期影响。"""
        df = pd.DataFrame([[1.0, 2.0], [100.0, 0.0]])
        r = rank(df)
        self.assertAlmostEqual(r.iloc[1, 0], 1.0)   # 第 2 行：100 是本行最大
        self.assertAlmostEqual(r.iloc[1, 1], 0.5)


class TestDecayLinear(unittest.TestCase):
    """线性衰减加权移动平均。"""

    def test_weights_sum_to_one_on_constant(self):
        """常数序列上结果恒等于该常数（证明权重和为 1）。"""
        df = pd.DataFrame(np.ones((10, 3)))
        out = decay_linear(df, window=5)
        # 前 4 行 warm-up 为 NaN
        self.assertTrue(out.iloc[:4].isna().all().all())
        np.testing.assert_allclose(out.iloc[4:].values, 1.0, atol=1e-12)

    def test_matches_manual_weighted_average(self):
        """与手工加权平均逐位一致。"""
        df = pd.DataFrame(np.arange(1.0, 6.0).reshape(5, 1))  # [1,2,3,4,5]
        w = np.arange(1, 4) / np.arange(1, 4).sum()            # [1/6, 2/6, 3/6]
        out = decay_linear(df, window=3)
        expected = np.dot([3.0, 4.0, 5.0], w)                  # 末窗口
        self.assertTrue(np.isnan(out.iloc[1, 0]))               # warm-up
        self.assertAlmostEqual(out.iloc[4, 0], expected, places=12)

    def test_recent_data_gets_higher_weight(self):
        """近期数据权重更高：上升序列的衰减均值应偏向新值。"""
        ramp = pd.DataFrame(np.arange(1.0, 11.0).reshape(10, 1))
        out = decay_linear(ramp, window=5).iloc[9, 0]
        # 5 日线性加权均值 < 但接近末端值 10（权重偏向近期）
        simple_mean = np.mean([6, 7, 8, 9, 10])
        self.assertGreater(out, simple_mean, "近期数据权重更高 -> 应高于简单平均")

    def test_nan_window_propagates(self):
        """窗口内任一 NaN -> 该点 NaN（一旦滑出窗口即恢复计算）。"""
        df = pd.DataFrame(np.ones((8, 1)))
        df.iloc[3, 0] = np.nan
        out = decay_linear(df, window=3)
        # NaN 位于 index 3：覆盖它的窗口为 index 3,4,5（窗口 3 天）
        self.assertTrue(out.iloc[3:6].isna().all().all(),
                        "含 NaN 的窗口应为 NaN")
        self.assertFalse(np.isnan(out.iloc[6, 0]),
                         "NaN 滑出窗口后应恢复计算")


class TestTSArgmax(unittest.TestCase):
    """时序最大值位置。"""

    def test_output_range(self):
        """输出 ∈ [0, window-1]，0=今天最大。"""
        df = _panel(rows=20, cols=5, seed=1)
        out = ts_argmax(df, window=5)
        valid = out.iloc[4:].dropna(how="all").values
        self.assertTrue(np.all(valid >= 0) and np.all(valid <= 4))

    def test_max_today_returns_zero(self):
        """今天（窗口末位）最大 -> 返回 0。"""
        df = pd.DataFrame(np.ones((6, 1)))
        df.iloc[5, 0] = 99.0  # 第 6 行最大，窗口 [4..9]... 用窗口 [1..5] 末位
        out = ts_argmax(df, window=5)
        # 窗口覆盖 index 1..5，末位 index5=99 最大
        self.assertAlmostEqual(out.iloc[5, 0], 0.0)

    def test_max_at_window_start(self):
        """窗口首日最大 -> 返回 window-1。"""
        df = pd.DataFrame(np.arange(6.0).reshape(6, 1))  # 递增，窗口首日最小
        # 构造：窗口 [1..5] 首日 index1 最大
        df.iloc[1, 0] = 99.0
        out = ts_argmax(df, window=5)
        self.assertAlmostEqual(out.iloc[5, 0], 4.0, msg="首日最大应返回 window-1=4")


class TestCorrelation(unittest.TestCase):
    """滚动时序相关系数。"""

    def test_matches_manual_pearson(self):
        rng = np.random.default_rng(5)
        a = pd.DataFrame(rng.normal(size=(20, 2)))
        b = pd.DataFrame(rng.normal(size=(20, 2)))
        out = correlation(a, b, window=10)
        for col in [0, 1]:
            manual = np.corrcoef(a[col].iloc[0:10], b[col].iloc[0:10])[0, 1]
            self.assertAlmostEqual(out.iloc[9, col], manual, places=12)
        self.assertTrue(out.iloc[:9].isna().all().all(), "warm-up 应为 NaN")

    def test_perfect_correlation(self):
        a = pd.DataFrame(np.arange(15.0).reshape(15, 1))
        b = a * 2 + 1
        out = correlation(a, b, window=5)
        np.testing.assert_allclose(out.iloc[4:].values, 1.0, atol=1e-12)


class TestDelayDelta(unittest.TestCase):
    """时序滞后与差分。"""

    def test_delay_is_shift(self):
        df = pd.DataFrame(np.arange(6.0).reshape(6, 1))
        out = delay(df, 2)
        self.assertTrue(out.iloc[:2].isna().all().all(), "前 d 行应为 NaN")
        self.assertAlmostEqual(out.iloc[4, 0], df.iloc[2, 0])
        self.assertAlmostEqual(out.iloc[5, 0], df.iloc[3, 0])

    def test_delta_is_current_minus_lagged(self):
        df = pd.DataFrame(np.arange(6.0).reshape(6, 1))  # 每次 +1
        out = delta(df, 1)
        self.assertAlmostEqual(out.iloc[5, 0], 1.0, msg="等差序列差分应为常数")
        self.assertTrue(np.isnan(out.iloc[0, 0]))

    def test_delta_window_equals_cumulative(self):
        df = pd.DataFrame(np.arange(10.0).reshape(10, 1))
        out = delta(df, 4)
        self.assertAlmostEqual(out.iloc[9, 0], df.iloc[9, 0] - df.iloc[5, 0])


class TestScale(unittest.TestCase):
    """横截面绝对值缩放。"""

    def test_row_abs_sum_equals_a(self):
        df = pd.DataFrame([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]])
        out = scale(df, a=1.0)
        for row in range(2):
            self.assertAlmostEqual(out.iloc[row].abs().sum(), 1.0, places=12)

    def test_default_scale_is_one(self):
        df = pd.DataFrame([[1.0, -2.0, 3.0]])
        out = scale(df)  # a 默认 1.0
        self.assertAlmostEqual(out.iloc[0].abs().sum(), 1.0)

    def test_proportionality_preserved(self):
        """缩放保持符号与比例，不改变横截面排序。"""
        df = pd.DataFrame([[1.0, -2.0, 3.0]])
        out = scale(df, a=100.0)
        np.testing.assert_allclose(out.iloc[0].values, [100 / 6, -200 / 6, 300 / 6])

    def test_zero_row_becomes_nan(self):
        """全零行 -> NaN（除以零保护）。"""
        df = pd.DataFrame([[0.0, 0.0], [1.0, 2.0]])
        out = scale(df)
        self.assertTrue(out.iloc[0].isna().all())
        self.assertFalse(out.iloc[1].isna().all())


if __name__ == "__main__":
    unittest.main()
