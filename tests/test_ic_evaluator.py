"""
test_ic_evaluator.py — IC 评估器的回归测试。

背景：backtest/ic_evaluator.py 为 2026-09-08 新增（commit c725602），
`CLAUDE_AGENT_SOP.md` Step 3 要求 Agent 优先调用它，但新增时**零测试、零引用**。

本测试重点守护**收益锚定方向**：`shift(-lag)` 必须与铁律的收益锚定约定一致，
即 lag=1 对应「t 时刻因子 → t→t+1 收益」。方向写反会得到同样"漂亮"的 IC 数值，
但结论完全错误（用过去收益解释当期因子）。

用法: python -m unittest tests.test_ic_evaluator -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backtest.ic_evaluator import compute_ic_decay, compute_rank_ic


def _make_predictive_panel(n_dates: int = 40, n_stocks: int = 20, seed: int = 7):
    """构造「因子 t 只预测 t→t+1 单日收益」的面板。

    返回 (factor, daily_ret)。daily_ret 为 pct_change 语义：
    daily_ret[t] = close[t]/close[t-1] - 1（t-1→t 的收益）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    cols = [f"S{i:03d}" for i in range(n_stocks)]

    factor = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=cols)

    daily_ret = pd.DataFrame(
        rng.normal(0, 0.01, size=(n_dates, n_stocks)), index=dates, columns=cols
    )
    # 关键：让 t 时刻的因子决定 t→t+1 的收益，即写入 daily_ret 的第 t+1 行
    for t in range(n_dates - 1):
        daily_ret.iloc[t + 1] = 0.02 * factor.iloc[t].values
    return factor, daily_ret


class TestRankIC(unittest.TestCase):
    """compute_rank_ic 基础行为。"""

    def test_perfect_positive_correlation(self):
        """因子与未来收益完全同序 -> IC = 1。"""
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        cols = ["A", "B", "C"]
        factor = pd.DataFrame([[1.0, 2.0, 3.0]] * 3, index=dates, columns=cols)
        fwd = pd.DataFrame([[10.0, 20.0, 30.0]] * 3, index=dates, columns=cols)
        ic = compute_rank_ic(factor, fwd)
        for v in ic:
            self.assertAlmostEqual(v, 1.0, places=9)

    def test_perfect_negative_correlation(self):
        """完全逆序 -> IC = -1。"""
        dates = pd.date_range("2023-01-01", periods=3, freq="B")
        cols = ["A", "B", "C"]
        factor = pd.DataFrame([[1.0, 2.0, 3.0]] * 3, index=dates, columns=cols)
        fwd = pd.DataFrame([[30.0, 20.0, 10.0]] * 3, index=dates, columns=cols)
        ic = compute_rank_ic(factor, fwd)
        for v in ic:
            self.assertAlmostEqual(v, -1.0, places=9)

    def test_spearman_is_rank_based(self):
        """Spearman 只看序，不看数值尺度（线性变换后 IC 不变）。"""
        rng = np.random.default_rng(0)
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        cols = [f"S{i}" for i in range(10)]
        factor = pd.DataFrame(rng.normal(size=(5, 10)), index=dates, columns=cols)
        fwd = pd.DataFrame(rng.normal(size=(5, 10)), index=dates, columns=cols)

        ic_raw = compute_rank_ic(factor, fwd)
        ic_scaled = compute_rank_ic(factor, fwd * 1000 + 5)
        pd.testing.assert_series_equal(ic_raw, ic_scaled)


class TestICDecayAnchoring(unittest.TestCase):
    """收益锚定方向 —— 本模块最易错、后果最严重的点。"""

    def test_lag1_maps_to_next_day_return(self):
        """lag=1 必须对应 t→t+1 收益（因子不动，未来收益移过来）。"""
        factor, daily_ret = _make_predictive_panel()
        decay = compute_ic_decay(factor, daily_ret, max_lag=3)

        # 因子被构造成只预测 t→t+1：lag=1 应接近 1，后续 lag 应明显衰减
        self.assertGreater(decay.loc[1, "Mean_IC"], 0.9,
                           "lag=1 应捕捉到 t→t+1 的预测力（锚定方向若反了会失败）")
        self.assertLess(abs(decay.loc[2, "Mean_IC"]), 0.5,
                        "lag=2 对应 t+1→t+2，本例中因子对其无预测力")

    def test_direction_matters_not_magnitude(self):
        """反向锚定（用过去收益）必须给出不同的 IC，证明方向敏感。"""
        factor, daily_ret = _make_predictive_panel()
        ic_future = compute_rank_ic(factor, daily_ret.shift(-1))   # t→t+1（正确）
        ic_past = compute_rank_ic(factor, daily_ret.shift(1))       # t-1→t（错误方向）

        self.assertGreater(ic_future.mean(), 0.9)
        self.assertLess(abs(ic_past.mean()), 0.5,
                        "用过去收益应几乎无 IC；若此处也高，说明锚定方向写反")

    def test_decay_is_monotonic_in_this_construct(self):
        """本例中 IC 应随 lag 衰减（lag1 显著大于 lag3）。"""
        factor, daily_ret = _make_predictive_panel()
        decay = compute_ic_decay(factor, daily_ret, max_lag=3)
        self.assertGreater(abs(decay.loc[1, "Mean_IC"]), abs(decay.loc[3, "Mean_IC"]))


class TestICDecayStructure(unittest.TestCase):
    """返回结构与边界。"""

    def test_shape_and_index(self):
        factor, daily_ret = _make_predictive_panel()
        decay = compute_ic_decay(factor, daily_ret, max_lag=5)
        self.assertEqual(list(decay.index), [1, 2, 3, 4, 5])
        self.assertEqual(
            set(decay.columns), {"Mean_IC", "IC_IR", "Pos_Rate"},
            "列应包含 Mean_IC / IC_IR / Pos_Rate",
        )

    def test_pos_rate_within_bounds(self):
        factor, daily_ret = _make_predictive_panel(seed=11)
        decay = compute_ic_decay(factor, daily_ret, max_lag=4)
        for lag in decay.index:
            pr = decay.loc[lag, "Pos_Rate"]
            self.assertGreaterEqual(pr, 0.0)
            self.assertLessEqual(pr, 1.0)

    def test_ic_ir_is_mean_over_std(self):
        """IC_IR = mean(IC) / std(IC)（需 IC 有波动，故注入噪声）。"""
        factor, daily_ret = _make_predictive_panel(seed=3)
        # 注入噪声使逐日 IC 产生波动（否则 std=0，IR 退化为 nan，见下一条测试）
        rng = np.random.default_rng(99)
        daily_ret = daily_ret + pd.DataFrame(
            rng.normal(0, 0.02, size=daily_ret.shape),
            index=daily_ret.index, columns=daily_ret.columns,
        )
        decay = compute_ic_decay(factor, daily_ret, max_lag=2)
        ic = compute_rank_ic(factor, daily_ret.shift(-1)).dropna()
        self.assertGreater(ic.std(), 1e-8, "本用例需要 IC 有波动")
        expected = ic.mean() / ic.std()
        self.assertAlmostEqual(decay.loc[1, "IC_IR"], expected, places=9)

    def test_ic_ir_nan_when_ic_constant(self):
        """IC 恒定（std=0）时 IR 返回 nan 而非 inf —— 记录既有行为。

        本例中因子完美预测 t→t+1，逐日 IC 恒为 1，std=0。
        源码用 `std_ic != 0` 判断并返回 np.nan，避免 inf 污染衰减报表。
        """
        factor, daily_ret = _make_predictive_panel(seed=3)
        ic = compute_rank_ic(factor, daily_ret.shift(-1)).dropna()
        self.assertAlmostEqual(ic.std(), 0.0, places=12, msg="前置条件：IC 应恒定")

        decay = compute_ic_decay(factor, daily_ret, max_lag=2)
        self.assertTrue(np.isnan(decay.loc[1, "IC_IR"]),
                        "std=0 时 IR 应为 nan（而非 inf）")
        self.assertAlmostEqual(decay.loc[1, "Mean_IC"], 1.0, places=9)

    def test_all_nan_input_does_not_crash(self):
        """全 NaN 输入不得抛异常。"""
        dates = pd.date_range("2023-01-01", periods=5, freq="B")
        cols = ["A", "B"]
        factor = pd.DataFrame(np.nan, index=dates, columns=cols)
        daily_ret = pd.DataFrame(np.nan, index=dates, columns=cols)
        decay = compute_ic_decay(factor, daily_ret, max_lag=3)
        self.assertEqual(len(decay), 3)
        self.assertTrue(decay["Mean_IC"].isna().all())


if __name__ == "__main__":
    unittest.main()
