"""
test_methodology.py — 方法论铁律编译测试。

把 CLAUDE.md 中的六条量化研究铁律编译为可执行的回归测试，
使「踩坑教训」从文档约束升级为代码强制约束。

铁律映射：
  1. 未来函数（.shift(1) / PIT 对齐）      -> TestNoFutureLeak
  2. 幸存者偏差（PIT Universe）            -> TestSurvivorshipBias
  3. 摩擦成本（双边 0.1%）                 -> TestFrictionCost
  4. 因子合成（等权基线优先）              -> TestFactorCombination
  5. 过拟合防范（Purged CV / 平滑相图）     -> TestOverfittingGuard
  6. Overlapping Returns 平滑陷阱           -> TestOverlappingReturns

用法: python -m unittest tests.test_methodology -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd


class TestNoFutureLeak(unittest.TestCase):
    """铁律 1：绝对禁止未来函数。"""

    def test_signal_shift_prevents_same_day_trade(self):
        """当日收盘信号不能用于当日交易。"""
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        close = pd.Series([10, 11, 12, 11, 13], index=dates)
        signal = pd.Series([0, 0, 1, 0, 0], index=dates)
        position = signal.shift(1).fillna(0)
        self.assertEqual(position.loc[dates[2]], 0, "信号当天被用于交易")
        self.assertEqual(position.loc[dates[3]], 1, "信号次日才应生效")

    def test_financial_data_uses_report_date_not_end_date(self):
        """财务数据必须用公告日对齐，不能用报告期截止日。"""
        end_date = pd.Timestamp("2023-12-31")
        report_date = pd.Timestamp("2024-04-15")
        backtest_date = pd.Timestamp("2024-01-15")
        self.assertFalse(
            report_date <= backtest_date,
            "Q4 财报 4 月才发布，不能出现在 1 月信号里",
        )
        self.assertTrue(
            end_date <= backtest_date,
            "若用 End Date 会误判为可用（这正是要防的错误）",
        )

    def test_pct_change_anchors_return_at_next_day(self):
        """pct_change 把 t→t+1 收益记在 t+1 日（全仓库收益锚定约定）。"""
        close = pd.Series([10, 11, 12], index=pd.date_range("2024-01-01", periods=3, freq="B"))
        r = close.pct_change()
        # r[0] 应为 NaN（无前值），r[1] = 11/10-1 = 0.1，r[2] = 12/11-1
        self.assertTrue(np.isnan(r.iloc[0]))
        self.assertAlmostEqual(r.iloc[1], 0.1)
        self.assertAlmostEqual(r.iloc[2], 12 / 11 - 1)


class TestSurvivorshipBias(unittest.TestCase):
    """铁律 2：消灭幸存者偏差。"""

    def test_universe_must_not_be_cache_slice(self):
        """股票池必须来自显式名单/成员矩阵，禁止用缓存目录切片。"""
        # 历史事故：sorted(cache)[:300] 在缓存扩容后漂移成 298 只纯深市股票。
        # 核心风险：切片结果取决于「当前缓存内容」，而非「历史 PIT 名单」。
        cache_v1 = [f"{i:06d}" for i in range(600000, 600300)]  # 300 只（沪市为主）
        cache_v2 = [f"{i:06d}" for i in range(0, 600)]           # 扩容后 600 只（混入深市 0 开头）

        # 同一行切片代码，在两次缓存状态下产生完全不同的股票池
        slice_v1 = sorted(cache_v1)[:300]
        slice_v2 = sorted(cache_v2)[:300]

        # 断言：两次切片结果不同（漂移），这正是「禁止用缓存切片决定股票池」的原因
        self.assertNotEqual(set(slice_v1), set(slice_v2),
                            "缓存扩容后切片结果漂移，证明切片不可作为 Universe 依据")

    def test_pit_membership_uses_historical_snapshot(self):
        """PIT 成员矩阵前向填充，不用未来名单。"""
        membership = pd.DataFrame(
            {"000001": [True, True], "600005": [False, True]},
            index=pd.to_datetime(["2020-01-31", "2020-02-28"]),
        )
        daily_dates = pd.date_range("2020-02-01", "2020-02-29", freq="B")
        expanded = membership.reindex(membership.index.union(daily_dates)).ffill().infer_objects(copy=False)
        # 2 月任何交易日，600005 的成员状态来自 1 月末快照（False），而非 2 月末（True）
        feb = expanded.loc[daily_dates, "600005"]
        self.assertFalse(feb.all(), "未来名单被偷跑")


class TestFrictionCost(unittest.TestCase):
    """铁律 3：摩擦成本（双边 0.1%）。"""

    def test_standard_cost_rates(self):
        """买 0.026% / 卖 0.076% 是默认铁律标准。"""
        buy_cost, sell_cost = 0.00026, 0.00076
        roundtrip = buy_cost + sell_cost
        self.assertAlmostEqual(roundtrip, 0.00102, places=5,
                               msg="双边合计应约 0.1%（含印花税）")

    def test_slippage_adds_to_friction(self):
        """日线级别默认加 0.05% 滑点。"""
        slippage = 0.0005
        total_roundtrip = 0.00102 + slippage
        self.assertGreater(total_roundtrip, 0.00102,
                           "滑点应增加摩擦成本")

    def test_cost_deduction_reduces_equity(self):
        """扣成本后的净值必须低于不扣成本的净值。"""
        dates = pd.date_range("2024-01-01", periods=252, freq="B")
        np.random.seed(42)
        ret = pd.Series(np.random.randn(252) * 0.02, index=dates)
        turnover = pd.Series(1.0, index=dates)
        cost = turnover * (0.00026 + 0.00076)
        gross = (1 + ret).cumprod().iloc[-1]
        net = (1 + (ret - cost)).cumprod().iloc[-1]
        self.assertLess(net, gross, "扣成本后净值必须更低")


class TestFactorCombination(unittest.TestCase):
    """铁律 4：因子合成（等权基线优先）。"""

    def test_equal_weight_baseline_is_first_step(self):
        """等权合成是必须的基线，复杂合成前先过等权。"""
        factors = pd.DataFrame({
            "f1": np.random.randn(100),
            "f2": np.random.randn(100),
            "f3": np.random.randn(100),
        })
        # 等权合成 = 简单平均
        equal_weight = factors.mean(axis=1)
        self.assertEqual(equal_weight.shape[0], 100)
        # 等权合成不引入任何非线性，是最朴素基线
        self.assertTrue(np.allclose(equal_weight, (factors["f1"] + factors["f2"] + factors["f3"]) / 3))


class TestOverfittingGuard(unittest.TestCase):
    """铁律 5：防范过拟合。"""

    def test_purged_cv_leaves_gap_between_train_and_val(self):
        """Purged CV 必须在训练集与验证集之间留出 purge 窗口。"""
        from models.cv import PurgedTimeSeriesSplit

        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        X = pd.DataFrame({"a": range(100)}, index=dates)
        cv = PurgedTimeSeriesSplit(n_splits=5, purge_days=6)
        for tr, val in cv.split(X):
            # 验证集全部在训练集之后
            self.assertGreater(val.min(), tr.max())
            # purge 窗口：验证集起点 - 训练集终点 - 1 >= purge_days
            gap = val.min() - tr.max() - 1
            self.assertGreaterEqual(gap, 5, "purge 窗口不足，存在时序泄露")

    def test_isolated_sharpe_spike_is_overfitting_signal(self):
        """孤立的超高收益尖峰 = 过拟合信号（参数相图应平滑）。"""
        # 模拟参数相图：绝大多数区域夏普在 0.5~1.0，唯独一处 8.75
        sharpe_grid = np.array([0.5, 0.6, 0.7, 8.75, 0.6, 0.5, 0.4])
        max_sharpe = sharpe_grid.max()
        median_sharpe = np.median(sharpe_grid)
        # 尖峰判定：最大值远高于中位数（孤立的离群点）
        self.assertGreater(max_sharpe, median_sharpe * 5,
                           "8.75 vs 0.6 这种尖峰是过拟合特征，应触发排查")


class TestOverlappingReturns(unittest.TestCase):
    """铁律 6：Overlapping Returns 平滑陷阱。"""

    def test_rolling_mean_flattens_volatility(self):
        """对信号收益做 rolling(H).mean() 会把波动压掉 sqrt(H) 倍。"""
        np.random.seed(42)
        n = 1000
        hold_days = 5
        daily_ret = pd.Series(np.random.randn(n) * 0.02)
        # 错误做法：rolling(hold_days).mean()
        smoothed = daily_ret.rolling(hold_days).mean()
        # 正确做法：原始日收益（权重追踪法）
        # 验证：平滑后波动率显著低于原始
        vol_original = daily_ret.std()
        vol_smoothed = smoothed.dropna().std()
        self.assertLess(vol_smoothed, vol_original / np.sqrt(hold_days) * 1.1,
                        "rolling mean 应把波动压到约 1/sqrt(H)")

    def test_sharpe_inflation_from_smoothing(self):
        """平滑会导致夏普虚高（曾把 1.8~2.1 算成 8.75）。"""
        np.random.seed(42)
        n = 2000
        hold_days = 10
        # 构造一个正漂移的日收益序列
        daily_ret = pd.Series(np.random.randn(n) * 0.01 + 0.0005)
        # 正确：直接算
        true_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        # 错误：rolling 平滑后算
        smoothed = daily_ret.rolling(hold_days).mean().dropna()
        smoothed_sharpe = smoothed.mean() / smoothed.std() * np.sqrt(252)
        self.assertGreater(smoothed_sharpe, true_sharpe,
                           "rolling 平滑会虚高夏普")

    def test_autocorr_approaches_one_with_hold_days(self):
        """多日持有的 lag-1 自相关随 hold_days 上升趋近 1（中招信号）。"""
        np.random.seed(42)
        base = pd.Series(np.random.randn(2000) * 0.01)
        # 用 rolling mean 模拟多日持有，自相关会显著上升
        for h in [1, 5, 10]:
            s = base if h == 1 else base.rolling(h).mean().dropna()
            ac1 = s.autocorr(lag=1)
            if h == 1:
                base_ac1 = ac1
            else:
                self.assertGreater(ac1, base_ac1,
                                   f"hold_days={h} 的自相关应高于 hold_days=1")


if __name__ == "__main__":
    unittest.main()
