"""
test_orthogonalize.py — 横截面正交化的回归测试。

⚠️ 2026-09-09 重构：原文件为 pytest 风格裸函数（`def test_xxx():` + assert），
而本项目其余测试均为 unittest，且环境未安装 pytest —— 导致 `python -m unittest
tests.test_orthogonalize` 收集到 **0 个测试**，这两个用例长期游离于测试套件之外。
现改写为 unittest.TestCase，使其真正被收集执行；原有断言逻辑全部保留。

同时在 TestOrthogonalizeFailureVisibility 中覆盖 2026-09-09 的修复：
原实现 `except Exception: pass` 会静默吞掉求解失败，使失败日被当成"无数据"。

用法: python -m unittest tests.test_orthogonalize -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from signals.orthogonalize import cross_sectional_orthogonalize


class TestOrthogonality(unittest.TestCase):
    """正交化后残差必须与基准因子严格无关。"""

    def test_residual_uncorrelated_with_base(self):
        """残差与基准因子的截面相关系数为 0。"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=10)
        stocks = ["AAPL", "MSFT", "GOOG"]

        base_factor = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=stocks)
        noise = pd.DataFrame(np.random.randn(10, 3) * 0.1, index=dates, columns=stocks)
        target_factor = base_factor * 2 + noise

        residual = cross_sectional_orthogonalize(target_factor, {"size": base_factor})

        for date in dates:
            x = base_factor.loc[date].values
            y_res = residual.loc[date].values
            corr = np.corrcoef(x, y_res)[0, 1]
            self.assertLess(
                abs(corr), 1e-10,
                f"截面 {date} 相关性未完全剥离: corr={corr}",
            )


class TestOrthogonalizeLookAhead(unittest.TestCase):
    """正交化不得引入未来函数。"""

    def test_truncation_does_not_change_earlier_values(self):
        """截取前 N 天重算，前 N 天结果必须与全量计算完全一致。"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=5)
        stocks = ["S1", "S2", "S3"]

        base = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=stocks)
        target = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=stocks)

        res_full = cross_sectional_orthogonalize(target, {"b": base})
        res_partial = cross_sectional_orthogonalize(
            target.iloc[:3], {"b": base.iloc[:3]}
        )

        diff = np.abs(res_partial.values - res_full.iloc[:3].values).max()
        self.assertLess(
            diff, 1e-10,
            f"未来函数检查失败！不同长度输入的正交结果存在差异，最大值: {diff}",
        )


class TestOrthogonalizeFailureVisibility(unittest.TestCase):
    """失败与降级不得静默（2026-09-09 修复：原 except Exception: pass）。

    注：np.linalg.lstsq 在秩亏（基准因子共线）时不抛异常，而是返回最小范数解，
    残差本身仍然唯一且有效 —— 因此 `failed_dates` 通常为空。
    真正频繁发生、且过去被静默吞掉的是「有效样本不足」。
    """

    def test_attrs_always_present(self):
        """返回值必须携带诊断字段，便于调用方检查。"""
        dates = pd.date_range("2023-01-01", periods=5)
        stocks = ["S1", "S2", "S3", "S4"]
        base = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=stocks)
        target = pd.DataFrame(np.random.randn(5, 4), index=dates, columns=stocks)

        res = cross_sectional_orthogonalize(target, {"b": base})
        self.assertIn("failed_dates", res.attrs)
        self.assertIn("insufficient_obs_dates", res.attrs)
        self.assertEqual(res.attrs["failed_dates"], [])

    def test_insufficient_sample_is_tracked_not_silent(self):
        """有效样本不足的日期必须被记录，而非静默留 NaN。"""
        dates = pd.date_range("2023-01-01", periods=3)
        stocks = ["S1", "S2", "S3"]

        base = pd.DataFrame(np.random.randn(3, 3), index=dates, columns=stocks)
        target = pd.DataFrame(np.random.randn(3, 3), index=dates, columns=stocks)
        # 只留 1 只有效样本（其余置 NaN），低于 required = n_base + 2 = 3
        target.iloc[0, 1:] = np.nan

        res = cross_sectional_orthogonalize(target, {"b": base})

        self.assertIn(
            dates[0], res.attrs["insufficient_obs_dates"],
            "样本不足的日期必须出现在诊断信息中，不能静默跳过",
        )
        self.assertTrue(
            res.loc[dates[0]].isna().all(),
            "未做回归的日期残差应为 NaN",
        )
        # 其余日期正常求解
        self.assertFalse(res.loc[dates[1]].isna().all())

    def test_empty_base_factors_returns_copy(self):
        """base_factors 为空时返回副本，不修改原对象。"""
        dates = pd.date_range("2023-01-01", periods=3)
        target = pd.DataFrame(np.random.randn(3, 2), index=dates, columns=["A", "B"])
        original = target.copy()

        res = cross_sectional_orthogonalize(target, {})
        pd.testing.assert_frame_equal(res, original)
        self.assertIsNot(res, target, "应返回副本而非原对象引用")


if __name__ == "__main__":
    unittest.main()
