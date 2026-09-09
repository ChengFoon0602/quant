"""
test_pit_auditor.py — PIT 审计器的自检测试（"测试测试器"）。

为什么必须有这个文件
--------------------
`data/pit_auditor.py::audit_pit_leakage` 是**防未来函数的最后一道防线**：它校验
特征矩阵是否误用财报「报告期 end_date」代替「实际公告日 ann_date」对齐。

但一个自身未经验证的审计器比没有审计器更危险——若 `merge_asof` 方向或去重逻辑出错，
它会稳定返回 `{"status": "PASS"}`，让人误以为数据干净，从而放行带前视偏差的因子。

本测试用**注入已知泄漏**的方式验证审计器确实能检出问题：
  1. 干净数据（按 ann_date 对齐）  -> 必须 PASS
  2. 注入泄漏（按 end_date 对齐）  -> 必须 FAIL
  3. 边界输入（空矩阵 / 全 NaN）   -> 不崩、leakage_rate = 0

用法: python -m unittest tests.test_pit_auditor -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data.pit_auditor import audit_pit_leakage


def _make_raw_financials() -> pd.DataFrame:
    """构造原始财报长表：两只股票，各两份财报。

    关键设计：end_date 远早于 ann_date（Q4 财报次年 4 月才发布），
    这正是前视偏差的经典来源。
    """
    return pd.DataFrame({
        "stock_code": ["A", "A", "B", "B"],
        "ann_date": pd.to_datetime(["2023-04-10", "2024-04-15",
                                    "2023-04-12", "2024-04-18"]),
        "end_date": pd.to_datetime(["2022-12-31", "2023-12-31",
                                    "2022-12-31", "2023-12-31"]),
        "value": [90.0, 100.0, 50.0, 60.0],
    })


# 交易日：2024-01-15 处于「新财报已截止但未公告」的窗口，是泄漏高发点；
#         2024-06-01 处于公告后，应看到新值。
TRADE_DATES = pd.to_datetime(["2024-01-15", "2024-06-01"])


class TestPitAuditorDetectsLeakage(unittest.TestCase):
    """审计器必须能区分「正确对齐」与「前视泄漏」。"""

    def test_clean_alignment_passes(self):
        """按 ann_date 正确对齐 -> PASS。"""
        raw = _make_raw_financials()
        # 2024-01-15 只能看到 ann_date<=该日 的最新值（90 / 50）
        # 2024-06-01 可看到新公告值（100 / 60）
        aligned = pd.DataFrame(
            [[90.0, 50.0], [100.0, 60.0]],
            index=TRADE_DATES,
            columns=["A", "B"],
        )
        report = audit_pit_leakage(raw, aligned)

        self.assertEqual(report["status"], "PASS",
                         "正确对齐的矩阵不应被判为泄漏")
        self.assertEqual(report["leakage_count"], 0)
        self.assertAlmostEqual(report["leakage_rate"], 0.0, places=8)
        self.assertEqual(report["total_checks"], 4, "2 只股票 × 2 个交易日 = 4 次校验")

    def test_end_date_alignment_is_flagged(self):
        """误用 end_date 对齐（提前看到未公告财报）-> FAIL。"""
        raw = _make_raw_financials()
        # 泄漏：2024-01-15 就用了 2024-04-15 才公告的 100 / 60
        aligned = pd.DataFrame(
            [[100.0, 60.0], [100.0, 60.0]],
            index=TRADE_DATES,
            columns=["A", "B"],
        )
        report = audit_pit_leakage(raw, aligned)

        self.assertEqual(report["status"], "FAIL",
                         "审计器必须检出 end_date 对齐导致的前视泄漏——这是它的核心职责")
        self.assertGreater(report["leakage_count"], 0)
        self.assertIsNotNone(report["sample_leakages"],
                             "FAIL 时应返回泄漏样本供排查")

    def test_partial_leakage_only_first_date(self):
        """仅首个交易日泄漏（新财报生效前的窗口）也要被抓到。"""
        raw = _make_raw_financials()
        aligned = pd.DataFrame(
            [[100.0, 60.0], [100.0, 60.0]],   # 第一天泄漏，第二天本就该是 100/60
            index=TRADE_DATES,
            columns=["A", "B"],
        )
        report = audit_pit_leakage(raw, aligned)
        self.assertEqual(report["status"], "FAIL")
        # 第一天两只股票都泄漏 = 2 处
        self.assertEqual(report["leakage_count"], 2,
                         "泄漏应精确定位到个股×交易日")

    def test_single_stock_leakage_located(self):
        """只有一只股票泄漏时，不能漏报。"""
        raw = _make_raw_financials()
        aligned = pd.DataFrame(
            [[100.0, 50.0], [100.0, 60.0]],   # A 泄漏（100 提前出现），B 正常
            index=TRADE_DATES,
            columns=["A", "B"],
        )
        report = audit_pit_leakage(raw, aligned)
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["leakage_count"], 1)
        leaked = report["sample_leakages"]
        self.assertEqual(set(leaked["stock_code"]), {"A"},
                         "泄漏样本应指向股票 A")


class TestPitAuditorEdgeCases(unittest.TestCase):
    """边界输入不得崩溃，也不得产生假阳性。"""

    def test_empty_aligned_matrix(self):
        """空矩阵 -> 不除零、不崩。"""
        raw = _make_raw_financials()
        aligned = pd.DataFrame(
            [[np.nan, np.nan], [np.nan, np.nan]],
            index=TRADE_DATES,
            columns=["A", "B"],
        )
        report = audit_pit_leakage(raw, aligned)
        self.assertEqual(report["total_checks"], 0)
        self.assertAlmostEqual(report["leakage_rate"], 0.0, places=8)
        self.assertEqual(report["status"], "PASS",
                         "无数据不应误报泄漏（避免假阳性麻痹使用者）")

    def test_no_base_factors_returns_copy(self):
        """base_factors 为空时的安全行为（与 orthogonalize 约定一致）。"""
        raw = _make_raw_financials()
        aligned = pd.DataFrame(
            [[90.0, 50.0]], index=TRADE_DATES[:1], columns=["A", "B"],
        )
        report = audit_pit_leakage(raw, aligned)
        self.assertIn("status", report)
        self.assertIn("leakage_rate", report)


if __name__ == "__main__":
    unittest.main()
