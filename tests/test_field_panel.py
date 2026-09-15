"""
test_field_panel.py — 多标的面板构建契约（data/fetcher.py::load_field_panel）。

该函数统一了此前散落在各离线运行器里的「逐票读 cache → 拼宽表」逻辑，
因此需要一份显式契约：字段白名单、列序保持、缺失标的跳过、短样本过滤、日期过滤。

用法: python -m unittest tests.test_field_panel -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from data import fetcher

# 110 个交易日 > 默认 min_rows=100，从而让测试同时覆盖默认阈值行为
N_DAYS = 110


def _daily_frame(base: float = 100.0, n_days: int = N_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    close = pd.Series(base + rng.normal(0, 1, n_days), index=dates)
    return pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1e6, "amount": 1e8,
    }, index=dates)


class TestLoadFieldPanel(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.dir = Path(self.td.name)
        for sym, base in (("600000", 100.0), ("600001", 200.0)):
            _daily_frame(base).reset_index(names="date").to_csv(
                self.dir / f"{sym}.csv", index=False)

        patcher = mock.patch.object(
            fetcher, "_cache_path",
            side_effect=lambda s, adjust="2": self.dir / f"{s}.csv")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_returns_one_frame_per_field(self):
        panel = fetcher.load_field_panel(["600000"], fields=("open", "close"))
        self.assertEqual(set(panel), {"open", "close"})
        self.assertEqual(panel["open"].shape, panel["close"].shape)

    def test_column_order_follows_input_order(self):
        """列序必须与传入的 symbols 顺序一致 —— 调用方依赖它做对齐。"""
        panel = fetcher.load_field_panel(["600001", "600000"], fields=("close",))
        self.assertEqual(list(panel["close"].columns), ["600001", "600000"])

    def test_index_is_sorted(self):
        panel = fetcher.load_field_panel(["600000"], fields=("close",))
        self.assertTrue(panel["close"].index.is_monotonic_increasing)

    def test_missing_symbol_is_skipped(self):
        panel = fetcher.load_field_panel(["600000", "999999"], fields=("close",))
        self.assertEqual(list(panel["close"].columns), ["600000"])

    def test_min_rows_uses_available_history_not_filtered_window(self):
        """min_rows 按**可得历史长度**判断 —— 窄日期窗口不得把标的判为无效。"""
        panel = fetcher.load_field_panel(
            ["600000"], fields=("close",), start="2021-01-02", end="2021-01-06")
        self.assertEqual(len(panel["close"]), 3)  # 交易日 01-04 / 01-05 / 01-06

    def test_min_rows_filters_short_history(self):
        panel = fetcher.load_field_panel(["600000"], fields=("close",), min_rows=10_000)
        self.assertEqual(panel["close"].shape[1], 0)

    def test_unknown_field_raises(self):
        with self.assertRaises(ValueError):
            fetcher.load_field_panel(["600000"], fields=("vwap",))

    def test_non_price_fields_supported(self):
        panel = fetcher.load_field_panel(["600000"], fields=("volume", "amount"))
        self.assertGreater(int(panel["volume"].to_numpy().sum()), 0)
        self.assertEqual(panel["volume"].shape, panel["amount"].shape)


if __name__ == "__main__":
    unittest.main()
