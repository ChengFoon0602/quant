"""
test_price_adjust.py — 复权口径的自动断言（铁律 3.5 / `docs/回测语义对照表.md` §1）。

守护三件事：

  1. **路径映射**：前复权沿用旧路径 `{symbol}.csv`，其余用 `{symbol}_adj{adjust}.csv`；
  2. **三种复权互不共用文件** —— 这才是铁律 3.5 的真正风险：前/后复权互相覆盖，
     会让截面因子的时点可比性被无声污染；
  3. **读写同映射**：`load_daily` 与 `download_daily` 必须以同一参数调用同一 `_cache_path`，
     否则会出现「写进 A 文件、读的是 B 文件」。

用法: python -m unittest tests.test_price_adjust -v
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from data import fetcher

ADJ_MODES = ("1", "2", "3")  # 后复权 / 前复权 / 不复权


def _daily_frame(n_days: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    close = pd.Series(100.0 + rng.normal(0, 1, n_days), index=dates)
    return pd.DataFrame({
        "open": close, "high": close + 1.0, "low": close - 1.0,
        "close": close, "volume": 1e6, "amount": 1e8,
    }, index=dates)


class TestCachePathMapping(unittest.TestCase):
    def test_front_adjust_uses_legacy_path(self):
        """前复权沿用旧路径，向后兼容存量缓存。"""
        self.assertEqual(fetcher._cache_path("000001", adjust="2").name, "000001.csv")

    def test_other_adjust_uses_suffix(self):
        self.assertEqual(fetcher._cache_path("000001", adjust="1").name, "000001_adj1.csv")
        self.assertEqual(fetcher._cache_path("000001", adjust="3").name, "000001_adj3.csv")

    def test_default_adjust_is_front(self):
        self.assertEqual(fetcher._cache_path("000001"),
                         fetcher._cache_path("000001", adjust="2"))

    def test_three_adjust_modes_never_share_a_file(self):
        """三种复权必须落在三个不同文件 —— 否则互相覆盖会污染因子时点可比性。"""
        paths = {a: fetcher._cache_path("600519", adjust=a) for a in ADJ_MODES}
        self.assertEqual(len(set(paths.values())), 3,
                         f"复权缓存路径发生碰撞：{ {k: str(v) for k, v in paths.items()} }")

    def test_different_symbols_never_share_a_file(self):
        a = fetcher._cache_path("000001", adjust="1")
        b = fetcher._cache_path("000002", adjust="1")
        self.assertNotEqual(a, b)


class TestReadWriteShareMapping(unittest.TestCase):
    """load_daily 与 download_daily 必须走同一映射。"""

    def test_load_daily_passes_adjust_through(self):
        calls = []

        def spy(symbol, adjust="2"):
            calls.append((symbol, adjust))
            return Path("no") / "such" / "file.csv"

        with mock.patch.object(fetcher, "_cache_path", side_effect=spy):
            self.assertIsNone(fetcher.load_daily("000001", adjust="1"))
        self.assertEqual(calls, [("000001", "1")])

    def test_download_daily_passes_adjust_through(self):
        """缓存已全覆盖时不触网，可直接观察映射调用参数。"""
        df = _daily_frame()
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "000001_adj1.csv"
            df.reset_index(names="date").to_csv(p, index=False)
            calls = []

            def spy(symbol, adjust="2"):
                calls.append((symbol, adjust))
                return p

            with mock.patch.object(fetcher, "_cache_path", side_effect=spy):
                out = fetcher.download_daily(
                    "000001", start="2021-01-02", end="2021-01-13", adjust="1")

            self.assertEqual(calls[0], ("000001", "1"))
            self.assertEqual(len(out), 8)  # 2021-01-02 ~ 2021-01-13 的交易日


if __name__ == "__main__":
    unittest.main()
