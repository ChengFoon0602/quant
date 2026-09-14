"""
test_reconciliation_real_data.py — 真实数据账本对账（轻量采样版）。

分工
----
    本文件              50 票采样，秒级，随测试套件运行 —— 意在「每次都跑」
    run_reconciliation.py  全量 790 票，离线按需运行 —— 意在「出结论前跑」

采样刻意保留**真实数据形态**（NaN 稀疏结构、停牌、离散跳空），
这是合成数据无法复现的；对账在这一层通过，才算「真实报告数据自证」。

用法: python -m unittest tests.test_reconciliation_real_data -v
"""

from __future__ import annotations

import unittest

import numpy as np

from backtest.reconciliation import assert_books_equal, assert_closed
from risk.portfolio import build_weight_portfolio

try:
    from models.portfolio_backtest import build_portfolio as _models_build_portfolio
except Exception:  # noqa: BLE001
    _models_build_portfolio = None

try:
    from run_reconciliation import load_real_panel

    _REAL_OK = True
except Exception:  # noqa: BLE001
    _REAL_OK = False

N_SYMS = 50
HOLD_DAYS = 5


@unittest.skipUnless(_REAL_OK, "真实数据加载链路不可用")
class TestRealDataReconciliation(unittest.TestCase):
    """真实价格 + 真实 OOF 预测上的账本对账。"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.pred, cls.close = load_real_panel(n_syms=N_SYMS)
            cls.available = cls.close.shape[1] > 0
        except Exception:  # noqa: BLE001
            cls.available = False

    def setUp(self):
        if not getattr(self, "available", False):
            self.skipTest("真实缓存与 OOF 预测的交集为空")

    def test_real_panel_keeps_real_shape(self):
        """采样必须保留真实形态：含 NaN（停牌/未上市）、日期有序、无负价。"""
        nan_frac = float(self.close.isna().to_numpy().mean())
        self.assertGreater(nan_frac, 0.0, "真实面板应含 NaN —— 否则采样失真")
        self.assertTrue(self.close.index.is_monotonic_increasing)
        finite = self.close.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        self.assertTrue((finite >= 0).all())

    def test_risk_book_closes_on_real_data(self):
        res, W = build_weight_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS, return_weights=True)
        gaps = assert_closed(res, W, self.close)
        self.assertLess(max(gaps.values()), 1e-9)

    @unittest.skipIf(_models_build_portfolio is None, "models.portfolio_backtest 不可导入")
    def test_models_book_closes_on_real_data(self):
        res, W = _models_build_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS, return_weights=True)
        gaps = assert_closed(res, W, self.close)
        self.assertLess(max(gaps.values()), 1e-9)

    @unittest.skipIf(_models_build_portfolio is None, "models.portfolio_backtest 不可导入")
    def test_cross_implementation_agrees_when_mult_unified(self):
        """门槛对齐后，两条实现在真实数据上必须逐位一致。

        不对齐时二者本就可能有差异（`min_stocks_mult` 默认 2 vs 3，
        在稀疏截面上会显著）——这正是 `TestKnownDivergence` 钉住的那条。
        """
        res_r, _ = build_weight_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS,
            min_stocks_mult=3, return_weights=True)
        res_m, _ = _models_build_portfolio(
            self.pred, self.close, hold_days=HOLD_DAYS,
            min_stocks_mult=3, return_weights=True)
        assert_books_equal(res_r, res_m, tol=1e-9)


if __name__ == "__main__":
    unittest.main()
