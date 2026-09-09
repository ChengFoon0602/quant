"""
test_weight_vectorization.py — build_portfolio 权重构造向量化的等价性回归。

背景（CLAUDE.md TODO #5，2026-09-09）：
  原实现逐日 Python 循环构造目标权重（3886 日 × 1326 股约 8.3s），违反「向量化优先」铁律。
  实测向量化版本 0.31s（27×）且输出逐位等价。本测试在模块内冻结**原始循环实现**为参考，
  断言向量化后的 build_portfolio 在多种配置下与之完全一致（防止重构引入细微语义漂移）。

用法: python -m unittest tests.test_weight_vectorization -v
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.portfolio_backtest import build_portfolio


def _reference_build(
    pred_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
    long_only: bool = False,
    short_only: bool = False,
    top_q: float = 0.20,
    bottom_q: float = 0.20,
    buy_cost: float = 0.00026,
    sell_cost: float = 0.00076,
    hold_days: int = 5,
    gate: pd.Series | None = None,
    position_scale: pd.Series | None = None,
):
    """原始逐日循环实现（冻结 2026-09-09 前的语义，仅作等价参考）。"""
    daily_ret = close_matrix.pct_change()
    common_dates = pred_df.index.intersection(daily_ret.index)
    common_cols = pred_df.columns.intersection(daily_ret.columns)
    p = pred_df.loc[common_dates, common_cols]
    r = daily_ret.loc[common_dates, common_cols]

    W_target = pd.DataFrame(0.0, index=common_dates, columns=common_cols)
    for d in common_dates:
        pv = p.loc[d]
        mask = pv.notna()
        if mask.sum() < max(int(1 / top_q), int(1 / bottom_q)) * 3:
            continue
        valid_p = pv[mask]
        top_thr = valid_p.quantile(1 - top_q)
        bottom_thr = valid_p.quantile(bottom_q)
        top = valid_p[valid_p >= top_thr].index
        bottom = valid_p[valid_p <= bottom_thr].index
        if not short_only and len(top):
            W_target.loc[d, top] = 1.0 / len(top)
        if not long_only and len(bottom):
            W_target.loc[d, bottom] = -1.0 / len(bottom)

    if gate is not None:
        g = gate.reindex(W_target.index).ffill().fillna(0.0)
        W_target = W_target.mul(g, axis=0)

    W_held = W_target.rolling(hold_days, min_periods=1).mean()
    if position_scale is not None:
        W_held = W_held.mul(position_scale.reindex(W_held.index).ffill().fillna(1.0), axis=0)

    W_lag = W_held.shift(1)
    port_gross = (W_lag * r).sum(axis=1, min_count=1)
    delta_w = W_held - W_held.shift(1)
    buy_t = delta_w.clip(lower=0.0).sum(axis=1)
    sell_t = (-delta_w).clip(lower=0.0).sum(axis=1)
    port_ret = port_gross - buy_t * buy_cost - sell_t * sell_cost
    port_ret = port_ret.iloc[hold_days:].dropna()
    return port_ret, W_held.loc[port_ret.index]


def _make_panel(n_dates=120, n_stocks=40, seed=0, nan_frac=0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    cols = [f"S{i:04d}" for i in range(n_stocks)]
    pred = pd.DataFrame(rng.normal(size=(n_dates, n_stocks)), index=dates, columns=cols)
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.015, (n_dates, n_stocks)), axis=0)),
        index=dates, columns=cols)
    if nan_frac > 0:
        mask = rng.random((n_dates, n_stocks)) < nan_frac
        pred[mask] = np.nan
    return pred, close


class TestVectorizedEquivalence(unittest.TestCase):
    """向量化 vs 原始逐日循环：输出必须逐位一致。"""

    def _assert_equivalent(self, pred, close, **kw):
        ref_ret, ref_W = _reference_build(pred, close, **kw)
        out = build_portfolio(pred, close, return_weights=True, **kw)
        if isinstance(out, tuple):
            df, W = out
        else:
            df, W = out, None
        new_ret = df["port_ret"]
        # 数值逐位一致 + 索引一致（避免 name/freq 等元数据干扰）
        pd.testing.assert_index_equal(new_ret.index, ref_ret.index)
        np.testing.assert_allclose(new_ret.values, ref_ret.values,
                                   atol=1e-14, rtol=1e-12,
                                   err_msg=f"port_ret 不一致 kw={kw}")
        if W is not None:
            pd.testing.assert_index_equal(W.index, ref_W.index)
            pd.testing.assert_index_equal(W.columns, ref_W.columns)
            np.testing.assert_allclose(W.values, ref_W.values,
                                       atol=1e-14, rtol=1e-12,
                                       err_msg=f"W_held 不一致 kw={kw}")

    def test_long_short_default(self):
        pred, close = _make_panel(seed=1)
        self._assert_equivalent(pred, close)

    def test_long_only(self):
        pred, close = _make_panel(seed=2)
        self._assert_equivalent(pred, close, long_only=True)

    def test_short_only_with_gate(self):
        pred, close = _make_panel(seed=3)
        gate = pd.Series(1.0, index=pred.index)
        gate.iloc[20:40] = 0.0
        self._assert_equivalent(pred, close, short_only=True, gate=gate, hold_days=10)

    def test_heavy_nan(self):
        """40% 缺失值下分位与归一仍一致。"""
        pred, close = _make_panel(n_dates=200, n_stocks=60, seed=4, nan_frac=0.4)
        self._assert_equivalent(pred, close)
        self._assert_equivalent(pred, close, long_only=True)

    def test_position_scale(self):
        pred, close = _make_panel(seed=5)
        ps = pd.Series(0.5, index=pred.index)
        ps.iloc[:30] = 1.0
        self._assert_equivalent(pred, close, position_scale=ps)

    def test_hold_variants(self):
        pred, close = _make_panel(seed=6)
        for h in [1, 5, 20]:
            self._assert_equivalent(pred, close, hold_days=h)


class TestRiskBuildWeightEquivalence(unittest.TestCase):
    """risk/portfolio.py::build_weight_portfolio 的向量化等价性。"""

    def _assert_equivalent(self, pred, close, **kw):
        from risk.portfolio import build_weight_portfolio

        def reference():
            p = pred
            r = close.pct_change()
            cd = p.index.intersection(r.index)
            cc = p.columns.intersection(r.columns)
            p, r = p.loc[cd, cc], r.loc[cd, cc]
            min_stocks = max(int(1.0 / kw.get("top_q", 0.2)), int(1.0 / kw.get("bottom_q", 0.2))) * 2
            W = pd.DataFrame(0.0, index=cd, columns=cc)
            for d in cd:
                pv = p.loc[d]
                m = pv.notna()
                if m.sum() < min_stocks:
                    continue
                v = pv[m]
                tt, bt = v.quantile(1 - kw.get("top_q", 0.2)), v.quantile(kw.get("bottom_q", 0.2))
                top = v[v >= tt].index
                bot = v[v <= bt].index
                if not kw.get("short_only") and len(top):
                    W.loc[d, top] = 1.0 / len(top)
                if not kw.get("long_only") and len(bot):
                    W.loc[d, bot] = -1.0 / len(bot)
            if kw.get("gate") is not None:
                W = W.mul(kw["gate"].reindex(cd).fillna(1.0), axis=0)
            Wh = W.rolling(kw.get("hold_days", 5), min_periods=1).mean()
            if kw.get("position_scale") is not None:
                Wh = Wh.mul(kw["position_scale"].reindex(cd).fillna(1.0), axis=0)
            Wl = Wh.shift(1).fillna(0.0)
            gross = (Wl * r).sum(axis=1)
            dW = Wh - Wh.shift(1).fillna(0.0)
            cost = dW.clip(lower=0).sum(axis=1) * kw.get("buy_cost", 0.00026) \
                + (-dW).clip(lower=0).sum(axis=1) * kw.get("sell_cost", 0.00076)
            pr = (gross - cost).iloc[kw.get("hold_days", 5):].dropna()
            return pr, Wh.loc[pr.index]

        ref_ret, ref_W = reference()
        out = build_weight_portfolio(pred, close, return_weights=True, **kw)
        df, W = out
        np.testing.assert_allclose(df["port_ret"].values, ref_ret.values,
                                   atol=1e-14, rtol=1e-12)
        pd.testing.assert_index_equal(df["port_ret"].index, ref_ret.index)
        np.testing.assert_allclose(W.values, ref_W.values, atol=1e-14, rtol=1e-12)

    def test_risk_ls(self):
        pred, close = _make_panel(seed=7, n_dates=100, n_stocks=30)
        self._assert_equivalent(pred, close)

    def test_risk_lo_nan(self):
        pred, close = _make_panel(seed=8, n_dates=150, n_stocks=40, nan_frac=0.35)
        self._assert_equivalent(pred, close, long_only=True)

    def test_risk_gate(self):
        pred, close = _make_panel(seed=9, n_dates=120, n_stocks=30)
        gate = pd.Series(1.0, index=pred.index)
        gate.iloc[10:25] = 0.0
        self._assert_equivalent(pred, close, gate=gate, hold_days=3)


if __name__ == "__main__":
    unittest.main()
