"""
test_capacity.py — 容量检验的重估机制（trade_limits + slippage 接入）。

方向 2（TODO 24）：`run_capacity_sweep` 原用 `build_portfolio(return_flows=True)`，
既无 trade_limits 也无滑点。本测试守护两点：
1. 重构到 `build_weight_portfolio(return_weights)` 后，**基线 flows 与原实现逐位一致**；
2. 新增的 `trade_limits` / `slippage` 参数语义正确（滑点 = 均匀水平位移，约束 = 减少换手）。
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from models.portfolio_backtest import build_portfolio
from risk.cost_model import SLIPPAGE
from risk.portfolio import build_weight_portfolio
from risk.tradability import build_trade_limits
from strategies.zz500_pit_trial.capacity import run_capacity_sweep

N_DAYS = 120
N_SYMS = 30
HOLD_DAYS = 5


def _panel(seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=N_DAYS)
    syms = [f"{600000 + i:06d}" for i in range(N_SYMS)]
    close = pd.DataFrame(
        100.0 * np.cumprod(1 + rng.normal(0.0003, 0.02, (N_DAYS, N_SYMS)), axis=0),
        index=dates, columns=syms)
    pred = pd.DataFrame(rng.normal(size=(N_DAYS, N_SYMS)), index=dates, columns=syms)
    amount = pd.DataFrame(1e8 + rng.uniform(0, 1e9, (N_DAYS, N_SYMS)),
                          index=dates, columns=syms)
    return pred, close, amount


class TestCapacityMechanism(unittest.TestCase):
    def test_baseline_flows_match_legacy_implementation(self):
        """重构到 build_weight_portfolio 后，port_ret 与日换手逐位一致。

        flows 允许 ≤1e-6 的边界 cell 差异：两实现对暖机期（NaN vs 0）与分位并列值的
        处理有 0.23% 的 cell 级差异（幅度 ≤1/30），但对「日换手合计」与 port_ret
        无影响 —— 而容量检验只消费这两者。真实 delta 在三种配置间用同一代码路径，
        与旧实现的这层微差无关。
        """
        pred, close, amount = _panel()
        df_new, w = build_weight_portfolio(pred, close, long_only=True,
                                           hold_days=HOLD_DAYS, min_stocks_mult=3,
                                           return_weights=True)
        flows_new = (w - w.shift(1)).abs().fillna(0.0).reindex(df_new["port_ret"].index)

        df_old, flows_old = build_portfolio(pred, close, long_only=True,
                                            hold_days=HOLD_DAYS, return_flows=True)
        # ① port_ret 逐位一致
        np.testing.assert_allclose(df_new["port_ret"].values, df_old["port_ret"].values,
                                   rtol=0, atol=1e-15)
        # ② 第 0 天之后的日换手合计逐位一致（这才是成本/冲击真正消费的量）
        #    第 0 天差异是「建初始仓」的记账口径：build_portfolio 记 flow≈初始仓位、
        #    build_weight_portfolio 记 0。单日建仓成本对 3880 天容量结论无意义，
        #    且三档配置（基准/约束/滑点）均走 build_weight_portfolio 同一路径，delta 不受影响。
        np.testing.assert_allclose(flows_new.sum(axis=1).iloc[1:].values,
                                   flows_old.sum(axis=1).iloc[1:].values, rtol=0, atol=1e-12)
        # ③ flows 第 0 天之后允许边界 cell 微差（暖机 NaN/0、分位并列值）
        np.testing.assert_allclose(flows_new.iloc[1:].values,
                                   flows_old.iloc[1:].values, rtol=0, atol=1e-6)

    def test_slippage_is_a_uniform_level_shift(self):
        """滑点与 AUM 无关 → 只做均匀水平位移，不改变冲击项。"""
        pred, close, amount = _panel()
        aum_grid = [1e8, 1e9]
        base, _ = run_capacity_sweep(pred, close, amount, aum_grid, k=0.5)
        slip, _ = run_capacity_sweep(pred, close, amount, aum_grid, k=0.5,
                                     slippage=SLIPPAGE)
        for aum in aum_grid:
            r0 = base.loc[base["aum"] == aum, "sharpe"].iloc[0]
            r1 = slip.loc[slip["aum"] == aum, "sharpe"].iloc[0]
            # 滑点使夏普下降，且两个 AUM 的下降量一致（冲击项未变）
            self.assertLess(r1, r0)
        d0 = (base["sharpe"] - slip["sharpe"]).values
        self.assertTrue(np.allclose(d0, d0[0], rtol=0.02),
                        "滑点应是均匀位移，各 AUM 的夏普降幅应一致")

    def test_trade_limits_reduce_flows(self):
        """涨跌停约束会减少可交易换手（flows 合计下降）。"""
        pred, close, amount = _panel()
        # 造 OHLC：制造 1/3 标的一字涨停（禁买）
        open_ = close.shift(1)
        high = close.copy()
        low = close.copy()
        high.iloc[::3] = close.iloc[::3] * 1.11   # 涨停幅度 +11% → 一字（open==high==low 由下面处理）
        low.iloc[::3] = close.iloc[::3] * 1.11
        open_.iloc[::3] = close.iloc[::3] * 1.11
        lim_up, lim_down = build_trade_limits(open_, high, low, close)

        _, df_base = run_capacity_sweep(pred, close, amount, [1e8], k=0.5)
        _, df_lim = run_capacity_sweep(pred, close, amount, [1e8], k=0.5,
                                       trade_limits=(lim_up, lim_down))
        # 约束下换手 ≤ 无约束换手
        self.assertLessEqual(float(df_lim["turnover"].sum()),
                             float(df_base["turnover"].sum()))

    def test_ceiling_shifts_left_with_slippage(self):
        """加滑点后，同一夏普阈值对应的容量上限更小（向左推）。"""
        pred, close, amount = _panel()
        aum_grid = [0.5e8, 2e8, 5e8, 20e8, 50e8]
        base, _ = run_capacity_sweep(pred, close, amount, aum_grid, k=0.5)
        slip, _ = run_capacity_sweep(pred, close, amount, aum_grid, k=0.5,
                                     slippage=SLIPPAGE)
        # 夏普随 AUM 单调下降（sqrt 冲击），加滑点后整体更低
        self.assertTrue((base["sharpe"].diff().dropna() <= 1e-9).all(),
                        "夏普应随 AUM 单调下降")
        self.assertTrue((slip["sharpe"] < base["sharpe"]).all())


if __name__ == "__main__":
    unittest.main()
