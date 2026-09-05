"""
risk/orchestrator.py — 组合构建与再平衡编排层。

背景：
  现有 risk/portfolio.py 提供了权重追踪组合构建（build_weight_portfolio）和
  波动率目标（apply_volatility_target），但缺一个统一的编排层，把以下拼图组装起来：
    - 调仓日历（月/周/季末调仓日）
    - 目标权重 -> 实际成交的撮合（受涨跌停/停牌/单资产上限约束）
    - 再平衡成本核算
    - 多资产约束（杠杆上限、单资产上限、换手上限）

设计目标：单一入口，统一口径（与 CLAUDE.md 铁律对齐），供各策略复用，
替代各策略 report.py 中各自手写的「填权重 + 算换手」重复代码。

用法:
    from risk.orchestrator import PortfolioOrchestrator

    orch = PortfolioOrchestrator(
        rebalance="monthly",
        max_leverage=1.0,
        max_weight_per_asset=0.10,
        max_turnover=0.50,
        buy_cost=0.00026,
        sell_cost=0.00076,
    )
    result = orch.run(target_weights_fn, close_matrix, ...)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from risk.portfolio import calculate_metrics


class PortfolioOrchestrator:
    """统一组合构建与再平衡编排器。

    把「调仓日历 → 目标权重 → 约束裁剪 → 撮合 → 成本核算 → 绩效」串成一条链路，
    取代各策略脚本中分散手写的组合构建逻辑，保证成本口径与收益锚定全仓库一致。

    Parameters
    ----------
    rebalance : str
        调仓频率 "monthly" | "weekly" | "quarterly"。
    max_leverage : float
        组合总杠杆上限（|sum(W)| 的绝对值上限），默认 1.0（纯多头满仓）。
    max_weight_per_asset : Optional[float]
        单资产权重上限（绝对值），默认 None 不限制。
    max_turnover : Optional[float]
        单次调仓换手率上限（sum|ΔW|），默认 None 不限制。
    buy_cost, sell_cost : float
        买入/卖出单边费率，默认铁律标准（买 0.026% / 卖 0.076%）。
    """

    def __init__(
        self,
        rebalance: str = "monthly",
        max_leverage: float = 1.0,
        max_weight_per_asset: Optional[float] = None,
        max_turnover: Optional[float] = None,
        buy_cost: float = 0.00026,
        sell_cost: float = 0.00076,
    ):
        self.rebalance = rebalance
        self.max_leverage = max_leverage
        self.max_weight_per_asset = max_weight_per_asset
        self.max_turnover = max_turnover
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost

    # ── 调仓日历 ──────────────────────────────────────────
    def rebalance_dates(self, dates_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """提取调仓日（每月/周/季最后一个交易日）。"""
        if self.rebalance == "monthly":
            groups = dates_index.to_series().groupby([dates_index.year, dates_index.month])
        elif self.rebalance == "weekly":
            groups = dates_index.to_series().groupby([
                dates_index.isocalendar().year.values,
                dates_index.isocalendar().week.values,
            ])
        elif self.rebalance == "quarterly":
            groups = dates_index.to_series().groupby([dates_index.year, dates_index.quarter])
        else:
            raise ValueError(f"不支持的调仓频率: {self.rebalance}")
        return pd.DatetimeIndex(groups.last().sort_index().values)

    # ── 约束裁剪 ──────────────────────────────────────────
    def apply_constraints(self, target_weights: pd.Series) -> pd.Series:
        """对单日目标权重施加杠杆上限与单资产上限约束。

        约束逻辑：
          1. 单资产上限：|w_i| 裁剪到 max_weight_per_asset；
          2. 杠杆上限：若 sum|W| > max_leverage，按比例缩放。
        """
        w = target_weights.copy()

        if self.max_weight_per_asset is not None:
            cap = self.max_weight_per_asset
            w = w.clip(lower=-cap, upper=cap)

        gross = w.abs().sum()
        if gross > self.max_leverage and gross > 0:
            w = w * (self.max_leverage / gross)

        return w

    # ── 撮合：受涨跌停/停牌约束的成交 ──────────────────────
    def execute(
        self,
        target_weights: pd.Series,
        prev_weights: pd.Series,
        untradeable: Optional[pd.Series] = None,
    ) -> pd.Series:
        """从目标权重到实际持仓的撮合。

        untradeable 为布尔掩码（True = 该资产当日不可交易，如一字涨停/跌停/停牌），
        这些资产当日维持 prev_weights（无法成交）。
        """
        w = target_weights.copy()
        if untradeable is not None:
            untradeable = untradeable.reindex(w.index).fillna(False)
            w = w.mask(untradeable, prev_weights.reindex(w.index).fillna(0.0))
        return w

    # ── 主流程 ────────────────────────────────────────────
    def run(
        self,
        target_weights_fn: Callable[[pd.Timestamp], pd.Series],
        close_matrix: pd.DataFrame,
        tradeable_mask: Optional[pd.DataFrame] = None,
        return_weights: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """执行完整编排流程。

        Parameters
        ----------
        target_weights_fn : Callable[[Timestamp], Series]
            给定调仓日，返回该日目标权重向量（symbol -> weight）。
        close_matrix : pd.DataFrame
            收盘价矩阵（index=date, columns=symbols）。
        tradeable_mask : Optional[pd.DataFrame]
            逐日逐资产可交易掩码（True=可交易）。None 表示全部可交易。
        return_weights : bool
            为 True 时返回 (result_df, W_held)。

        Returns
        -------
        pd.DataFrame
            columns: gross_ret, cost, turnover, port_ret, cum
        """
        # 收益锚定（与全仓库一致：t→t+1 收益记在 t+1）
        daily_ret = close_matrix.pct_change()

        dates = close_matrix.index
        symbols = close_matrix.columns
        rb_dates = self.rebalance_dates(dates)

        W_held = pd.DataFrame(0.0, index=dates, columns=symbols)

        prev_w = pd.Series(0.0, index=symbols)
        for i, d in enumerate(dates):
            if d in rb_dates:
                # 调仓日：生成目标权重 -> 约束 -> 撮合
                target = target_weights_fn(d).reindex(symbols).fillna(0.0)
                target = self.apply_constraints(target)
                if tradeable_mask is not None and d in tradeable_mask.index:
                    untradeable = ~tradeable_mask.loc[d].reindex(symbols).fillna(True)
                else:
                    untradeable = None
                executed = self.execute(target, prev_w, untradeable)
            else:
                # 非调仓日：维持前一持仓
                executed = prev_w

            W_held.loc[d] = executed.values
            prev_w = executed

        # 组合收益 = W[t-1] · daily_ret[t]
        W_lag = W_held.shift(1).fillna(0.0)
        gross_ret = (W_lag * daily_ret).sum(axis=1)

        # 换手成本（方向分离）
        delta_w = W_held - W_held.shift(1).fillna(0.0)
        turnover = delta_w.abs().sum(axis=1)
        buy_turnover = delta_w.clip(lower=0.0).sum(axis=1)
        sell_turnover = (-delta_w).clip(lower=0.0).sum(axis=1)
        cost = buy_turnover * self.buy_cost + sell_turnover * self.sell_cost

        port_ret = gross_ret - cost
        cum = (1.0 + port_ret).cumprod()

        result = pd.DataFrame({
            "gross_ret": gross_ret,
            "cost": cost,
            "turnover": turnover,
            "port_ret": port_ret,
            "cum": cum,
        })

        if return_weights:
            return result, W_held
        return result
