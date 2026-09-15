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
        max_leverage=2.0,            # LS 组合 Σ|W| 约为 2.0；默认 1.0 会被缩放到半仓
        max_weight_per_asset=0.10,
        buy_cost=0.00026,
        sell_cost=0.00076,
    )
    result = orch.run(target_weights_fn, close_matrix, ...)

    ⚠️ max_turnover 尚未实现（传入即 raise）。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest.invariants import (  # 输入契约断言（零项目内依赖）
    assert_cost_params,
    assert_price_panel,
    assert_result_sane,
    assert_weight_matrix,
)
from risk.cost_model import BUY_COST, SELL_COST  # 费率单一真源（2026-09-09 迁移）
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
        组合**总杠杆**上限（`Σ|W|`，即 gross leverage），默认 1.0（纯多头满仓）。
        ⚠️ 注意与「净敞口 `|ΣW|`」的区别：多空对冲组合的 `Σ|W|` 约为两腿之和，
        各 1.0 时合计 2.0。因此 LS 策略须显式传 `max_leverage=2.0`，
        否则会被按比例缩放到半个仓位（约束是静默生效的，不会报错）。
    max_weight_per_asset : Optional[float]
        单资产权重上限（绝对值），默认 None 不限制。
    max_turnover : Optional[float]
       单次调仓换手率上限（sum|ΔW|）。**当前未实现** —— 传入非 None 会立即
       `raise NotImplementedError`。此前该参数被声明、写进 docstring 示例、
       存为 `self.max_turnover`，却在 `apply_constraints` / `run` 中**从未被读取**，
       传了等于没传且毫无提示（详见 CLAUDE.md TODO 18）。
    buy_cost, sell_cost : float
        买入/卖出单边费率，默认铁律标准（买 0.026% / 卖 0.076%）。
    """

    def __init__(
        self,
        rebalance: str = "monthly",
        max_leverage: float = 1.0,
        max_weight_per_asset: Optional[float] = None,
        max_turnover: Optional[float] = None,
        buy_cost: float = BUY_COST,
        sell_cost: float = SELL_COST,
        slippage: float = 0.0,
    ):
        self.rebalance = rebalance
        self.max_leverage = max_leverage
        self.max_weight_per_asset = max_weight_per_asset
        if max_turnover is not None:
            # 唯一真源：CLAUDE.md TODO 18 —— 该参数此前被静默忽略（声明 + 文档示例 +
            # 存 self，但 apply_constraints / run 从不读取）。按工程铁律 E1，
            # 不确定的行为必须**响亮地失败**，而不是静默放行。
            raise NotImplementedError(
                "max_turnover（单次调仓换手率上限）尚未实现；此前该参数被静默忽略。"
                "请勿传入（保持 None），或先实现裁剪逻辑再移除本守卫。见 CLAUDE.md TODO 18。"
            )
        self.max_turnover: None = None  # 恒为 None —— 上限未实现
        self.buy_cost = buy_cost
        self.sell_cost = sell_cost
        self.slippage = slippage

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

        # ── 输入契约断言（事前拦截，失败直接中断而非降级成日志）──
        assert_price_panel(close_matrix)
        assert_cost_params(self.buy_cost, self.sell_cost)

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

        # 持仓契约：总杠杆与单资产上限（apply_constraints 已强制，此处自证）
        assert_weight_matrix(
            W_held,
            max_gross=self.max_leverage,
            max_weight_per_asset=self.max_weight_per_asset,
            name="W_held",
        )

        # 组合收益 = W[t-1] · daily_ret[t]
        W_lag = W_held.shift(1).fillna(0.0)
        gross_ret = (W_lag * daily_ret).sum(axis=1)

        # 换手成本（方向分离）
        delta_w = W_held - W_held.shift(1).fillna(0.0)
        turnover = delta_w.abs().sum(axis=1)
        buy_turnover = delta_w.clip(lower=0.0).sum(axis=1)
        sell_turnover = (-delta_w).clip(lower=0.0).sum(axis=1)
        cost = buy_turnover * self.buy_cost + sell_turnover * self.sell_cost
        if self.slippage:
            # 滑点按**总换手**计提（单边费率口径）：|ΔW| = 买入换手 + 卖出换手
            cost = cost + turnover * self.slippage

        port_ret = gross_ret - cost
        cum = (1.0 + port_ret).cumprod()

        result = pd.DataFrame({
            "gross_ret": gross_ret,
            "cost": cost,
            "turnover": turnover,
            "port_ret": port_ret,
            "cum": cum,
        })

        # 输出端契约：结果合理性（只断言必然错误，不判断指标好坏）
        assert_result_sane(result)

        if return_weights:
            return result, W_held
        return result
