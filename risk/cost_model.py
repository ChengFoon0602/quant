"""
risk/cost_model.py — 交易摩擦成本单一真源（Single Source of Truth）。

存在理由
--------
2026-09 事故：`models/portfolio_backtest.py` 的 `COST_BPS = 0.003`（双边 0.3%）是
铁律 0.1% 的 3 倍，导致「多空 alpha 不成立」的结论被压成错误结果（LS 0.036 → 1.175）。

根因不是某个人写错数字，而是**成本参数在 8 处硬编码、无单一真源**：

    backtest/engine.py:22-23            risk/orchestrator.py:23-24,65-66
    backtest/cross_section.py:27-28     models/evaluate.py:25-28
    risk/portfolio.py:80-81             models/portfolio_backtest.py:50-52,117-118
    strategies/etf_momentum_crowding/report.py（ETF 用 0.0004，属正当差异）

本模块把铁律第 3 条固化为可导入的常量，供 `tests/test_cost_model.py` 断言各处默认值
未发生漂移。**新增回测代码应引用本模块常量，而非重新书写字面量。**

费率口径（2026 年 A 股标准，与 CLAUDE.md 铁律第 3 条一致）
----------------------------------------------------------
- 买入 0.026%：佣金万 2.5（0.025%）+ 过户费（0.001%）
- 卖出 0.076%：佣金万 2.5 + 印花税 0.05% + 过户费
- 双边合计 ≈ 0.102%
- 滑点：日线级别默认 0.05%（另计，不并入 ROUND_TRIP）
"""

from __future__ import annotations

from dataclasses import dataclass

# ── 铁律费率常量（唯一定义处）──────────────────────────────
BUY_COST: float = 0.00026   # 买入单边：佣金 + 过户费
SELL_COST: float = 0.00076  # 卖出单边：佣金 + 印花税 + 过户费
SLIPPAGE: float = 0.0005    # 日线级别默认滑点（另计）
ROUND_TRIP: float = BUY_COST + SELL_COST  # 双边合计 ≈ 0.00102

# 已知的正当偏离（不属于 bug，测试需豁免）
# ETF / 低摩擦资产走 build_weight_portfolio(cost=...) 对半拆口径
ETF_ROUND_TRIP: float = 0.0004


@dataclass(frozen=True)
class CostModel:
    """不可变成本模型，便于把费率作为参数整体传递。

    Examples
    --------
    >>> cm = CostModel()
    >>> cm.round_trip == ROUND_TRIP
    True
    >>> cm.deduction(buy_turnover=1.0, sell_turnover=1.0) == ROUND_TRIP
    True
    """

    buy_cost: float = BUY_COST
    sell_cost: float = SELL_COST
    slippage: float = SLIPPAGE

    @property
    def round_trip(self) -> float:
        """双边合计成本（不含滑点）。"""
        return self.buy_cost + self.sell_cost

    def deduction(self, buy_turnover: float, sell_turnover: float) -> float:
        """按买入/卖出方向分别计提的成本额。

        Parameters
        ----------
        buy_turnover : float
            当日增仓换手（Σ max(ΔW, 0)）。
        sell_turnover : float
            当日减仓换手（Σ max(-ΔW, 0)）。
        """
        return buy_turnover * self.buy_cost + sell_turnover * self.sell_cost

    def with_slippage(self, turnover: float) -> float:
        """在方向成本之外叠加滑点（按总换手计提）。"""
        return turnover * self.slippage
