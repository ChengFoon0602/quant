"""
risk — 风险控制、组合持仓、交易限制与市场结构/拥挤度分析模块。
"""

from risk.portfolio import (
    build_weight_portfolio,
    detect_limit_moves,
    apply_volatility_target,
    calculate_metrics,
    bootstrap_sharpe_test,
)

from risk.orchestrator import PortfolioOrchestrator

from risk.crowding import (
    month_end_dates,
    wide_to_long,
    align_direction,
    factor_exposure_extreme_ratio,
    factor_monthly_returns,
    style_homogeneity,
    turnover_crowding,
    factor_return_spike,
    compute_crowding_indicators,
    compute_composite_crowding,
    detect_extreme_events,
)

__all__ = [
    "build_weight_portfolio",
    "detect_limit_moves",
    "apply_volatility_target",
    "calculate_metrics",
    "bootstrap_sharpe_test",
    "PortfolioOrchestrator",
    "month_end_dates",
    "wide_to_long",
    "align_direction",
    "factor_exposure_extreme_ratio",
    "factor_monthly_returns",
    "style_homogeneity",
    "turnover_crowding",
    "factor_return_spike",
    "compute_crowding_indicators",
    "compute_composite_crowding",
    "detect_extreme_events",
]
