"""
strategies/all_weather_risk_parity/optimizer.py — 多资产配置与等风险贡献（Risk Parity）凸优化求解器。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_risk_contributions(weights: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    """计算给定权重与协方差矩阵下，各资产的总风险贡献 (TRC) 占组合波动的比例。

    Returns
    -------
    np.ndarray
        各资产的风险贡献占比 (sum = 1.0)。
    """
    w = np.asarray(weights)
    cov = np.asarray(cov_matrix)
    port_var = float(w.T @ cov @ w)
    if port_var <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    port_vol = np.sqrt(port_var)
    mrc = (cov @ w) / port_vol  # 边际风险贡献
    trc = w * mrc               # 总风险贡献
    trc_ratio = trc / port_vol  # 占比
    return trc_ratio


def solve_equal_risk_contribution(
    cov_matrix: np.ndarray,
    risk_budgets: Optional[np.ndarray] = None,
    tol: float = 1e-8,
) -> np.ndarray:
    """求解等风险贡献（Equal Risk Contribution / Risk Parity）最优权重。

    Parameters
    ----------
    cov_matrix : np.ndarray
        资产收益率协方差矩阵 (N x N)。
    risk_budgets : Optional[np.ndarray]
        目标风险贡献占比向量 (sum = 1.0)。默认 None 时为标准等风险贡献 (b_i = 1/N)。
    tol : float
        优化容差。

    Returns
    -------
    np.ndarray
        最优多头权重向量 w (w_i >= 0, sum(w_i) = 1.0)。
    """
    cov = np.asarray(cov_matrix)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    if risk_budgets is None:
        b = np.full(n, 1.0 / n)
    else:
        b = np.asarray(risk_budgets)
        b = b / np.sum(b)

    # 初始点：使用波动率倒数加权
    vols = np.sqrt(np.diag(cov))
    vols[vols <= 1e-8] = 1e-4
    w0 = (1.0 / vols) / np.sum(1.0 / vols)

    def objective(w: np.ndarray) -> float:
        port_var = float(w.T @ cov @ w)
        if port_var <= 1e-14:
            return 1.0
        # 各资产风险贡献: w_i * (cov @ w)_i
        mrc_times_w = w * (cov @ w)
        # 目标: mrc_times_w_i / port_var == b_i
        diffs = (mrc_times_w / port_var) - b
        return float(np.sum(diffs ** 2))

    bounds = [(0.0, 1.0) for _ in range(n)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=tol,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    if res.success:
        w_opt = res.x
        w_opt[w_opt < 0] = 0.0
        return w_opt / np.sum(w_opt)
    else:
        # 回退使用波动率倒数加权
        return w0


def optimize_inverse_volatility(cov_matrix: np.ndarray) -> np.ndarray:
    """波动率倒数加权（启发式加权）。"""
    cov = np.asarray(cov_matrix)
    vols = np.sqrt(np.diag(cov))
    vols[vols <= 1e-8] = 1e-4
    inv_v = 1.0 / vols
    return inv_v / np.sum(inv_v)


def optimize_60_40_blend(asset_symbols: List[str]) -> np.ndarray:
    """经典股债 60/40 组合：权益类资产分配 60% 权重，固定收益/黄金/现金类分配 40% 权重。"""
    n = len(asset_symbols)
    weights = np.zeros(n)

    equity_syms = ["510300", "513100", "510500"]
    non_equity_syms = ["511010", "518880", "159980", "511880"]

    eq_indices = [i for i, s in enumerate(asset_symbols) if s in equity_syms]
    non_eq_indices = [i for i, s in enumerate(asset_symbols) if s in non_equity_syms]

    if eq_indices and non_eq_indices:
        weights[eq_indices] = 0.60 / len(eq_indices)
        weights[non_eq_indices] = 0.40 / len(non_eq_indices)
    elif eq_indices:
        weights[eq_indices] = 1.0 / len(eq_indices)
    elif non_eq_indices:
        weights[non_eq_indices] = 1.0 / len(non_eq_indices)
    else:
        weights[:] = 1.0 / n

    return weights / np.sum(weights)


def determine_macro_risk_budgets(
    close_subset: pd.DataFrame,
    asset_symbols: List[str],
) -> np.ndarray:
    """宏观四象限自适应风险预算调整器。

    根据资产趋势与大类相对强弱，自适应调节目标风险预算 b_i。
    """
    n = len(asset_symbols)
    b = np.ones(n)

    for i, sym in enumerate(asset_symbols):
        if sym in close_subset.columns:
            s = close_subset[sym].dropna()
            if len(s) >= 20:
                ma20 = s.rolling(20).mean().iloc[-1]
                cur = s.iloc[-1]
                # 若资产处于 20日均线上升趋势，给予更多风险预算；下行趋势压低风险预算
                if cur > ma20:
                    b[i] = 1.5
                else:
                    b[i] = 0.5

    # 结构性加权：在下行期提高国债 (511010) 与黄金 (518880) 的基础风险配额
    for i, sym in enumerate(asset_symbols):
        if sym in ["511010", "518880"]:
            b[i] *= 1.5

    return b / np.sum(b)
